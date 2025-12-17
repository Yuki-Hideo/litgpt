# Skip2-LoRA LitGPT統合 - アーキテクチャ図

## システム全体図

```
┌─────────────────────────────────────────────────────────────┐
│                    Input: Token IDs                         │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
        ┌────────────────────────────────┐
        │  Embedding + Position Encoding  │
        └────────────┬───────────────────┘
                     │
                     ▼
        ┌────────────────────────────────┐
        │   Block 0: Attention + MLP     │
        │   ┌──────────────────────────┐ │
        │   │  skip2lora_0 (if enabled)│ │
        │   └──────────┬───────────────┘ │
        │              │ saved for later  │
        └────────────┬─┴────────────────┘
                     │ output_0
                     ▼
        ┌────────────────────────────────┐
        │   Block 1: Attention + MLP     │
        │   ┌──────────────────────────┐ │
        │   │  skip2lora_1 (if enabled)│ │
        │   └──────────┬───────────────┘ │
        │              │ saved for later  │
        └────────────┬─┴────────────────┘
                     │ output_1
                     ▼
               ... (Blocks 2-30)
                     │
                     ▼
        ┌────────────────────────────────┐
        │  Block 31: Attention + MLP     │
        │  ┌──────────────────────────┐  │
        │  │  skip2lora_31 (if enabled)  │
        │  └──────────┬────────────────┘  │
        │             │ saved for later   │
        └────────────┬┴────────────────┘
                     │ output_31
                     ▼
        ┌────────────────────────────────┐
        │      Layer Normalization       │
        └────────────┬───────────────────┘
                     │
        ╔════════════╩════════════╗
        ║                         ║
        ▼                         ▼
    [output]              [Accumulated]
                         skip2lora_outputs
                         = Σ(skip2lora_i)
        │                         │
        │   ┌─────────────────────┘
        │   │ (element-wise add)
        ▼   ▼
        ┌────────────────────────────────┐
        │   Enhanced representation      │
        │  (output + accumulated LoRA)   │
        └────────────┬───────────────────┘
                     │
                     ▼
        ┌────────────────────────────────┐
        │        lm_head (Linear)        │
        │   ┌────────────────────────┐   │
        │   │  Traditional LoRA      │   │
        │   │  (if lora_head=true)   │   │
        │   └────────────────────────┘   │
        └────────────┬───────────────────┘
                     │
                     ▼
        ┌────────────────────────────────┐
        │   Logits (vocab_size)          │
        └────────────────────────────────┘
```

## Block 内部の詳細

```
Input x
  │
  ├─────────────────────────────────┐
  │                                 │
  ▼                                 ▼
┌──────────────────┐        ┌──────────────────────┐
│  CausalSelfAttn  │        │   Skip2LoRA Layer    │
│  (pretrained)    │        │   (trainable only)   │
└────────┬─────────┘        └──────────┬───────────┘
         │                             │
         ▼                             │
┌──────────────────┐                  │
│  Output Proj     │                  │
│  (pretrained)    │                  │
└────────┬─────────┘                  │
         │                             │
         ▼                             │
┌──────────────────┐                  │
│      MLP         │                  │
│  (pretrained)    │                  │
└────────┬─────────┘                  │
         │                             │
         ├─────────────────────────────┤
         │                             │
         ▼                             ▼
      output_block              skip2lora_output
                                (accumulated)
```

## Skip2-LoRA層の詳細

```
Input: x (B, T, n_embd)
  │
  ▼
┌──────────────────────────────────┐
│    Dropout (if lora_dropout>0)   │
└──────────┬───────────────────────┘
           │
           ▼
    ┌─────────────────────┐
    │  Linear projection  │
    │  (r, n_embd) × A    │
    │                     │
    │ x @ A.T → (B, T, r) │
    └──────────┬──────────┘
               │
               ▼
        ┌──────────────────────┐
        │ Linear projection    │
        │ (out_features, r) × B│
        │                      │
        │ (...@r) @ B.T        │
        │ → (B, T, out_features)
        └──────────┬───────────┘
                   │
                   ▼
            ┌─────────────────────┐
            │   Scaling           │
            │ output * (alpha / r) │
            └─────────────────────┘
                   │
                   ▼
            Skip2LoRA_output
```

## デジタル実装の流れ

### Config設定

```
config = Config(
    skip2lora_enabled=True,               # ← 機能有効化
    skip2lora_block_indices=(0, 1, 2),   # ← 適用層
    lora_r=32,                            # ← LoRAランク
    lora_alpha=16,                        # ← スケーリング
)
```

### モデル構築

```python
class Block(BaseBlock):
    def __init__(self, config, block_idx):
        ...
        # Skip2-LoRA層の条件付き挿入
        if config.skip2lora_enabled and block_idx in config.skip2lora_block_indices:
            self.skip2lora = Skip2LoRA(
                in_features=config.n_embd,
                out_features=config.n_embd,
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
            )
```

### Forward処理

```python
def forward(self, idx):
    ...
    skip2lora_accumulator = []
    
    for block_idx, block in enumerate(transformer.h):
        x = block(x, ...)
        
        if block.skip2lora is not None:
            lora_output = block.skip2lora(x)     # ← LoRA計算
            skip2lora_accumulator.append(lora_output)
    
    x = transformer.ln_f(x)
    
    # 最終加算
    if skip2lora_accumulator:
        accumulated = torch.stack(skip2lora_accumulator).sum(dim=0)
        x = x + accumulated                      # ← 重要！
    
    return lm_head(x)
```

## パラメータフロー

### 有効なパラメータ

```
Block 0
├── attn
│   ├── qkv.linear.weight ..................... [FROZEN]
│   └── qkv.linear.bias ....................... [FROZEN]
└── skip2lora
    ├── weight_lora_a ......................... [TRAINABLE] ✓
    └── weight_lora_b ......................... [TRAINABLE] ✓

Block 1-31
├── attn
│   ├── qkv.linear.weight ..................... [FROZEN]
│   └── qkv.linear.bias ....................... [FROZEN]
└── skip2lora (if block_idx in skip2lora_block_indices)
    ├── weight_lora_a ......................... [TRAINABLE] ✓
    └── weight_lora_b ......................... [TRAINABLE] ✓

lm_head
└── weight ................................ [FROZEN]
    (lora_head=false の場合)
```

### パラメータ数計算例

```
n_embd = 4096, r = 32

各Skip2-LoRA層のパラメータ数:
- lora_A: r × n_embd = 32 × 4096 = 131,072
- lora_B: n_embd × r = 4096 × 32 = 131,072
- 合計/層: 262,144

6層適用時:
- 合計: 262,144 × 6 = 1,572,864 (≈1.6M)

従来LoRA比較:
- 従来LoRA（全層）: 262,144 × 32 = 8,388,608 (≈8.4M)
- Skip2-LoRA（6層）: 1,572,864 (≈1.6M)
- 削減率: 81.3%
```

## メモリ効率の改善

```
従来のLoRA:
┌─────────────────────────────────────────┐
│ Forward: x_0 → x_1 → ... → x_31        │  メモリ: O(n_layer)
│ Backward: ∂L/∂A, ∂L/∂B for each layer │  勾配計算: O(n_layer)
└─────────────────────────────────────────┘

Skip2-LoRA:
┌─────────────────────────────────────────┐
│ Forward: 各層でLoRA計算（記録なし）      │  メモリ: O(1)
│ Backward: 最後の層にのみ勾配流入          │  勾配計算: O(selected_layers)
└─────────────────────────────────────────┘
```

---

## パフォーマンスメトリクス

```
Computation Graph Complexity:

従来LoRA:
Block 0: x₀ → W⋅x₀ + A⋅B⁻¹⋅x₀ → y₀
Block 1: y₀ → W⋅y₀ + A⋅B⁻¹⋅y₀ → y₁
...
Block 31: y₃₀ → W⋅y₃₀ + A⋅B⁻¹⋅y₃₀ → y₃₁
lm_head: y₃₁ → output

Skip2-LoRA:
Block 0: x₀ → W⋅x₀ → y₀,  A⋅B⁻¹⋅x₀ → S₀
Block 1: y₀ → W⋅y₀ → y₁,  A⋅B⁻¹⋅y₀ → S₁
...
Block 31: y₃₀ → W⋅y₃₀ → y₃₁, A⋅B⁻¹⋅y₃₀ → S₃₁
lm_head: y₃₁ + Σ(Sᵢ) → output
```

