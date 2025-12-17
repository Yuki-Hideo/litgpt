# Skip2-LoRA の LitGPT 統合計画

## 概要

Skip2-LoRAは、従来のLoRAの各層への適用ではなく、**複数層から集約されたLoRA出力を最終層のみに加算**する手法です。
これにより以下の効果が期待できます：

- **逆伝播コスト削減**: 中間層でのLoRA勾配計算が不要
- **順伝播コスト削減**: 各層で中間計算を減らす

## 現在の実装構造

### sugiura/lora.py のSkip2-LoRA実装

```
SkipLoRA クラス
├─ in_features: 入力次元
├─ out_features: 最終層の出力次元
├─ forward(): x @ A.T @ B.T を計算（最終層の次元に投影）
└─ skip_lora_output を各層で集約

insert_skip_lora_layers()
└─ 指定層に skip_lora 属性を自動挿入
```

### LitGPT の現在のLoRA実装

```
Block
├─ CausalSelfAttention
│  ├─ qkv: LoRAQKVLinear (各層個別に適用)
│  └─ proj: LoRALinear
└─ MLP
   ├─ fc: LoRALinear
   └─ proj: LoRALinear
```

## 統合方法（推奨）

### 1. Config 拡張（litgpt/lora.py の Config クラス）

```yaml
# Skip2-LoRA 専用パラメータ
skip2lora_enabled: bool = False          # Skip2-LoRA の有効化
skip2lora_positions: List[str] = field   # Skip2-LoRA を挿入する層
                                          # 例: ["transformer.h.0", "transformer.h.1"]
skip2lora_output_layer: str = ""         # 最終層の層指定
                                          # 例: "lm_head" または "transformer.h.31"
```

### 2. Skip2LoRA クラスの実装

**LitGPTに合わせた簡潔なSkipLoRA実装**:

```python
class Skip2LoRA(LoRALayer):
    """Skip-connected LoRA: 複数層から最終層にのみLoRA出力を集約"""
    
    def __init__(self, in_features: int, out_features: int, 
                 r: int = 0, lora_alpha: int = 1, lora_dropout: float = 0.0):
        super().__init__(r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        self.in_features = in_features
        self.out_features = out_features
        
        if r > 0:
            self.lora_A = nn.Parameter(torch.empty((r, in_features)))
            self.lora_B = nn.Parameter(torch.empty((out_features, r)))
            self.scaling = lora_alpha / r
            self.reset_parameters()
    
    def reset_parameters(self) -> None:
        if hasattr(self, "lora_A"):
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Skip-connectedなLoRA出力（プリタイムしない）"""
        if self.r == 0:
            return None
        return (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) 
                @ self.lora_B.transpose(0, 1)) * self.scaling
```

### 3. Model 統合方法

**2つのアプローチがあります**：

#### アプローチA: 最小侵襲（推奨）
- 各 `Block` に `skip2lora: Optional[Skip2LoRA]` を追加
- `CausalSelfAttention` の最後で Skip2-LoRA 出力を保存
- `lm_head` に最終加算ロジックを追加

```python
class Block(BaseBlock):
    def __init__(self, config: Config, block_idx: int):
        super().__init__(config, block_idx)
        self.attn = CausalSelfAttention(config, block_idx)
        self.mlp = config.mlp_class(config)
        
        # Skip2-LoRA 層の追加
        if config.skip2lora_enabled and block_idx in config.skip2lora_block_indices:
            self.skip2lora = Skip2LoRA(
                in_features=config.n_embd,
                out_features=config.n_embd,  # 通常は変わらない
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout
            )
        else:
            self.skip2lora = None
```

#### アプローチB: 統一的な設計
- `transformer.h` にスキップ接続バッファを保持
- `forward` 時に累積

### 4. Forward ロジック

```python
# in GPT._forward_impl or similar
skip2lora_accumulator = []

for block in transformer.h:
    x = block(x)
    if block.skip2lora is not None:
        skip2lora_accumulator.append(block.skip2lora(x))

# lm_head での最終加算
if skip2lora_accumulator:
    accumulated = torch.stack(skip2lora_accumulator).sum(dim=0)
    lm_output = self.lm_head(x) + accumulated
else:
    lm_output = self.lm_head(x)
```

## 実装ステップ

### Phase 1: SkipLoRA クラス追加
1. `litgpt/lora.py` に `Skip2LoRA` クラスを追加
2. `mark_only_lora_as_trainable()` で Skip2-LoRA パラメータも対象化

### Phase 2: Config 拡張
3. `Config` クラスに Skip2-LoRA 用パラメータを追加
4. デフォルト値を設定（無効化）

### Phase 3: Block 統合
5. `Block` の `__init__` で Skip2-LoRA 層を条件付き挿入
6. Block の forward で出力を記録

### Phase 4: GPT 統合
7. `GPT._forward_impl` で Skip2-LoRA 出力の累積
8. `lm_head` での最終加算

### Phase 5: テスト＆チューニング
9. 設定ファイル例を作成
10. 動作確認

## YAML設定例

```yaml
# config_hub/finetune/llama-2-7b/skip2lora.yaml
checkpoint_dir: checkpoints/meta-llama/Llama-2-7b-hf
out_dir: out/finetune/skip2lora-llama2-7b

precision: bf16-true

lora_r: 32
lora_alpha: 16
lora_dropout: 0.05

# 従来のLoRA（無効化）
lora_query: false
lora_key: false
lora_value: false
lora_projection: false
lora_mlp: false
lora_head: false

# Skip2-LoRA 設定
skip2lora_enabled: true
skip2lora_positions: [0, 1, 2, 3, 4, 5]  # 最初の6層
skip2lora_output_layer: "lm_head"
```

## メリット＆デメリット

### メリット
- メモリ使用量削減（中間層のLoRA出力保持が不要）
- 勾配計算量削減（複数層で計算量削減）
- パラメータ数は従来LoRAと同じ

### デメリット
- 複数層の特徴を最終層で融合するため、表現力低下の可能性
- チューニングパラメータ増加（どの層まで Skip2-LoRA を適用するか）

## 関連論文

- Skip2-LoRA: "Efficient Fine-tuning with Skip-Connected LoRA"
- 逆伝播効率化: 重みを各層で更新せず、最終層でのみ更新

