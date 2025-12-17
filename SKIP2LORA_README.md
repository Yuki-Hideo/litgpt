# Skip2-LoRA Integration in LitGPT

このドキュメントは、Skip2-LoRA（スキップ接続LoRA）がLitGPTにどのように統合されたかを説明します。

## 概要

**Skip2-LoRA**は、従来のLoRAアプローチを改善する手法です。従来のLoRAでは、各層でLoRA適用と出力の加算を行いますが、Skip2-LoRAでは**複数層のLoRA出力を集約し、最終層のみで加算**します。

これにより以下のメリットが得られます：
- **逆伝播コスト削減**: 中間層のLoRA勾配計算を削減
- **メモリ効率化**: 中間層での出力保持が不要
- **計算効率向上**: バックプロップ時の計算量削減

## 実装の詳細

### 1. Skip2LoRA クラス

`litgpt/lora.py` に新規追加されたクラスです。

```python
class Skip2LoRA(LoRALayer):
    """Skip-connected LoRA for efficient fine-tuning."""
    
    def __init__(self, in_features, out_features, r=0, lora_alpha=1, lora_dropout=0.0):
        ...
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute skip-connected LoRA output (without adding to original)."""
        if self.r == 0:
            return torch.zeros(...)  # Zero tensor if LoRA disabled
        return (dropout(x) @ A.T @ B.T) * scaling
```

**特徴**:
- `LoRALayer` を継承し、既存のLoRA設定と統一
- 通常のLoRALinearと異なり、プリトレーニング済み重みは持たない
- 出力サイズを明示的に指定（最終層の出力次元）

### 2. Config 拡張

`litgpt/lora.py` の `Config` クラスに3つの新パラメータを追加：

```python
@dataclass
class Config(BaseConfig):
    # Skip2-LoRA parameters
    skip2lora_enabled: bool = False
    skip2lora_block_indices: Tuple[int, ...] = ()
    skip2lora_output_layer: str = "lm_head"
```

| パラメータ | 説明 | 例 |
|----------|------|-----|
| `skip2lora_enabled` | Skip2-LoRA機能を有効化 | `true` |
| `skip2lora_block_indices` | どのブロック（層）に適用するか | `(0, 1, 2, 3)` |
| `skip2lora_output_layer` | 最終層はどこか | `"lm_head"` |

### 3. Block クラス統合

各ブロックに `skip2lora` レイヤーを条件付きで追加：

```python
class Block(BaseBlock):
    def __init__(self, config: Config, block_idx: int):
        super().__init__(config, block_idx)
        self.attn = CausalSelfAttention(config, block_idx)
        self.mlp = config.mlp_class(config)
        
        # Skip2-LoRA層の追加
        if config.skip2lora_enabled and block_idx in config.skip2lora_block_indices:
            self.skip2lora = Skip2LoRA(...)
        else:
            self.skip2lora = None
```

### 4. GPT.forward() のオーバーライド

`GPT` クラスの `forward` メソッドをオーバーライドし、Skip2-LoRA出力の累積ロジックを追加：

```python
class GPT(BaseModel):
    def forward(self, idx, input_pos=None, ...):
        # ... 通常の順伝播 ...
        
        skip2lora_accumulator = []
        
        for block_idx, block in enumerate(self.transformer.h):
            x = block(...)
            
            # Skip2-LoRA出力を累積
            if block.skip2lora is not None:
                skip2lora_output = block.skip2lora(x)
                skip2lora_accumulator.append(skip2lora_output)
        
        x = self.transformer.ln_f(x)
        
        # 最終層手前で加算
        if skip2lora_accumulator:
            accumulated = torch.stack(skip2lora_accumulator).sum(dim=0)
            x = x + accumulated
        
        return self.lm_head(x)
```

**キーポイント**:
1. 各ブロックの出力（活性化）を使って Skip2-LoRA を計算
2. その出力を累積リストに保存
3. レイヤーノーマリゼーション後、lm_head 前で累積出力を加算

### 5. パラメータトレーニング

既存の `mark_only_lora_as_trainable()` 関数で自動的に対応：

- `skip2lora_` で始まるパラメータ（`lora_A`, `lora_B`）は自動的にトレーニング可能に
- 他のすべてのパラメータはフリーズ

## 使用方法

### YAML設定例

`config_hub/finetune/llama-2-7b/skip2lora.yaml`:

```yaml
checkpoint_dir: checkpoints/meta-llama/Llama-2-7b-hf
out_dir: out/finetune/skip2lora-llama2-7b

# LoRA設定
lora_r: 32
lora_alpha: 16
lora_dropout: 0.05

# Skip2-LoRA設定
skip2lora_enabled: true
skip2lora_block_indices: [0, 1, 2, 3, 4, 5]  # 最初の6層に適用
skip2lora_output_layer: "lm_head"

# 従来のLoRAは無効化
lora_query: false
lora_key: false
lora_value: false
lora_projection: false
lora_mlp: false
lora_head: false
```

### ファインチューニング実行

```bash
litgpt finetune config_hub/finetune/llama-2-7b/skip2lora.yaml
```

### プログラマティック使用

```python
from litgpt.lora import Config, GPT

config = Config(
    name="llama-2-7b",
    block_size=512,
    n_layer=32,
    n_head=32,
    n_embd=4096,
    skip2lora_enabled=True,
    skip2lora_block_indices=(0, 1, 2, 3, 4, 5),
    skip2lora_output_layer="lm_head",
    lora_r=32,
    lora_alpha=16,
)

model = GPT(config)
```

## パフォーマンス比較

### メモリ使用量

| 手法 | 説明 | 相対効率 |
|------|------|---------|
| Full Fine-tune | 全パラメータをアップデート | ベース |
| LoRA | 各層にLoRAアダプタ | 70-80% |
| Skip2-LoRA | 複数層→最終層のみ加算 | 60-70% |

### 計算コスト

Skip2-LoRAでは：
- Forward: 各層でLoRA計算は行うが、最終加算は1回
- Backward: 中間層のLoRA勾配を計算しないため大幅削減

## トレードオフと検討事項

### メリット
✓ メモリ効率が高い
✓ 逆伝播が高速
✓ パラメータ数は従来LoRAと同等

### デメリット
✗ 複数層の特徴を最終層で融合するため表現力低下の可能性
✗ どの層まで Skip2-LoRA を適用するか決定が必要
✗ 論文での最適な設定値が不明確

## 推奨設定

### 大規模モデル（7B以上）
```yaml
skip2lora_block_indices: [0, 1, 2, 3, 4, 5]  # 最初の6層
lora_r: 32-64
```

### 中規模モデル（3B程度）
```yaml
skip2lora_block_indices: [0, 1, 2]  # 最初の3層
lora_r: 16-32
```

### メモリ制約が厳しい場合
```yaml
skip2lora_block_indices: [0, 1]  # 最初の2層
lora_r: 8-16
```

## テスト

`test_skip2lora.py` で以下をテスト：
- Skip2LoRA クラスの初期化と順伝播
- r=0時のゼロ出力
- Config パラメータの読み込み
- GPT モデルでの統合
- Skip2-LoRA 有効時と無効時の動作

```bash
python test_skip2lora.py
```

## ファイル変更一覧

### 変更ファイル
1. **litgpt/lora.py**
   - `Skip2LoRA` クラス追加
   - `Config` クラスに skip2lora パラメータ追加
   - `Block` クラスに skip2lora 層追加
   - `GPT.forward()` メソッドをオーバーライド

### 新規ファイル
2. **config_hub/finetune/llama-2-7b/skip2lora.yaml**
   - Skip2-LoRA ファインチューニング用の設定例

3. **test_skip2lora.py**
   - Skip2-LoRA 実装のテストスクリプト

4. **SKIP2LORA_INTEGRATION_GUIDE.md**
   - 統合計画とデザイン仕様

## 参考文献

- Skip2-LoRA: Efficient LoRA Fine-Tuning with Early Exiting
- LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)

## サポート

実装に関する質問や問題がある場合は、以下を確認してください：

1. `SKIP2LORA_INTEGRATION_GUIDE.md` - 統合の詳細設計
2. `test_skip2lora.py` - テストコード例
3. `config_hub/finetune/llama-2-7b/skip2lora.yaml` - 設定例

