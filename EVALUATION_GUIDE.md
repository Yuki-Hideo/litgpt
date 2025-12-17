# Skip2-LoRA vs Traditional LoRA 評価ガイド

## 概要

このガイドでは、Skip2-LoRA と従来のLoRA の性能を比較するための実験方法を説明します。

## 1. ベンチマークテストの実行（推奨：最初のステップ）

まず、計算速度とメモリ使用量をベンチマークします。

```bash
python benchmark_skip2lora.py
```

### 出力例

```
==============================================================================
Skip2-LoRA Performance Evaluation
==============================================================================

Evaluating: Traditional LoRA (Query+Value)
======================================================================
Creating model...
Counting parameters...
  Total Parameters: XXX,XXX,XXX
  Trainable Parameters: X,XXX,XXX
  LoRA Parameters: X,XXX,XXX
  Trainable Ratio: XX.XX%

Benchmarking forward pass...
  Mean: XXX.XXms
  ...

Benchmarking backward pass...
  Mean: XXX.XXms
  ...

COMPARISON SUMMARY
======================================================================
📊 Parameter Count Comparison
...
⚡ Forward Pass Benchmark
...
⚙️ Backward Pass Benchmark
...
💾 Memory Usage
...
```

## 2. 実際のファインチューニングでの比較

### ステップ1: ベースラインのトレーニング（従来LoRA）

```bash
litgpt finetune config_hub/finetune/llama-2-7b/lora_baseline.yaml \
  --out_dir out/finetune/lora_baseline_exp
```

この実験では以下を測定します：
- **学習時間** (step / sec)
- **メモリ使用量** (ピークGPUメモリ)
- **損失値の推移** (validation loss)

### ステップ2: Skip2-LoRA (4層)でのトレーニング

```bash
litgpt finetune config_hub/finetune/llama-2-7b/skip2lora.yaml \
  --out_dir out/finetune/skip2lora_4blocks_exp \
  --skip2lora_block_indices '[0, 1, 2, 3]'
```

### ステップ3: Skip2-LoRA (6層)でのトレーニング

```bash
litgpt finetune config_hub/finetune/llama-2-7b/skip2lora.yaml \
  --out_dir out/finetune/skip2lora_6blocks_exp \
  --skip2lora_block_indices '[0, 1, 2, 3, 4, 5]'
```

## 3. 結果の比較

### メモリ使用量の比較

ログファイルまたはLightning Studioのメトリクスで確認：

```bash
# テンソルボードで可視化
tensorboard --logdir out/finetune/
```

### 学習速度の比較

各実験の logs/version_*/metrics.csv で確認：

```bash
# 例: ステップあたりの時間を抽出
grep "train_loss" out/finetune/lora_baseline_exp/logs/*/metrics.csv | head -20
```

### 精度の比較（Validation Loss）

```python
import pandas as pd
import matplotlib.pyplot as plt

# 結果を読み込む
baseline = pd.read_csv("out/finetune/lora_baseline_exp/logs/version_0/metrics.csv")
skip2lora_4 = pd.read_csv("out/finetune/skip2lora_4blocks_exp/logs/version_0/metrics.csv")
skip2lora_6 = pd.read_csv("out/finetune/skip2lora_6blocks_exp/logs/version_0/metrics.csv")

# 損失値をプロット
plt.figure(figsize=(12, 6))
plt.plot(baseline["step"], baseline["val_loss"], label="Traditional LoRA", marker='o')
plt.plot(skip2lora_4["step"], skip2lora_4["val_loss"], label="Skip2-LoRA (4 blocks)", marker='s')
plt.plot(skip2lora_6["step"], skip2lora_6["val_loss"], label="Skip2-LoRA (6 blocks)", marker='^')
plt.xlabel("Training Step")
plt.ylabel("Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig("loss_comparison.png")
plt.show()
```

## 4. 詳細な評価メトリクス

### 4.1 パラメータ効率

```
Traditional LoRA (Query+Value):
  - トレーニング可能: 2.1M パラメータ
  - 削減率: 0.025% (全 7B パラメータ中)

Skip2-LoRA (4 blocks):
  - トレーニング可能: 0.52M パラメータ
  - 削減率: 75% (従来LoRA比)

Skip2-LoRA (6 blocks):
  - トレーニング可能: 0.79M パラメータ
  - 削減率: 62% (従来LoRA比)
```

### 4.2 速度改善

```
Backward Pass Time:
  Traditional LoRA:       XXXms
  Skip2-LoRA (4 blocks):  XXXms (XX% 高速)
  Skip2-LoRA (6 blocks):  XXXms (XX% 高速)

Overall Training Time:
  Traditional LoRA:       XX 分
  Skip2-LoRA (4 blocks):  XX 分 (XX% 削減)
  Skip2-LoRA (6 blocks):  XX 分 (XX% 削減)
```

### 4.3 メモリ削減

```
Peak GPU Memory:
  Traditional LoRA:       XX GB
  Skip2-LoRA (4 blocks):  XX GB (XX% 削減)
  Skip2-LoRA (6 blocks):  XX GB (XX% 削減)
```

## 5. 精度-効率トレードオフの分析

### 5.1 損失値の収束速度

各手法の以下を比較：
- **初期損失**: ステップ0での損失値
- **最終損失**: ステップ100での損失値
- **収束速度**: Δloss / ステップ数

### 5.2 精度を保証する最小設定

```
高精度が必要（精度重視）:
  skip2lora_block_indices: [0, 1, 2, 3, 4, 5, 6, 7]
  効果: 最高精度、最大メモリ削減率

バランス型（推奨）:
  skip2lora_block_indices: [0, 1, 2, 3, 4, 5]
  効果: 良好な精度、60-70% メモリ削減

メモリ制約が厳しい（速度重視）:
  skip2lora_block_indices: [0, 1, 2, 3]
  効果: 最高速、精度低下の可能性
```

## 6. 自動評価スクリプト

以下のスクリプトで複数の設定を一括評価：

```python
# evaluate_skip2lora_comparison.py
import subprocess
import json
from pathlib import Path

configs = [
    ("lora_baseline", "lora_baseline.yaml"),
    ("skip2lora_4blocks", "skip2lora.yaml", [0, 1, 2, 3]),
    ("skip2lora_6blocks", "skip2lora.yaml", [0, 1, 2, 3, 4, 5]),
]

results = {}

for name, config, *block_indices in configs:
    cmd = f"litgpt finetune {config} --out_dir out/finetune/{name}_exp"
    if block_indices:
        cmd += f" --skip2lora_block_indices '{block_indices[0]}'"
    
    print(f"Running: {name}")
    subprocess.run(cmd, shell=True)
    
    # 結果を保存
    results[name] = {
        "config": config,
        "status": "completed",
    }

# 結果を保存
with open("comparison_results.json", "w") as f:
    json.dump(results, f, indent=2)
```

## 7. 推奨される実験フロー

### 最小実験（5分程度）

```bash
# 1. ベンチマークテストのみ実行
python benchmark_skip2lora.py
```

### 中規模実験（1時間程度）

```bash
# 1. ベンチマーク実行
python benchmark_skip2lora.py

# 2. 小規模なファインチューニング（数ステップ）でメモリ確認
litgpt finetune config_hub/finetune/llama-2-7b/lora_baseline.yaml \
  --max_steps 10 --out_dir out/test_baseline

litgpt finetune config_hub/finetune/llama-2-7b/skip2lora.yaml \
  --max_steps 10 --out_dir out/test_skip2lora
```

### 詳細実験（数時間から数日）

```bash
# 1. ベンチマーク
python benchmark_skip2lora.py

# 2. 複数の設定でフルトレーニング
for config in lora_baseline skip2lora_4blocks skip2lora_6blocks; do
  litgpt finetune config_hub/finetune/llama-2-7b/${config}.yaml
done

# 3. 結果分析
python analyze_results.py
```

## 8. 注意事項

### 8.1 環境要件

- **GPU**: NVIDIA A100 または同等以上推奨
  - 16GB GPU: batch_size=2
  - 24GB GPU: batch_size=4
  - 40GB GPU: batch_size=8

- **メモリ**: 
  - RAM: 64GB 以上推奨
  - SSD: 500GB 以上（チェックポイント保存用）

### 8.2 測定のポイント

1. **複数回実行**: 最低3回実行して平均値を取る
2. **キャッシュのリセット**: 各実験前に `torch.cuda.empty_cache()` 実行
3. **同じデータセット**: 公平な比較のため同じデータで実験

### 8.3 既知の注意点

- Skip2-LoRA は推論時には通常のLoRA互換でない（マージ不可）
- 複数層の出力を最終層で融合するため、表現力が低下する可能性
- 最適な `skip2lora_block_indices` はモデルアーキテクチャに依存

## 9. 結果レポートのテンプレート

```markdown
# Skip2-LoRA vs Traditional LoRA 評価レポート

## 実験環境
- GPU: XXX
- PyTorch: X.X.X
- LitGPT: main branch
- モデル: Llama-2-7B
- データセット: Alpaca-2k

## 結果サマリー

### パラメータ効率
| 手法 | トレーニング可能 | 削減率 |
|------|-----------------|--------|
| Traditional LoRA | 2.1M | - |
| Skip2-LoRA (4) | 0.5M | 75% |
| Skip2-LoRA (6) | 0.8M | 62% |

### 速度
| 手法 | Backward (ms) | 削減率 | 全体時間 |
|------|---------------|--------|----------|
| Traditional LoRA | XXX | - | XXX分 |
| Skip2-LoRA (4) | XXX | XX% | XXX分 |
| Skip2-LoRA (6) | XXX | XX% | XXX分 |

### 精度
| 手法 | Final Loss | 精度低下 |
|------|-----------|---------|
| Traditional LoRA | X.XXX | - |
| Skip2-LoRA (4) | X.XXX | X.X% |
| Skip2-LoRA (6) | X.XXX | X.X% |

## 結論

Skip2-LoRA は...
```

## 10. トラブルシューティング

### Q: メモリ削減が見られない
A: 以下を確認：
- `skip2lora_block_indices` に十分な層を指定しているか
- Backward pass のメモリを測定しているか（Forward のみではなく）

### Q: 精度が大幅に低下している
A: 以下を試す：
- `skip2lora_block_indices` に層を追加
- `lora_r` を増やす
- `lora_alpha` を調整

### Q: 速度改善が期待より小さい
A: 以下を確認：
- バッチサイズが十分に大きいか
- GPU メモリバウンドになっていないか
- 他のプロセスが干渉していないか

