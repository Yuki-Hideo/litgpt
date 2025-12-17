#!/usr/bin/env python3
"""
Skip2-LoRA Performance Evaluation Script

このスクリプトは、Skip2-LoRA の有無でファインチューニングの速度と性能を比較します。
"""

import time
import json
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List

from litgpt.lora import Config, GPT, mark_only_lora_as_trainable


def create_config(
    model_name: str,
    use_skip2lora: bool = False,
    lora_r: int = 32,
    lora_alpha: int = 16,
    skip2lora_blocks: Tuple[int, ...] = (0, 1, 2, 3),
) -> Config:
    """LoRA設定を作成"""
    return Config(
        name=model_name,
        block_size=512,
        vocab_size=50000,
        padded_vocab_size=50256,
        n_layer=32,
        n_head=32,
        n_embd=2048,
        head_size=64,
        intermediate_size=5632,  # LLaMA style
        
        # LoRA設定
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        
        # 従来のLoRA設定
        lora_query=not use_skip2lora,  # Skip2-LoRA使用時は無効化
        lora_key=False,
        lora_value=not use_skip2lora,
        lora_projection=False,
        lora_mlp=False,
        lora_head=False,
        
        # Skip2-LoRA設定
        skip2lora_enabled=use_skip2lora,
        skip2lora_block_indices=skip2lora_blocks,
        skip2lora_output_layer="lm_head",
    )


def count_trainable_parameters(model: nn.Module) -> Dict[str, int]:
    """トレーニング可能なパラメータ数をカウント"""
    total_params = 0
    trainable_params = 0
    lora_params = 0
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        
        if param.requires_grad:
            trainable_params += num_params
            if "lora_" in name:
                lora_params += num_params
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "lora": lora_params,
        "frozen": total_params - trainable_params,
    }


def count_model_parameters(model: nn.Module) -> Dict[str, Dict[str, int]]:
    """各層のパラメータ数を詳細にカウント"""
    details = {}
    
    # Attention層
    lora_params = 0
    for i, block in enumerate(model.transformer.h):
        if block.skip2lora is not None:
            if hasattr(block.skip2lora, "lora_A"):
                lora_params += block.skip2lora.lora_A.numel()
                lora_params += block.skip2lora.lora_B.numel()
    
    details["skip2lora_layers"] = lora_params
    
    return details


def benchmark_forward_pass(
    model: GPT,
    batch_size: int = 4,
    seq_length: int = 512,
    num_iterations: int = 10,
) -> Dict[str, float]:
    """Forward passの速度を測定"""
    model.eval()
    
    # ダミー入力
    idx = torch.randint(0, model.config.padded_vocab_size, (batch_size, seq_length))
    
    # ウォームアップ
    with torch.no_grad():
        for _ in range(3):
            _ = model(idx)
    
    # ベンチマーク
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start = time.time()
            _ = model(idx)
            end = time.time()
            times.append(end - start)
    
    return {
        "mean": sum(times) / len(times),
        "min": min(times),
        "max": max(times),
        "std": (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5,
    }


def benchmark_backward_pass(
    model: GPT,
    batch_size: int = 4,
    seq_length: int = 512,
    num_iterations: int = 10,
) -> Dict[str, float]:
    """Backward passの速度を測定"""
    model.train()
    
    # パラメータをトレーニングモードに
    mark_only_lora_as_trainable(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    idx = torch.randint(0, model.config.padded_vocab_size, (batch_size, seq_length))
    
    # ウォームアップ
    for _ in range(3):
        logits = model(idx)
        loss = logits.mean()  # ダミー損失
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # ベンチマーク
    times = []
    for _ in range(num_iterations):
        start = time.time()
        logits = model(idx)
        loss = logits.mean()
        optimizer.zero_grad()
        loss.backward()
        end = time.time()
        times.append(end - start)
    
    return {
        "mean": sum(times) / len(times),
        "min": min(times),
        "max": max(times),
        "std": (sum((t - sum(times)/len(times))**2 for t in times) / len(times))**0.5,
    }


def measure_memory_usage(
    model: GPT,
    batch_size: int = 4,
    seq_length: int = 512,
) -> Dict[str, float]:
    """メモリ使用量を測定"""
    model.eval()
    
    # GPUメモリをリセット
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    idx = torch.randint(0, model.config.padded_vocab_size, (batch_size, seq_length))
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            _ = model(idx)
        peak_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
    else:
        peak_memory = 0.0  # CPU使用時は測定しない
    
    return {
        "peak_memory_gb": peak_memory,
    }


def evaluate_configuration(
    config: Config,
    config_name: str,
    batch_size: int = 4,
    seq_length: int = 512,
) -> Dict:
    """設定の評価を実行"""
    print(f"\n{'='*70}")
    print(f"Evaluating: {config_name}")
    print(f"{'='*70}")
    
    # モデルを作成
    print("Creating model...")
    model = GPT(config)
    mark_only_lora_as_trainable(model)
    
    # パラメータ数をカウント
    print("Counting parameters...")
    param_counts = count_trainable_parameters(model)
    param_details = count_model_parameters(model)
    
    print(f"  Total Parameters: {param_counts['total']:,}")
    print(f"  Trainable Parameters: {param_counts['trainable']:,}")
    print(f"  LoRA Parameters: {param_counts['lora']:,}")
    print(f"  Frozen Parameters: {param_counts['frozen']:,}")
    print(f"  Trainable Ratio: {100*param_counts['trainable']/param_counts['total']:.2f}%")
    
    # Forward passベンチマーク
    print(f"\nBenchmarking forward pass ({batch_size} batch, {seq_length} seq)...")
    forward_times = benchmark_forward_pass(model, batch_size, seq_length, num_iterations=10)
    print(f"  Mean: {forward_times['mean']*1000:.2f}ms")
    print(f"  Min: {forward_times['min']*1000:.2f}ms")
    print(f"  Max: {forward_times['max']*1000:.2f}ms")
    print(f"  Std: {forward_times['std']*1000:.2f}ms")
    
    # Backward passベンチマーク
    print(f"\nBenchmarking backward pass ({batch_size} batch, {seq_length} seq)...")
    backward_times = benchmark_backward_pass(model, batch_size, seq_length, num_iterations=10)
    print(f"  Mean: {backward_times['mean']*1000:.2f}ms")
    print(f"  Min: {backward_times['min']*1000:.2f}ms")
    print(f"  Max: {backward_times['max']*1000:.2f}ms")
    print(f"  Std: {backward_times['std']*1000:.2f}ms")
    
    # メモリ使用量
    print(f"\nMeasuring memory usage...")
    memory = measure_memory_usage(model, batch_size, seq_length)
    if memory["peak_memory_gb"] > 0:
        print(f"  Peak Memory: {memory['peak_memory_gb']:.2f} GB")
    else:
        print(f"  Peak Memory: Not available (CPU mode)")
    
    return {
        "config_name": config_name,
        "skip2lora_enabled": config.skip2lora_enabled,
        "skip2lora_blocks": config.skip2lora_block_indices,
        "lora_r": config.lora_r,
        "parameters": param_counts,
        "forward_benchmark": forward_times,
        "backward_benchmark": backward_times,
        "memory": memory,
    }


def compare_results(results: List[Dict]):
    """結果を比較"""
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}\n")
    
    # 列ラベルの準備
    labels = [r["config_name"] for r in results]
    
    # パラメータ数比較
    print("📊 Parameter Count Comparison")
    print("-" * 70)
    print(f"{'Config':<25} {'Trainable':<20} {'Ratio':<15}")
    print("-" * 70)
    for result in results:
        trainable = result["parameters"]["trainable"]
        total = result["parameters"]["total"]
        ratio = 100 * trainable / total
        print(f"{result['config_name']:<25} {trainable:>15,} {ratio:>13.2f}%")
    
    # Forward pass比較
    print(f"\n⚡ Forward Pass Benchmark (ms)")
    print("-" * 70)
    print(f"{'Config':<25} {'Mean':<15} {'Min':<15} {'Max':<15}")
    print("-" * 70)
    for result in results:
        mean = result["forward_benchmark"]["mean"] * 1000
        min_t = result["forward_benchmark"]["min"] * 1000
        max_t = result["forward_benchmark"]["max"] * 1000
        print(f"{result['config_name']:<25} {mean:>13.2f} {min_t:>13.2f} {max_t:>13.2f}")
    
    # Backward pass比較
    print(f"\n⚙️ Backward Pass Benchmark (ms)")
    print("-" * 70)
    print(f"{'Config':<25} {'Mean':<15} {'Min':<15} {'Max':<15}")
    print("-" * 70)
    backward_times_list = []
    for result in results:
        mean = result["backward_benchmark"]["mean"] * 1000
        min_t = result["backward_benchmark"]["min"] * 1000
        max_t = result["backward_benchmark"]["max"] * 1000
        backward_times_list.append(mean)
        print(f"{result['config_name']:<25} {mean:>13.2f} {min_t:>13.2f} {max_t:>13.2f}")
    
    # 速度改善率を計算
    if len(backward_times_list) >= 2:
        speedup = backward_times_list[0] / backward_times_list[1]
        improvement = (1 - backward_times_list[1] / backward_times_list[0]) * 100
        print(f"\n  Speed improvement (Skip2-LoRA vs Traditional LoRA):")
        print(f"    Backward pass: {improvement:.1f}% faster" if improvement > 0 else f"    Backward pass: {-improvement:.1f}% slower")
        print(f"    Speedup ratio: {speedup:.2f}x")
    
    # メモリ比較
    print(f"\n💾 Memory Usage (GB)")
    print("-" * 70)
    print(f"{'Config':<25} {'Peak Memory':<20}")
    print("-" * 70)
    for result in results:
        mem = result["memory"]["peak_memory_gb"]
        if mem > 0:
            print(f"{result['config_name']:<25} {mem:>18.2f}")
        else:
            print(f"{result['config_name']:<25} {'N/A (CPU mode)':<20}")
    
    # スケーリング効率
    print(f"\n📈 Parameter Efficiency (Trainable Params)")
    print("-" * 70)
    base_params = results[0]["parameters"]["trainable"]
    for result in results:
        params = result["parameters"]["trainable"]
        reduction = (1 - params / base_params) * 100
        if reduction > 0:
            print(f"{result['config_name']:<25} {reduction:>13.1f}% reduction")
        else:
            print(f"{result['config_name']:<25} {'(baseline)':<20}")


def save_results(results: List[Dict], output_file: Path = None):
    """結果をJSONファイルに保存"""
    if output_file is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = Path(f"skip2lora_evaluation_{timestamp}.json")
    
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")


def main():
    """メイン評価ルーチン"""
    print("\n" + "="*70)
    print("Skip2-LoRA Performance Evaluation")
    print("="*70)
    
    # 評価する設定のリスト
    configs_to_evaluate = [
        ("Traditional LoRA (Query+Value)", create_config(
            "test_model",
            use_skip2lora=False,
            lora_r=32,
        )),
        ("Skip2-LoRA (4 blocks)", create_config(
            "test_model",
            use_skip2lora=True,
            lora_r=32,
            skip2lora_blocks=(0, 1, 2, 3),
        )),
        ("Skip2-LoRA (6 blocks)", create_config(
            "test_model",
            use_skip2lora=True,
            lora_r=32,
            skip2lora_blocks=(0, 1, 2, 3, 4, 5),
        )),
        ("Skip2-LoRA (2 blocks)", create_config(
            "test_model",
            use_skip2lora=True,
            lora_r=32,
            skip2lora_blocks=(0, 1),
        )),
    ]
    
    results = []
    for config_name, config in configs_to_evaluate:
        try:
            result = evaluate_configuration(config, config_name, batch_size=4, seq_length=512)
            results.append(result)
        except Exception as e:
            print(f"❌ Error evaluating {config_name}: {e}")
            continue
    
    # 結果を比較
    if results:
        compare_results(results)
        save_results(results)
    else:
        print("❌ No results to compare")


if __name__ == "__main__":
    main()
