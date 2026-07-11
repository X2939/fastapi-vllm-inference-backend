"""Experiment: GPU Memory Budget comparison.

Tests different GPU memory budgets and measures:
  - Throughput (tokens/sec)
  - Latency (avg, p50, p95, p99)
  - GPU Occupancy
  - Memory utilization

Usage:
    python3 -m experiments.experiment_gpu_budget
"""
from __future__ import annotations

import csv
import json
import os
import random
import sys
import time
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.inference_engine import InferenceEngine


def run_experiment(
    gpu_memory_budget: int,
    num_requests: int = 30,
    seed: int = 42,
) -> Dict:
    """Run a single experiment with a given memory budget.

    Args:
        gpu_memory_budget: GPU memory budget in tokens.
        num_requests: Number of requests to simulate.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary of result metrics.
    """
    random.seed(seed)
    engine = InferenceEngine(
        max_num_seqs=256,
        gpu_memory_budget=gpu_memory_budget,
        block_size=16,
        num_blocks=2048,
        prefill_cost_per_token=0.0008,
        decode_cost_per_token=0.0004,
        batching_mode="mixed",
    )

    common_prefix = list(range(1, 16))
    for i in range(num_requests):
        prompt_length = random.randint(20, 100)
        max_new_tokens = random.randint(10, 50)
        if i % 3 == 0:
            prompt_tokens = common_prefix + list(
                range(1000 + i, 1000 + i + prompt_length - len(common_prefix))
            )
        else:
            prompt_tokens = list(range(2000 + i * 100, 2000 + i * 100 + prompt_length))
        engine.add_request(
            prompt_length=prompt_length,
            max_new_tokens=max_new_tokens,
            prompt_tokens=prompt_tokens,
        )

    engine.run(verbose=False)
    return engine._get_final_results()


def main():
    results_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "results",
    )
    os.makedirs(results_dir, exist_ok=True)

    budgets = [4000, 8000, 16000, 24000]
    budget_labels = ["4K", "8K", "16K", "24K"]

    print("=" * 70)
    print("  Experiment: GPU Memory Budget Comparison")
    print("=" * 70)
    print()

    all_results: List[Dict] = []

    for budget, label in zip(budgets, budget_labels):
        print(f"Running with budget={label} tokens...")
        result = run_experiment(gpu_memory_budget=budget)
        result["budget_label"] = label
        result["budget_tokens"] = budget
        all_results.append(result)
        print(f"  Throughput:    {result['throughput']:.1f} tok/s")
        print(f"  Avg Latency:   {result['avg_latency']:.4f}s")
        print(f"  P99 Latency:   {result['p99_latency']:.4f}s")
        print(f"  GPU Occupancy: {result['gpu_occupancy']*100:.1f}%")
        print()

    csv_path = os.path.join(results_dir, "experiment_gpu_budget.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        if all_results:
            writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_results)
    print(f"Saved: {csv_path}")

    json_path = os.path.join(results_dir, "experiment_gpu_budget.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"Saved: {json_path}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        labels = [r["budget_label"] for r in all_results]

        ax = axes[0, 0]
        ax.bar(labels, [r["throughput"] for r in all_results], color="#2196F3")
        ax.set_title("Throughput vs GPU Budget")
        ax.set_ylabel("Tokens / sec")
        ax.grid(True, alpha=0.3, axis="y")

        ax = axes[0, 1]
        ax.bar(labels, [r["avg_latency"] for r in all_results],
               label="Avg", alpha=0.7, color="#4CAF50")
        ax.bar(labels, [r["p99_latency"] for r in all_results],
               label="P99", alpha=0.7, color="#F44336")
        ax.set_title("Latency vs GPU Budget")
        ax.set_ylabel("Latency (s)")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        ax = axes[1, 0]
        ax.bar(labels, [r["gpu_occupancy"] * 100 for r in all_results], color="#FF9800")
        ax.set_title("GPU Occupancy vs GPU Budget")
        ax.set_ylabel("Occupancy (%)")
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3, axis="y")

        ax = axes[1, 1]
        ax.bar(labels, [r["avg_batch_size"] for r in all_results], color="#9C27B0")
        ax.set_title("Avg Batch Size vs GPU Budget")
        ax.set_ylabel("Avg Batch Size")
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        png_path = os.path.join(results_dir, "experiment_gpu_budget.png")
        plt.savefig(png_path, dpi=150)
        plt.close(fig)
        print(f"Saved: {png_path}")
    except Exception as e:
        print(f"Warning: Could not generate plot: {e}")

    print()
    print("=" * 70)
    print("  GPU Budget Experiment Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
