"""Experiment: Prefix Sharing On vs Off

Usage:
    python3 -m experiments.experiment_prefix_share
"""
from __future__ import annotations

import csv
import json
import logging
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from engine.inference_engine import InferenceEngine
from engine.policy import (
    MemoryBudgetPolicy, MaxSeqPolicy, PriorityPolicy, CompositePolicy
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)
os.makedirs(RESULTS_DIR, exist_ok=True)


def save_results(name, results, plots_func=None):
    """Save experiment results to CSV, JSON, and optionally PNG."""
    csv_path = os.path.join(RESULTS_DIR, f"{name}.csv")
    json_path = os.path.join(RESULTS_DIR, f"{name}.json")

    if results:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        if plots_func:
            png_path = os.path.join(RESULTS_DIR, f"{name}.png")
            plots_func(results, png_path)

    print(f"  Saved: {csv_path}")
    print(f"  Saved: {json_path}")


def run_with_sharing(enabled, num_requests=40, seed=42):
    """Run engine with prefix sharing on or off."""
    random.seed(seed)
    engine = InferenceEngine(
        block_size=16, num_blocks=2048,
        prefill_cost_per_token=0.0008,
        decode_cost_per_token=0.0004,
        policy=MemoryBudgetPolicy(memory_budget=50000),
        enable_prefix_sharing=enabled,
    )

    shared = list(range(1, 51))  # 50-token shared prefix
    for i in range(num_requests):
        pl = random.randint(60, 150)
        mt = random.randint(10, 40)
        if i % 2 == 0:
            pt = shared + list(range(1000+i, 1000+i+pl-len(shared)))
        else:
            pt = list(range(2000+i*100, 2000+i*100+pl))
        engine.add_request(prompt_length=pl, max_new_tokens=mt, prompt_tokens=pt)

    engine.run(max_steps=100000)
    r = engine.get_results()
    r["prefix_sharing"] = "ON" if enabled else "OFF"
    return r


def main():
    print("=" * 60)
    print("  Experiment: Prefix Sharing Comparison")
    print("=" * 60)
    print()

    all_results = []
    for label, enabled in [("OFF", False), ("ON", True)]:
        print(f"  Running prefix_sharing={label}...")
        r = run_with_sharing(enabled)
        all_results.append(r)
        print(f"    Peak Shared: {r['kv_peak_shared_blocks']}, "
              f"TTFT: {r['ttft']['avg']:.4f}s, "
              f"Throughput: {r['throughput']:.1f} tok/s")

    save_results("experiment_prefix_share", all_results, _plot)


def _plot(results, path):
    labels = [r["prefix_sharing"] for r in results]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0,0].bar(labels, [r["kv_peak_shared_blocks"] for r in results], color="#9C27B0")
    axes[0,0].set_title("Peak Shared Blocks")
    axes[0,0].grid(True, alpha=0.3, axis="y")

    axes[0,1].bar(labels, [r["ttft"]["avg"] for r in results], color="#2196F3")
    axes[0,1].set_title("Average TTFT (s)")
    axes[0,1].grid(True, alpha=0.3, axis="y")

    axes[1,0].bar(labels, [r["throughput"] for r in results], color="#4CAF50")
    axes[1,0].set_title("Throughput (tok/s)")
    axes[1,0].grid(True, alpha=0.3, axis="y")

    axes[1,1].bar(labels, [r["gpu_occupancy"]*100 for r in results], color="#FF9800")
    axes[1,1].set_title("GPU Occupancy (%)")
    axes[1,1].set_ylim(0, 105)
    axes[1,1].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


if __name__ == "__main__":
    main()
