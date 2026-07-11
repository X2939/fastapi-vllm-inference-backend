"""Experiment: Admission Policy Comparison

Usage:
    python3 -m experiments.experiment_admission_policy
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


def run_with_policy(policy_name, policy, num_requests=30, seed=42):
    """Run engine with a given admission policy."""
    random.seed(seed)
    engine = InferenceEngine(
        block_size=16, num_blocks=2048,
        prefill_cost_per_token=0.0008,
        decode_cost_per_token=0.0004,
        policy=policy,
        enable_prefix_sharing=True,
    )

    shared = list(range(1, 11))
    for i in range(num_requests):
        pl = random.randint(20, 80)
        mt = random.randint(10, 30)
        if i % 3 == 0:
            pt = shared + list(range(100+i, 100+i+pl-len(shared)))
        else:
            pt = list(range(200+i*50, 200+i*50+pl))
        engine.add_request(prompt_length=pl, max_new_tokens=mt, prompt_tokens=pt)

    engine.run(max_steps=100000)
    r = engine.get_results()
    r["policy"] = policy_name
    return r


def main():
    print("=" * 60)
    print("  Experiment: Admission Policy Comparison")
    print("=" * 60)
    print()

    configs = [
        ("MemoryBudget_4K", MemoryBudgetPolicy(memory_budget=4000)),
        ("MemoryBudget_8K", MemoryBudgetPolicy(memory_budget=8000)),
        ("MaxSeq_16", MaxSeqPolicy(max_num_seqs=16)),
        ("MaxSeq_32", MaxSeqPolicy(max_num_seqs=32)),
        ("Priority", PriorityPolicy(max_num_seqs=64, max_per_priority=16)),
        ("Composite", CompositePolicy([
            MemoryBudgetPolicy(memory_budget=8000),
            MaxSeqPolicy(max_num_seqs=32),
        ])),
    ]

    all_results = []
    for name, policy in configs:
        print(f"  Running {name}...")
        r = run_with_policy(name, policy)
        all_results.append(r)
        print(f"    Throughput: {r['throughput']:.1f} tok/s, "
              f"Latency: {r['latency']['avg']:.4f}s, "
              f"Admission: {r['admission_rate']*100:.1f}%")

    save_results("experiment_admission_policy", all_results, _plot)


def _plot(results, path):
    """Generate comparison bar charts."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    labels = [r["policy"] for r in results]

    axes[0,0].barh(labels, [r["throughput"] for r in results], color="#2196F3")
    axes[0,0].set_title("Throughput (tok/s)")
    axes[0,0].grid(True, alpha=0.3, axis="x")

    axes[0,1].barh(labels, [r["latency"]["avg"] for r in results], color="#F44336")
    axes[0,1].set_title("Avg Latency (s)")
    axes[0,1].grid(True, alpha=0.3, axis="x")

    axes[1,0].barh(labels, [r["gpu_occupancy"]*100 for r in results], color="#4CAF50")
    axes[1,0].set_title("GPU Occupancy (%)")
    axes[1,0].set_xlim(0, 105)
    axes[1,0].grid(True, alpha=0.3, axis="x")

    axes[1,1].barh(labels, [r["admission_rate"]*100 for r in results], color="#FF9800")
    axes[1,1].set_title("Admission Rate (%)")
    axes[1,1].set_xlim(0, 105)
    axes[1,1].grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


if __name__ == "__main__":
    main()
