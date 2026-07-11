"""vLLM-style Inference Engine - Main Entry Point.

A production-grade simulation of vLLM core architectures:
    - PagedAttention (Block Pool + Block Table + Prefix Sharing)
    - Continuous Batching (Mixed Prefill + Decode)
    - Pluggable Admission Policy (Strategy Pattern)
    - GPU Executor with Pipeline (Prefill/Decode separation)
    - Comprehensive Benchmarking & Visualization

Usage:
    python3 main.py
"""
from __future__ import annotations

import json
import logging
import os
import random
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine.inference_engine import InferenceEngine
from engine.policy import MemoryBudgetPolicy, MaxSeqPolicy, CompositePolicy
from benchmarks.collector import BenchmarkCollector

# ============================================================================
# Logging Setup (no print() - use logging everywhere)
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


def create_results_dir() -> str:
    """Create and return the results directory."""
    results_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results"
    )
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def print_timeline_step(info: dict) -> None:
    """Print one step's timeline in the required format.

    Args:
        info: Step info dictionary from Engine.run_step().
    """
    step = info["step"]
    batch_id = info["batch_id"]
    waiting = info["waiting"]
    running = info["running"]
    finished = info["finished"]
    prefill = info["prefill"]
    decode = info["decode"]
    gpu_occ = info["gpu_occupancy"] * 100
    mem_used = info["memory_used"]
    mem_budget = info["memory_budget"]
    mem_pct = (mem_used / mem_budget * 100) if mem_budget > 0 else 0
    tokens = info["total_tokens"]

    print("-" * 60)
    print(f"STEP {step}")
    print(f"Batch #{batch_id}")
    print(f"Waiting : {waiting}")
    print(f"Running : {running}")
    print(f"Finished: {finished}")
    print(f"Prefill : {prefill}")
    print(f"Decode  : {decode}")
    print(f"GPU Occ : {gpu_occ:.0f}%")
    print(f"Memory  : {mem_pct:.0f}%")
    print(f"Tokens  : {tokens}")
    print("-" * 60)


def main():
    """Run the main inference demo."""
    print("=" * 60)
    print("  vLLM-style Inference Engine")
    print("  PagedAttention + Continuous Batching + Mixed P/D")
    print("=" * 60)
    print()

    # Create policy
    policy = MemoryBudgetPolicy(memory_budget=80000)

    # Create engine
    engine = InferenceEngine(
        block_size=16,
        num_blocks=1024,
        prefill_cost_per_token=0.001,
        decode_cost_per_token=0.0005,
        policy=policy,
        enable_prefix_sharing=True,
    )

    print("Configuration:")
    print(f"  Memory Budget:  {policy.memory_budget} tokens")
    print(f"  Block Size:     {engine.kv_cache.block_size}")
    print(f"  Num Blocks:     {engine.kv_cache.num_blocks}")
    print(f"  Prefill Cost:   {engine._prefill_cost}s/token")
    print(f"  Decode Cost:    {engine._decode_cost}s/token")
    print()

    # Generate requests
    random.seed(42)
    num_requests = 20
    print(f"Generating {num_requests} requests...")

    shared_prefix = list(range(1, 9))  # 8-token shared prefix
    for i in range(num_requests):
        prompt_length = random.randint(10, 60)
        max_new_tokens = random.randint(5, 25)

        if i % 4 == 0:
            # Use shared prefix
            prompt_tokens = shared_prefix + list(
                range(100 + i, 100 + i + prompt_length - len(shared_prefix))
            )
        else:
            prompt_tokens = list(
                range(200 + i * 100, 200 + i * 100 + prompt_length)
            )

        engine.add_request(
            prompt_length=prompt_length,
            max_new_tokens=max_new_tokens,
            prompt_tokens=prompt_tokens,
        )

    print()
    print("Timeline:")
    print()

    # Create benchmark collector
    benchmark = BenchmarkCollector()
    benchmark.start()

    # Run engine with timeline callback
    engine.run(
        max_steps=100000,
        callback=print_timeline_step,
    )

    benchmark.end()

    # Collect results
    finished = engine.scheduler.get_finished_requests()
    benchmark.process_requests(finished)

    # Save results
    print()
    print("=" * 60)
    print("  Results")
    print("=" * 60)

    results_dir = create_results_dir()

    # Summary JSON
    stats = benchmark.compute_stats()
    engine_results = engine.get_results()
    # Merge engine-level stats
    stats["gpu_occupancy"] = engine_results["gpu_occupancy"]
    stats["admission_rate"] = engine_results["admission_rate"]
    stats["total_steps"] = engine_results["total_steps"]
    stats["kv_stats"] = {
        "memory_utilization": engine_results["kv_memory_utilization"],
        "peak_shared_blocks": engine_results["kv_peak_shared_blocks"],
        "prefix_cache_hits": engine_results["kv_prefix_cache_hits"],
        "prefix_cache_hit_rate": engine_results["kv_prefix_cache_hit_rate"],
    }

    summary_path = os.path.join(results_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"  Saved: {summary_path}")

    # Per-request metrics CSV
    csv_path = os.path.join(results_dir, "metrics.csv")
    benchmark.export_csv(csv_path)

    # Timeline CSV
    timeline_path = os.path.join(results_dir, "timeline.csv")
    # Re-record steps - but we already have them in engine
    # Let's build timeline from step records
    gpu_records = engine.gpu_monitor.get_step_records()
    if gpu_records:
        import csv as csv_mod
        fields = ["step", "batch_id", "duration", "prefill_count",
                   "decode_count", "batch_size", "memory_used",
                   "busy", "tokens_generated"]
        with open(timeline_path, "w", newline="", encoding="utf-8") as f:
            writer = csv_mod.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for r in gpu_records:
                writer.writerow({
                    "step": r.step,
                    "batch_id": r.batch_id,
                    "duration": f"{r.duration:.6f}",
                    "prefill_count": r.prefill_count,
                    "decode_count": r.decode_count,
                    "batch_size": r.batch_size,
                    "memory_used": r.memory_used,
                    "busy": r.busy,
                    "tokens_generated": r.tokens_generated,
                })
        print(f"  Saved: {timeline_path}")

    # Generate plots
    print()
    print("  Generating plots...")
    try:
        from visualization.plot_metrics import plot_all
        plot_all(results_dir)
        print(f"  Plots saved to {results_dir}/")
    except Exception as e:
        print(f"  Warning: Could not generate plots: {e}")

    # Print summary
    print()
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    print(f"  Total Requests:  {stats['total_requests']}")
    print(f"  Total Tokens:    {stats['total_tokens']}")
    print(f"  Total Time:      {stats['total_time']:.4f}s")
    print(f"  Throughput:      {stats['throughput']:.2f} tok/s")
    print(f"  GPU Occupancy:   {stats['gpu_occupancy']*100:.1f}%")
    print(f"  Admission Rate:  {stats['admission_rate']*100:.1f}%")
    print(f"  Total Steps:     {stats['total_steps']}")
    print(f"  TTFT  avg/p50/p95/p99: "
          f"{stats['ttft']['avg']:.4f} / {stats['ttft']['p50']:.4f} / "
          f"{stats['ttft']['p95']:.4f} / {stats['ttft']['p99']:.4f} s")
    print(f"  TPOT  avg/p50/p95/p99: "
          f"{stats['tpot']['avg']:.4f} / {stats['tpot']['p50']:.4f} / "
          f"{stats['tpot']['p95']:.4f} / {stats['tpot']['p99']:.4f} s")
    print(f"  Lat   avg/p50/p95/p99: "
          f"{stats['latency']['avg']:.4f} / {stats['latency']['p50']:.4f} / "
          f"{stats['latency']['p95']:.4f} / {stats['latency']['p99']:.4f} s")
    print()
    print("=" * 60)
    print("  Done! All results saved to results/")
    print("=" * 60)


if __name__ == "__main__":
    main()
