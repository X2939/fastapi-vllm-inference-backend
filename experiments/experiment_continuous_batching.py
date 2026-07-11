"""Experiment: Continuous Batching vs Static Batching.

This experiment compares true continuous batching against traditional static
batching using the same InferenceEngine, Executor and KV Cache:

  - **Static**: requests are grouped into fixed-size batches. The next batch
    cannot start until every request in the current batch finishes.
  - **Continuous**: requests arrive over time; after every decode step finished
    requests are removed and newly arrived requests are admitted. The GPU never
    waits for a whole batch to finish.

Metrics collected:
  - Throughput (tokens / sec)
  - TTFT avg / p50 / p95
  - End-to-end latency avg / p50 / p95
  - Average batch size and GPU occupancy

Outputs:
  - results/experiment_continuous_batching.csv
  - results/experiment_continuous_batching.json
  - results/experiment_continuous_batching.png

Usage:
    python3 -m experiments.experiment_continuous_batching
"""
from __future__ import annotations

import csv
import json
import logging
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.inference_engine import InferenceEngine
from engine.policy import MemoryBudgetPolicy
from engine.continuous_batching import (
    BatchingResult,
    ContinuousBatchingRunner,
    StaticBatchingRunner,
    create_workload,
)
from visualization.plot_continuous_batching import plot_continuous_batching_results

logger = logging.getLogger(__name__)


def run_comparison(
    num_requests: int = 64,
    seed: int = 42,
) -> Tuple[BatchingResult, BatchingResult]:
    """Run continuous and static batching on the same synthetic workload.

    Args:
        num_requests: Number of requests in the workload.
        seed: Random seed for workload generation.

    Returns:
        Tuple of (continuous_result, static_result).
    """
    workload = create_workload(
        num_requests=num_requests,
        seed=seed,
        prompt_min=20,
        prompt_max=80,
        output_min=10,
        output_max=30,
        mean_inter_arrival=0.05,
    )
    # Pass the same arrival times to the static runner so it waits for a
    # full batch to arrive before starting it. This makes the comparison
    # reflect real online serving instead of assuming all requests are
    # available at time zero.
    static_requests = [(pl, mt, pt, at) for pl, mt, pt, at in workload]

    common_config = {
        "block_size": 16,
        "num_blocks": 2048,
        "prefill_cost_per_token": 0.0008,
        "decode_cost_per_token": 0.0004,
        "policy": MemoryBudgetPolicy(memory_budget=100000),
        "enable_prefix_sharing": True,
    }

    logger.info("Running Continuous Batching...")
    engine_cb = InferenceEngine(backend="pytorch", **common_config)
    runner_cb = ContinuousBatchingRunner(engine_cb)
    for prompt_length, max_new_tokens, prompt_tokens, arrival_time in workload:
        runner_cb.add_request(
            prompt_length=prompt_length,
            max_new_tokens=max_new_tokens,
            prompt_tokens=prompt_tokens,
            arrival_time=arrival_time,
        )
    continuous_result = runner_cb.run()
    logger.info(
        "Continuous -> throughput=%.1f tok/s  ttft_p95=%.4fs  latency_p95=%.4fs  "
        "avg_batch=%.1f  steps=%d",
        continuous_result.throughput,
        continuous_result.ttft_p95,
        continuous_result.latency_p95,
        continuous_result.avg_batch_size,
        continuous_result.total_steps,
    )

    logger.info("Running Static Batching...")
    runner_static = StaticBatchingRunner(**common_config)
    static_result = runner_static.run(static_requests, batch_size=16)
    logger.info(
        "Static     -> throughput=%.1f tok/s  ttft_p95=%.4fs  latency_p95=%.4fs  "
        "avg_batch=%.1f  steps=%d",
        static_result.throughput,
        static_result.ttft_p95,
        static_result.latency_p95,
        static_result.avg_batch_size,
        static_result.total_steps,
    )

    return continuous_result, static_result


def save_csv(results: List[Dict], filepath: str) -> None:
    """Save results to CSV file.

    Args:
        results: List of result dicts.
        filepath: Output CSV path.
    """
    if not results:
        logger.warning("No results to save to CSV")
        return

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fieldnames = list(results[0].keys())
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    logger.info("Saved CSV: %s", filepath)


def save_json(results: List[Dict], filepath: str) -> None:
    """Save results to JSON file.

    Args:
        results: List of result dicts.
        filepath: Output JSON path.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Saved JSON: %s", filepath)


def print_summary(
    continuous_result: BatchingResult,
    static_result: BatchingResult,
) -> None:
    """Print a formatted summary table to stdout."""
    print()
    print("=" * 72)
    print("  Continuous Batching vs Static Batching")
    print("=" * 72)
    print()
    header = (
        "  {:<12} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}"
    ).format(
        "Mode", "Throughput", "TTFT avg", "TTFT p95", "Lat avg", "Lat p95", "Avg Batch"
    )
    print(header)
    print("  " + "-" * 70)

    def row(label: str, r: BatchingResult) -> None:
        print(
            "  {:<12} {:>9.1f} {:>9.4f} {:>9.4f} {:>9.4f} {:>9.4f} {:>9.1f}".format(
                label,
                r.throughput,
                r.ttft_avg,
                r.ttft_p95,
                r.latency_avg,
                r.latency_p95,
                r.avg_batch_size,
            )
        )

    row("Static", static_result)
    row("Continuous", continuous_result)
    print()

    if static_result.throughput > 0:
        speedup = continuous_result.throughput / static_result.throughput
        print(f"  Throughput speedup: {speedup:.2f}x")
    if static_result.latency_p95 > 0:
        lat_ratio = continuous_result.latency_p95 / static_result.latency_p95
        print(f"  P95 latency ratio:  {lat_ratio:.2f}x (continuous / static)")
    print()
    print(f"  Static steps:     {static_result.total_steps}")
    print(f"  Continuous steps: {continuous_result.total_steps}")
    print()
    print("=" * 72)


def main() -> None:
    """Run the continuous batching comparison experiment."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)

    print()
    print("=" * 72)
    print("  Experiment: Continuous Batching vs Static Batching")
    print("=" * 72)
    print()

    continuous_result, static_result = run_comparison(num_requests=64, seed=42)

    results = [continuous_result.to_dict(), static_result.to_dict()]
    csv_path = os.path.join(output_dir, "experiment_continuous_batching.csv")
    json_path = os.path.join(output_dir, "experiment_continuous_batching.json")
    png_path = os.path.join(output_dir, "experiment_continuous_batching.png")

    save_csv(results, csv_path)
    save_json(results, json_path)

    try:
        plot_continuous_batching_results(results, png_path)
        logger.info("Saved PNG: %s", png_path)
    except Exception as e:
        logger.warning("Failed to generate plot: %s", e)

    print_summary(continuous_result, static_result)

    print(f"  CSV: {csv_path}")
    print(f"  JSON: {json_path}")
    print(f"  PNG: {png_path}")
    print()


if __name__ == "__main__":
    main()
