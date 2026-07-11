"""Chunked Prefill experiment: compare different chunk sizes.

Tests Short, Medium, and Long prompts across multiple chunk sizes
and measures TTFT, Latency, Throughput, and GPU Utilization.

Outputs: CSV, JSON, PNG
"""
from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.inference_engine import InferenceEngine
from engine.policy import MemoryBudgetPolicy
from visualization.plot_chunked_prefill import plot_chunked_prefill_results

logger = logging.getLogger(__name__)


@dataclass
class ChunkedPrefillResult:
    """Result of a single chunked prefill experiment run.

    Attributes:
        chunk_size: Chunk size used (0 = no chunking / standard).
        prompt_category: "short", "medium", or "long".
        avg_prompt_length: Average prompt length.
        num_requests: Number of requests.
        avg_ttft: Average time to first token (seconds).
        p50_ttft: P50 TTFT.
        p95_ttft: P95 TTFT.
        avg_latency: Average end-to-end latency (seconds).
        p50_latency: P50 latency.
        p95_latency: P95 latency.
        throughput: Tokens per second.
        gpu_occupancy: Average GPU occupancy (0-1).
        total_steps: Total engine steps.
        total_chunks: Total prefill chunks processed.
        total_tokens: Total output tokens generated.
        total_time: Total wall-clock time.
    """
    chunk_size: int
    prompt_category: str
    avg_prompt_length: int
    num_requests: int
    avg_ttft: float = 0.0
    p50_ttft: float = 0.0
    p95_ttft: float = 0.0
    avg_latency: float = 0.0
    p50_latency: float = 0.0
    p95_latency: float = 0.0
    throughput: float = 0.0
    gpu_occupancy: float = 0.0
    total_steps: int = 0
    total_chunks: int = 0
    total_tokens: int = 0
    total_time: float = 0.0


def generate_prompts(
    category: str,
    num_requests: int,
    seed: int = 42,
) -> List[Tuple[int, int, List[int]]]:
    """Generate prompt configurations for a given category.

    Args:
        category: "short", "medium", or "long".
        num_requests: Number of requests to generate.
        seed: Random seed.

    Returns:
        List of (prompt_length, max_new_tokens, prompt_tokens).
    """
    rng = np.random.default_rng(seed)

    if category == "short":
        prompt_lengths = rng.integers(32, 64, size=num_requests)
        max_new_tokens_list = rng.integers(10, 20, size=num_requests)
    elif category == "medium":
        prompt_lengths = rng.integers(256, 512, size=num_requests)
        max_new_tokens_list = rng.integers(20, 50, size=num_requests)
    elif category == "long":
        prompt_lengths = rng.integers(1024, 2048, size=num_requests)
        max_new_tokens_list = rng.integers(30, 80, size=num_requests)
    else:
        raise ValueError(f"Unknown category: {category}")

    result = []
    for i in range(num_requests):
        pl = int(prompt_lengths[i])
        mt = int(max_new_tokens_list[i])
        pt = list(range(pl))
        result.append((pl, mt, pt))

    return result


def run_single(
    chunk_size: int,
    requests: List[Tuple[int, int, List[int]]],
    prompt_category: str,
    prefill_cost: float = 0.0002,
    decode_cost: float = 0.0001,
    max_steps: int = 5000,
) -> ChunkedPrefillResult:
    """Run a single experiment configuration.

    Args:
        chunk_size: Prefill chunk size (0 = standard / no chunking).
        requests: List of (prompt_length, max_new_tokens, prompt_tokens).
        prompt_category: Label for the prompt category.
        prefill_cost: Simulated prefill cost per token.
        decode_cost: Simulated decode cost per token.
        max_steps: Safety limit.

    Returns:
        ChunkedPrefillResult with all metrics.
    """
    chunked = chunk_size > 0
    engine = InferenceEngine(
        block_size=16,
        num_blocks=4096,
        prefill_cost_per_token=prefill_cost,
        decode_cost_per_token=decode_cost,
        policy=MemoryBudgetPolicy(memory_budget=200000),
        chunked_prefill=chunked,
        prefill_chunk_size=chunk_size if chunked else 128,
    )

    for pl, mt, pt in requests:
        engine.add_request(
            prompt_length=pl,
            max_new_tokens=mt,
            prompt_tokens=pt,
        )

    start_time = time.perf_counter()
    steps = engine.run(max_steps=max_steps)
    total_time = time.perf_counter() - start_time

    results = engine.get_results()
    finished = engine.scheduler.get_finished_requests()

    # Compute total output tokens from finished requests (more reliable
    # than gpu.total_tokens which counts prefill steps).
    total_output_tokens = sum(r.generated_tokens for r in finished)

    chunk_stats = 0
    if engine.scheduler.chunk_helper is not None:
        chunk_stats = engine.scheduler.chunk_helper.stats.total_chunks

    avg_prompt = int(np.mean([pl for pl, _, _ in requests]))

    return ChunkedPrefillResult(
        chunk_size=chunk_size,
        prompt_category=prompt_category,
        avg_prompt_length=avg_prompt,
        num_requests=len(finished),
        avg_ttft=results["ttft"]["avg"],
        p50_ttft=results["ttft"]["p50"],
        p95_ttft=results["ttft"]["p95"],
        avg_latency=results["latency"]["avg"],
        p50_latency=results["latency"]["p50"],
        p95_latency=results["latency"]["p95"],
        throughput=total_output_tokens / total_time if total_time > 0 else 0,
        gpu_occupancy=results["gpu_occupancy"],
        total_steps=len(steps),
        total_chunks=chunk_stats,
        total_tokens=total_output_tokens,
        total_time=total_time,
    )


def run_experiment(
    categories: List[str] = None,
    chunk_sizes: List[int] = None,
    num_requests: int = 16,
    seed: int = 42,
    output_dir: str = "results",
) -> List[Dict]:
    """Run the full chunked prefill experiment.

    Args:
        categories: Prompt categories to test.
        chunk_sizes: Chunk sizes to test (0 = standard no chunking).
        num_requests: Number of requests per configuration.
        seed: Random seed.
        output_dir: Directory to save results.

    Returns:
        List of result dicts.
    """
    if categories is None:
        categories = ["short", "medium", "long"]
    if chunk_sizes is None:
        chunk_sizes = [0, 64, 128, 256, 512]

    os.makedirs(output_dir, exist_ok=True)
    all_results: List[Dict] = []

    for category in categories:
        logger.info("=== Category: %s ===", category)
        requests = generate_prompts(category, num_requests, seed=seed)

        for chunk_size in chunk_sizes:
            label = f"standard" if chunk_size == 0 else f"chunk={chunk_size}"
            logger.info("  Running %s...", label)

            result = run_single(chunk_size, requests, category)
            result_dict = {
                "category": category,
                "avg_prompt_length": result.avg_prompt_length,
                "chunk_size": result.chunk_size,
                "num_requests": result.num_requests,
                "avg_ttft_s": round(result.avg_ttft, 6),
                "p50_ttft_s": round(result.p50_ttft, 6),
                "p95_ttft_s": round(result.p95_ttft, 6),
                "avg_latency_s": round(result.avg_latency, 6),
                "p50_latency_s": round(result.p50_latency, 6),
                "p95_latency_s": round(result.p95_latency, 6),
                "throughput_tok_s": round(result.throughput, 2),
                "gpu_occupancy": round(result.gpu_occupancy, 4),
                "total_steps": result.total_steps,
                "total_chunks": result.total_chunks,
                "total_tokens": result.total_tokens,
                "total_time_s": round(result.total_time, 6),
            }
            all_results.append(result_dict)

            logger.info(
                "    TTFT=%.4fs  Latency=%.4fs  Throughput=%.1f tok/s  "
                "GPU=%.1f%%  Steps=%d  Chunks=%d",
                result.avg_ttft, result.avg_latency, result.throughput,
                result.gpu_occupancy * 100, result.total_steps, result.total_chunks,
            )

    # Save CSV
    csv_path = os.path.join(output_dir, "experiment_chunked_prefill.csv")
    if all_results:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_results[0].keys()))
            writer.writeheader()
            writer.writerows(all_results)
        logger.info("Saved CSV: %s", csv_path)

    # Save JSON
    json_path = os.path.join(output_dir, "experiment_chunked_prefill.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Saved JSON: %s", json_path)

    # Save PNG
    png_path = os.path.join(output_dir, "experiment_chunked_prefill.png")
    plot_chunked_prefill_results(all_results, png_path)
    logger.info("Saved PNG: %s", png_path)

    return all_results


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    results = run_experiment(
        categories=["short", "medium", "long"],
        chunk_sizes=[0, 64, 128, 256, 512],
        num_requests=16,
        seed=42,
        output_dir="results",
    )
    print(f"\nTotal configurations: {len(results)}")
    print("Results saved to results/experiment_chunked_prefill.{csv,json,png}")
