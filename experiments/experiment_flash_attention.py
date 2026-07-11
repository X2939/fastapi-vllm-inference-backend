"""FlashAttention vs Standard Attention experiment.

Compares latency, peak memory, and memory savings across multiple
sequence lengths and block sizes. Outputs CSV, JSON, and PNG.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.flash_attention import (
    AttentionStats,
    attention_flash,
    attention_standard,
)
from visualization.plot_flash_attention import plot_flash_attention_results

logger = logging.getLogger(__name__)


def generate_qkv(
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate random Q, K, V tensors.

    Args:
        batch_size: Batch size.
        num_heads: Number of attention heads.
        seq_len: Sequence length.
        head_dim: Head dimension.
        seed: Random seed.

    Returns:
        (Q, K, V) each of shape [batch, heads, seq, dim].
    """
    rng = np.random.default_rng(seed)
    Q = rng.standard_normal(
        (batch_size, num_heads, seq_len, head_dim)
    ).astype(np.float32)
    K = rng.standard_normal(
        (batch_size, num_heads, seq_len, head_dim)
    ).astype(np.float32)
    V = rng.standard_normal(
        (batch_size, num_heads, seq_len, head_dim)
    ).astype(np.float32)
    return Q, K, V


def run_comparison(
    batch_size: int = 1,
    num_heads: int = 8,
    seq_len: int = 1024,
    head_dim: int = 128,
    block_size: int = 64,
    num_warmup: int = 2,
    num_runs: int = 5,
) -> Dict:
    """Run Standard vs Flash attention comparison.

    Args:
        batch_size: Batch size.
        num_heads: Number of heads.
        seq_len: Sequence length.
        head_dim: Head dimension.
        block_size: FlashAttention block size.
        num_warmup: Number of warmup runs.
        num_runs: Number of timed runs.

    Returns:
        Dict with results for both backends.
    """
    Q, K, V = generate_qkv(batch_size, num_heads, seq_len, head_dim)

    # --- Standard Attention ---
    logger.info("Running Standard Attention (seq=%d)...", seq_len)

    for _ in range(num_warmup):
        _, _ = attention_standard(Q, K, V)

    std_times: List[float] = []
    std_stats: Optional[AttentionStats] = None
    for i in range(num_runs):
        start = time.perf_counter()
        out_std, stats_std = attention_standard(Q, K, V)
        elapsed = time.perf_counter() - start
        std_times.append(elapsed)
        if i == 0:
            std_stats = stats_std

    # --- Flash Attention ---
    logger.info(
        "Running FlashAttention (seq=%d, block=%d)...",
        seq_len, block_size,
    )

    for _ in range(num_warmup):
        _, _ = attention_flash(Q, K, V, block_size=block_size)

    flash_times: List[float] = []
    flash_stats: Optional[AttentionStats] = None
    for i in range(num_runs):
        start = time.perf_counter()
        out_flash, stats_flash = attention_flash(
            Q, K, V, block_size=block_size,
        )
        elapsed = time.perf_counter() - start
        flash_times.append(elapsed)
        if i == 0:
            flash_stats = stats_flash

    # Verify correctness
    max_diff = float(np.max(np.abs(out_std - out_flash)))
    correct = max_diff < 1e-3

    result = {
        "seq_len": seq_len,
        "head_dim": head_dim,
        "num_heads": num_heads,
        "batch_size": batch_size,
        "block_size": block_size,
        "standard_latency_avg": float(np.mean(std_times)),
        "standard_latency_min": float(np.min(std_times)),
        "standard_latency_max": float(np.max(std_times)),
        "standard_peak_mb": float(std_stats.memory_peak_bytes / (1024 * 1024)),
        "standard_qk_mb": float(std_stats.memory_qk / (1024 * 1024)),
        "standard_softmax_mb": float(std_stats.memory_softmax / (1024 * 1024)),
        "flash_latency_avg": float(np.mean(flash_times)),
        "flash_latency_min": float(np.min(flash_times)),
        "flash_latency_max": float(np.max(flash_times)),
        "flash_peak_mb": float(flash_stats.memory_peak_bytes / (1024 * 1024)),
        "flash_saved_mb": float(flash_stats.memory_saved_bytes / (1024 * 1024)),
        "flash_saved_pct": float(flash_stats.memory_saved_pct),
        "flash_num_tiles_q": flash_stats.num_tiles_q,
        "flash_num_tiles_k": flash_stats.num_tiles_k,
        "flash_num_tiles_total": flash_stats.num_tiles_total,
        "max_diff": max_diff,
        "correct": correct,
        "speedup": float(np.mean(std_times) / np.mean(flash_times))
        if np.mean(flash_times) > 0
        else 0.0,
    }

    logger.info(
        "seq=%d: std=%.3fs flash=%.3fs speedup=%.2fx saved=%.1f%% correct=%s",
        seq_len,
        np.mean(std_times),
        np.mean(flash_times),
        result["speedup"],
        flash_stats.memory_saved_pct,
        correct,
    )

    return result


def run_sequence_length_sweep(
    seq_lengths: List[int],
    batch_size: int = 1,
    num_heads: int = 8,
    head_dim: int = 128,
    block_size: int = 64,
    output_dir: str = "results",
) -> List[Dict]:
    """Run comparison across multiple sequence lengths.

    Args:
        seq_lengths: List of sequence lengths to test.
        batch_size: Batch size.
        num_heads: Number of heads.
        head_dim: Head dimension.
        block_size: FlashAttention block size.
        output_dir: Output directory.

    Returns:
        List of result dicts.
    """
    results: List[Dict] = []
    for seq_len in seq_lengths:
        result = run_comparison(
            batch_size=batch_size,
            num_heads=num_heads,
            seq_len=seq_len,
            head_dim=head_dim,
            block_size=block_size,
        )
        results.append(result)
    return results


def save_csv(results: List[Dict], filepath: str) -> None:
    """Save results to CSV file.

    Args:
        results: List of result dicts.
        filepath: Output file path.
    """
    if not results:
        logger.warning("No results to save to CSV")
        return

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fieldnames = list(results[0].keys())
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)
    logger.info("Saved CSV: %s", filepath)


def save_json(results: List[Dict], filepath: str) -> None:
    """Save results to JSON file.

    Args:
        results: List of result dicts.
        filepath: Output file path.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Saved JSON: %s", filepath)


def main() -> None:
    """Run the FlashAttention experiment."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    print()
    print("=" * 60)
    print("  Experiment: Standard Attention vs FlashAttention")
    print("=" * 60)
    print()

    # Configuration
    seq_lengths = [64, 128, 256, 512, 1024, 2048]
    batch_size = 1
    num_heads = 8
    head_dim = 128
    block_size = 64
    output_dir = "results"

    print(f"  Batch size:   {batch_size}")
    print(f"  Num heads:    {num_heads}")
    print(f"  Head dim:     {head_dim}")
    print(f"  Block size:   {block_size}")
    print(f"  Seq lengths:  {seq_lengths}")
    print()

    # Run experiments
    results = run_sequence_length_sweep(
        seq_lengths=seq_lengths,
        batch_size=batch_size,
        num_heads=num_heads,
        head_dim=head_dim,
        block_size=block_size,
        output_dir=output_dir,
    )

    # Save outputs
    csv_path = os.path.join(output_dir, "experiment_flash_attention.csv")
    json_path = os.path.join(output_dir, "experiment_flash_attention.json")
    png_path = os.path.join(output_dir, "experiment_flash_attention.png")

    save_csv(results, csv_path)
    save_json(results, json_path)

    # Generate plots
    try:
        plot_flash_attention_results(results, png_path)
        logger.info("Saved PNG: %s", png_path)
    except Exception as e:
        logger.warning("Failed to generate plots: %s", e)

    # Summary table
    print()
    print("=" * 60)
    print("  Summary")
    print("=" * 60)
    print()
    header = "  {:>6} {:>10} {:>10} {:>8} {:>10} {:>10} {:>8}".format(
        "Seq", "Std(ms)", "Flash(ms)", "Speedup", "Std Mem", "Flash Mem", "Saved"
    )
    print(header)
    print("  " + "-" * 72)
    for r in results:
        print(
            "  {:>6d} {:>10.2f} {:>10.2f} {:>7.2f}x {:>8.2f}MB {:>8.2f}MB {:>7.1f}%".format(
                r["seq_len"],
                r["standard_latency_avg"] * 1000,
                r["flash_latency_avg"] * 1000,
                r["speedup"],
                r["standard_peak_mb"],
                r["flash_peak_mb"],
                r["flash_saved_pct"],
            )
        )
    print()
    print(f"  CSV:  {csv_path}")
    print(f"  JSON: {json_path}")
    print(f"  PNG:  {png_path}")
    print()
    print("=" * 60)
    print("  FlashAttention Experiment Complete!")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
