"""Chunked Prefill visualization.

Plots:
  1. Timeline: prefill/decode interleaving for a sample long prompt
  2. Chunk size vs TTFT
  3. Chunk size vs Throughput
  4. GPU Occupancy comparison
"""
from __future__ import annotations

import logging
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_chunked_prefill_results(
    results: List[Dict],
    output_path: str,
    title: str = "Chunked Prefill Experiment Results",
) -> None:
    """Plot chunked prefill comparison across chunk sizes and categories.

    Args:
        results: List of result dicts from experiment.
        output_path: Path to save the PNG.
        title: Plot title.
    """
    categories = sorted(set(r["category"] for r in results))
    chunk_sizes = sorted(set(r["chunk_size"] for r in results))

    fig = plt.figure(figsize=(16, 12))

    # Grid layout: 2 rows x 2 cols
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)

    # ---- Subplot 1: Chunk Size vs TTFT (P95) ----
    ax1 = fig.add_subplot(gs[0, 0])
    _plot_ttft_by_chunk(ax1, results, categories, chunk_sizes)

    # ---- Subplot 2: Chunk Size vs Throughput ----
    ax2 = fig.add_subplot(gs[0, 1])
    _plot_throughput_by_chunk(ax2, results, categories, chunk_sizes)

    # ---- Subplot 3: GPU Occupancy ----
    ax3 = fig.add_subplot(gs[1, 0])
    _plot_gpu_occupancy(ax3, results, categories, chunk_sizes)

    # ---- Subplot 4: Latency (P95) ----
    ax4 = fig.add_subplot(gs[1, 1])
    _plot_latency_by_chunk(ax4, results, categories, chunk_sizes)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved plot: %s", output_path)


def _plot_ttft_by_chunk(ax, results, categories, chunk_sizes):
    """Plot TTFT P95 vs chunk size for each category."""
    colors = {"short": "#4CAF50", "medium": "#2196F3", "long": "#F44336"}
    markers = {"short": "o", "medium": "s", "long": "^"}

    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        cat_results.sort(key=lambda x: x["chunk_size"])
        x = [r["chunk_size"] if r["chunk_size"] > 0 else "std" for r in cat_results]
        y = [r["p95_ttft_s"] * 1000 for r in cat_results]
        ax.plot(
            range(len(x)), y,
            marker=markers.get(cat, "o"),
            color=colors.get(cat, "#333"),
            label=f"{cat.capitalize()} (avg_prompt={cat_results[0]['avg_prompt_length']})",
            linewidth=2,
            markersize=7,
        )

    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x)
    ax.set_xlabel("Chunk Size (tokens)", fontsize=10)
    ax.set_ylabel("TTFT P95 (ms)", fontsize=10)
    ax.set_title("Time To First Token (P95)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)


def _plot_throughput_by_chunk(ax, results, categories, chunk_sizes):
    """Plot Throughput vs chunk size for each category."""
    colors = {"short": "#4CAF50", "medium": "#2196F3", "long": "#F44336"}
    markers = {"short": "o", "medium": "s", "long": "^"}

    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        cat_results.sort(key=lambda x: x["chunk_size"])
        x = [r["chunk_size"] if r["chunk_size"] > 0 else "std" for r in cat_results]
        y = [r["throughput_tok_s"] for r in cat_results]
        ax.plot(
            range(len(x)), y,
            marker=markers.get(cat, "o"),
            color=colors.get(cat, "#333"),
            label=f"{cat.capitalize()}",
            linewidth=2,
            markersize=7,
        )

    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x)
    ax.set_xlabel("Chunk Size (tokens)", fontsize=10)
    ax.set_ylabel("Throughput (tokens/s)", fontsize=10)
    ax.set_title("Throughput", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)


def _plot_gpu_occupancy(ax, results, categories, chunk_sizes):
    """Plot GPU Occupancy comparison."""
    colors = {"short": "#4CAF50", "medium": "#2196F3", "long": "#F44336"}
    bar_width = 0.25
    x_pos = np.arange(len(chunk_sizes))

    for i, cat in enumerate(categories):
        cat_results = [r for r in results if r["category"] == cat]
        cat_results.sort(key=lambda x: x["chunk_size"])
        y = [r["gpu_occupancy"] * 100 for r in cat_results]
        ax.bar(
            x_pos + i * bar_width, y, bar_width,
            color=colors.get(cat, "#333"),
            alpha=0.8,
            label=f"{cat.capitalize()}",
        )

    labels = [str(cs) if cs > 0 else "std" for cs in sorted(chunk_sizes)]
    ax.set_xticks(x_pos + bar_width * (len(categories) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Chunk Size (tokens)", fontsize=10)
    ax.set_ylabel("GPU Occupancy (%)", fontsize=10)
    ax.set_title("GPU Utilization", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")


def _plot_latency_by_chunk(ax, results, categories, chunk_sizes):
    """Plot Latency P95 vs chunk size."""
    colors = {"short": "#4CAF50", "medium": "#2196F3", "long": "#F44336"}
    markers = {"short": "o", "medium": "s", "long": "^"}

    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        cat_results.sort(key=lambda x: x["chunk_size"])
        x = [r["chunk_size"] if r["chunk_size"] > 0 else "std" for r in cat_results]
        y = [r["p95_latency_s"] * 1000 for r in cat_results]
        ax.plot(
            range(len(x)), y,
            marker=markers.get(cat, "o"),
            color=colors.get(cat, "#333"),
            label=f"{cat.capitalize()}",
            linewidth=2,
            markersize=7,
        )

    ax.set_xticks(range(len(x)))
    ax.set_xticklabels(x)
    ax.set_xlabel("Chunk Size (tokens)", fontsize=10)
    ax.set_ylabel("Latency P95 (ms)", fontsize=10)
    ax.set_title("End-to-End Latency (P95)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, alpha=0.3)


def plot_prefill_decode_timeline(
    steps: List[Dict],
    output_path: str,
    title: str = "Prefill / Decode Timeline",
) -> None:
    """Plot a timeline showing prefill and decode phases across steps.

    This visualizes how Chunked Prefill interleaves prefill chunks with
    decode steps.

    Args:
        steps: List of step info dicts from engine.run().
        output_path: Path to save the PNG.
        title: Plot title.
    """
    fig, ax = plt.subplots(figsize=(14, 5))

    # Build cumulative time axis
    prefill_heights = []
    decode_heights = []
    step_labels = []
    cumulative_time = 0.0
    time_points = [0.0]

    for i, step in enumerate(steps):
        prefill_count = step.get("prefill", 0)
        decode_count = step.get("decode", 0)
        prefill_heights.append(prefill_count)
        decode_heights.append(decode_count)
        step_labels.append(f"S{i+1}")
        cumulative_time += step.get("step_duration", 0.001)
        time_points.append(cumulative_time)

    x = np.arange(len(steps))
    width = 0.6

    bars_prefill = ax.bar(
        x, prefill_heights, width,
        label="Prefill Requests",
        color="#F44336",
        alpha=0.8,
    )
    bars_decode = ax.bar(
        x, decode_heights, width,
        bottom=prefill_heights,
        label="Decode Requests",
        color="#4CAF50",
        alpha=0.8,
    )

    ax.set_xlabel("Step", fontsize=10)
    ax.set_ylabel("Number of Requests", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Label steps with step numbers
    if len(steps) <= 30:
        ax.set_xticks(x)
        ax.set_xticklabels(step_labels, fontsize=8)
    else:
        step = max(1, len(steps) // 20)
        ax.set_xticks(x[::step])
        ax.set_xticklabels(step_labels[::step], fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved timeline plot: %s", output_path)
