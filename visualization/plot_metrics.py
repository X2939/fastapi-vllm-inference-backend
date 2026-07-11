"""Unified visualization entry point.

Generates all standard plots from results directory.
Each plot type is in its own module for single responsibility.
"""
from __future__ import annotations

import csv
import logging
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def _load_csv(filepath: str) -> List[Dict]:
    """Load a CSV file into a list of dicts."""
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def plot_all(results_dir: str) -> None:
    """Generate all standard plots from results directory.

    Args:
        results_dir: Directory containing metrics.csv and timeline.csv.
    """
    # Load data
    metrics = _load_csv(os.path.join(results_dir, "metrics.csv"))
    timeline = _load_csv(os.path.join(results_dir, "timeline.csv"))

    # 1. Latency distribution
    if metrics:
        latencies = [float(r["latency"]) for r in metrics
                     if r.get("latency") and r["latency"] not in ("", "None")]
        ttfts = [float(r["ttft"]) for r in metrics
                 if r.get("ttft") and r["ttft"] not in ("", "None")]
        tpots = [float(r["tpot"]) for r in metrics
                 if r.get("tpot") and r["tpot"] not in ("", "None")]

        _plot_distribution(latencies, os.path.join(results_dir, "latency_distribution.png"),
                           "End-to-End Latency", "Latency (s)")
        _plot_distribution(ttfts, os.path.join(results_dir, "ttft_distribution.png"),
                           "Time To First Token (TTFT)", "TTFT (s)")
        _plot_distribution(tpots, os.path.join(results_dir, "tpot_distribution.png"),
                           "Time Per Output Token (TPOT)", "TPOT (s)")

    # 2. Timeline plots
    if timeline:
        steps = [int(r["step"]) for r in timeline]
        batch_sizes = [int(r["batch_size"]) for r in timeline]
        prefill_counts = [int(r["prefill_count"]) for r in timeline]
        decode_counts = [int(r["decode_count"]) for r in timeline]
        memory = [int(r["memory_used"]) for r in timeline]
        busy = [1 if r["busy"] == "True" else 0 for r in timeline]

        # GPU Usage
        _plot_line(steps, [b * 100 for b in busy],
                   os.path.join(results_dir, "gpu_usage.png"),
                   "GPU Usage Over Time", "Step", "GPU Busy (%)")

        # Batch Size
        _plot_stacked_bar(steps, prefill_counts, decode_counts,
                          os.path.join(results_dir, "batch_size.png"),
                          "Batch Size Over Time", "Prefill", "Decode")

        # Memory Usage
        _plot_line(steps, memory,
                   os.path.join(results_dir, "memory_usage.png"),
                   "Memory Usage Over Time", "Step", "Memory (tokens)")

        # Queue length (from timeline if available)
        if "waiting" in timeline[0]:
            waiting = [int(r.get("waiting", 0)) for r in timeline]
            running = [int(r.get("running", 0)) for r in timeline]
            finished = [int(r.get("finished", 0)) for r in timeline]
            _plot_multi_line(steps, [waiting, running, finished],
                             ["Waiting", "Running", "Finished"],
                             os.path.join(results_dir, "queue_length.png"),
                             "Queue Lengths Over Time", "Step", "Requests")

    # 3. Throughput / TTFT / TPOT / P95 / P99 bar charts
    summary_path = os.path.join(results_dir, "summary.json")
    if os.path.exists(summary_path):
        import json
        with open(summary_path, "r") as f:
            summary = json.load(f)
        _plot_summary_bars(summary, results_dir)

    logger.info(f"All plots saved to {results_dir}/")


def _plot_distribution(data: List[float], path: str, title: str, xlabel: str):
    """Plot histogram + CDF."""
    if not data:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.hist(data, bins=30, color="#2196F3", alpha=0.7, edgecolor="white")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Count")
    ax1.set_title(f"{title} - Histogram")
    ax1.grid(True, alpha=0.3)

    sorted_data = sorted(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data) * 100
    ax2.plot(sorted_data, cdf, color="#4CAF50", linewidth=2)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("CDF (%)")
    ax2.set_title(f"{title} - CDF")
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def _plot_line(x, y, path, title, xlabel, ylabel):
    """Plot a simple line chart."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, y, color="#2196F3", linewidth=2)
    ax.fill_between(x, y, alpha=0.1, color="#2196F3")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def _plot_stacked_bar(x, y1, y2, path, title, label1, label2):
    """Plot stacked bar chart."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(x, y1, label=label1, color="#FF9800", alpha=0.8)
    ax.bar(x, y2, bottom=y1, label=label2, color="#4CAF50", alpha=0.8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Batch Size")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def _plot_multi_line(x, ys, labels, path, title, xlabel, ylabel):
    """Plot multiple lines on one chart."""
    colors = ["#F44336", "#FF9800", "#4CAF50", "#2196F3", "#9C27B0"]
    fig, ax = plt.subplots(figsize=(10, 4))
    for y, label, color in zip(ys, labels, colors):
        ax.plot(x, y, label=label, color=color, linewidth=2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def _plot_summary_bars(summary: dict, results_dir: str):
    """Plot summary metric bars (throughput, TTFT, TPOT, P95, P99)."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    # Throughput
    ax = axes[0, 0]
    ax.bar(["Throughput"], [summary.get("throughput", 0)], color="#2196F3")
    ax.set_title("Throughput (tok/s)")
    ax.grid(True, alpha=0.3, axis="y")

    # TTFT
    ax = axes[0, 1]
    ttft = summary.get("ttft", {})
    ax.bar(["avg", "p50", "p95", "p99"],
           [ttft.get("avg", 0), ttft.get("p50", 0),
            ttft.get("p95", 0), ttft.get("p99", 0)],
           color="#4CAF50")
    ax.set_title("TTFT (s)")
    ax.grid(True, alpha=0.3, axis="y")

    # TPOT
    ax = axes[0, 2]
    tpot = summary.get("tpot", {})
    ax.bar(["avg", "p50", "p95", "p99"],
           [tpot.get("avg", 0), tpot.get("p50", 0),
            tpot.get("p95", 0), tpot.get("p99", 0)],
           color="#FF9800")
    ax.set_title("TPOT (s)")
    ax.grid(True, alpha=0.3, axis="y")

    # Latency
    ax = axes[1, 0]
    lat = summary.get("latency", {})
    ax.bar(["avg", "p50", "p95", "p99"],
           [lat.get("avg", 0), lat.get("p50", 0),
            lat.get("p95", 0), lat.get("p99", 0)],
           color="#F44336")
    ax.set_title("Latency (s)")
    ax.grid(True, alpha=0.3, axis="y")

    # GPU Occupancy
    ax = axes[1, 1]
    occ = summary.get("gpu_occupancy", 0) * 100
    ax.bar(["GPU Occ"], [occ], color="#9C27B0")
    ax.set_title("GPU Occupancy (%)")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis="y")

    # Admission Rate
    ax = axes[1, 2]
    adm = summary.get("admission_rate", 0) * 100
    ax.bar(["Admission"], [adm], color="#00BCD4")
    ax.set_title("Admission Rate (%)")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "summary_bars.png"), dpi=150)
    plt.close(fig)
