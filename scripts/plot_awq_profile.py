#!/usr/bin/env python3
"""Create publication-ready summary plots for the matched AWQ profile."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ORDER = ("bf16", "awq_int4", "awq_marlin")
LABELS = ("BF16", "AWQ (generic)", "AWQ-Marlin")
COLORS = ("#4C78A8", "#E45756", "#54A24B")


def extract(pattern: str, text: str) -> float:
    match = re.search(pattern, text)
    if not match:
        raise RuntimeError(f"missing log field: {pattern}")
    return float(match.group(1).replace(",", ""))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    analysis = json.loads((root / "analysis.json").read_text(encoding="utf-8"))
    summaries = {item["variant"]: item for item in analysis["variants"]}

    latency_ms = []
    model_gib = []
    kv_tokens = []
    kernel_families = []
    family_order = ("quantized_linear", "dense_linear", "attention", "memory_copy", "elementwise", "other")
    for name in ORDER:
        metadata = json.loads((root / name / "metadata.json").read_text(encoding="utf-8"))
        latency_ms.append(metadata["warmups"][-1]["elapsed_s_with_profiler_overhead"] * 1000)
        log = (root / name / "server.log").read_text(encoding="utf-8", errors="replace")
        model_gib.append(extract(r"Model loading took ([0-9.]+) GiB", log))
        kv_tokens.append(extract(r"GPU KV cache size: ([0-9,]+) tokens", log))
        families = summaries[name]["kernel_families"]
        kernel_families.append([families.get(family, {}).get("total_ms", 0.0) for family in family_order])

    plt.rcParams.update({"font.size": 10, "axes.titleweight": "bold", "figure.dpi": 150})
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)

    x = np.arange(len(ORDER))
    bars = axes[0].bar(x, latency_ms, color=COLORS)
    axes[0].set_title("Second warm-up request latency (unprofiled)")
    axes[0].set_ylabel("Elapsed time (ms, lower is better)")
    axes[0].set_xticks(x, LABELS, rotation=15, ha="right")
    axes[0].bar_label(bars, fmt="%.0f ms", padding=3)
    axes[0].set_ylim(0, max(latency_ms) * 1.17)

    family_colors = ("#F58518", "#4C78A8", "#72B7B2", "#B279A2", "#FF9DA6", "#9D755D")
    bottom = np.zeros(len(ORDER))
    matrix = np.array(kernel_families)
    for index, family in enumerate(family_order):
        values = matrix[:, index]
        axes[1].bar(x, values, bottom=bottom, label=family.replace("_", " "), color=family_colors[index])
        bottom += values
    axes[1].set_title("CUDA kernel time composition")
    axes[1].set_ylabel("Aggregated kernel time (ms)")
    axes[1].set_xticks(x, LABELS, rotation=15, ha="right")
    axes[1].legend(fontsize=8, frameon=False, loc="upper right")

    width = 0.36
    axes[2].bar(x - width / 2, model_gib, width, label="Model loading memory (GiB)", color="#4C78A8")
    ax2 = axes[2].twinx()
    capacity_bars = ax2.bar(x + width / 2, np.array(kv_tokens) / 1000, width, label="KV token capacity (thousands)", color="#54A24B")
    axes[2].set_title("Weight memory becomes KV capacity")
    axes[2].set_ylabel("Model memory (GiB)")
    ax2.set_ylabel("KV cache capacity (thousand tokens)")
    ax2.set_ylim(0, max(np.array(kv_tokens) / 1000) * 1.13)
    axes[2].set_xticks(x, LABELS, rotation=15, ha="right")
    handles1, labels1 = axes[2].get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    axes[2].legend(handles1 + handles2, labels1 + labels2, fontsize=8, frameon=False, loc="upper left")
    ax2.bar_label(capacity_bars, fmt="%.1fK", padding=3, fontsize=8)

    for axis in axes[:2]:
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].set_axisbelow(True)

    fig.suptitle("Qwen2.5-1.5B on RTX 3060 Laptop: AWQ backend root-cause profile", fontsize=13, fontweight="bold")
    for suffix in ("svg", "png"):
        fig.savefig(root / f"awq_profile_summary.{suffix}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
