"""FlashAttention visualization: memory curve, tile flow, memory compare.

Generates a multi-panel figure with:
  1. Latency comparison (Standard vs Flash) across sequence lengths
  2. Peak Memory comparison (Standard vs Flash) across sequence lengths
  3. Memory Saved % across sequence lengths
  4. Speedup curve
  5. Memory growth curve (O(N^2) vs O(N*B))
  6. Tile execution flow diagram
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def plot_flash_attention_results(
    results: List[Dict],
    output_path: str,
    title: Optional[str] = None,
) -> None:
    """Generate comprehensive FlashAttention comparison plots.

    Args:
        results: List of result dictionaries from the experiment.
        output_path: Path to save the PNG file.
        title: Optional plot title.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        logger.warning("matplotlib not available, skipping plots")
        return

    seq_lens = [r["seq_len"] for r in results]
    std_lat = [r["standard_latency_avg"] * 1000 for r in results]
    flash_lat = [r["flash_latency_avg"] * 1000 for r in results]
    std_mem = [r["standard_peak_mb"] for r in results]
    flash_mem = [r["flash_peak_mb"] for r in results]
    saved_pct = [r["flash_saved_pct"] for r in results]
    speedup = [r["speedup"] for r in results]
    num_tiles = [r["flash_num_tiles_total"] for r in results]
    block_size = results[0]["block_size"] if results else 64

    # Create figure with 3 rows x 2 cols
    fig = plt.figure(figsize=(14, 16))
    if title is None:
        title = "Standard Attention vs FlashAttention"
    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)

    x = np.arange(len(seq_lens))
    width = 0.35

    # ---- Panel 1: Latency comparison ----
    ax1 = plt.subplot(3, 2, 1)
    ax1.bar(x - width / 2, std_lat, width, label="Standard",
            color="#FF6B6B", alpha=0.85)
    ax1.bar(x + width / 2, flash_lat, width, label="FlashAttention",
            color="#4ECDC4", alpha=0.85)
    ax1.set_xlabel("Sequence Length")
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("Latency Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(s) for s in seq_lens])
    ax1.legend(fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # ---- Panel 2: Peak Memory comparison ----
    ax2 = plt.subplot(3, 2, 2)
    ax2.bar(x - width / 2, std_mem, width, label="Standard",
            color="#FF6B6B", alpha=0.85)
    ax2.bar(x + width / 2, flash_mem, width, label="FlashAttention",
            color="#4ECDC4", alpha=0.85)
    ax2.set_xlabel("Sequence Length")
    ax2.set_ylabel("Peak Memory (MB)")
    ax2.set_title("Peak Memory Comparison")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(s) for s in seq_lens])
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    # ---- Panel 3: Memory Saved % ----
    ax3 = plt.subplot(3, 2, 3)
    ax3.plot(seq_lens, saved_pct, "o-", color="#2ECC71", linewidth=2,
             markersize=8)
    ax3.fill_between(seq_lens, saved_pct, alpha=0.2, color="#2ECC71")
    ax3.set_xlabel("Sequence Length")
    ax3.set_ylabel("Memory Saved (%)")
    ax3.set_title("Memory Saved by FlashAttention")
    ax3.grid(alpha=0.3)
    ax3.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    # ---- Panel 4: Speedup ----
    ax4 = plt.subplot(3, 2, 4)
    ax4.plot(seq_lens, speedup, "s-", color="#9B59B6", linewidth=2,
             markersize=8)
    ax4.fill_between(seq_lens, speedup, alpha=0.2, color="#9B59B6")
    ax4.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="1x (baseline)")
    ax4.set_xlabel("Sequence Length")
    ax4.set_ylabel("Speedup (x)")
    ax4.set_title("Speedup (Standard / Flash)")
    ax4.legend(fontsize=9)
    ax4.grid(alpha=0.3)

    # ---- Panel 5: Memory growth curve (O(N^2) vs O(N*B)) ----
    ax5 = plt.subplot(3, 2, 5)
    N = np.linspace(64, 4096, 200)
    d = 128
    B = block_size
    # Standard: QK + softmax + output
    std_curve = (2 * N * N + N * d) * 4 / (1024 * 1024)
    # Flash: tile QK + tile softmax + tile output + running state
    flash_curve = (2 * N * B + N * d + N + N) * 4 / (1024 * 1024)

    ax5.plot(N, std_curve, label="Standard (O(N^2))",
             color="#FF6B6B", linewidth=2.5)
    ax5.plot(N, flash_curve, label="FlashAttention (O(N*B))",
             color="#4ECDC4", linewidth=2.5)
    ax5.fill_between(N, flash_curve, std_curve, alpha=0.15, color="green",
                     label="Memory Saved")
    # Mark actual data points
    ax5.scatter(seq_lens, std_mem, color="#FF6B6B", zorder=5, s=50,
                edgecolors="black", linewidth=0.5)
    ax5.scatter(seq_lens, flash_mem, color="#4ECDC4", zorder=5, s=50,
                edgecolors="black", linewidth=0.5)
    ax5.set_xlabel("Sequence Length N")
    ax5.set_ylabel("Memory (MB)")
    ax5.set_title(f"Memory Growth: O(N^2) vs O(N * {B})")
    ax5.legend(fontsize=9)
    ax5.grid(alpha=0.3)

    # ---- Panel 6: Tile execution flow ----
    ax6 = plt.subplot(3, 2, 6)
    _draw_tile_diagram(ax6, seq_len=max(seq_lens) if seq_lens else 1024,
                       block_size=B, num_tiles=max(num_tiles) if num_tiles else 256)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Plot saved to %s", output_path)


def _draw_tile_diagram(
    ax,
    seq_len: int,
    block_size: int,
    num_tiles: int,
) -> None:
    """Draw a tile-based computation flow diagram.

    Args:
        ax: Matplotlib axes to draw on.
        seq_len: Total sequence length.
        block_size: Tile/block size.
        num_tiles: Total number of tiles.
    """
    from matplotlib.patches import Rectangle, FancyArrowPatch

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")

    # Draw Q tiles (rows) on the left
    n_q_tiles = min(6, (seq_len + block_size - 1) // block_size)
    q_x = 0.3
    q_w = 1.8
    q_h = 6.5 / n_q_tiles
    for i in range(n_q_tiles):
        y = 8.2 - (i + 1) * q_h
        rect = Rectangle(
            (q_x, y), q_w, q_h * 0.88,
            facecolor="#FFEAA7", edgecolor="#FDCB6E", linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(q_x + q_w / 2, y + q_h * 0.44, f"Q{i}",
                ha="center", va="center", fontsize=8, fontweight="bold",
                color="#E17055")

    # Draw K tiles (columns) on top
    n_k_tiles = min(6, (seq_len + block_size - 1) // block_size)
    k_y = 8.7
    k_w = 6.2 / n_k_tiles
    k_h = 0.9
    for j in range(n_k_tiles):
        x = 2.6 + j * k_w
        rect = Rectangle(
            (x, k_y), k_w * 0.9, k_h,
            facecolor="#A29BFE", edgecolor="#6C5CE7", linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(x + k_w * 0.45, k_y + k_h / 2, f"K{j}",
                ha="center", va="center", fontsize=8, fontweight="bold",
                color="white")

    # Draw attention tile grid
    grid_x = 2.6
    grid_y = 1.7
    grid_w = 6.2
    grid_h = 6.5
    tile_w = grid_w / n_k_tiles
    tile_h = grid_h / n_q_tiles

    # Highlight tiles in computation order (first 6 highlighted)
    tile_idx = 0
    for i in range(n_q_tiles):
        for j in range(n_k_tiles):
            tx = grid_x + j * tile_w
            ty = grid_y + (n_q_tiles - 1 - i) * tile_h
            if tile_idx < 6:
                alpha = 0.8 - tile_idx * 0.1
                color = "#00B894"
            else:
                alpha = 0.2
                color = "#B2BEC3"
            rect = Rectangle(
                (tx + 0.03, ty + 0.03),
                tile_w * 0.94, tile_h * 0.94,
                facecolor=color, alpha=alpha,
                edgecolor="#2D3436", linewidth=0.5,
            )
            ax.add_patch(rect)
            tile_idx += 1

    # Arrow showing computation flow direction
    ax.annotate(
        "", xy=(grid_x + grid_w + 0.2, grid_y + grid_h / 2),
        xytext=(grid_x - 0.3, grid_y + grid_h / 2),
        arrowprops=dict(arrowstyle="->", color="#E17055", lw=2),
    )
    ax.text(grid_x + grid_w / 2, grid_y - 0.5,
            f"Tiles: {n_q_tiles} x {n_k_tiles} = {n_q_tiles * n_k_tiles}",
            ha="center", fontsize=9, color="#2D3436", fontweight="bold")

    ax.set_title("Tile Execution Flow")
    ax.axis("off")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#FFEAA7", edgecolor="#FDCB6E", label="Q Tiles"),
        Patch(facecolor="#A29BFE", edgecolor="#6C5CE7", label="K Tiles"),
        Patch(facecolor="#00B894", alpha=0.7, label="Computed Tile"),
        Patch(facecolor="#B2BEC3", alpha=0.2, label="Remaining Tile"),
    ]
    ax.legend(handles=legend_elements, loc="lower center",
              bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=7, frameon=False)


__all__ = ["plot_flash_attention_results"]
