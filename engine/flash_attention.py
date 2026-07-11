"""FlashAttention educational implementation (CPU version).

Implements both standard attention and FlashAttention with tiling + online softmax,
with detailed memory tracking for educational purposes.

Standard Attention:
    Scores -> Softmax -> Attention (three explicit steps, full matrices stored)

FlashAttention:
    Tiled QK computation + online softmax + incremental output accumulation.
    Never stores full QK or Softmax matrices.

Memory is measured in bytes, modeling GPU HBM usage.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AttentionStats:
    """Statistics from an attention forward pass."""

    latency_seconds: float = 0.0
    memory_peak_bytes: int = 0
    memory_q: int = 0
    memory_k: int = 0
    memory_v: int = 0
    memory_qk: int = 0
    memory_softmax: int = 0
    memory_output: int = 0
    memory_saved_bytes: int = 0
    memory_saved_pct: float = 0.0
    num_tiles_q: int = 0
    num_tiles_k: int = 0
    num_tiles_total: int = 0
    block_size: int = 0
    seq_len_q: int = 0
    seq_len_k: int = 0
    head_dim: int = 0
    num_heads: int = 0
    batch_size: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "latency_seconds": self.latency_seconds,
            "memory_peak_bytes": self.memory_peak_bytes,
            "memory_peak_mb": self.memory_peak_bytes / (1024 * 1024),
            "memory_q_mb": self.memory_q / (1024 * 1024),
            "memory_k_mb": self.memory_k / (1024 * 1024),
            "memory_v_mb": self.memory_v / (1024 * 1024),
            "memory_qk_mb": self.memory_qk / (1024 * 1024),
            "memory_softmax_mb": self.memory_softmax / (1024 * 1024),
            "memory_output_mb": self.memory_output / (1024 * 1024),
            "memory_saved_bytes": self.memory_saved_bytes,
            "memory_saved_mb": self.memory_saved_bytes / (1024 * 1024),
            "memory_saved_pct": self.memory_saved_pct,
            "num_tiles_q": self.num_tiles_q,
            "num_tiles_k": self.num_tiles_k,
            "num_tiles_total": self.num_tiles_total,
            "block_size": self.block_size,
            "seq_len_q": self.seq_len_q,
            "seq_len_k": self.seq_len_k,
            "head_dim": self.head_dim,
            "num_heads": self.num_heads,
            "batch_size": self.batch_size,
        }


class StandardAttention:
    """Standard scaled dot-product attention.

    Three explicit steps:
        1. Q @ K^T / sqrt(d)  -> Scores matrix (full N x N)
        2. softmax(Scores)     -> Softmax matrix (full N x N)
        3. Softmax @ V         -> Output matrix

    Stores all intermediate matrices (QK, Softmax) for educational clarity.
    Memory complexity: O(N^2) dominated by attention scores matrix.
    """

    def __init__(self, dtype: np.dtype = np.float32):
        """Initialize standard attention.

        Args:
            dtype: Data type for computations.
        """
        self.dtype = dtype
        self.qk_matrix: Optional[np.ndarray] = None
        self.softmax_matrix: Optional[np.ndarray] = None
        self.output_matrix: Optional[np.ndarray] = None
        logger.debug("StandardAttention initialized (dtype=%s)", dtype)

    def forward(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, AttentionStats]:
        """Forward pass of standard attention.

        Args:
            Q: Query tensor of shape [batch, num_heads, seq_q, head_dim].
            K: Key tensor of shape [batch, num_heads, seq_k, head_dim].
            V: Value tensor of shape [batch, num_heads, seq_k, head_dim].
            mask: Optional boolean mask [seq_q, seq_k], True = attend.

        Returns:
            Tuple of (output, AttentionStats).
        """
        stats = AttentionStats()
        start = time.time()

        batch_size, num_heads, seq_q, head_dim = Q.shape
        _, _, seq_k, _ = K.shape

        stats.batch_size = batch_size
        stats.num_heads = num_heads
        stats.seq_len_q = seq_q
        stats.seq_len_k = seq_k
        stats.head_dim = head_dim

        # Memory for input tensors (Q, K, V)
        stats.memory_q = Q.nbytes
        stats.memory_k = K.nbytes
        stats.memory_v = V.nbytes

        logger.debug(
            "StandardAttention forward: Q=%s K=%s V=%s",
            Q.shape, K.shape, V.shape,
        )

        # Step 1: Q @ K^T / sqrt(d)  ->  QK matrix [batch, heads, seq_q, seq_k]
        scale = 1.0 / math.sqrt(head_dim)
        qk = np.matmul(Q, K.swapaxes(-2, -1)) * scale

        if mask is not None:
            qk = np.where(mask, qk, -np.inf)

        self.qk_matrix = qk
        stats.memory_qk = qk.nbytes

        # Peak memory so far: Q + K + V + QK
        current_peak = (
            stats.memory_q + stats.memory_k + stats.memory_v + stats.memory_qk
        )
        stats.memory_peak_bytes = current_peak
        logger.debug("Step 1 (QK): peak=%.2f MB", current_peak / (1024 * 1024))

        # Step 2: Softmax
        qk_max = np.max(qk, axis=-1, keepdims=True)
        qk_exp = np.exp(qk - qk_max)
        qk_sum = np.sum(qk_exp, axis=-1, keepdims=True)
        softmax = qk_exp / qk_sum

        self.softmax_matrix = softmax
        stats.memory_softmax = softmax.nbytes

        # Peak: add softmax (qk still alive)
        peak_with_softmax = current_peak + stats.memory_softmax
        stats.memory_peak_bytes = peak_with_softmax
        logger.debug(
            "Step 2 (Softmax): peak=%.2f MB",
            peak_with_softmax / (1024 * 1024),
        )

        # Step 3: Softmax @ V  ->  Output [batch, heads, seq_q, head_dim]
        output = np.matmul(softmax, V)
        self.output_matrix = output
        stats.memory_output = output.nbytes

        # Peak: output is added (qk + softmax may be freed but we count max alive)
        peak_with_output = (
            stats.memory_q + stats.memory_k + stats.memory_v
            + stats.memory_qk + stats.memory_softmax + stats.memory_output
        )
        stats.memory_peak_bytes = peak_with_output
        logger.debug(
            "Step 3 (Output): peak=%.2f MB",
            peak_with_output / (1024 * 1024),
        )

        stats.latency_seconds = time.time() - start
        stats.memory_saved_pct = 0.0  # baseline

        logger.info(
            "StandardAttention done: %.4fs, peak=%.2f MB",
            stats.latency_seconds,
            stats.memory_peak_bytes / (1024 * 1024),
        )

        return output, stats


class FlashAttentionCPU:
    """FlashAttention educational implementation (CPU, numpy).

    Key ideas:
    - Split Q and K/V into tiles (blocks)
    - For each Q tile, iterate over K/V tiles
    - Compute partial attention scores for this tile
    - Use online softmax to merge results across K tiles
    - Accumulate output incrementally
    - NEVER materialize the full N x N attention matrix

    Online Softmax algorithm:
    - Track running max (m) and running sum (l) per row
    - When a new tile arrives, update m and l with proper rescaling
    - Output is accumulated with same rescaling factor

    Memory complexity: O(N * d * B) where B = block_size, much smaller than O(N^2).
    """

    def __init__(
        self,
        block_size: int = 64,
        dtype: np.dtype = np.float32,
    ):
        """Initialize FlashAttention.

        Args:
            block_size: Tile/block size for Q and K tiling.
            dtype: Data type for computations.
        """
        self.block_size = block_size
        self.dtype = dtype
        logger.debug(
            "FlashAttentionCPU initialized (block_size=%d, dtype=%s)",
            block_size, dtype,
        )

    def forward(
        self,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, AttentionStats]:
        """Forward pass of FlashAttention with tiling + online softmax.

        Args:
            Q: Query tensor [batch, num_heads, seq_q, head_dim].
            K: Key tensor [batch, num_heads, seq_k, head_dim].
            V: Value tensor [batch, num_heads, seq_k, head_dim].
            mask: Optional boolean mask [seq_q, seq_k].

        Returns:
            Tuple of (output, AttentionStats).
        """
        stats = AttentionStats()
        start = time.time()

        batch_size, num_heads, seq_q, head_dim = Q.shape
        _, _, seq_k, _ = K.shape

        stats.batch_size = batch_size
        stats.num_heads = num_heads
        stats.seq_len_q = seq_q
        stats.seq_len_k = seq_k
        stats.head_dim = head_dim
        stats.block_size = self.block_size

        # Input memory
        stats.memory_q = Q.nbytes
        stats.memory_k = K.nbytes
        stats.memory_v = V.nbytes

        Bq = self.block_size  # block size for Q rows
        Bk = self.block_size  # block size for K columns

        num_tiles_q = (seq_q + Bq - 1) // Bq
        num_tiles_k = (seq_k + Bk - 1) // Bk
        stats.num_tiles_q = num_tiles_q
        stats.num_tiles_k = num_tiles_k
        stats.num_tiles_total = num_tiles_q * num_tiles_k

        logger.debug(
            "FlashAttention: seq_q=%d seq_k=%d dim=%d tiles_q=%d tiles_k=%d total_tiles=%d",
            seq_q, seq_k, head_dim,
            num_tiles_q, num_tiles_k, num_tiles_q * num_tiles_k,
        )

        # Output accumulator [batch, heads, seq_q, head_dim]
        output = np.zeros_like(Q, dtype=self.dtype)
        stats.memory_output = output.nbytes

        # Running online softmax state:
        #   m_i = running max of row i
        #   l_i = running sum of exp(s - m_i) for row i
        m = np.full(
            (batch_size, num_heads, seq_q, 1),
            -np.inf,
            dtype=self.dtype,
        )
        l = np.zeros(
            (batch_size, num_heads, seq_q, 1),
            dtype=self.dtype,
        )

        # Track peak memory
        # Q, K, V + output + m + l + tile_qk + tile_softmax + tile_output
        tile_qk_size = batch_size * num_heads * Bq * Bk * np.dtype(self.dtype).itemsize
        tile_softmax_size = tile_qk_size
        tile_output_size = batch_size * num_heads * Bq * head_dim * np.dtype(self.dtype).itemsize

        peak_tile_allocation = (
            stats.memory_q + stats.memory_k + stats.memory_v
            + stats.memory_output
            + m.nbytes + l.nbytes
            + tile_qk_size + tile_softmax_size + tile_output_size
        )
        stats.memory_peak_bytes = peak_tile_allocation
        stats.memory_qk = 0  # never stores full QK
        stats.memory_softmax = 0  # never stores full softmax

        scale = 1.0 / math.sqrt(head_dim)

        # Outer loop: iterate over Q tiles (rows)
        for qi in range(num_tiles_q):
            q_start = qi * Bq
            q_end = min(q_start + Bq, seq_q)
            q_tile = Q[:, :, q_start:q_end, :]

            # Inner loop: iterate over K/V tiles (columns)
            for kj in range(num_tiles_k):
                k_start = kj * Bk
                k_end = min(k_start + Bk, seq_k)
                k_tile = K[:, :, k_start:k_end, :]
                v_tile = V[:, :, k_start:k_end, :]

                # Step 1: Compute tile QK = Q_tile @ K_tile^T / sqrt(d)
                tile_qk = np.matmul(q_tile, k_tile.swapaxes(-2, -1)) * scale

                # Apply mask if provided
                if mask is not None:
                    tile_mask = mask[q_start:q_end, k_start:k_end]
                    tile_qk = np.where(tile_mask, tile_qk, -np.inf)

                # Step 2: Online softmax update
                # New block max
                tile_max = np.max(tile_qk, axis=-1, keepdims=True)

                # Compute exp of tile scores (numerically stable)
                tile_exp = np.exp(tile_qk - tile_max)
                tile_sum = np.sum(tile_exp, axis=-1, keepdims=True)

                # Update running max and sum (online softmax)
                # new_m = max(old_m, tile_max)
                # scale_old = exp(old_m - new_m)
                # scale_new = exp(tile_max - new_m)
                # new_l = old_l * scale_old + tile_sum * scale_new
                new_m = np.maximum(m[:, :, q_start:q_end, :], tile_max)

                old_scale = np.exp(
                    m[:, :, q_start:q_end, :] - new_m
                )
                new_scale = np.exp(tile_max - new_m)

                new_l = (
                    l[:, :, q_start:q_end, :] * old_scale
                    + tile_sum * new_scale
                )

                # Step 3: Accumulate output
                # tile_output_tile = tile_exp @ V_tile
                tile_output = np.matmul(tile_exp, v_tile)

                # Rescale old output and add new tile contribution
                output[:, :, q_start:q_end, :] = (
                    output[:, :, q_start:q_end, :] * old_scale
                    + tile_output * new_scale
                )

                # Update running state
                m[:, :, q_start:q_end, :] = new_m
                l[:, :, q_start:q_end, :] = new_l

        # Final normalization: divide by running sum
        # Avoid division by zero
        safe_l = np.where(l > 0, l, 1.0)
        output = output / safe_l

        stats.memory_peak_bytes = peak_tile_allocation
        stats.latency_seconds = time.time() - start

        # Memory saved vs standard attention
        # Standard peak = Q + K + V + QK + Softmax + Output
        qk_full_size = (
            batch_size * num_heads * seq_q * seq_k
            * np.dtype(self.dtype).itemsize
        )
        softmax_full_size = qk_full_size
        standard_peak = (
            stats.memory_q + stats.memory_k + stats.memory_v
            + qk_full_size + softmax_full_size + stats.memory_output
        )
        stats.memory_saved_bytes = standard_peak - peak_tile_allocation
        stats.memory_saved_pct = (
            stats.memory_saved_bytes / standard_peak * 100.0
            if standard_peak > 0
            else 0.0
        )

        logger.info(
            "FlashAttention done: %.4fs, peak=%.2f MB, saved=%.1f%%, tiles=%d",
            stats.latency_seconds,
            stats.memory_peak_bytes / (1024 * 1024),
            stats.memory_saved_pct,
            stats.num_tiles_total,
        )

        return output, stats


def attention_standard(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: Optional[np.ndarray] = None,
    dtype: np.dtype = np.float32,
) -> Tuple[np.ndarray, AttentionStats]:
    """Convenience function for standard attention.

    Args:
        Q: Query [batch, heads, seq_q, dim].
        K: Key [batch, heads, seq_k, dim].
        V: Value [batch, heads, seq_k, dim].
        mask: Optional boolean mask.
        dtype: Computation dtype.

    Returns:
        (output, stats).
    """
    attn = StandardAttention(dtype=dtype)
    return attn.forward(Q, K, V, mask=mask)


def attention_flash(
    Q: np.ndarray,
    K: np.ndarray,
    V: np.ndarray,
    mask: Optional[np.ndarray] = None,
    block_size: int = 64,
    dtype: np.dtype = np.float32,
) -> Tuple[np.ndarray, AttentionStats]:
    """Convenience function for FlashAttention.

    Args:
        Q: Query [batch, heads, seq_q, dim].
        K: Key [batch, heads, seq_k, dim].
        V: Value [batch, heads, seq_k, dim].
        mask: Optional boolean mask.
        block_size: Tile size.
        dtype: Computation dtype.

    Returns:
        (output, stats).
    """
    attn = FlashAttentionCPU(block_size=block_size, dtype=dtype)
    return attn.forward(Q, K, V, mask=mask)


def verify_attention(
    seq_len: int = 256,
    head_dim: int = 64,
    num_heads: int = 2,
    batch_size: int = 1,
    block_size: int = 64,
    atol: float = 1e-4,
) -> bool:
    """Verify that FlashAttention matches StandardAttention.

    Args:
        seq_len: Sequence length.
        head_dim: Head dimension.
        num_heads: Number of attention heads.
        batch_size: Batch size.
        block_size: FlashAttention block size.
        atol: Absolute tolerance for comparison.

    Returns:
        True if outputs match within tolerance.
    """
    np.random.seed(42)
    Q = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
    K = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)
    V = np.random.randn(batch_size, num_heads, seq_len, head_dim).astype(np.float32)

    out_std, stats_std = attention_standard(Q, K, V)
    out_flash, stats_flash = attention_flash(Q, K, V, block_size=block_size)

    max_diff = np.max(np.abs(out_std - out_flash))
    match = max_diff < atol

    logger.info(
        "Verification: max_diff=%.2e, match=%s",
        max_diff, match,
    )
    logger.info(
        "Standard peak: %.2f MB | Flash peak: %.2f MB | saved: %.1f%%",
        stats_std.memory_peak_bytes / (1024 * 1024),
        stats_flash.memory_peak_bytes / (1024 * 1024),
        stats_flash.memory_saved_pct,
    )

    return match


__all__ = [
    "StandardAttention",
    "FlashAttentionCPU",
    "AttentionStats",
    "attention_standard",
    "attention_flash",
    "verify_attention",
]
