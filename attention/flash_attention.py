
from __future__ import annotations
import math
import time
from typing import Dict, List


def flash_attention_simulated(
    q: list[list[float]],
    k: list[list[float]],
    v: list[list[float]],
    block_size: int = 64,
) -> list[list[float]]:
    """Simulated FlashAttention with tiling.

    Key optimization:
    - Process attention in blocks to fit GPU SRAM
    - Avoid writing full attention matrix to HBM
    - Compute softmax in blocks and merge results

    This simulation models the tiling behavior without actual GPU kernels.
    Real FlashAttention uses CUDA kernels for this.
    """
    seq_len = len(q)
    d_k = len(q[0]) if q else 0

    # Process in blocks (tiling)
    output = [[0.0] * d_k for _ in range(seq_len)]
    row_sums = [0.0] * seq_len
    row_maxs = [float("-inf")] * seq_len

    num_blocks = (seq_len + block_size - 1) // block_size

    for block_start in range(0, seq_len, block_size):
        block_end = min(block_start + block_size, seq_len)

        for i in range(seq_len):
            # Compute attention for this block
            block_scores = []
            for j in range(block_start, block_end):
                score = sum(q[i][m] * k[j][m] for m in range(d_k))
                score = score / math.sqrt(d_k)
                block_scores.append(score)

            if not block_scores:
                continue

            # Local softmax
            block_max = max(block_scores)
            block_exp = [math.exp(s - block_max) for s in block_scores]
            block_sum = sum(block_exp)

            # Update global max and sum
            new_max = max(row_maxs[i], block_max)
            scale = math.exp(row_maxs[i] - new_max)
            block_scale = math.exp(block_max - new_max)

            row_sums[i] = row_sums[i] * scale + block_sum * block_scale
            row_maxs[i] = new_max

            # Accumulate output
            for m in range(d_k):
                weighted_sum = sum(block_exp[j] * v[block_start + j][m]
                                 for j in range(len(block_scores)))
                output[i][m] = output[i][m] * scale + weighted_sum * block_scale

    # Final normalization
    for i in range(seq_len):
        if row_sums[i] > 0:
            for m in range(d_k):
                output[i][m] = output[i][m] / row_sums[i]

    return output


def benchmark_flash_attention(
    seq_lengths: List[int],
    d: int = 64,
    block_size: int = 64,
) -> Dict:
    """Benchmark FlashAttention simulation."""
    import random
    results = {}

    for seq_len in seq_lengths:
        q = [[random.gauss(0, 1) for _ in range(d)] for _ in range(seq_len)]
        k = [[random.gauss(0, 1) for _ in range(d)] for _ in range(seq_len)]
        v = [[random.gauss(0, 1) for _ in range(d)] for _ in range(seq_len)]

        start = time.time()
        flash_attention_simulated(q, k, v, block_size)
        elapsed = time.time() - start

        # FlashAttention has better memory complexity
        # O(seq_len * d * (seq_len / block_size)) instead of O(seq_len^2)
        num_blocks = (seq_len + block_size - 1) // block_size
        memory_ops = seq_len * d * num_blocks * 2

        results[seq_len] = {
            "time_seconds": elapsed,
            "flops_approx": seq_len * seq_len * d * 6,
            "memory_intensity": memory_ops / (seq_len * seq_len * d),
            "tiling_blocks": num_blocks,
        }

    return results
