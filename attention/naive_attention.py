
from __future__ import annotations
import math
import time
from typing import Tuple


def naive_attention(
    q: list[list[float]],
    k: list[list[float]],
    v: list[list[float]],
    mask: list[bool] = None,
) -> list[list[float]]:
    """Standard scaled dot-product attention.

    This is the textbook implementation:
    Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

    Time complexity: O(seq_len^2 * d)
    Memory complexity: O(seq_len^2) for attention scores
    """
    batch_size, seq_len_q, d_k = len(q), len(q[0]) if q else 0, len(q[0][0]) if q and q[0] else 0
    seq_len_k = len(k[0]) if k else 0

    # Q @ K^T
    attention_scores = []
    for i in range(seq_len_q):
        row = []
        for j in range(seq_len_k):
            score = sum(q[i][m] * k[j][m] for m in range(d_k))
            score = score / math.sqrt(d_k)

            if mask and not mask[j]:
                score = float("-inf")

            row.append(score)
        attention_scores.append(row)

    # Softmax
    attention_weights = []
    for i in range(seq_len_q):
        max_score = max(attention_scores[i])
        exp_scores = [math.exp(attention_scores[i][j] - max_score) for j in range(seq_len_k)]
        sum_exp = sum(exp_scores)
        attention_weights.append([e / sum_exp for e in exp_scores])

    # Weighted sum
    output = []
    for i in range(seq_len_q):
        out_vec = []
        for m in range(d_k):
            val = sum(attention_weights[i][j] * v[j][m] for j in range(seq_len_k))
            out_vec.append(val)
        output.append(out_vec)

    return output


def benchmark_naive_attention(seq_lengths: list[int], d: int = 64) -> dict:
    """Benchmark naive attention with different sequence lengths."""
    import random
    results = {}

    for seq_len in seq_lengths:
        # Generate random Q, K, V
        q = [[random.gauss(0, 1) for _ in range(d)] for _ in range(seq_len)]
        k = [[random.gauss(0, 1) for _ in range(d)] for _ in range(seq_len)]
        v = [[random.gauss(0, 1) for _ in range(d)] for _ in range(seq_len)]

        start = time.time()
        naive_attention(q, k, v)
        elapsed = time.time() - start

        # Calculate memory operations (approximation)
        memory_ops = seq_len * seq_len * d * 2  # Read Q, K, V, write output

        results[seq_len] = {
            "time_seconds": elapsed,
            "flops_approx": seq_len * seq_len * d * 6,
            "memory_intensity": memory_ops / (seq_len * seq_len * d),
        }

    return results
