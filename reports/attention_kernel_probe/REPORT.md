# Attention Kernel Probe

> This probe uses PyTorch CUDA SDPA backend selection. It is a lower-level execution experiment, not a custom CUDA kernel implementation.

## Workload

- Sequence lengths: `[128, 256, 512, 1024]`
- Batch size: `1`
- Heads: `8`
- Head dim: `64`
- Dtype: `fp16`
- Causal: `True`
- Warmup / runs: `5` / `20`

## Results

| Backend | Seq | Success | P95 latency (ms) | Peak memory (MiB) | Max abs diff vs math | Error |
|---|---:|---:|---:|---:|---:|---|
| math | 128 | True | 0.759 | 10.8 | 0 |  |
| flash | 128 | True | 0.146 | 9.1 | 0.0009766 |  |
| math | 256 | True | 0.744 | 16.0 | 0 |  |
| flash | 256 | True | 0.152 | 10.1 | 0.0009766 |  |
| math | 512 | True | 1.032 | 33.4 | 0 |  |
| flash | 512 | True | 0.198 | 12.1 | 0.0009766 |  |
| math | 1024 | True | 2.553 | 96.6 | 0 |  |
| flash | 1024 | True | 0.336 | 16.2 | 0.0009766 |  |

## How to Read It

- `math` is the conservative reference path and is useful for numerical comparison.
- `flash` attempts to use the fused FlashAttention-style SDPA kernel when the GPU, dtype and shape support it.
- Lower latency usually comes from avoiding materializing the full attention matrix and reducing HBM traffic, but backend availability depends on CUDA, PyTorch, GPU architecture, dtype, head dimension and mask pattern.

This result complements the vLLM benchmark: serving metrics such as TPOT and throughput are affected by scheduler and KV Cache behavior, but the decode/prefill compute path still depends on attention kernel efficiency.
