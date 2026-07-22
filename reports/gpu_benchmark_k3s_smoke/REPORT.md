# Real GPU Benchmark Report

> This report is generated from a real OpenAI-compatible streaming endpoint, not the CPU simulation benchmark.

## Workload

- Model: `/models/Qwen2.5-1.5B-Instruct`
- Prompt: `medium` / `unique`
- Requests per concurrency: `2`
- Warmup per run: `0`
- Repetitions: `1`
- Max completion tokens: `16`

## Environment

- vLLM: `0.19.0`
- PyTorch: `2.10.0`
- GPU: `NVIDIA GeForce RTX 3060 Laptop GPU, driver 551.61, 5749 / 6144 MiB used`
- CUDA: `12.8`

## Aggregate Results

Each value is the arithmetic mean of the corresponding per-run result; P95 is computed within each run before averaging.

| Concurrency | Success | Tokens/s | P95 TTFT (s) | P95 TPOT (s/token) | P95 E2E (s) | Error rate |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 2.0 | 12.15 | 1.413 | 0.0554 | 2.245 | 0.00% |

## Metric Definition

- TTFT: request start to first non-empty generated content chunk.
- TPOT: `(E2E - TTFT) / (completion_tokens - 1)` for requests with at least two output tokens.
- E2E: request start to streaming completion.
- Tokens/s: all successful completion tokens divided by the workload wall time.

Raw per-run summaries are in `summary.csv`; request-level samples are in `requests.csv`.
