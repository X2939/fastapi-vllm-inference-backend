# Real GPU Benchmark Report

> This report is generated from a real OpenAI-compatible streaming endpoint, not the CPU simulation benchmark.

## Workload

- Model: `/home/xxx/models/Qwen2.5-1.5B-Instruct`
- Prompt: `medium` / `unique`
- Requests per concurrency: `20`
- Warmup per run: `3`
- Repetitions: `3`
- Max completion tokens: `128`

## Environment

- vLLM: `0.19.0`
- PyTorch: `2.10.0`
- GPU: `NVIDIA GeForce RTX 3060 Laptop GPU, driver 551.61, 5878 / 6144 MiB used`
- CUDA: `12.8`

## Aggregate Results

Each value is the arithmetic mean of the corresponding per-run result; P95 is computed within each run before averaging.

| Concurrency | Success | Tokens/s | P95 TTFT (s) | P95 TPOT (s/token) | P95 E2E (s) | Error rate |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20.0 | 57.90 | 0.051 | 0.0211 | 2.764 | 0.00% |
| 2 | 20.0 | 104.61 | 0.064 | 0.0203 | 2.625 | 0.00% |
| 4 | 20.0 | 199.08 | 0.075 | 0.0213 | 2.771 | 0.00% |

## Metric Definition

- TTFT: request start to first non-empty generated content chunk.
- TPOT: `(E2E - TTFT) / (completion_tokens - 1)` for requests with at least two output tokens.
- E2E: request start to streaming completion.
- Tokens/s: all successful completion tokens divided by the workload wall time.

Raw per-run summaries are in `summary.csv`; request-level samples are in `requests.csv`.
