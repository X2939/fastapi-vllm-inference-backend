# Real GPU Benchmark Report

> This report is generated from a real OpenAI-compatible streaming endpoint, not the CPU simulation benchmark.

## Workload

- Model: `/home/xxx/models/Qwen2.5-1.5B-Instruct`
- Prompt: `long` / `shared_prefix`
- Requests per concurrency: `16`
- Warmup per run: `3`
- Repetitions: `3`
- Max completion tokens: `64`

## Environment

- vLLM: `0.19.0`
- PyTorch: `2.10.0`
- GPU: `NVIDIA GeForce RTX 3060 Laptop GPU, 551.61, 6144 MiB`
- CUDA: `12.8`

## Aggregate Results

Each value is the arithmetic mean of the corresponding per-run result; P95 is computed within each run before averaging.

| Concurrency | Success | Tokens/s | P95 TTFT (s) | P95 TPOT (s/token) | P95 E2E (s) | Error rate |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 16.0 | 75.66 | 0.058 | 0.0129 | 0.868 | 0.00% |
| 2 | 16.0 | 127.98 | 0.098 | 0.0153 | 1.035 | 0.00% |
| 4 | 16.0 | 232.26 | 0.177 | 0.0165 | 1.118 | 0.00% |

## Metric Definition

- TTFT: request start to first non-empty generated content chunk.
- TPOT: `(E2E - TTFT) / (completion_tokens - 1)` for requests with at least two output tokens.
- E2E: request start to streaming completion.
- Tokens/s: all successful completion tokens divided by the workload wall time.

Raw per-run summaries are in `summary.csv`; request-level samples are in `requests.csv`.
