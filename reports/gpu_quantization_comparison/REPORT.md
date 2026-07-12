# BF16 vs AWQ Real GPU Comparison

> Positive delta means the AWQ value is higher than BF16. For TTFT, TPOT and E2E latency, a negative delta is an improvement.

## Controlled Variables

- `prompt_type`: `medium`
- `prompt_mode`: `unique`
- `requests_per_level`: `20`
- `warmup`: `3`
- `runs`: `3`
- `max_tokens`: `128`
- BF16 model: `/home/xxx/models/Qwen2.5-1.5B-Instruct`
- AWQ model: `/home/xxx/models/Qwen2.5-1.5B-Instruct-AWQ`
- BF16 server args: `effective: enable_prefix_caching=true, enable_chunked_prefill=true`
- AWQ server args: `--quantization awq`
- BF16 memory after run: `5457 / 6144 MiB`
- AWQ memory after run: `5645 / 6144 MiB`

## Results

| Concurrency | Tokens/s BF16 → AWQ | Δ | P95 TTFT BF16 → AWQ | Δ | P95 TPOT BF16 → AWQ | Δ | P95 E2E BF16 → AWQ | Δ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 73.20 → 21.14 | -71.1% | 37.7 → 69.6 ms | +84.5% | 14.56 → 50.42 ms | +246.3% | 1884.7 → 6469.3 ms | +243.3% |
| 2 | 110.23 → 41.27 | -62.6% | 54.1 → 155.3 ms | +186.8% | 19.90 → 49.19 ms | +147.2% | 2572.7 → 6395.7 ms | +148.6% |
| 4 | 211.61 → 77.37 | -63.4% | 66.5 → 513.4 ms | +671.9% | 19.09 → 53.91 ms | +182.4% | 2479.8 → 7238.1 ms | +191.9% |

## Interpretation Boundary

AWQ reduces weight precision, but it does not shrink KV Cache at the same ratio. On this RTX 3060 Laptop GPU run, AWQ saves model weight storage but decode latency is worse because INT4 dequantization and kernel support dominate this small-batch workload. Treat this as a real tradeoff measurement, not as a blanket claim that quantization always improves throughput.
