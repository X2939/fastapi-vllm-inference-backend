# BF16 vs AWQ-Marlin Real GPU Comparison

> Positive delta means the AWQ-Marlin value is higher than BF16. For TTFT, TPOT and E2E latency, a negative delta is an improvement.

## Controlled Variables

- `prompt_type`: `medium`
- `prompt_mode`: `unique`
- `requests_per_level`: `20`
- `warmup`: `3`
- `runs`: `3`
- `max_tokens`: `128`
- BF16 model: `/home/xxx/models/Qwen2.5-1.5B-Instruct`
- AWQ model: `/home/xxx/models/Qwen2.5-1.5B-Instruct-AWQ`
- BF16 server args: `effective: quantization=none, enable_prefix_caching=true, enable_chunked_prefill=true`
- AWQ server args: `--quantization awq_marlin`
- BF16 memory after run: `5727 / 6144 MiB`
- AWQ memory after run: `5616 / 6144 MiB`

## Results

| Concurrency | Tokens/s BF16 -> AWQ-Marlin | Delta | P95 TTFT BF16 -> AWQ-Marlin | Delta | P95 TPOT BF16 -> AWQ-Marlin | Delta | P95 E2E BF16 -> AWQ-Marlin | Delta |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 67.48 -> 132.10 | +95.8% | 38.0 -> 41.5 ms | +9.2% | 15.23 -> 7.83 ms | -48.6% | 1968.6 -> 1028.4 ms | -47.8% |
| 2 | 119.43 -> 245.31 | +105.4% | 52.1 -> 96.5 ms | +85.5% | 16.94 -> 8.93 ms | -47.3% | 2192.5 -> 1204.2 ms | -45.1% |
| 4 | 230.39 -> 481.77 | +109.1% | 71.7 -> 58.9 ms | -17.9% | 17.59 -> 8.28 ms | -52.9% | 2297.4 -> 1093.5 ms | -52.4% |

## Findings

- AWQ-Marlin changed token throughput by +95.8% to +109.1% across the tested concurrency levels.
- P95 TPOT changed by -52.9% to -47.3%, and P95 E2E changed by -52.4% to -45.1%.
- P95 TTFT was mixed (-17.9% to +85.5%). A faster decode kernel can improve TPOT without guaranteeing lower prefill or queueing tail latency.

## Interpretation Boundary

AWQ is weight-only quantization, so it does not shrink KV Cache at the same ratio. With a fixed vLLM GPU-memory-utilization budget, lower weight memory can be reassigned to KV capacity; similar total nvidia-smi memory after startup is therefore expected. The result is specific to this checkpoint, backend, GPU, software stack, and workload.
