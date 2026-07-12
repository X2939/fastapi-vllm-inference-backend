# Prefix Cache Real GPU A/B

> Positive delta means the ON value is higher than OFF. For TTFT, TPOT and E2E latency, a negative delta is an improvement.

## Controlled Variables

- `model`: `/home/xxx/models/Qwen2.5-1.5B-Instruct`
- `prompt_type`: `long`
- `prompt_mode`: `shared_prefix`
- `requests_per_level`: `16`
- `warmup`: `3`
- `runs`: `3`
- `max_tokens`: `64`
- OFF server args: `--no-enable-prefix-caching`
- ON server args: `--enable-prefix-caching`

## Results

| Concurrency | Tokens/s OFF → ON | Δ | P95 TTFT OFF → ON | Δ | P95 TPOT OFF → ON | Δ | P95 E2E OFF → ON | Δ |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 75.66 → 66.28 | -12.4% | 57.7 → 38.6 ms | -33.2% | 12.87 → 15.68 ms | +21.9% | 867.8 → 1023.6 ms | +18.0% |
| 2 | 127.98 → 114.07 | -10.9% | 97.9 → 60.3 ms | -38.3% | 15.27 → 17.96 ms | +17.6% | 1034.6 → 1172.9 ms | +13.4% |
| 4 | 232.26 → 211.09 | -9.1% | 176.6 → 64.9 ms | -63.3% | 16.49 → 19.13 ms | +16.0% | 1118.3 → 1267.7 ms | +13.4% |

## Interpretation Boundary

Prefix Cache skips repeated prefill work, so TTFT is its primary expected benefit. TPOT mainly reflects decode work and should not be presented as a direct Prefix Cache gain. Throughput and E2E latency depend on scheduler batch composition, cache lookup overhead and the host/GPU runtime; they must be interpreted from the measured deltas rather than assumed.
