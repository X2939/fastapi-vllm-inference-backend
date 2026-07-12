# BF16 vs AWQ Quality Smoke Comparison

> This is a fixed prompt regression smoke test, not a formal accuracy benchmark.

## Runs

- BF16: `/home/xxx/models/Qwen2.5-1.5B-Instruct`
- AWQ: `/home/xxx/models/Qwen2.5-1.5B-Instruct-AWQ`

## Summary

| Metric | BF16 | AWQ |
|---|---:|---:|
| Overall pass | 90.0% | 90.0% |
| JSON parse pass | 90.0% | 90.0% |
| Expected substring pass | 100.0% | 100.0% |

## Case Diff

| ID | BF16 | AWQ |
|---|---:|---:|
| arith_add | pass | pass |
| arith_mul | pass | pass |
| kv_cache | pass | pass |
| lowercase | pass | fail |
| metric_tpot | pass | pass |
| metric_ttft | pass | pass |
| paged_attention | fail | pass |
| prefix_cache | pass | pass |
| scheduler | pass | pass |
| sort_numbers | pass | pass |
