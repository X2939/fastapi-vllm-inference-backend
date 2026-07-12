# Quality Smoke Report

> This is a small regression smoke test for endpoint sanity, not a formal accuracy benchmark.

## Run

- Label: `awq_int4`
- Model: `/home/xxx/models/Qwen2.5-1.5B-Instruct-AWQ`
- Cases: `10`
- Max tokens: `96`

## Summary

- Overall pass: `9/10` (90.0%)
- JSON parse pass: `9/10` (90.0%)
- Expected substring pass: `10/10` (100.0%)
- Average latency: `0.614s`

## Cases

| ID | Category | OK | JSON | Expected | Latency (s) |
|---|---|---:|---:|---:|---:|
| arith_add | arithmetic | yes | yes | yes | 1.412 |
| arith_mul | arithmetic | yes | yes | yes | 0.357 |
| sort_numbers | format | yes | yes | yes | 0.735 |
| lowercase | format | no | no | yes | 0.242 |
| metric_ttft | inference | yes | yes | yes | 0.610 |
| metric_tpot | inference | yes | yes | yes | 0.605 |
| kv_cache | inference | yes | yes | yes | 0.731 |
| scheduler | inference | yes | yes | yes | 0.483 |
| paged_attention | inference | yes | yes | yes | 0.288 |
| prefix_cache | inference | yes | yes | yes | 0.678 |
