# Quality Smoke Report

> This is a small regression smoke test for endpoint sanity, not a formal accuracy benchmark.

## Run

- Label: `bf16_baseline`
- Model: `/home/xxx/models/Qwen2.5-1.5B-Instruct`
- Cases: `10`
- Max tokens: `96`

## Summary

- Overall pass: `9/10` (90.0%)
- JSON parse pass: `9/10` (90.0%)
- Expected substring pass: `10/10` (100.0%)
- Average latency: `0.213s`

## Cases

| ID | Category | OK | JSON | Expected | Latency (s) |
|---|---|---:|---:|---:|---:|
| arith_add | arithmetic | yes | yes | yes | 0.461 |
| arith_mul | arithmetic | yes | yes | yes | 0.227 |
| sort_numbers | format | yes | yes | yes | 0.267 |
| lowercase | format | yes | yes | yes | 0.103 |
| metric_ttft | inference | yes | yes | yes | 0.207 |
| metric_tpot | inference | yes | yes | yes | 0.196 |
| kv_cache | inference | yes | yes | yes | 0.218 |
| scheduler | inference | yes | yes | yes | 0.158 |
| paged_attention | inference | no | no | yes | 0.069 |
| prefix_cache | inference | yes | yes | yes | 0.222 |
