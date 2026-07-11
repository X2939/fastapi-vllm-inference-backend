# Benchmark Notes

## Environment

- OS: WSL2 Ubuntu
- Model: `/home/xxx/models/Qwen2.5-1.5B-Instruct`
- vLLM endpoint: `http://127.0.0.1:8000/v1`
- FastAPI endpoint: `http://127.0.0.1:9000/chat`
- API Key: `token-abc123`
- vLLM launch args:
  - `--gpu-memory-utilization 0.65`
  - `--max-model-len 2048`

## Basic Service Checks

- vLLM `/v1/models`: success
- vLLM `/v1/chat/completions`: success
- vLLM `stream=true`: success
- FastAPI `/health`: success
- FastAPI `/chat`: success
- FastAPI `/chat/stream`: success

## Multi-Concurrency Inference Benchmark

### Test Setup

- Backend: FastAPI `/chat`
- Prompt type: `medium`
- Requests per concurrency level: `10`
- Max tokens: `128`
- Concurrency levels: `1 / 2 / 4`

### Results

| Concurrency | Requests | Success | Failed | Avg Latency | P50 Latency | P95 Latency | Throughput | Tokens/s |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10 | 10 | 0 | 1.64s | 1.61s | 1.77s | 0.61 req/s | 78.19 |
| 2 | 10 | 10 | 0 | 1.83s | 1.83s | 1.87s | 1.09 req/s | 139.73 |
| 4 | 10 | 10 | 0 | 1.87s | 1.88s | 1.90s | 1.79 req/s | 229.45 |

### Observation

在 `medium prompt`、`max_tokens=128`、每组 `10` 个请求的测试中，随着并发数从 `1` 提升到 `4`，单请求平均延迟从 `1.64s` 增加到 `1.87s`，P95 延迟从 `1.77s` 增加到 `1.90s`。

同时，系统吞吐量从 `0.61 req/s` 提升到 `1.79 req/s`，输出速度从 `78.19 tokens/s` 提升到 `229.45 tokens/s`。

该结果说明，在本地低并发场景下，vLLM 服务能够通过请求调度和 batching 提升整体吞吐，但并发增加也会带来一定的单请求延迟上升。这个现象符合在线大模型推理服务中“吞吐量”和“单请求延迟”之间的常见权衡。

需要注意的是，本测试运行在本地 WSL2 + 单机 GPU 环境下，模型规模较小，请求量也较少，因此结果主要用于工程验证和趋势观察，不代表生产环境性能上限。

## Streaming Benchmark

### Result

| Metric | Value |
|---|---:|
| TTFT | 0.33s |
| Total streaming time | 1.90s |
| Output chars | 266 |
| Chars/s | 140.20 |

### Observation

流式输出的 TTFT 约为 `0.33s`，完整输出耗时约为 `1.90s`。这说明流式输出不会必然缩短完整生成时间，但可以显著降低用户感知等待时间，使用户在完整回答生成结束前就能看到模型输出。

## Early Basic Checks

以下是早期接口联调阶段的基础结果，仅作为服务可用性记录，不作为当前主性能结论。

- Non-stream elapsed: `1.56s`
- First token latency: `0.70s`
- Total stream time: `2.39s`
- Serial test average latency: `0.50s`
- Low-concurrency test: `2` workers, throughput about `4.76 req/s`
