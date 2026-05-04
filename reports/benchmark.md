# Benchmark Notes

## Environment
- OS: WSL2 Ubuntu
- Model: Qwen/Qwen2.5-1.5B-Instruct
- Endpoint: http://127.0.0.1:8000/v1
- FastAPI Endpoint: http://127.0.0.1:9000/chat
- API Key: token-abc123
- vLLM launch args:
  - --gpu-memory-utilization 0.65
  - --max-model-len 2048

## Basic Checks
- /v1/models: success
- /v1/chat/completions: success
- stream=true: success
- FastAPI /health: success
- FastAPI /chat: success

## Sample Results
- non-stream elapsed: 1.56s
- first token latency: 0.70s
- total stream time: 2.39s

## Basic Benchmark
### Serial Test
- times: [1.14, 0.38, 0.31, 0.24, 0.39]
- average latency: 0.50s

### Concurrent Test
- workers: 2
- times: [0.30, 0.52, 0.32, 0.32, 0.43]
- average single request latency: 0.38s
- total wall time: 1.05s
- throughput: 4.76 req/s

## Observations
- The first request is slower than later requests, likely due to warm-up overhead.
- Under low concurrency, the service remains stable and can sustain basic parallel request processing.
- Due to limited local GPU memory, gpu_memory_utilization and max_model_len were reduced to make deployment stable.
