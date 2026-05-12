# FastAPI vLLM Inference Backend

一个基于 **FastAPI + vLLM** 的本地大模型推理服务后端。底层使用 vLLM 部署本地开源模型，上层使用 FastAPI 封装业务接口，并通过脚本验证普通调用、流式输出和基础低并发压测。

## 项目目标

- 使用 vLLM 部署本地开源大模型推理服务。
- 使用 OpenAI-compatible API 调用本地模型。
- 使用 FastAPI 封装面向业务系统的 `/chat` 和 `/chat/stream` 接口。
- 验证普通调用、流式输出、首 token 延迟和完整响应耗时。
- 记录串行与低并发场景下的简单吞吐表现。
- 理解 KV Cache、PagedAttention、continuous batching 在推理服务中的工程作用。
- 跟踪 vLLM 请求链路，理解 OpenAI-compatible API 到 scheduler、KV Cache 和输出返回的核心流程。

## 技术栈

- Python 3.10+
- FastAPI / Uvicorn
- vLLM
- OpenAI Python SDK
- Qwen2.5-1.5B-Instruct
- StreamingResponse
- requests / concurrent.futures

## 架构

```text
Client / scripts
      |
      v
FastAPI backend :9000
  - GET  /health
  - POST /chat
  - POST /chat/stream
      |
      v
vLLM OpenAI-compatible server :8000
  - GET  /v1/models
  - POST /v1/chat/completions
      |
      v
Qwen/Qwen2.5-1.5B-Instruct
```

FastAPI 层不是简单转发。它负责把业务侧的简化请求转换为模型所需的 `messages` 结构，并统一返回 `answer`、`elapsed` 和 `usage` 等字段。

## 项目结构

```text
vllm-demo/
├── app/
│   └── main.py             # FastAPI 后端接口
├── docs/
│   ├── project_intro.md    # 项目介绍文档
│   └── vllm_request_lifecycle.md # vLLM 请求链路学习笔记
├── reports/
│   └── benchmark.md        # 基础测试结果
├── scripts/
│   ├── chat_demo.py        # 非流式调用脚本
│   ├── stream_demo.py      # 流式调用脚本，记录首 token 延迟
│   ├── bench_demo.py       # 串行和低并发测试脚本
│   ├── inference_bench.py  # 多并发推理压测脚本
│   └── stream_bench.py     # 流式输出指标测试脚本
├── .env.example
├── requirements.txt
└── README.md
```

## 环境说明

- OS: WSL2 Ubuntu
- Python environment: `~/venvs/vllm-env`
- Model: `/home/xxx/models/Qwen2.5-1.5B-Instruct`
- vLLM endpoint: `http://127.0.0.1:8000/v1`
- FastAPI endpoint: `http://127.0.0.1:9000`

## 配置

项目通过环境变量配置 vLLM 地址、API Key、模型名称和 system prompt。

```bash
cp .env.example .env
set -a
source .env
set +a
```

核心配置：

- `VLLM_BASE_URL`: FastAPI 调用 vLLM 的 OpenAI-compatible API 地址。
- `VLLM_API_KEY`: vLLM 服务启动时配置的 API Key。
- `MODEL_NAME`: vLLM 加载的模型名称。
- `SYSTEM_PROMPT`: FastAPI 封装层统一注入的 system prompt。
- `BASE_URL` / `API_KEY`: `scripts/chat_demo.py` 和 `scripts/stream_demo.py` 直接调用 vLLM 时使用。

## 快速启动

### 1. 安装依赖

```bash
cd ~/vllm_projects/vllm-demo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果本地已有 vLLM 环境，也可以直接使用：

```bash
source ~/venvs/vllm-env/bin/activate
```

### 2. 启动 vLLM

```bash
vllm serve /home/xxx/models/Qwen2.5-1.5B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key token-abc123 \
  --gpu-memory-utilization 0.65 \
  --max-model-len 2048
```

参数说明：

- `--gpu-memory-utilization 0.65`: 降低 vLLM 可用显存比例，减少本地显存不足导致的启动失败。
- `--max-model-len 2048`: 限制最大上下文长度，降低 KV Cache 压力。

如果使用 Hugging Face 在线模型名，也可以把模型路径替换为：

```text
Qwen/Qwen2.5-1.5B-Instruct
```

但 FastAPI `.env` 里的 `MODEL_NAME` 必须和 `curl /v1/models` 返回的 `id` 保持一致。

### 3. 启动 FastAPI

另开一个终端：

```bash
cd ~/vllm_projects/vllm-demo
source ~/venvs/vllm-env/bin/activate
set -a
source .env
set +a
uvicorn app.main:app --host 0.0.0.0 --port 9000
```

接口文档：

```text
http://127.0.0.1:9000/docs
```

## 接口示例

### vLLM model list

```bash
curl -i http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer token-abc123"
```

### FastAPI health check

```bash
curl -i http://127.0.0.1:9000/health
```

返回示例：

```json
{
  "status": "ok"
}
```

### FastAPI chat

```bash
curl -i http://127.0.0.1:9000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "请解释什么是 KV Cache",
    "temperature": 0.7,
    "max_tokens": 128
  }'
```

返回结构示例：

```json
{
  "answer": "KV Cache 是在大模型自回归生成时缓存历史 token 的 Key 和 Value...",
  "elapsed": 1.56,
  "usage": {
    "prompt_tokens": 31,
    "completion_tokens": 95,
    "total_tokens": 126
  }
}
```

### FastAPI streaming chat

```bash
curl -N http://127.0.0.1:9000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "message": "请解释什么是 attention",
    "temperature": 0.7,
    "max_tokens": 128
  }'
```

`/chat/stream` 使用 vLLM 的 `stream=True`，通过 FastAPI `StreamingResponse` 将模型输出逐段返回给客户端。

## 脚本

```bash
python scripts/chat_demo.py
python scripts/stream_demo.py
python scripts/bench_demo.py
```

- `chat_demo.py`: 直接调用 vLLM 的非流式接口，记录完整响应耗时和 token usage。
- `stream_demo.py`: 直接调用 vLLM 的流式接口，记录首 token 延迟和总耗时。
- `bench_demo.py`: 调用 FastAPI `/chat`，进行串行和低并发测试，记录平均耗时和简单吞吐。
- `inference_bench.py`: 调用 FastAPI `/chat`，支持多并发档位，统计 avg、p50、p95、throughput 和 tokens/s。
- `stream_bench.py`: 调用 FastAPI `/chat/stream`，统计 TTFT、完整流式耗时和字符输出速度。

### 多并发推理压测

```bash
python scripts/inference_bench.py \
  --concurrency 1,2,4,8 \
  --requests 10 \
  --prompt-type medium \
  --max-tokens 128 \
  --output reports/inference_benchmark.csv
```

输出指标说明：

- `avg_latency`: 成功请求的平均延迟。
- `p50_latency`: 50% 请求低于该延迟。
- `p95_latency`: 95% 请求低于该延迟，常用于观察尾延迟。
- `throughput`: 每秒完成的成功请求数。
- `tokens_per_second`: 按完成 token 数估算的整体输出速度。
- `failed`: 失败请求数，用于观察并发升高后的服务稳定性。

### 流式输出指标测试

```bash
python scripts/stream_bench.py \
  --max-tokens 128
```

关键指标：

- `ttft`: Time To First Token，客户端收到第一段流式输出的时间。
- `total_time`: 完整流式输出结束的总耗时。
- `chars_per_second`: 按字符数估算的输出速度。

## 标准验证流程

建议按下面顺序验证，方便定位问题：

```bash
# 1. 确认 vLLM 服务可用
curl -i http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer token-abc123"

# 2. 确认 FastAPI 服务可用
curl -i http://127.0.0.1:9000/health

# 3. 验证 FastAPI 普通聊天接口
curl -i http://127.0.0.1:9000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "请用一句话解释什么是 attention",
    "temperature": 0.7,
    "max_tokens": 64
  }'

# 4. 验证流式接口
curl -N http://127.0.0.1:9000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{
    "message": "请解释什么是 KV Cache",
    "temperature": 0.7,
    "max_tokens": 128
  }'

# 5. 执行基础低并发测试
python scripts/bench_demo.py

# 6. 执行多并发推理压测
python scripts/inference_bench.py --concurrency 1,2,4 --requests 10

# 7. 执行流式输出指标测试
python scripts/stream_bench.py
```

## Benchmark Summary

### Multi-Concurrency Inference Benchmark

测试条件：

- Prompt type: `medium`
- Requests per concurrency level: `10`
- Max tokens: `128`
- Concurrency levels: `1 / 2 / 4`

| Concurrency | Avg Latency | P50 Latency | P95 Latency | Throughput | Tokens/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.64s | 1.61s | 1.77s | 0.61 req/s | 78.19 |
| 2 | 1.83s | 1.83s | 1.87s | 1.09 req/s | 139.73 |
| 4 | 1.87s | 1.88s | 1.90s | 1.79 req/s | 229.45 |

在 `medium prompt`、`max_tokens=128`、每组 `10` 个请求的测试中，随着并发数从 `1` 提升到 `4`，平均延迟从 `1.64s` 增加到 `1.87s`，P95 从 `1.77s` 增加到 `1.90s`；同时吞吐量从 `0.61 req/s` 提升到 `1.79 req/s`，tokens/s 从 `78.19` 提升到 `229.45`。

该结果说明，在本地低并发场景下，vLLM 服务能够通过请求调度和 batching 提升整体推理吞吐，但并发增加也会带来一定单请求延迟上升。

### Streaming Benchmark

| Metric | Value |
|---|---:|
| TTFT | 0.33s |
| Total streaming time | 1.90s |
| Output chars | 266 |
| Chars/s | 140.20 |

流式输出的 TTFT 约为 `0.33s`，完整输出耗时约为 `1.90s`。这说明流式输出主要优化用户感知等待时间，使用户在完整回答生成结束前就能看到模型输出。

## 关键理解

- vLLM 负责底层模型推理、OpenAI-compatible API、KV Cache 管理和请求调度。
- FastAPI 负责业务接口封装，将简化请求转换为底层模型调用所需的 `messages` 结构。
- OpenAI `messages` 不是模型直接理解的输入，vLLM 会通过 chat template 和 tokenizer 转换成模型需要的 token ids。
- 普通采样请求会构造 `SamplingParams`，beam search 请求才会走 `BeamSearchParams`。
- vLLM V1 scheduler 源码里没有简单硬编码“prefill 阶段”和“decode 阶段”，而是按每个请求还需要计算的 token 数进行统一调度。
- 流式输出主要改善用户感知响应速度，不一定缩短完整生成时间。
- 首 token 延迟决定用户多久开始看到内容，完整耗时决定一次回答何时结束。
- 本地显存限制会影响最大上下文长度、并发能力和服务稳定性。
- P50/P95、吞吐量、TTFT 和 tokens/s 比单次响应耗时更适合描述推理服务性能。

更完整的源码链路笔记见：

```text
docs/vllm_request_lifecycle.md
```

## 面试可讲点

- 为什么要在 vLLM 外面包一层 FastAPI：业务系统通常不直接暴露底层模型接口，FastAPI 可以统一请求结构、system prompt、结果格式和耗时统计。
- vLLM 和 FastAPI 各自负责什么：vLLM 负责模型推理服务，FastAPI 负责业务接口和调用链封装。
- 流式和非流式有什么区别：流式更早返回中间 token，优化用户感知时延；非流式等待完整结果后一次性返回。
- `gpu_memory_utilization` 和 `max_model_len` 为什么影响稳定性：它们会影响显存占用和 KV Cache 规模。
- PagedAttention 解决什么问题：优化 KV Cache 的显存管理，减少碎片和浪费。
- continuous batching 的作用：在线请求持续到来时动态调度，提高 GPU 利用率和整体吞吐。
- vLLM 请求链路怎么走：`/v1/chat/completions` 进入 OpenAI serving 层，经 chat template、tokenizer、SamplingParams、AsyncLLM、EngineCore、scheduler、model executor、OutputProcessor，最终返回 JSON 或 SSE 流式输出。
- 为什么压测和 vLLM 有关系：压测观察的是 vLLM 调度与 batching 在不同并发下对吞吐、P95、tokens/s 和 TTFT 的影响。

## 已知限制

- Hugging Face 模型拉取阶段可能受网络影响。
- 小模型对部分专业概念的回答质量有限。
- 当前压测是本地低并发性能观测，不是完整生产环境性能上限评估。
- 当前项目未接入鉴权、限流、日志系统和持久化对话历史。

## 常见问题

### vLLM 启动时 Hugging Face 超时

原因通常是网络访问 Hugging Face 不稳定。如果模型已经完整缓存过，可以尝试离线启动：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 vllm serve /home/xxx/models/Qwen2.5-1.5B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key token-abc123 \
  --gpu-memory-utilization 0.65 \
  --max-model-len 2048
```

### `Connection refused`

通常是目标服务没有启动或端口写错：

- `127.0.0.1:8000`: vLLM 服务。
- `127.0.0.1:9000`: FastAPI 服务。

先用 `/v1/models` 检查 vLLM，再用 `/health` 检查 FastAPI。

### `Unauthorized`

通常是 `Authorization: Bearer token-abc123` 写错，或者 FastAPI 的 `VLLM_API_KEY` 和 vLLM 启动参数 `--api-key` 不一致。

### FastAPI 返回 `500 Internal Server Error`

优先看启动 FastAPI 的终端日志。常见原因包括 vLLM 没启动、模型名不一致、端口错误、API Key 错误或请求参数不合法。

## 后续扩展

- 增加 pytest 自动化测试。
- 增加请求日志和异常日志。
- 增加接口鉴权和限流。
- 接入 Redis 或数据库保存对话历史。
- 增加 RAG 文档问答能力。
- 增加更系统的并发压测和指标记录。
