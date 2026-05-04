# FastAPI vLLM Inference Backend

一个基于 **FastAPI + vLLM** 的本地大模型推理服务后端。底层使用 vLLM 部署本地开源模型，上层使用 FastAPI 封装业务接口，并通过脚本验证普通调用、流式输出和基础低并发压测。

## 项目目标

- 使用 vLLM 部署本地开源大模型推理服务。
- 使用 OpenAI-compatible API 调用本地模型。
- 使用 FastAPI 封装面向业务系统的 `/chat` 和 `/chat/stream` 接口。
- 验证普通调用、流式输出、首 token 延迟和完整响应耗时。
- 记录串行与低并发场景下的简单吞吐表现。
- 理解 KV Cache、PagedAttention、continuous batching 在推理服务中的工程作用。

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
│   └── project_intro.md    # 项目介绍文档
├── reports/
│   └── benchmark.md        # 基础测试结果
├── scripts/
│   ├── chat_demo.py        # 非流式调用脚本
│   ├── stream_demo.py      # 流式调用脚本，记录首 token 延迟
│   └── bench_demo.py       # 串行和低并发测试脚本
├── .env.example
├── requirements.txt
└── README.md
```

## 环境说明

- OS: WSL2 Ubuntu
- Python environment: `~/venvs/vllm-env`
- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- vLLM endpoint: `http://127.0.0.1:8000/v1`
- FastAPI endpoint: `http://127.0.0.1:9000`

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
vllm serve Qwen/Qwen2.5-1.5B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --api-key token-abc123 \
  --gpu-memory-utilization 0.65 \
  --max-model-len 2048
```

参数说明：

- `--gpu-memory-utilization 0.65`: 降低 vLLM 可用显存比例，减少本地显存不足导致的启动失败。
- `--max-model-len 2048`: 限制最大上下文长度，降低 KV Cache 压力。

### 3. 启动 FastAPI

另开一个终端：

```bash
cd ~/vllm_projects/vllm-demo
source ~/venvs/vllm-env/bin/activate
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

## 基础测试结果

当前基础测试结果记录在 `reports/benchmark.md`：

- 非流式完整响应耗时：`1.56s`
- 首 token 延迟：`0.70s`
- 流式完整输出耗时：`2.39s`
- 串行测试平均耗时：`0.50s`
- 低并发测试：`2` workers，吞吐约 `4.76 req/s`

这些结果只代表本地小规模验证，不等同于完整生产压测。

## 关键理解

- vLLM 负责底层模型推理、OpenAI-compatible API、KV Cache 管理和请求调度。
- FastAPI 负责业务接口封装，将简化请求转换为底层模型调用所需的 `messages` 结构。
- 流式输出主要改善用户感知响应速度，不一定缩短完整生成时间。
- 首 token 延迟决定用户多久开始看到内容，完整耗时决定一次回答何时结束。
- 本地显存限制会影响最大上下文长度、并发能力和服务稳定性。

## 面试可讲点

- 为什么要在 vLLM 外面包一层 FastAPI：业务系统通常不直接暴露底层模型接口，FastAPI 可以统一请求结构、system prompt、结果格式和耗时统计。
- vLLM 和 FastAPI 各自负责什么：vLLM 负责模型推理服务，FastAPI 负责业务接口和调用链封装。
- 流式和非流式有什么区别：流式更早返回中间 token，优化用户感知时延；非流式等待完整结果后一次性返回。
- `gpu_memory_utilization` 和 `max_model_len` 为什么影响稳定性：它们会影响显存占用和 KV Cache 规模。
- PagedAttention 解决什么问题：优化 KV Cache 的显存管理，减少碎片和浪费。
- continuous batching 的作用：在线请求持续到来时动态调度，提高 GPU 利用率和整体吞吐。

## 已知限制

- Hugging Face 模型拉取阶段可能受网络影响。
- 小模型对部分专业概念的回答质量有限。
- 当前压测是基础低并发验证，不是完整性能上限评估。
- 当前项目未接入鉴权、限流、日志系统和持久化对话历史。

## 后续扩展

- 增加 pytest 自动化测试。
- 增加请求日志和异常日志。
- 增加接口鉴权和限流。
- 接入 Redis 或数据库保存对话历史。
- 增加 RAG 文档问答能力。
- 增加更系统的并发压测和指标记录。
