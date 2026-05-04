# FastAPI + vLLM Demo

基于 FastAPI 和 vLLM 的本地大模型推理服务后端。底层使用 vLLM 部署本地模型推理服务，上层使用 FastAPI 封装业务接口，并通过脚本验证普通调用、流式输出和基础低并发压测。

## Project Goal

- 使用 vLLM 部署本地开源大模型推理服务
- 使用 OpenAI-compatible API 调用本地模型
- 使用 FastAPI 封装面向业务调用的 `/chat` 接口
- 验证普通调用、流式输出和基础时延指标
- 记录串行和低并发场景下的简单吞吐表现

## Architecture

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
      |
      v
Qwen/Qwen2.5-1.5B-Instruct
```

## Environment

- OS: WSL2 Ubuntu
- Python environment: `~/venvs/vllm-env`
- Model: `Qwen/Qwen2.5-1.5B-Instruct`
- vLLM endpoint: `http://127.0.0.1:8000/v1`
- FastAPI endpoint: `http://127.0.0.1:9000`

## Start vLLM

```bash
source ~/venvs/vllm-env/bin/activate

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

## Start FastAPI

另开一个终端：

```bash
source ~/venvs/vllm-env/bin/activate
cd ~/vllm_projects/vllm-demo

uvicorn app.main:app --host 0.0.0.0 --port 9000
```

## API Checks

### vLLM model list

```bash
curl -i http://127.0.0.1:8000/v1/models \
  -H "Authorization: Bearer token-abc123"
```

### FastAPI health check

```bash
curl -i http://127.0.0.1:9000/health
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

## Scripts

```bash
python ~/vllm_projects/vllm-demo/scripts/chat_demo.py
python ~/vllm_projects/vllm-demo/scripts/stream_demo.py
python ~/vllm_projects/vllm-demo/scripts/bench_demo.py
```

- `chat_demo.py`: 直接调用 vLLM 的非流式接口，记录完整响应耗时和 token usage。
- `stream_demo.py`: 直接调用 vLLM 的流式接口，记录首 token 延迟和总耗时。
- `bench_demo.py`: 调用 FastAPI `/chat`，进行串行和低并发测试，记录平均耗时和简单吞吐。

## Benchmark Summary

当前基础测试结果：

- 非流式完整响应耗时：`1.56s`
- 首 token 延迟：`0.70s`
- 流式完整输出耗时：`2.39s`
- 串行测试平均耗时：`0.50s`
- 低并发测试：`2` workers，吞吐约 `4.76 req/s`

这些结果只代表本地小规模验证，不等同于完整生产压测。

## Key Takeaways

- vLLM 负责底层模型推理、OpenAI-compatible API、KV Cache 管理和请求调度。
- FastAPI 负责业务接口封装，将简化请求转换为底层模型调用所需的 `messages` 结构。
- 流式输出主要改善用户感知响应速度，不一定缩短完整生成时间。
- 本地显存限制会影响最大上下文长度、并发能力和服务稳定性。

## Known Issues

- Hugging Face 模型拉取阶段可能受网络影响。
- 小模型对部分专业概念的回答质量有限。
- 当前压测是基础低并发验证，不是完整性能上限评估。
