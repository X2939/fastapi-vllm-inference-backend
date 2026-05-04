# 基于 FastAPI 和 vLLM 的本地大模型推理服务后端

## 项目背景

为了理解大模型在线推理服务的工程链路，我基于 vLLM 部署本地开源模型推理服务，并在其上使用 FastAPI 封装业务后端接口，验证普通调用、流式输出和基础低并发测试。

## 技术选型

- `vLLM`: 本地大模型推理服务框架
- `Qwen/Qwen2.5-1.5B-Instruct`: 本地部署的开源指令模型
- `FastAPI`: 业务后端接口层
- `OpenAI-compatible API`: 统一模型调用格式
- `Python`: 客户端脚本、流式调用和基础压测
- `WSL2 Ubuntu`: 本地运行环境

## 系统架构

```text
Client / scripts
      |
      v
FastAPI backend :9000
      |
      v
vLLM server :8000
      |
      v
Qwen model
```

## 我做的工作

- 在 WSL2 环境中部署 vLLM 服务，并使用本地 GPU 运行 `Qwen/Qwen2.5-1.5B-Instruct`。
- 使用 vLLM 提供的 OpenAI-compatible API 验证 `/v1/models`、`/v1/chat/completions` 和流式输出。
- 基于 FastAPI 封装 `/health`、`/chat` 和 `/chat/stream` 接口，将简化业务请求转换为底层模型调用所需的 `messages` 结构。
- 编写 `chat_demo.py`、`stream_demo.py`、`bench_demo.py`，验证普通调用、流式调用和基础低并发测试。
- 在本地显存受限条件下，调整 `gpu_memory_utilization` 和 `max_model_len`，提高服务启动稳定性。

## 当前验证结果

- vLLM `/v1/models`: success
- vLLM `/v1/chat/completions`: success
- FastAPI `/health`: success
- FastAPI `/chat`: success
- FastAPI `/chat/stream`: success
- 串行和低并发基础测试已完成

## 基础测试结果

- 非流式完整响应耗时：`1.56s`
- 首 token 延迟：`0.70s`
- 流式完整输出耗时：`2.39s`
- 串行测试平均耗时：`0.50s`
- 低并发测试：`2` workers，吞吐约 `4.76 req/s`

## 关键理解

- vLLM 是推理与服务框架，不是模型本身。
- OpenAI-compatible API 兼容的是接口协议，不是 OpenAI 模型。
- FastAPI 封装层负责业务接口、请求适配、结果封装和后续扩展点。
- LLM 推理主要分为 `prefill` 和 `decode`，`KV Cache` 用于减少 decode 阶段重复计算。
- `PagedAttention` 主要优化 KV Cache 的显存管理方式，`continuous batching` 主要优化在线请求调度。
- 流式输出改善的是首 token 可见时间和用户感知响应速度。

## 遇到的问题

- Hugging Face 模型拉取阶段网络不稳定，导致启动过程出现超时和 SSL 错误。
- 本地显存有限，需要限制最大上下文长度和显存利用率。
- FastAPI 和 vLLM 需要分端口运行，vLLM 使用 `8000`，FastAPI 使用 `9000`。
- 开发过程中出现过 SDK 方法名、字段名、缩进和 HTTP/HTTPS 配置错误，最终通过查看服务日志逐项定位。

## 面试表述

这个项目底层使用 vLLM 部署本地大模型推理服务，上层通过 FastAPI 封装业务接口。客户端请求先进入 FastAPI 的 `/chat` 或 `/chat/stream`，后端将简化请求转换成 OpenAI-compatible 的 `messages` 结构，再调用 vLLM 服务完成模型推理，并将结果统一封装返回。项目中我完成了部署、接口联调、流式调用、基础压测和显存参数调优。
