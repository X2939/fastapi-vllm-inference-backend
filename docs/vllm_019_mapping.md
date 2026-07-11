# 与 vLLM 0.19.0 的架构映射

本项目不是 vLLM 的替代实现。目标是让教学代码中的关键边界能与 vLLM V1 EngineCore 一一对照，然后回到官方源码继续阅读。

参考版本：vLLM `v0.19.0`。

## 主循环映射

```python
scheduler_output = scheduler.schedule()
model_runner_output = worker.execute_model(scheduler_output)
engine_outputs = scheduler.update_from_output(
    scheduler_output,
    model_runner_output,
)
```

| vLLM 0.19.0 | 本项目 | 保留的语义 | 有意省略 |
|---|---|---|---|
| `EngineCore.step()` | `InferenceEngine.run_step()` | schedule → execute → update | async future、batch queue、ZMQ |
| `SchedulerOutput` | `engine.outputs.SchedulerOutput` | new/cached 请求、逐请求 token、finished IDs | multimodal、LoRA、spec decode、connectors |
| `ModelRunnerOutput` | `engine.outputs.ModelRunnerOutput` | req index、sampled IDs、执行信息 | logprobs tensor、pooler、draft tokens |
| `Scheduler.update_from_output()` | 同名方法 | computed token、输出 token、finish、KV free | stop strings、grammar、preemption |
| `GPUWorker` | `engine.worker.GPUWorker` | 设备/执行环境边界 | 进程、rank、distributed init |
| `GPUModelRunner` | `engine.model_runner.ModelRunner` | input prep、attention、sampling | weights、CUDA graph、真实 tensor batch |
| Attention Backend | `engine.attention_backend` | backend protocol + metadata + forward | CUDA kernel、FlashInfer、真实 KV layout |
| KV Cache Manager | `KVCacheManager` | block table、引用计数、prefix sharing | 真实 K/V tensor、hybrid cache groups |

## SchedulerOutput 字段对照

官方 v0.19.0 通过 `scheduled_new_reqs` 和 `scheduled_cached_reqs` 避免每 step 重发完整请求；本项目使用 `new_requests` 和 `cached_requests` 表达相同边界。

| 教学字段 | 含义 |
|---|---|
| `scheduled_request_ids` | 本轮真正送入 ModelRunner 的请求顺序 |
| `num_scheduled_tokens` | 每个请求本轮计算的 token 数 |
| `new_requests` | 第一次进入 Worker，需要完整 prompt 和 block table |
| `cached_requests` | Worker 已缓存静态信息，只发送计数和 block 更新 |
| `finished_request_ids` | 通知 Worker 清理请求侧缓存 |
| `block_table_updates` | logical → physical block IDs |
| `total_num_scheduled_tokens` | 全局 token budget 的实际消耗 |

## Token 调度

官方 V1 Scheduler 的核心不是“先判断 prefill 还是 decode”，而是比较 `num_computed_tokens` 与当前需要计算的 token 数，并在全局 token budget 内推进。本项目保留这一思想：

```text
remaining prompt = prompt_length - num_computed_tokens

PREFILL: schedule min(remaining prompt, chunk limit, token budget)
DECODE : schedule 1 token
```

因此 Chunked Prefill 不再由 Executor 私自拆分，而是 Scheduler 的 token allocation 结果。

## 状态所有权

```text
Scheduler owns:
  Request status / counters / timestamps
  KV allocate / append / free
  finish decision

ModelRunner owns:
  input preparation
  attention backend invocation
  simulated forward and sampled token IDs

Worker owns:
  execution environment boundary
```

`tests/test_engine_core_boundaries.py` 专门验证 ModelRunner 执行前后 Request 不会被修改，只有 `update_from_output()` 能推进状态。

## 官方源码入口

- Scheduler output: <https://github.com/vllm-project/vllm/blob/v0.19.0/vllm/v1/core/sched/output.py>
- Scheduler: <https://github.com/vllm-project/vllm/blob/v0.19.0/vllm/v1/core/sched/scheduler.py>
- EngineCore: <https://github.com/vllm-project/vllm/blob/v0.19.0/vllm/v1/engine/core.py>
- ModelRunnerOutput: <https://github.com/vllm-project/vllm/blob/v0.19.0/vllm/v1/outputs.py>
- GPU Worker: <https://github.com/vllm-project/vllm/blob/v0.19.0/vllm/v1/worker/gpu_worker.py>
- GPU Model Runner: <https://github.com/vllm-project/vllm/blob/v0.19.0/vllm/v1/worker/gpu_model_runner.py>

## 不在本项目复刻

- EngineCore 与 API Server 的多进程/ZMQ 通信。
- TP/PP/DP 和每 GPU Worker 进程。
- CUDA Graph、speculative decoding、multimodal、LoRA。
- 真实按 layer/head/dtype 布局的 GPU KV Tensor。
- CUDA、Triton、FlashInfer kernel。

这些内容直接阅读官方实现更有效；本项目只负责提供进入源码前的结构导航。
