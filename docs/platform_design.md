# AI Infra 实验平台设计说明

## 1. 定位转变

| 之前 | 现在 |
|------|------|
| FastAPI + vLLM 推理 Demo | AI Infra **实验平台** |
| 脚本能压测就行 | 每个实验有 **假设、变量、报告、分析结论** |
| 功能模块堆叠 | 围绕 Scheduler、KV Cache、Metrics、Benchmark 构建 |

## 2. 为什么这样设计

早期版本的核心缺口是：**生命周期不完整、缺少可复现的性能实验**。

因此平台围绕三个可观测的推理机制组织：

### Scheduler（exp002）
- **假设**：提高并发 → 吞吐上升，P95 可能上升
- **变量**：concurrency 1/2/4/8，unique prompt（隔离 prefix cache）
- **指标**：throughput、P95、tokens/s、error_rate
- **分析重点**：continuous batching、GPU 利用率、尾延迟

### KV Cache（exp003）
- **假设**：shared_prefix 与 unique 的 prefill 行为不同
- **变量**：prompt_mode、max_tokens（故意偏小以提高 prefill 占比）
- **对照**：同一配置改 `prompt_mode: unique` 再跑一遍
- **分析重点**：prefill vs decode、PagedAttention vs prefix caching

### Metrics
- **HTTP 层**：QPS、P95 — 用户感知
- **Inference 层**：prompt/completion tokens — 负载与产出
- **边界**：不等于 vLLM 内部 kernel 耗时；实验报告需说明

## 3. 目录职责

```text
benchmarks/          # 实验内核：统计、prompt 构造、run_case
experiments/
  configs/           # 实验定义（假设、变量、分析 notes）
  runner.py          # 读 YAML → 跑实验 → 写 reports/
app/core/metrics.py  # Prometheus 指标（支撑压测期间观测）
reports/exp00x_*/    # 每次实验的 CSV + report + meta
```

## 4. PromptMode 设计

| 模式 | 用途 |
|------|------|
| `unique` | 每个请求不同 suffix，隔离 prefix cache，适合 scheduler 实验 |
| `shared_prefix` | 共享长前缀 + 小变体，观察 KV Cache / prefix caching |

## 5. 使用流程

```bash
make serve-vllm    # 终端 1
make serve-api     # 终端 2
make compose-up    # 可选：Prometheus + Grafana

python -m experiments.runner experiments/configs/exp002_scheduler_sweep.yaml
# 压测期间打开 http://127.0.0.1:3000 观察 QPS / P95 / tokens
```

## 6. 不要做的事

- 不要为「看起来完整」而加鉴权、用户系统等与实验无关的模块
- 不要把 TPOT 说成 vLLM 内部 decode 耗时（客户端估算）
- 不要在没有对照实验的情况下结论「量化一定更快」
