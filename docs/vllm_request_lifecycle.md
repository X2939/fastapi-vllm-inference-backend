# vLLM Request Lifecycle Notes

This note records the request path studied in vLLM `0.19.0`. It is not a source-code modification record. Its goal is to explain how an OpenAI-compatible chat request enters vLLM, becomes engine input, is scheduled, executes model inference, and is finally returned as JSON or streaming chunks.

## Overall Flow

```text
Client / FastAPI
  -> POST /v1/chat/completions
  -> OpenAI-compatible serving layer
  -> render_chat_request
  -> tokenizer / chat template / engine_input
  -> AsyncLLM.generate
  -> EngineCore.add_request
  -> Scheduler.schedule
  -> model_executor.execute_model
  -> Scheduler.update_from_output
  -> OutputProcessor / detokenizer
  -> ChatCompletionResponse or SSE stream
```

## 1. OpenAI-Compatible API Entry

Key files:

- `vllm/entrypoints/openai/chat_completion/api_router.py`
- `vllm/entrypoints/openai/chat_completion/serving.py`

The route `POST /v1/chat/completions` receives an OpenAI-style `ChatCompletionRequest`.

At this layer, vLLM does not directly run model inference. It first parses request fields such as:

- `model`
- `messages`
- `temperature`
- `max_tokens`
- `stream`

Then it calls the chat serving handler to continue the request.

This is why the client can use OpenAI-style request code while the backend runs a local model.

## 2. Messages To Engine Input

Key behavior:

```text
OpenAI messages JSON
  -> chat template
  -> prompt text
  -> tokenizer
  -> token ids
  -> engine_input
```

The model itself does not understand the OpenAI `messages` JSON structure. vLLM's serving layer converts it into the prompt format expected by the selected chat/instruct model.

During this stage, vLLM also constructs generation strategy parameters:

- `SamplingParams`: used by normal sampling generation, such as temperature and max tokens.
- `BeamSearchParams`: used when beam search is enabled.

In this project, normal chat requests use `SamplingParams`.

## 3. AsyncLLM And EngineCore

Key files:

- `vllm/v1/engine/async_llm.py`
- `vllm/v1/engine/core.py`

`AsyncLLM.generate()` is the async boundary between the OpenAI serving layer and the engine.

The core responsibilities are:

- register the request output queue
- convert processed input into `EngineCoreRequest`
- submit the request to `EngineCore`
- yield generated `RequestOutput` back to the serving layer

In simplified form:

```text
AsyncLLM.generate
  -> add_request
  -> input_processor.process_inputs
  -> engine_core.add_request_async
  -> wait for RequestOutput from output queue
  -> yield output to serving layer
```

## 4. Scheduler And Continuous Batching

Key file:

- `vllm/v1/core/sched/scheduler.py`

New requests enter the scheduler's waiting queue. The scheduler repeatedly decides which requests can run in the current engine step.

Important concepts:

- `WAITING`: request has entered the system but is not scheduled yet.
- `RUNNING`: request is being served by the engine.
- `token_budget`: how many tokens can be scheduled in the current step.
- KV cache slots: GPU memory blocks used to store previous tokens' K/V tensors.

vLLM V1 scheduler does not hard-code a separate "prefill phase" and "decode phase" in the scheduler logic. Instead, it tracks how many tokens each request has already computed and how many still need computation.

Conceptually:

- Prefill computes prompt tokens and builds KV cache.
- Decode generates new tokens one step at a time while reusing KV cache.

Implementation-wise, the scheduler unifies them as token scheduling work.

This design supports continuous batching:

```text
engine step N:
  keep running unfinished requests
  add newly available waiting requests
  remove finished requests
  schedule tokens within budget and KV cache limits
```

So vLLM does not wait for one fixed batch to fully finish before accepting more work.

## 5. Model Execution And Output Update

Key files:

- `vllm/v1/engine/core.py`
- `vllm/v1/core/sched/scheduler.py`

Each engine step roughly follows this flow:

```text
scheduler_output = scheduler.schedule()
model_output = model_executor.execute_model(scheduler_output)
engine_core_outputs = scheduler.update_from_output(scheduler_output, model_output)
```

`model_executor.execute_model()` runs the model forward pass and produces sampled token ids.

`scheduler.update_from_output()` then:

- appends new token ids to the request state
- checks stop conditions
- marks finished requests
- frees KV cache blocks for finished requests
- creates `EngineCoreOutput`

Stop conditions include:

- EOS token
- configured stop token
- `max_tokens`
- `max_model_len`

## 6. OutputProcessor And Response Return

Key file:

- `vllm/v1/engine/output_processor.py`

The engine returns token ids. The client needs text. `OutputProcessor` uses the detokenizer to convert new token ids into incremental text.

For non-streaming requests:

```text
collect all RequestOutput
  -> build ChatCompletionResponse
  -> return one JSON response
```

For streaming requests:

```text
each RequestOutput
  -> build ChatCompletionStreamResponse chunk
  -> yield "data: {...}\n\n"
  -> final "data: [DONE]\n\n"
```

This is why `/chat/stream` can print model output token by token before the full answer is complete.

## 7. Relation To This Project's Benchmark

Observed local benchmark:

| Concurrency | Avg Latency | P95 Latency | Throughput | Tokens/s |
|---:|---:|---:|---:|---:|
| 1 | 1.64s | 1.77s | 0.61 req/s | 78.19 |
| 2 | 1.83s | 1.87s | 1.09 req/s | 139.73 |
| 4 | 1.87s | 1.90s | 1.79 req/s | 229.45 |

The result shows a common inference-serving tradeoff:

- Higher concurrency improved overall throughput and tokens/s.
- Single-request latency and P95 increased slightly.

This does not prove a production-level performance limit. It is a local observation consistent with vLLM's request scheduling and batching design.

Streaming benchmark:

| Metric | Value |
|---|---:|
| TTFT | 0.33s |
| Total streaming time | 1.90s |

TTFT includes queueing, prefill, first decode step, detokenization, and network return to the client.

## 8. Interview Explanation

A concise project explanation:

> I built a local inference service based on vLLM and FastAPI. The FastAPI layer provides business-facing `/chat` and `/chat/stream` APIs, while vLLM provides the OpenAI-compatible model serving layer. I also traced the vLLM request path from `/v1/chat/completions` through request parsing, chat template rendering, `SamplingParams`, `AsyncLLM`, `EngineCore`, scheduler, KV cache allocation, model execution, detokenization, and final JSON/SSE response. Based on this, I added simple benchmark scripts to observe latency, P50/P95, throughput, tokens/s, and TTFT under local low-concurrency conditions.

Important boundary:

This project is not a vLLM kernel or scheduler source-code optimization project. It is a local inference service and vLLM request-lifecycle learning project with performance observation.
