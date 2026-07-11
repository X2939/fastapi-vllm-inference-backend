"""ModelRunner boundary: input preparation, attention, and token sampling."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from engine.attention_backend import (
    AttentionBackend,
    AttentionMetadata,
    create_attention_backend,
)
from engine.outputs import ExecutionTiming, ModelRunnerOutput, SchedulerOutput


@dataclass(frozen=True)
class ModelInput:
    """Prepared request metadata consumed by the simulated forward pass."""

    req_id: int
    prompt_length: int
    num_computed_tokens: int
    num_output_tokens: int
    num_scheduled_tokens: int
    block_ids: list[int]

    @property
    def prefill_tokens(self) -> int:
        remaining_prompt = max(0, self.prompt_length - self.num_computed_tokens)
        return min(remaining_prompt, self.num_scheduled_tokens)

    @property
    def reaches_sampling_point(self) -> bool:
        return self.num_computed_tokens + self.num_scheduled_tokens >= self.prompt_length


class ModelRunner:
    """Single-process educational counterpart of vLLM's GPUModelRunner.

    It owns no scheduling state and never mutates Request objects. The runner
    prepares per-step inputs, invokes the selected attention backend on a tiny
    representative tensor, simulates batch compute, and returns sampled IDs.
    """

    def __init__(
        self,
        *,
        prefill_cost_per_token: float,
        decode_cost_per_token: float,
        attention_backend: str = "flash_numpy",
    ) -> None:
        self.prefill_cost_per_token = prefill_cost_per_token
        self.decode_cost_per_token = decode_cost_per_token
        self.attention_backend: AttentionBackend = create_attention_backend(
            attention_backend
        )
        self._request_cache: dict[int, ModelInput] = {}
        self._total_tokens_generated = 0

    @property
    def total_tokens_generated(self) -> int:
        return self._total_tokens_generated

    def prepare_inputs(self, scheduler_output: SchedulerOutput) -> list[ModelInput]:
        """Convert new/cached scheduler records into model-runner inputs."""
        records: dict[int, ModelInput] = {}
        for request in scheduler_output.new_requests:
            records[request.req_id] = ModelInput(
                req_id=request.req_id,
                prompt_length=request.prompt_length,
                num_computed_tokens=request.num_computed_tokens,
                num_output_tokens=request.num_output_tokens,
                num_scheduled_tokens=scheduler_output.num_scheduled_tokens[request.req_id],
                block_ids=request.block_ids,
            )
        for request in scheduler_output.cached_requests:
            records[request.req_id] = ModelInput(
                req_id=request.req_id,
                prompt_length=request.prompt_length,
                num_computed_tokens=request.num_computed_tokens,
                num_output_tokens=request.num_output_tokens,
                num_scheduled_tokens=scheduler_output.num_scheduled_tokens[request.req_id],
                block_ids=request.block_ids,
            )
        self._request_cache.update(records)
        for request_id in scheduler_output.finished_request_ids:
            self._request_cache.pop(request_id, None)
        return [records[request_id] for request_id in scheduler_output.scheduled_request_ids]

    def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        """Run one simulated model step without changing scheduler state."""
        started = time.perf_counter()
        model_inputs = self.prepare_inputs(scheduler_output)
        if not model_inputs:
            return ModelRunnerOutput(req_ids=[], req_id_to_index={})

        prefill_inputs = [item for item in model_inputs if item.prefill_tokens > 0]
        decode_inputs = [item for item in model_inputs if item.prefill_tokens == 0]

        attention_started = time.perf_counter()
        self._run_attention_probe(model_inputs, scheduler_output)
        attention_time = time.perf_counter() - attention_started

        prefill_time = self._simulate_prefill(prefill_inputs)
        decode_time = self._simulate_decode(decode_inputs)

        sampled_token_ids: list[list[int]] = []
        num_computed_tokens: dict[int, int] = {}
        for item in model_inputs:
            num_computed_tokens[item.req_id] = item.num_scheduled_tokens
            if item.reaches_sampling_point:
                sampled_token_ids.append([32_000 + item.num_output_tokens])
            else:
                sampled_token_ids.append([])

        generated = sum(len(tokens) for tokens in sampled_token_ids)
        self._total_tokens_generated += generated
        timing = ExecutionTiming(
            prefill_time=prefill_time,
            decode_time=decode_time,
            attention_time=attention_time,
            total_time=time.perf_counter() - started,
            prefill_count=len(prefill_inputs),
            decode_count=len(decode_inputs),
            tokens_generated=generated,
        )
        req_ids = [item.req_id for item in model_inputs]
        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={request_id: index for index, request_id in enumerate(req_ids)},
            sampled_token_ids=sampled_token_ids,
            num_computed_tokens=num_computed_tokens,
            timing=timing,
            attention_backend=self.attention_backend.name,
        )

    def _run_attention_probe(
        self,
        model_inputs: list[ModelInput],
        scheduler_output: SchedulerOutput,
    ) -> None:
        """Exercise the selected backend on a bounded representative tensor."""
        sequence_length = max(1, min(4, max(item.num_scheduled_tokens for item in model_inputs)))
        query = np.ones((1, 1, sequence_length, 8), dtype=np.float32)
        key = np.ones_like(query)
        value = np.ones_like(query)
        metadata = AttentionMetadata(
            request_ids=scheduler_output.scheduled_request_ids,
            num_scheduled_tokens=scheduler_output.num_scheduled_tokens,
            block_tables=scheduler_output.block_table_updates,
            is_prefill=any(item.prefill_tokens > 0 for item in model_inputs),
        )
        self.attention_backend.forward(query, key, value, metadata)

    def _simulate_prefill(self, inputs: list[ModelInput]) -> float:
        if not inputs:
            return 0.0
        token_work = sum(item.prefill_tokens for item in inputs)
        duration = token_work * self.prefill_cost_per_token / math.pow(len(inputs), 0.65)
        time.sleep(duration)
        return duration

    def _simulate_decode(self, inputs: list[ModelInput]) -> float:
        if not inputs:
            return 0.0
        duration = len(inputs) * self.decode_cost_per_token / math.pow(len(inputs), 0.80)
        time.sleep(duration)
        return duration

    def reset(self) -> None:
        self._request_cache.clear()
        self._total_tokens_generated = 0


class LegacyExecutorModelRunner:
    """Adapter that exposes ONNX/TensorRT compute through ModelRunnerOutput.

    The legacy executor is used only for its compute methods. Its state-update
    methods are deliberately not called; Scheduler.update_from_output remains
    the single owner of request and KV-cache lifecycle.
    """

    def __init__(self, executor: object, backend_name: str) -> None:
        self.executor = executor
        self.backend_name = backend_name
        self._total_tokens_generated = 0

    @property
    def total_tokens_generated(self) -> int:
        return self._total_tokens_generated

    def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        started = time.perf_counter()
        batch = scheduler_output.batch
        prefill_started = time.perf_counter()
        prefill_time = (
            self.executor.run_prefill(batch.prefill_requests)
            if batch.prefill_requests
            else 0.0
        )
        prefill_elapsed = time.perf_counter() - prefill_started
        decode_started = time.perf_counter()
        decode_time = (
            self.executor.run_decode(batch.decode_requests)
            if batch.decode_requests
            else 0.0
        )
        decode_elapsed = time.perf_counter() - decode_started

        sampled_by_request: dict[int, list[int]] = {}
        computed_tokens: dict[int, int] = {}
        for request in batch.all_requests():
            scheduled = scheduler_output.num_scheduled_tokens[request.id]
            computed_tokens[request.id] = scheduled
            if request.num_computed_tokens + scheduled >= request.prompt_length:
                sampled_by_request[request.id] = [32_000 + request.num_output_tokens]
            else:
                sampled_by_request[request.id] = []

        req_ids = scheduler_output.scheduled_request_ids
        sampled_token_ids = [sampled_by_request[request_id] for request_id in req_ids]
        generated = sum(len(tokens) for tokens in sampled_token_ids)
        self._total_tokens_generated += generated
        timing = ExecutionTiming(
            prefill_time=prefill_time or prefill_elapsed,
            decode_time=decode_time or decode_elapsed,
            total_time=time.perf_counter() - started,
            prefill_count=len(batch.prefill_requests),
            decode_count=len(batch.decode_requests),
            tokens_generated=generated,
        )
        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={request_id: index for index, request_id in enumerate(req_ids)},
            sampled_token_ids=sampled_token_ids,
            num_computed_tokens=computed_tokens,
            timing=timing,
            attention_backend=self.backend_name,
        )

    def reset(self) -> None:
        self._total_tokens_generated = 0
        if hasattr(self.executor, "reset"):
            self.executor.reset()
