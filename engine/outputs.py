"""Boundary objects mirroring the vLLM V1 engine-core data flow.

These types intentionally keep only the fields needed by this educational
engine. They are semantic counterparts, not drop-in vLLM API replacements.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from engine.batch import Batch


@dataclass(frozen=True)
class NewRequestData:
    """Full request metadata sent to a model runner for the first time."""

    req_id: int
    prompt_token_ids: list[int]
    block_ids: list[int]
    num_computed_tokens: int
    num_output_tokens: int
    max_new_tokens: int

    @property
    def prompt_length(self) -> int:
        return len(self.prompt_token_ids)


@dataclass(frozen=True)
class CachedRequestData:
    """Small per-step update for a request already cached by the worker."""

    req_id: int
    block_ids: list[int]
    num_computed_tokens: int
    num_output_tokens: int
    prompt_length: int
    max_new_tokens: int


@dataclass
class SchedulerOutput:
    """Everything a model worker needs for one execution step."""

    scheduled_request_ids: list[int]
    num_scheduled_tokens: dict[int, int]
    new_requests: list[NewRequestData]
    cached_requests: list[CachedRequestData]
    finished_request_ids: set[int]
    total_num_scheduled_tokens: int
    block_table_updates: dict[int, list[int]]
    batch: Batch

    @classmethod
    def empty(cls, batch_id: int, finished_request_ids: set[int] | None = None) -> "SchedulerOutput":
        return cls(
            scheduled_request_ids=[],
            num_scheduled_tokens={},
            new_requests=[],
            cached_requests=[],
            finished_request_ids=finished_request_ids or set(),
            total_num_scheduled_tokens=0,
            block_table_updates={},
            batch=Batch.empty(batch_id),
        )


@dataclass
class ExecutionTiming:
    """Model execution timing returned across the runner boundary."""

    prefill_time: float = 0.0
    decode_time: float = 0.0
    attention_time: float = 0.0
    total_time: float = 0.0
    prefill_count: int = 0
    decode_count: int = 0
    tokens_generated: int = 0

    @property
    def total_count(self) -> int:
        return self.prefill_count + self.decode_count


@dataclass
class ModelRunnerOutput:
    """Pure model-execution result consumed by Scheduler.update_from_output."""

    req_ids: list[int]
    req_id_to_index: dict[int, int]
    sampled_token_ids: list[list[int]] = field(default_factory=list)
    num_computed_tokens: dict[int, int] = field(default_factory=dict)
    timing: ExecutionTiming = field(default_factory=ExecutionTiming)
    attention_backend: str = ""


@dataclass(frozen=True)
class EngineOutput:
    """User-facing delta emitted after scheduler state is updated."""

    request_id: int
    token_ids: list[int]
    finished: bool
    finish_reason: str | None = None
