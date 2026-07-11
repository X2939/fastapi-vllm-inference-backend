"""Request data model and state machine.

Represents a single inference request flowing through the engine.
States follow vLLM's lifecycle: WAITING -> PREFILL -> DECODE -> FINISHED.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

logger = logging.getLogger(__name__)


class RequestStatus(Enum):
    """Lifecycle states for an inference request."""
    WAITING = "waiting"
    PREFILL = "prefill"
    DECODE = "decode"
    FINISHED = "finished"


@dataclass
class Request:
    """A single inference request.

    Attributes:
        id: Unique request identifier.
        arrival_time: Wall-clock time when the request was created.
        prompt_length: Number of tokens in the prompt.
        max_new_tokens: Maximum number of tokens to generate.
        prompt_tokens: The actual prompt token IDs (for prefix sharing).
        generated_tokens: Number of tokens generated so far.
        status: Current lifecycle state.
        priority: Scheduling priority (lower = higher priority).
        prefill_finish_time: When prefill completed (for TTFT).
        finish_time: When the request finished (for latency).
        admission_time: When the request was admitted from waiting queue.
        cached_prompt_tokens: Prompt tokens whose KV state was reused.
        num_computed_tokens: Total prompt/output positions processed by the model.
        output_token_ids: Sampled token IDs accepted by the scheduler.
        has_been_scheduled: Whether full metadata was sent to the model runner.
    """
    id: int
    arrival_time: float
    prompt_length: int
    max_new_tokens: int
    prompt_tokens: List[int] = field(default_factory=list)
    generated_tokens: int = 0
    status: RequestStatus = RequestStatus.WAITING
    priority: int = 0
    prefill_finish_time: Optional[float] = None
    finish_time: Optional[float] = None
    admission_time: Optional[float] = None
    cached_prompt_tokens: int = 0
    num_computed_tokens: int = 0
    output_token_ids: List[int] = field(default_factory=list)
    has_been_scheduled: bool = False

    @property
    def is_finished(self) -> bool:
        """Whether this request has finished generating."""
        return self.generated_tokens >= self.max_new_tokens

    @property
    def total_length(self) -> int:
        """Total tokens (prompt + generated)."""
        return self.prompt_length + self.generated_tokens

    @property
    def estimated_kv_size(self) -> int:
        """Estimated KV cache size needed (prompt + max output)."""
        return self.prompt_length + self.max_new_tokens

    @property
    def prefill_tokens(self) -> int:
        """Prompt tokens that still require prefill computation."""
        return max(0, self.prompt_length - self.cached_prompt_tokens)

    @property
    def remaining_prefill_tokens(self) -> int:
        """Prompt tokens not yet accounted for by model execution."""
        accounted = max(self.cached_prompt_tokens, self.num_computed_tokens)
        return max(0, self.prompt_length - accounted)

    @property
    def num_output_tokens(self) -> int:
        """Number of sampled output tokens accepted by the scheduler."""
        return len(self.output_token_ids)

    def __repr__(self) -> str:
        return (
            f"Request(id={self.id}, status={self.status.value}, "
            f"prompt={self.prompt_length}, gen={self.generated_tokens}/"
            f"{self.max_new_tokens})"
        )
