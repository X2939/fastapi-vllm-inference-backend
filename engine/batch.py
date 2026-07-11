"""Batch data structures for the inference engine.

A Batch represents a single GPU step's workload. It may contain both
prefill requests (newly admitted, processing full prompt) and decode
requests (already running, generating one token each). This mixed
batching is vLLM's core optimization.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from engine.request import Request


@dataclass
class BatchMetadata:
    """Pre-computed metadata describing a batch.

    Computed at batch construction time so consumers can read
    properties without re-iterating request lists.

    Attributes:
        batch_id: Unique batch identifier.
        create_time: Wall-clock time when the batch was created.
        prefill_tokens: Total prompt tokens across all prefill requests.
        decode_tokens: Number of decode requests (each generates 1 token).
        estimated_compute_cost: Rough compute estimate (prefill * 1.0 + decode * 0.5).
        estimated_memory: Estimated KV cache memory usage at batch creation.
    """
    batch_id: int
    create_time: float
    prefill_tokens: int = 0
    decode_tokens: int = 0
    estimated_compute_cost: float = 0.0
    estimated_memory: int = 0

    @property
    def total_tokens(self) -> int:
        """Total tokens processed in this batch."""
        return self.prefill_tokens + self.decode_tokens

    @property
    def is_empty(self) -> bool:
        """Whether this batch has no work."""
        return self.total_tokens == 0


@dataclass
class Batch:
    """A single GPU step's workload with mixed prefill and decode requests.

    Attributes:
        metadata: Pre-computed batch metadata.
        prefill_requests: Requests in the prefill phase.
        decode_requests: Requests in the decode phase.
    """
    metadata: BatchMetadata
    prefill_requests: List[Request] = field(default_factory=list)
    decode_requests: List[Request] = field(default_factory=list)

    @property
    def total_requests(self) -> int:
        """Total number of requests in this batch."""
        return len(self.prefill_requests) + len(self.decode_requests)

    @property
    def total_tokens(self) -> int:
        """Total tokens processed in this batch."""
        return self.metadata.total_tokens

    @property
    def is_empty(self) -> bool:
        """Whether this batch has no work."""
        return self.metadata.is_empty

    @property
    def batch_id(self) -> int:
        """Convenience accessor for batch ID."""
        return self.metadata.batch_id

    def all_requests(self) -> List[Request]:
        """Return all requests (prefill + decode) in this batch."""
        return self.prefill_requests + self.decode_requests

    @classmethod
    def empty(cls, batch_id: int) -> "Batch":
        """Create an empty batch with the given ID."""
        return cls(
            metadata=BatchMetadata(
                batch_id=batch_id,
                create_time=time.time(),
            )
        )

    @classmethod
    def from_requests(
        cls,
        batch_id: int,
        prefill_requests: List[Request],
        decode_requests: List[Request],
        memory_usage: int = 0,
        num_scheduled_tokens: Optional[Dict[int, int]] = None,
    ) -> "Batch":
        """Build a Batch from request lists, computing metadata.

        Args:
            batch_id: Unique batch identifier.
            prefill_requests: Requests entering prefill phase.
            decode_requests: Requests continuing decode phase.
            memory_usage: Current KV cache memory usage.

        Returns:
            A new Batch with pre-computed metadata.
        """
        if num_scheduled_tokens is None:
            prefill_tokens = sum(r.prompt_length for r in prefill_requests)
            decode_tokens = len(decode_requests)
        else:
            prefill_tokens = sum(
                num_scheduled_tokens.get(request.id, 0)
                for request in prefill_requests
            )
            decode_tokens = sum(
                num_scheduled_tokens.get(request.id, 0)
                for request in decode_requests
            )
        estimated_compute = prefill_tokens * 1.0 + decode_tokens * 0.5

        metadata = BatchMetadata(
            batch_id=batch_id,
            create_time=time.time(),
            prefill_tokens=prefill_tokens,
            decode_tokens=decode_tokens,
            estimated_compute_cost=estimated_compute,
            estimated_memory=memory_usage,
        )
        return cls(
            metadata=metadata,
            prefill_requests=list(prefill_requests),
            decode_requests=list(decode_requests),
        )

    def __repr__(self) -> str:
        return (
            f"Batch(id={self.metadata.batch_id}, "
            f"prefill={len(self.prefill_requests)}, "
            f"decode={len(self.decode_requests)})"
        )
