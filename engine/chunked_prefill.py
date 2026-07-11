"""Chunked-prefill progress accounting owned by the Scheduler."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from engine.request import Request


@dataclass
class PrefillChunkInfo:
    """Per-request progress through uncached prompt tokens."""

    request_id: int
    total_tokens: int
    processed_tokens: int = 0
    chunk_count: int = 0
    is_complete: bool = False

    @property
    def remaining_tokens(self) -> int:
        return max(0, self.total_tokens - self.processed_tokens)

    @property
    def progress_fraction(self) -> float:
        if self.total_tokens == 0:
            return 1.0
        return self.processed_tokens / self.total_tokens


@dataclass
class ChunkedPrefillStats:
    """Aggregate chunk counters used by experiments."""

    total_chunks: int = 0
    total_prefill_requests: int = 0
    total_chunk_tokens: int = 0

    @property
    def avg_chunks_per_request(self) -> float:
        if self.total_prefill_requests == 0:
            return 0.0
        return self.total_chunks / self.total_prefill_requests


class ChunkedPrefillHelper:
    """Track progress while Scheduler enforces the per-step token budget."""

    def __init__(self, chunk_size: int = 128) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = chunk_size
        self._progress: Dict[int, PrefillChunkInfo] = {}
        self._stats = ChunkedPrefillStats()

    @property
    def stats(self) -> ChunkedPrefillStats:
        return self._stats

    def register_request(self, request: Request) -> None:
        if request.id in self._progress:
            return
        total = request.prefill_tokens
        self._progress[request.id] = PrefillChunkInfo(
            request_id=request.id,
            total_tokens=total,
            is_complete=total == 0,
        )
        self._stats.total_prefill_requests += 1

    def get_next_chunk_size(self, request: Request) -> int:
        info = self._progress.get(request.id)
        if info is None or info.is_complete:
            return 0
        return min(self.chunk_size, info.remaining_tokens)

    def get_remaining_tokens(self, request: Request) -> int:
        info = self._progress.get(request.id)
        return info.remaining_tokens if info is not None else 0

    def is_prefill_complete(self, request: Request) -> bool:
        info = self._progress.get(request.id)
        return True if info is None else info.is_complete

    def mark_chunk_done(self, request: Request, tokens_processed: int) -> bool:
        info = self._progress.get(request.id)
        if info is None:
            raise ValueError(f"Request {request.id} is not registered")
        if tokens_processed <= 0:
            return info.is_complete
        info.processed_tokens = min(
            info.total_tokens,
            info.processed_tokens + tokens_processed,
        )
        info.chunk_count += 1
        self._stats.total_chunks += 1
        self._stats.total_chunk_tokens += tokens_processed
        info.is_complete = info.processed_tokens >= info.total_tokens
        return info.is_complete

    def remove_request(self, request_id: int) -> None:
        self._progress.pop(request_id, None)

    def reset(self) -> None:
        self._progress.clear()
        self._stats = ChunkedPrefillStats()
