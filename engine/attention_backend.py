"""Attention backend boundary used by the educational ModelRunner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from engine.flash_attention import FlashAttentionCPU, StandardAttention


@dataclass(frozen=True)
class AttentionMetadata:
    """Minimal metadata normally built from scheduler and block-table state."""

    request_ids: list[int]
    num_scheduled_tokens: dict[int, int]
    block_tables: dict[int, list[int]]
    is_prefill: bool


class AttentionBackend(Protocol):
    """Interface implemented by attention kernels/backends."""

    name: str

    def forward(
        self,
        query: np.ndarray,
        key_cache: np.ndarray,
        value_cache: np.ndarray,
        metadata: AttentionMetadata,
    ) -> np.ndarray:
        """Execute attention and return the output tensor."""
        ...


class NaiveAttentionBackend:
    """Materializes the full attention matrix via NumPy."""

    name = "naive_numpy"

    def __init__(self) -> None:
        self._implementation = StandardAttention()

    def forward(
        self,
        query: np.ndarray,
        key_cache: np.ndarray,
        value_cache: np.ndarray,
        metadata: AttentionMetadata,
    ) -> np.ndarray:
        output, _ = self._implementation.forward(query, key_cache, value_cache)
        return output


class NumpyFlashAttentionBackend:
    """Uses tiled online-softmax attention without a full score matrix."""

    name = "flash_numpy"

    def __init__(self, block_size: int = 16) -> None:
        self._implementation = FlashAttentionCPU(block_size=block_size)

    def forward(
        self,
        query: np.ndarray,
        key_cache: np.ndarray,
        value_cache: np.ndarray,
        metadata: AttentionMetadata,
    ) -> np.ndarray:
        output, _ = self._implementation.forward(query, key_cache, value_cache)
        return output


def create_attention_backend(name: str) -> AttentionBackend:
    """Build a supported educational attention backend by name."""
    normalized = name.lower().replace("-", "_")
    if normalized in {"naive", "naive_numpy", "standard"}:
        return NaiveAttentionBackend()
    if normalized in {"flash", "flash_numpy", "numpy_flash"}:
        return NumpyFlashAttentionBackend()
    raise ValueError(f"Unknown attention backend: {name}")
