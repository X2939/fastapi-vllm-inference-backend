"""Admission policy plugin system.

Implements the Strategy Pattern for admission control. The Scheduler
delegates admission decisions to a policy object, allowing different
admission strategies without modifying the Scheduler itself.

Policies:
    - MemoryBudgetPolicy: Admit based on KV cache memory budget
    - MaxSeqPolicy: Admit based on max concurrent sequences
    - PriorityPolicy: Admit based on request priority
    - CompositePolicy: Combine multiple policies (AND logic)
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import List, Optional

from engine.request import Request

logger = logging.getLogger(__name__)


class BaseAdmissionPolicy(ABC):
    """Abstract base class for admission policies.

    Subclasses must implement can_admit() and record_admission().
    """

    @abstractmethod
    def can_admit(
        self,
        request: Request,
        current_running: int,
        current_memory: int,
    ) -> bool:
        """Check if a request can be admitted.

        Args:
            request: The request to check.
            current_running: Number of currently running requests.
            current_memory: Current memory usage (in token units).

        Returns:
            True if the request can be admitted.
        """
        raise NotImplementedError

    @abstractmethod
    def record_admission(self, request: Request) -> None:
        """Record that a request was admitted.

        Called after the request is successfully admitted, so the
        policy can update its internal state.

        Args:
            request: The admitted request.
        """
        raise NotImplementedError

    @abstractmethod
    def record_completion(self, request: Request) -> None:
        """Record that a request completed and was freed.

        Args:
            request: The completed request.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """Reset policy state (for reuse across runs)."""
        pass


class MemoryBudgetPolicy(BaseAdmissionPolicy):
    """Admit requests based on a GPU memory budget.

    A request is admitted only if there is enough memory to hold
    its estimated KV cache (prompt_length + max_new_tokens).
    """

    def __init__(self, memory_budget: int = 80000):
        """Initialize with a memory budget.

        Args:
            memory_budget: Maximum total KV cache tokens.
        """
        self.memory_budget = memory_budget
        self._current_memory: int = 0

    @property
    def current_memory(self) -> int:
        """Current memory usage."""
        return self._current_memory

    @property
    def memory_utilization(self) -> float:
        """Fraction of memory budget used."""
        if self.memory_budget == 0:
            return 0.0
        return self._current_memory / self.memory_budget

    def can_admit(
        self,
        request: Request,
        current_running: int,
        current_memory: int,
    ) -> bool:
        required = request.estimated_kv_size
        return self._current_memory + required <= self.memory_budget

    def record_admission(self, request: Request) -> None:
        self._current_memory += request.estimated_kv_size
        logger.debug(
            f"MemoryBudgetPolicy: admitted req {request.id}, "
            f"mem={self._current_memory}/{self.memory_budget}"
        )

    def record_completion(self, request: Request) -> None:
        self._current_memory -= request.estimated_kv_size
        self._current_memory = max(0, self._current_memory)

    def reset(self) -> None:
        self._current_memory = 0


class MaxSeqPolicy(BaseAdmissionPolicy):
    """Admit requests based on max concurrent sequences.

    Simple policy: never exceed a fixed number of running requests.
    """

    def __init__(self, max_num_seqs: int = 256):
        """Initialize with a max sequence count.

        Args:
            max_num_seqs: Maximum number of concurrent running requests.
        """
        self.max_num_seqs = max_num_seqs
        self._current_running: int = 0

    @property
    def current_running(self) -> int:
        """Current running count."""
        return self._current_running

    def can_admit(
        self,
        request: Request,
        current_running: int,
        current_memory: int,
    ) -> bool:
        return self._current_running < self.max_num_seqs

    def record_admission(self, request: Request) -> None:
        self._current_running += 1

    def record_completion(self, request: Request) -> None:
        self._current_running = max(0, self._current_running - 1)

    def reset(self) -> None:
        self._current_running = 0


class PriorityPolicy(BaseAdmissionPolicy):
    """Admit requests based on priority.

    Higher-priority requests (lower priority value) are admitted first.
    Optionally enforces a max concurrent count per priority level.
    """

    def __init__(self, max_num_seqs: int = 256, max_per_priority: int = 64):
        """Initialize with priority limits.

        Args:
            max_num_seqs: Maximum total running requests.
            max_per_priority: Max running per priority level.
        """
        self.max_num_seqs = max_num_seqs
        self.max_per_priority = max_per_priority
        self._priority_counts: dict[int, int] = {}
        self._total_running: int = 0

    def can_admit(
        self,
        request: Request,
        current_running: int,
        current_memory: int,
    ) -> bool:
        if self._total_running >= self.max_num_seqs:
            return False
        count = self._priority_counts.get(request.priority, 0)
        return count < self.max_per_priority

    def record_admission(self, request: Request) -> None:
        self._priority_counts[request.priority] = (
            self._priority_counts.get(request.priority, 0) + 1
        )
        self._total_running += 1

    def record_completion(self, request: Request) -> None:
        self._priority_counts[request.priority] = max(
            0, self._priority_counts.get(request.priority, 0) - 1
        )
        self._total_running = max(0, self._total_running - 1)

    def reset(self) -> None:
        self._priority_counts.clear()
        self._total_running = 0


class CompositePolicy(BaseAdmissionPolicy):
    """Combine multiple policies with AND logic.

    A request is admitted only if ALL sub-policies agree.
    """

    def __init__(self, policies: List[BaseAdmissionPolicy]):
        """Initialize with a list of sub-policies.

        Args:
            policies: Sub-policies to evaluate (all must agree).
        """
        self._policies = policies

    def can_admit(
        self,
        request: Request,
        current_running: int,
        current_memory: int,
    ) -> bool:
        return all(
            p.can_admit(request, current_running, current_memory)
            for p in self._policies
        )

    def record_admission(self, request: Request) -> None:
        for p in self._policies:
            p.record_admission(request)

    def record_completion(self, request: Request) -> None:
        for p in self._policies:
            p.record_completion(request)

    def reset(self) -> None:
        for p in self._policies:
            p.reset()
