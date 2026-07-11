"""ONNX Runtime executor for inference.

Uses ONNX Runtime to run a real exported model, providing actual GPU/CPU
inference instead of simulated time.sleep(). The interface mirrors the
PyTorch Executor for seamless backend switching.

Key differences from PyTorch Executor:
    - Loads a real ONNX model
    - Runs actual inference via onnxruntime
    - Measures real compute time (no time.sleep simulation)
    - Still updates request state and KV cache

Usage:
    executor = ONNXExecutor(
        kv_cache=kv_cache,
        model_path="models/simple_model.onnx",
        prefill_cost_per_token=0.001,
        decode_cost_per_token=0.0005,
    )
    executor.load_model()
    executor.warmup()
    timing = executor.execute(batch)
"""
from __future__ import annotations

import logging
import os
import time
from typing import Callable, Dict, List, Optional

import numpy as np

from engine.request import Request, RequestStatus
from engine.batch import Batch
from engine.kv_cache import KVCacheManager
from engine.executor import ExecutionTiming

logger = logging.getLogger(__name__)


class ONNXExecutor:
    """ONNX Runtime-based GPU executor.

    Executes prefill and decode phases using a real ONNX model.
    Interface mirrors Executor for drop-in replacement.

    Args:
        kv_cache: The KV cache manager.
        model_path: Path to the ONNX model file.
        prefill_cost_per_token: Fallback simulated cost (if model fails).
        decode_cost_per_token: Fallback simulated cost (if model fails).
        mark_prefill_done_fn: Callback for prefill completion.
        mark_finished_fn: Callback for request completion.
        providers: ONNX Runtime execution providers.
        vocab_size: Model vocabulary size for token-ID boundary clipping.
    """

    def __init__(
        self,
        kv_cache: KVCacheManager,
        model_path: str = "models/simple_model.onnx",
        prefill_cost_per_token: float = 0.001,
        decode_cost_per_token: float = 0.0005,
        mark_prefill_done_fn: Optional[Callable] = None,
        mark_finished_fn: Optional[Callable] = None,
        providers: Optional[List[str]] = None,
        vocab_size: int = 4096,
    ):
        self.kv_cache = kv_cache
        self.model_path = model_path
        self.prefill_cost_per_token = prefill_cost_per_token
        self.decode_cost_per_token = decode_cost_per_token
        self._mark_prefill_done_fn = mark_prefill_done_fn
        self._mark_finished_fn = mark_finished_fn
        self._providers = providers or ["CPUExecutionProvider"]
        # Embedding vocab size; token IDs are mod-clipped at the ONNX
        # boundary to stay within range for the exported model.
        self._vocab_size = vocab_size

        self._session = None
        self._total_tokens_generated: int = 0
        self._model_loaded: bool = False

    @property
    def total_tokens_generated(self) -> int:
        """Total tokens generated across all steps."""
        return self._total_tokens_generated

    @property
    def is_loaded(self) -> bool:
        """Whether the ONNX model is loaded."""
        return self._model_loaded

    def load_model(self) -> bool:
        """Load the ONNX model for inference.

        Returns:
            True if model loaded successfully, False otherwise.
        """
        try:
            import onnxruntime as ort

            if not os.path.exists(self.model_path):
                logger.warning(
                    f"ONNX model not found at {self.model_path}. "
                    f"Falling back to simulated execution."
                )
                return False

            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            )

            self._session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=self._providers,
            )
            self._model_loaded = True
            logger.info(
                f"ONNX model loaded: {self.model_path} "
                f"(providers: {self._providers})"
            )
            return True

        except ImportError:
            logger.warning(
                "onnxruntime not installed. Falling back to simulated execution."
            )
            return False
        except Exception as e:
            logger.warning(f"Failed to load ONNX model: {e}")
            return False

    def warmup(self, num_warmup_runs: int = 3) -> None:
        """Warm up the ONNX model with dummy inputs.

        Args:
            num_warmup_runs: Number of warmup inference calls.
        """
        if not self._model_loaded:
            return

        for i in range(num_warmup_runs):
            batch_size = 1
            seq_len = 8
            dummy_input = np.random.randint(
                0, 1000, (batch_size, seq_len), dtype=np.int64
            )
            self._session.run(None, {"input_ids": dummy_input})
        logger.info(f"ONNX warmup complete ({num_warmup_runs} runs)")

    def execute(self, batch: Batch) -> ExecutionTiming:
        """Execute a full batch step (prefill + decode).

        If the ONNX model is loaded, runs actual inference. Otherwise,
        falls back to time.sleep simulation (same as PyTorch Executor).

        Args:
            batch: The batch to execute.

        Returns:
            ExecutionTiming with timing breakdown.
        """
        step_start = time.time()
        timing = ExecutionTiming()

        # Prefill phase
        prefill_requests = self.build_prefill_batch(batch)
        if prefill_requests:
            timing.prefill_time = self.run_prefill(prefill_requests)
            self.update_prefill(prefill_requests)
            timing.prefill_count = len(prefill_requests)

        # Decode phase
        decode_requests = self.build_decode_batch(batch)
        if decode_requests:
            timing.decode_time = self.run_decode(decode_requests)
            self.update_decode(decode_requests)
            timing.decode_count = len(decode_requests)

        timing.total_time = time.time() - step_start
        timing.tokens_generated = timing.prefill_count + timing.decode_count
        self._total_tokens_generated += timing.tokens_generated

        return timing

    # ========================================================================
    # Prefill Pipeline
    # ========================================================================

    def build_prefill_batch(self, batch: Batch) -> List[Request]:
        """Extract prefill requests from a batch.

        Args:
            batch: The full batch.

        Returns:
            List of requests to prefill.
        """
        return list(batch.prefill_requests)

    def run_prefill(self, requests: List[Request]) -> float:
        """Run prefill inference.

        If ONNX model is loaded, runs actual model inference on the
        prompt tokens. Otherwise, falls back to time.sleep simulation.

        Args:
            requests: Prefill requests to execute.

        Returns:
            Total prefill time in seconds.
        """
        if self._model_loaded:
            return self._run_onnx_prefill(requests)
        return self._run_simulated_prefill(requests)

    def _run_onnx_prefill(self, requests: List[Request]) -> float:
        """Run actual ONNX inference for prefill.

        Args:
            requests: Prefill requests.

        Returns:
            Total inference time.
        """
        total_time = 0.0
        for request in requests:
            # Prepare input: (1, seq_len). Token IDs are mod-clipped
            # at the ONNX boundary to stay within the model's vocab.
            raw_ids = np.array([request.prompt_tokens], dtype=np.int64)
            input_ids = np.mod(raw_ids, self._vocab_size)
            start = time.time()
            self._session.run(None, {"input_ids": input_ids})
            total_time += time.time() - start
        return total_time

    def _run_simulated_prefill(self, requests: List[Request]) -> float:
        """Fallback: simulate prefill with time.sleep.

        Args:
            requests: Prefill requests.

        Returns:
            Total simulated time.
        """
        total_time = 0.0
        for request in requests:
            compute_time = request.prompt_length * self.prefill_cost_per_token
            time.sleep(compute_time)
            total_time += compute_time
        return total_time

    def update_prefill(self, requests: List[Request]) -> None:
        """Update request state after prefill.

        Args:
            requests: Prefill requests that were executed.
        """
        for request in requests:
            request.generated_tokens += 1
            self.kv_cache.append_token(request.id)

            if self._mark_prefill_done_fn:
                self._mark_prefill_done_fn(request)

            if request.is_finished:
                if self._mark_finished_fn:
                    self._mark_finished_fn(request)

    # ========================================================================
    # Decode Pipeline
    # ========================================================================

    def build_decode_batch(self, batch: Batch) -> List[Request]:
        """Extract decode requests from a batch.

        Args:
            batch: The full batch.

        Returns:
            List of requests to decode.
        """
        return list(batch.decode_requests)

    def run_decode(self, requests: List[Request]) -> float:
        """Run decode inference.

        If ONNX model is loaded, runs actual model inference. Otherwise,
        falls back to time.sleep simulation.

        Args:
            requests: Decode requests to execute.

        Returns:
            Total decode time in seconds.
        """
        if self._model_loaded:
            return self._run_onnx_decode(requests)
        return self._run_simulated_decode(requests)

    def _run_onnx_decode(self, requests: List[Request]) -> float:
        """Run actual ONNX inference for decode.

        Each decode step generates one token. We run the model with
        the current sequence (prompt + generated tokens so far).

        Args:
            requests: Decode requests.

        Returns:
            Total inference time.
        """
        total_time = 0.0
        for request in requests:
            # Build current sequence: prompt + generated tokens. The
            # generated token IDs are placeholders outside the vocab
            # range, so they are mod-clipped at the ONNX boundary.
            current_tokens = (
                request.prompt_tokens[:request.prompt_length]
                + list(range(9000, 9000 + request.generated_tokens))
            )
            raw_ids = np.array([current_tokens], dtype=np.int64)
            input_ids = np.mod(raw_ids, self._vocab_size)
            start = time.time()
            self._session.run(None, {"input_ids": input_ids})
            total_time += time.time() - start
        return total_time

    def _run_simulated_decode(self, requests: List[Request]) -> float:
        """Fallback: simulate decode with time.sleep.

        Args:
            requests: Decode requests.

        Returns:
            Total simulated time.
        """
        total_time = 0.0
        for request in requests:
            compute_time = self.decode_cost_per_token
            time.sleep(compute_time)
            total_time += compute_time
        return total_time

    def update_decode(self, requests: List[Request]) -> None:
        """Update request state after decode.

        Args:
            requests: Decode requests that were executed.
        """
        for request in requests:
            request.generated_tokens += 1
            self.kv_cache.append_token(request.id)

            if request.is_finished:
                if self._mark_finished_fn:
                    self._mark_finished_fn(request)

    def reset(self) -> None:
        """Reset executor state."""
        self._total_tokens_generated = 0
