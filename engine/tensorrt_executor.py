"""TensorRT executor for inference.

Uses NVIDIA TensorRT to run an optimized engine, providing real GPU
inference with maximum performance. Device memory management is handled
via PyTorch CUDA tensors (no pycuda dependency required).

The interface mirrors Executor and ONNXExecutor for seamless backend
switching.

Key features:
    - Loads serialized TensorRT engines (.engine files)
    - Real GPU inference via TensorRT + PyTorch CUDA tensors
    - Dynamic shape support via optimization profiles
    - Graceful fallback to time.sleep simulation if TensorRT unavailable
    - Token ID mod-clipping for vocab boundary safety

Usage:
    executor = TensorRTExecutor(
        kv_cache=kv_cache,
        engine_path="models/simple_model.engine",
        prefill_cost_per_token=0.001,
        decode_cost_per_token=0.0005,
    )
    executor.load_engine()
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


class TensorRTExecutor:
    """TensorRT-based GPU executor.

    Executes prefill and decode phases using a real TensorRT engine.
    Interface mirrors Executor and ONNXExecutor for drop-in replacement.

    Uses PyTorch CUDA tensors for device memory management to avoid
    pycuda dependency issues.

    Args:
        kv_cache: The KV cache manager.
        engine_path: Path to the TensorRT engine file.
        prefill_cost_per_token: Fallback simulated cost (if engine fails).
        decode_cost_per_token: Fallback simulated cost (if engine fails).
        mark_prefill_done_fn: Callback for prefill completion.
        mark_finished_fn: Callback for request completion.
        vocab_size: Model vocabulary size for token-ID boundary clipping.
    """

    def __init__(
        self,
        kv_cache: KVCacheManager,
        engine_path: str = "models/simple_model.engine",
        prefill_cost_per_token: float = 0.001,
        decode_cost_per_token: float = 0.0005,
        mark_prefill_done_fn: Optional[Callable] = None,
        mark_finished_fn: Optional[Callable] = None,
        vocab_size: int = 4096,
    ):
        self.kv_cache = kv_cache
        self.engine_path = engine_path
        self.prefill_cost_per_token = prefill_cost_per_token
        self.decode_cost_per_token = decode_cost_per_token
        self._mark_prefill_done_fn = mark_prefill_done_fn
        self._mark_finished_fn = mark_finished_fn
        # Embedding vocab size; token IDs are mod-clipped at the TRT
        # boundary to stay within range for the engine's model.
        self._vocab_size = vocab_size

        # TensorRT state (lazily initialized)
        self._trt_logger = None
        self._runtime = None
        self._engine = None
        self._context = None
        self._engine_loaded = False
        self._cuda_available = False

        # Binding info (filled after load)
        self._input_idx: int = -1
        self._output_idx: int = -1
        self._input_name: str = ""
        self._output_name: str = ""

        self._total_tokens_generated: int = 0

    @property
    def total_tokens_generated(self) -> int:
        """Total tokens generated across all steps."""
        return self._total_tokens_generated

    @property
    def is_loaded(self) -> bool:
        """Whether the TensorRT engine is loaded."""
        return self._engine_loaded

    def load_engine(self) -> bool:
        """Load a serialized TensorRT engine.

        Uses PyTorch CUDA for device memory management. Falls back to
        simulated execution if TensorRT or CUDA is unavailable.

        Returns:
            True if engine loaded successfully, False otherwise.
        """
        try:
            import tensorrt as trt
            import torch

            # Check CUDA availability via torch
            if not torch.cuda.is_available():
                logger.warning(
                    "CUDA not available via torch. "
                    "Falling back to simulated execution."
                )
                return False
            self._cuda_available = True

            if not os.path.exists(self.engine_path):
                logger.warning(
                    f"TensorRT engine not found at {self.engine_path}. "
                    f"Falling back to simulated execution."
                )
                return False

            # Load engine
            self._trt_logger = trt.Logger(trt.Logger.WARNING)
            self._runtime = trt.Runtime(self._trt_logger)

            with open(self.engine_path, "rb") as f:
                self._engine = self._runtime.deserialize_cuda_engine(f.read())

            if self._engine is None:
                logger.error("Failed to deserialize TensorRT engine")
                return False

            self._context = self._engine.create_execution_context()

            # Resolve binding names and indices
            for i in range(self._engine.num_io_tensors):
                name = self._engine.get_tensor_name(i)
                mode = self._engine.get_tensor_mode(name)
                import tensorrt as trt
                if mode == trt.TensorIOMode.INPUT:
                    self._input_idx = i
                    self._input_name = name
                else:
                    self._output_idx = i
                    self._output_name = name

            if self._input_idx < 0 or self._output_idx < 0:
                logger.error("Could not find input/output bindings")
                return False

            self._engine_loaded = True
            logger.info(
                f"TensorRT engine loaded: {self.engine_path} "
                f"(input={self._input_name}, output={self._output_name})"
            )
            return True

        except ImportError as e:
            logger.warning(
                f"TensorRT not installed ({e}). "
                f"Falling back to simulated execution."
            )
            return False
        except Exception as e:
            logger.warning(f"Failed to load TensorRT engine: {e}")
            return False

    # Alias for interface consistency with ONNXExecutor
    def load_model(self) -> bool:
        """Load the TensorRT engine (alias for load_engine)."""
        return self.load_engine()

    def warmup(self, num_warmup_runs: int = 3) -> None:
        """Warm up the TensorRT engine with dummy inputs.

        Args:
            num_warmup_runs: Number of warmup inference calls.
        """
        if not self._engine_loaded:
            return

        import torch

        input_shape = (1, 32)
        input_data = np.random.randint(
            0, self._vocab_size, input_shape, dtype=np.int32
        )

        # Use torch CUDA tensors for device memory
        d_input = torch.from_numpy(input_data).cuda().int()
        # Allocate output buffer - first set shape to get output shape
        self._context.set_input_shape(self._input_name, input_shape)
        output_shape = tuple(
            self._context.get_tensor_shape(self._output_name)
        )
        d_output = torch.zeros(output_shape, dtype=torch.float32, device="cuda")

        self._context.set_tensor_address(self._input_name, d_input.data_ptr())
        self._context.set_tensor_address(self._output_name, d_output.data_ptr())

        for i in range(num_warmup_runs):
            self._context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        torch.cuda.current_stream().synchronize()

        # Synchronize to ensure warmup completes
        torch.cuda.synchronize()

        logger.info(f"TensorRT warmup complete ({num_warmup_runs} runs)")

    def execute(self, batch: Batch) -> ExecutionTiming:
        """Execute a full batch step (prefill + decode).

        If the TensorRT engine is loaded, runs actual GPU inference.
        Otherwise, falls back to time.sleep simulation.

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

        If TensorRT engine is loaded, runs actual GPU inference.
        Otherwise, falls back to time.sleep simulation.

        Args:
            requests: Prefill requests to execute.

        Returns:
            Total prefill time in seconds.
        """
        if self._engine_loaded:
            return self._run_trt_prefill(requests)
        return self._run_simulated_prefill(requests)

    def _run_trt_prefill(self, requests: List[Request]) -> float:
        """Run actual TensorRT inference for prefill.

        Each request is processed sequentially. Uses torch CUDA tensors
        for device memory and data transfer.

        Args:
            requests: Prefill requests.

        Returns:
            Total inference time.
        """
        import torch

        total_time = 0.0
        for request in requests:
            # Prepare input: (1, seq_len). Token IDs mod-clipped.
            raw_ids = np.array([request.prompt_tokens], dtype=np.int32)
            input_np = np.mod(raw_ids, self._vocab_size)
            input_shape = input_np.shape

            # Copy to GPU via torch tensor
            d_input = torch.from_numpy(input_np).cuda().int()

            # Set input shape for dynamic profile
            self._context.set_input_shape(self._input_name, input_shape)

            # Get output shape and allocate
            output_shape = tuple(
                self._context.get_tensor_shape(self._output_name)
            )
            d_output = torch.zeros(
                output_shape, dtype=torch.float32, device="cuda"
            )

            # Set tensor addresses
            self._context.set_tensor_address(self._input_name, d_input.data_ptr())
            self._context.set_tensor_address(self._output_name, d_output.data_ptr())

            # Run inference with timing
            torch.cuda.synchronize()
            start = time.time()
            self._context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
            torch.cuda.current_stream().synchronize()
            torch.cuda.synchronize()
            total_time += time.time() - start

            # (Optional) copy output back - not needed for benchmarking
            # output_np = d_output.cpu().numpy()

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

        If TensorRT engine is loaded, runs actual GPU inference.
        Otherwise, falls back to time.sleep simulation.

        Args:
            requests: Decode requests to execute.

        Returns:
            Total decode time in seconds.
        """
        if self._engine_loaded:
            return self._run_trt_decode(requests)
        return self._run_simulated_decode(requests)

    def _run_trt_decode(self, requests: List[Request]) -> float:
        """Run actual TensorRT inference for decode.

        Each decode step generates one token. Uses torch CUDA tensors
        for device memory management.

        Args:
            requests: Decode requests.

        Returns:
            Total inference time.
        """
        import torch

        total_time = 0.0
        for request in requests:
            # Build current sequence: prompt + generated tokens
            current_tokens = (
                request.prompt_tokens[:request.prompt_length]
                + list(range(9000, 9000 + request.generated_tokens))
            )
            raw_ids = np.array([current_tokens], dtype=np.int32)
            input_np = np.mod(raw_ids, self._vocab_size)
            input_shape = input_np.shape

            # Copy to GPU
            d_input = torch.from_numpy(input_np).cuda().int()

            # Set input shape
            self._context.set_input_shape(self._input_name, input_shape)

            # Allocate output
            output_shape = tuple(
                self._context.get_tensor_shape(self._output_name)
            )
            d_output = torch.zeros(
                output_shape, dtype=torch.float32, device="cuda"
            )

            # Set tensor addresses
            self._context.set_tensor_address(self._input_name, d_input.data_ptr())
            self._context.set_tensor_address(self._output_name, d_output.data_ptr())

            # Run with timing
            torch.cuda.synchronize()
            start = time.time()
            self._context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
            torch.cuda.current_stream().synchronize()
            torch.cuda.synchronize()
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
