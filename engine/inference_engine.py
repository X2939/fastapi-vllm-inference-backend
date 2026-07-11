"""Inference Engine: pure driver/orchestrator with backend selection.

The Engine is the top-level driver. It coordinates all subsystems but
contains NO business logic itself. Each step follows the vLLM V1 boundary:

    scheduler_output = scheduler.schedule()
    model_runner_output = worker.execute_model(scheduler_output)
    engine_outputs = scheduler.update_from_output(
        scheduler_output, model_runner_output
    )

Supports three backends:
    - "pytorch": ModelRunner + selectable AttentionBackend simulation
    - "onnx": ONNX compute adapter behind the ModelRunner boundary
    - "tensorrt": TensorRT compute adapter behind the ModelRunner boundary
"""
from __future__ import annotations

import logging
import time
from typing import Callable, Dict, List, Optional

from engine.request import Request
from engine.scheduler import Scheduler
from engine.onnx_executor import ONNXExecutor
from engine.tensorrt_executor import TensorRTExecutor
from engine.kv_cache import KVCacheManager
from engine.gpu_monitor import GPUMonitor
from engine.policy import BaseAdmissionPolicy, MemoryBudgetPolicy
from engine.model_runner import LegacyExecutorModelRunner, ModelRunner
from engine.worker import GPUWorker

logger = logging.getLogger(__name__)


class InferenceEngine:
    """End-to-end inference engine with backend selection.

    Orchestrates: Scheduler + Worker + ModelRunner + KVCache + GPUMonitor.
    The engine itself is stateless beyond step counting.

    Args:
        block_size: KV cache block size (tokens per block).
        num_blocks: Total physical blocks in KV cache.
        prefill_cost_per_token: Simulated prefill compute cost.
        decode_cost_per_token: Simulated decode compute cost.
        policy: Admission policy (defaults to MemoryBudgetPolicy).
        enable_prefix_sharing: Whether to enable prefix sharing.
        backend: Inference backend - "pytorch", "onnx", or "tensorrt".
        onnx_model_path: Path to ONNX model (for onnx backend).
        trt_engine_path: Path to TensorRT engine (for tensorrt backend).
        chunked_prefill: Whether to enable chunked prefill (pytorch backend).
        prefill_chunk_size: Max prompt tokens per prefill chunk.
        max_num_scheduled_tokens: Global token budget for each scheduler step.
        attention_backend: Educational attention backend used by ModelRunner.
    """

    def __init__(
        self,
        block_size: int = 16,
        num_blocks: int = 1024,
        prefill_cost_per_token: float = 0.001,
        decode_cost_per_token: float = 0.0005,
        policy: Optional[BaseAdmissionPolicy] = None,
        enable_prefix_sharing: bool = True,
        backend: str = "pytorch",
        onnx_model_path: str = "models/simple_model.onnx",
        trt_engine_path: str = "models/simple_model.engine",
        chunked_prefill: bool = False,
        prefill_chunk_size: int = 128,
        max_num_scheduled_tokens: int = 4096,
        attention_backend: str = "flash_numpy",
    ):
        self._chunked_prefill = chunked_prefill
        self._prefill_chunk_size = prefill_chunk_size
        # KV Cache
        self.kv_cache = KVCacheManager(
            block_size=block_size,
            num_blocks=num_blocks,
            enable_prefix_sharing=enable_prefix_sharing,
        )

        # Scheduler (with KV cache callbacks)
        self.scheduler = Scheduler(
            policy=policy,
            kv_alloc_fn=self.kv_cache.allocate_blocks,
            kv_append_fn=self.kv_cache.append_token,
            kv_free_fn=self.kv_cache.free_request,
            kv_cached_tokens_fn=self.kv_cache.get_cached_prompt_tokens,
            kv_block_table_fn=self.kv_cache.get_block_table,
            chunked_prefill=chunked_prefill,
            prefill_chunk_size=prefill_chunk_size,
            max_num_scheduled_tokens=max_num_scheduled_tokens,
        )

        # Select executor based on backend
        self.backend = backend
        self._prefill_cost = prefill_cost_per_token
        self._decode_cost = decode_cost_per_token

        if backend == "pytorch":
            self.model_runner = ModelRunner(
                prefill_cost_per_token=self._prefill_cost,
                decode_cost_per_token=self._decode_cost,
                attention_backend=attention_backend,
            )
        else:
            if chunked_prefill:
                raise ValueError(
                    "chunked_prefill is supported by the simulated ModelRunner only"
                )
            legacy_executor = self._create_executor(
                backend=backend,
                onnx_model_path=onnx_model_path,
                trt_engine_path=trt_engine_path,
            )
            self.model_runner = LegacyExecutorModelRunner(
                legacy_executor,
                backend_name=backend,
            )
        self.worker = GPUWorker(self.model_runner)
        # Compatibility alias for existing benchmark/result accessors.
        self.executor = self.model_runner

        # Monitor
        self.gpu_monitor = GPUMonitor()

        # State
        self._step_count: int = 0

    def _create_executor(
        self,
        backend: str,
        onnx_model_path: str,
        trt_engine_path: str,
    ) -> ONNXExecutor | TensorRTExecutor:
        """Create the appropriate executor based on backend.

        Args:
            backend: Backend name - "pytorch", "onnx", or "tensorrt".
            onnx_model_path: Path to ONNX model file.
            trt_engine_path: Path to TensorRT engine file.

        Returns:
            An executor instance with the standard interface.
        """
        if backend == "tensorrt":
            executor: TensorRTExecutor = TensorRTExecutor(
                kv_cache=self.kv_cache,
                engine_path=trt_engine_path,
                prefill_cost_per_token=self._prefill_cost,
                decode_cost_per_token=self._decode_cost,
            )
            loaded = executor.load_engine()
            if loaded:
                executor.warmup()
                logger.info("Using TensorRT backend")
            else:
                logger.info(
                    "TensorRT engine not available, using simulated execution"
                )
            return executor

        elif backend == "onnx":
            executor = ONNXExecutor(
                kv_cache=self.kv_cache,
                model_path=onnx_model_path,
                prefill_cost_per_token=self._prefill_cost,
                decode_cost_per_token=self._decode_cost,
            )
            loaded = executor.load_model()
            if loaded:
                executor.warmup()
                logger.info("Using ONNX Runtime backend")
            else:
                logger.info(
                    "ONNX model not available, using simulated execution"
                )
            return executor

        raise ValueError(f"Unsupported legacy backend: {backend}")

    @property
    def step_count(self) -> int:
        """Current step number."""
        return self._step_count

    @property
    def is_done(self) -> bool:
        """Whether all work is complete."""
        return self.scheduler.is_done

    def add_request(
        self,
        prompt_length: int,
        max_new_tokens: int,
        prompt_tokens: Optional[List[int]] = None,
        priority: int = 0,
    ) -> Request:
        """Add a new inference request.

        Args:
            prompt_length: Number of prompt tokens.
            max_new_tokens: Max tokens to generate.
            prompt_tokens: Actual token IDs (for prefix sharing).
            priority: Scheduling priority (lower = higher).

        Returns:
            The created Request.
        """
        return self.scheduler.add_request(
            prompt_length=prompt_length,
            max_new_tokens=max_new_tokens,
            prompt_tokens=prompt_tokens,
            priority=priority,
        )

    def run_step(self) -> Dict:
        """Execute one inference step.

        This is the core driver loop:
        1. Scheduler.schedule() -> SchedulerOutput
        2. Worker.execute_model(...) -> ModelRunnerOutput
        3. Scheduler.update_from_output(...) -> EngineOutput
        4. GPUMonitor.record_step(...)

        Returns:
            Dictionary with step info for timeline output.
        """
        self._step_count += 1
        step_start = time.time()

        # 1. Schedule
        sched_start = time.time()
        scheduler_output = self.scheduler.schedule()
        sched_time = time.time() - sched_start

        # 2. Execute
        exec_start = time.time()
        model_runner_output = self.worker.execute_model(scheduler_output)
        exec_time = time.time() - exec_start

        # 3. Apply output in the scheduler (the sole request-state owner).
        update_start = time.time()
        engine_outputs = self.scheduler.update_from_output(
            scheduler_output,
            model_runner_output,
        )
        update_time = time.time() - update_start
        batch = scheduler_output.batch
        timing = model_runner_output.timing

        # 4. Monitor
        step_duration = time.time() - step_start
        sched_state = self.scheduler.get_state()
        mem_used = 0
        if hasattr(self.scheduler.policy, 'current_memory'):
            mem_used = self.scheduler.policy.current_memory

        self.gpu_monitor.record_step(
            step=self._step_count,
            batch=batch,
            duration=step_duration,
            memory_usage=mem_used,
            queue_length=sched_state["waiting_size"],
            running_length=sched_state["running_size"],
        )

        gpu_stats = self.gpu_monitor.get_stats()
        return {
            "step": self._step_count,
            "batch_id": batch.batch_id,
            "prefill": len(batch.prefill_requests),
            "decode": len(batch.decode_requests),
            "batch_size": batch.total_requests,
            "waiting": sched_state["waiting_size"],
            "running": sched_state["running_size"],
            "finished": sched_state["finished_size"],
            "memory_used": mem_used,
            "memory_budget": getattr(
                self.scheduler.policy, 'memory_budget', 0
            ),
            "gpu_occupancy": gpu_stats.occupancy,
            "total_tokens": self.executor.total_tokens_generated,
            "step_duration": step_duration,
            "sched_time": sched_time,
            "exec_time": exec_time,
            "prefill_time": timing.prefill_time,
            "decode_time": timing.decode_time,
            "attention_time": timing.attention_time,
            "update_time": update_time,
            "scheduled_tokens": scheduler_output.total_num_scheduled_tokens,
            "num_scheduled_tokens": dict(scheduler_output.num_scheduled_tokens),
            "engine_outputs": engine_outputs,
            "attention_backend": model_runner_output.attention_backend,
        }

    def run(
        self,
        max_steps: int = 100000,
        callback: Optional[Callable[[Dict], None]] = None,
    ) -> List[Dict]:
        """Run the engine until all requests complete.

        Args:
            max_steps: Safety limit on steps.
            callback: Optional function called with step info each step.

        Returns:
            List of step info dictionaries.
        """
        steps: List[Dict] = []
        while not self.scheduler.is_done and self._step_count < max_steps:
            info = self.run_step()
            steps.append(info)
            if callback:
                callback(info)
        return steps

    def get_results(self) -> Dict:
        """Get final aggregated results from all subsystems."""
        gpu = self.gpu_monitor.get_stats()
        sched = self.scheduler.get_stats()
        kv = self.kv_cache.get_stats()
        finished = self.scheduler.get_finished_requests()

        # Compute per-request metrics
        ttfts = []
        tpots = []
        latencies = []
        for req in finished:
            if req.prefill_finish_time is not None:
                ttfts.append(req.prefill_finish_time - req.arrival_time)
            if (req.finish_time and req.prefill_finish_time
                    and req.generated_tokens > 1):
                dt = req.finish_time - req.prefill_finish_time
                dk = req.generated_tokens - 1
                if dk > 0:
                    tpots.append(dt / dk)
            if req.finish_time:
                latencies.append(req.finish_time - req.arrival_time)

        def pct(sorted_vals, p):
            if not sorted_vals:
                return 0.0
            k = (len(sorted_vals) - 1) * (p / 100.0)
            f = int(k)
            c = f + 1
            if c >= len(sorted_vals):
                return sorted_vals[-1]
            return sorted_vals[f] * (c - k) + sorted_vals[c] * (k - f)

        ttfts.sort()
        tpots.sort()
        latencies.sort()

        return {
            "backend": self.backend,
            "total_requests": len(finished),
            "total_tokens": gpu.total_tokens,
            "total_time": gpu.total_time,
            "ttft": {
                "avg": sum(ttfts) / len(ttfts) if ttfts else 0,
                "p50": pct(ttfts, 50),
                "p95": pct(ttfts, 95),
                "p99": pct(ttfts, 99),
            },
            "tpot": {
                "avg": sum(tpots) / len(tpots) if tpots else 0,
                "p50": pct(tpots, 50),
                "p95": pct(tpots, 95),
                "p99": pct(tpots, 99),
            },
            "latency": {
                "avg": sum(latencies) / len(latencies) if latencies else 0,
                "p50": pct(latencies, 50),
                "p95": pct(latencies, 95),
                "p99": pct(latencies, 99),
            },
            "throughput": gpu.tokens_per_second,
            "gpu_occupancy": gpu.occupancy,
            "gpu_busy_time": gpu.busy_time,
            "gpu_idle_time": gpu.idle_time,
            "peak_memory": gpu.peak_memory,
            "avg_memory": gpu.avg_memory,
            "peak_batch_size": gpu.peak_batch_size,
            "avg_batch_size": gpu.avg_batch_size,
            "tokens_per_second": gpu.tokens_per_second,
            "admission_rate": sched.admission_rate,
            "scheduled_tokens": sched.scheduled_tokens,
            "token_budget_exhaustions": sched.token_budget_exhaustions,
            "max_num_scheduled_tokens": self.scheduler.max_num_scheduled_tokens,
            "total_steps": self._step_count,
            "kv_memory_utilization": kv["memory_utilization"],
            "kv_peak_shared_blocks": kv["peak_shared_blocks"],
            "kv_prefix_cache_hits": kv["prefix_cache_hits"],
            "kv_prefix_cache_hit_rate": kv["prefix_cache_hit_rate"],
            "cached_prompt_tokens": sum(
                request.cached_prompt_tokens for request in finished
            ),
        }

    def reset(self) -> None:
        """Reset engine for a new run."""
        self.scheduler.reset()
        self.worker.reset()
        self.gpu_monitor.reset()
        self.kv_cache.reset()
        self._step_count = 0
