"""Worker boundary for device/environment ownership."""

from __future__ import annotations

from engine.model_runner import ModelRunner
from engine.outputs import ModelRunnerOutput, SchedulerOutput


class GPUWorker:
    """Single-process worker that delegates execution to one ModelRunner.

    Production vLLM uses workers to own accelerator state and participate in
    distributed execution. This teaching worker intentionally stays local.
    """

    def __init__(self, model_runner: ModelRunner, device: str = "cpu-sim") -> None:
        self.model_runner = model_runner
        self.device = device

    def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput:
        return self.model_runner.execute_model(scheduler_output)

    def reset(self) -> None:
        self.model_runner.reset()
