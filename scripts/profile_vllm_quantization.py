#!/usr/bin/env python3
"""Run matched BF16/AWQ/AWQ-Marlin vLLM profiler sessions on one GPU.

This script is intentionally separate from the performance benchmark: profiler
overhead makes its latency numbers unsuitable as production benchmark results.
The generated traces are used to compare operator/kernel composition only.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


PROMPT = (
    "Explain why weight-only INT4 quantization can reduce model weight memory "
    "without necessarily improving decode latency on a consumer GPU. Discuss "
    "kernel efficiency, batch shape, memory bandwidth, and KV cache."
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_text(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout.strip(),
        "stderr": result.stderr.strip(),
    }


def gpu_snapshot() -> dict[str, Any]:
    return run_text(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.used,memory.free,"
            "utilization.gpu,utilization.memory,power.draw",
            "--format=csv,noheader,nounits",
        ]
    )


class PowerSampler:
    def __init__(self, output_csv: Path, interval_s: float = 0.2) -> None:
        self.output_csv = output_csv
        self.interval_s = interval_s
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)

    def _run(self) -> None:
        fields = [
            "timestamp",
            "memory_used_mib",
            "gpu_utilization_percent",
            "memory_utilization_percent",
            "power_w",
        ]
        with self.output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            while not self.stop_event.is_set():
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=memory.used,utilization.gpu,"
                        "utilization.memory,power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                if result.returncode == 0 and result.stdout.strip():
                    parts = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
                    if len(parts) == 4:
                        writer.writerow(
                            {
                                "timestamp": time.time(),
                                "memory_used_mib": parts[0],
                                "gpu_utilization_percent": parts[1],
                                "memory_utilization_percent": parts[2],
                                "power_w": parts[3],
                            }
                        )
                        handle.flush()
                self.stop_event.wait(self.interval_s)


def auth_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def wait_for_server(base_url: str, api_key: str, process: subprocess.Popen, timeout_s: int) -> str:
    deadline = time.time() + timeout_s
    last_error = ""
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"vLLM exited during startup with code {process.returncode}")
        try:
            response = requests.get(
                f"{base_url}/v1/models",
                headers=auth_headers(api_key),
                timeout=3,
            )
            if response.ok:
                models = response.json().get("data", [])
                if not models:
                    raise RuntimeError("/v1/models returned no served model")
                return str(models[0]["id"])
            last_error = f"HTTP {response.status_code}: {response.text[:200]}"
        except requests.RequestException as exc:
            last_error = str(exc)
        time.sleep(2)
    raise TimeoutError(f"vLLM did not become ready: {last_error}")


def request_completion(
    base_url: str,
    api_key: str,
    served_model: str,
    max_tokens: int,
) -> dict[str, Any]:
    payload = {
        "model": served_model,
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "min_tokens": max_tokens,
        "ignore_eos": True,
        "stream": False,
    }
    started = time.perf_counter()
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        headers=auth_headers(api_key),
        json=payload,
        timeout=300,
    )
    elapsed = time.perf_counter() - started
    response.raise_for_status()
    body = response.json()
    usage = body.get("usage", {})
    return {
        "elapsed_s_with_profiler_overhead": elapsed,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "finish_reason": body.get("choices", [{}])[0].get("finish_reason"),
    }


def post_control(base_url: str, endpoint: str, api_key: str) -> None:
    response = requests.post(
        f"{base_url}/{endpoint}",
        headers=auth_headers(api_key),
        timeout=120,
    )
    response.raise_for_status()


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=45)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=15)


def tail(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return ""
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def run_variant(
    *,
    name: str,
    model: Path,
    quantization: str | None,
    args: argparse.Namespace,
    run_dir: Path,
) -> dict[str, Any]:
    variant_dir = run_dir / name
    trace_dir = variant_dir / "traces"
    variant_dir.mkdir(parents=True, exist_ok=True)
    trace_dir.mkdir(parents=True, exist_ok=True)
    log_path = variant_dir / "server.log"
    profiler_config = {
        "profiler": "torch",
        "torch_profiler_dir": str(trace_dir.resolve()),
        "torch_profiler_record_shapes": True,
        "torch_profiler_with_memory": True,
        "torch_profiler_with_stack": False,
        "torch_profiler_use_gzip": True,
        "ignore_frontend": True,
    }
    command = [
        str(args.vllm_bin),
        "serve",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--api-key",
        args.api_key,
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-model-len",
        str(args.max_model_len),
        "--profiler-config",
        json.dumps(profiler_config, separators=(",", ":")),
    ]
    if quantization:
        command.extend(["--quantization", quantization])

    metadata: dict[str, Any] = {
        "variant": name,
        "model": str(model),
        "quantization": quantization,
        "started_at": utc_now(),
        "gpu_before_server": gpu_snapshot(),
        "server_command": ["<API_KEY>" if part == args.api_key else part for part in command],
        "profiler_config": profiler_config,
        "warmup_requests": args.warmup_requests,
        "profile_requests": args.profile_requests,
        "max_tokens": args.max_tokens,
        "prompt": PROMPT,
    }
    log_handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    base_url = f"http://127.0.0.1:{args.port}"
    sampler: PowerSampler | None = None
    try:
        served_model = wait_for_server(base_url, args.api_key, process, args.startup_timeout)
        metadata["served_model"] = served_model
        metadata["gpu_after_server_ready"] = gpu_snapshot()
        metadata["warmups"] = [
            request_completion(base_url, args.api_key, served_model, args.max_tokens)
            for _ in range(args.warmup_requests)
        ]
        sampler = PowerSampler(variant_dir / "power_samples.csv")
        sampler.start()
        post_control(base_url, "start_profile", args.api_key)
        metadata["profile_started_at"] = utc_now()
        metadata["profiled_requests"] = [
            request_completion(base_url, args.api_key, served_model, args.max_tokens)
            for _ in range(args.profile_requests)
        ]
        post_control(base_url, "stop_profile", args.api_key)
        metadata["profile_stopped_at"] = utc_now()
        time.sleep(5)
        metadata["gpu_after_profile"] = gpu_snapshot()
    except Exception as exc:
        metadata["error"] = repr(exc)
        raise
    finally:
        if sampler is not None:
            sampler.stop()
        stop_process(process)
        log_handle.close()
        metadata["finished_at"] = utc_now()
        metadata["server_returncode"] = process.returncode
        metadata["trace_files"] = [str(path) for path in sorted(trace_dir.rglob("*")) if path.is_file()]
        metadata["server_log_tail"] = tail(log_path)
        metadata["gpu_after_server_stop"] = gpu_snapshot()
        (variant_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bf16-model", type=Path, required=True)
    parser.add_argument("--awq-model", type=Path, required=True)
    parser.add_argument("--vllm-bin", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--api-key", default="token-abc123")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.65)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--profile-requests", type=int, default=1)
    parser.add_argument("--startup-timeout", type=int, default=600)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=("bf16", "awq_int4", "awq_marlin"),
        default=("bf16", "awq_int4", "awq_marlin"),
        help="Variants to run; existing variants in session.json are preserved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.output_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    session_path = run_dir / "session.json"
    if session_path.exists():
        session = json.loads(session_path.read_text(encoding="utf-8"))
    else:
        session = {
            "generated_at": utc_now(),
            "purpose": (
                "Matched BF16 vs generic AWQ vs AWQ-Marlin torch-profiler "
                "diagnostic; not a latency benchmark"
            ),
            "variants": [],
        }
    variants = {
        "bf16": (args.bf16_model, None),
        "awq_int4": (args.awq_model, "awq"),
        "awq_marlin": (args.awq_model, "awq_marlin"),
    }
    for name in args.variants:
        model, quantization = variants[name]
        print(f"[{name}] starting", flush=True)
        result = run_variant(
            name=name,
            model=model,
            quantization=quantization,
            args=args,
            run_dir=run_dir,
        )
        session["variants"] = [
            item for item in session["variants"] if item.get("variant") != name
        ]
        session["variants"].append(result)
        session["updated_at"] = utc_now()
        session_path.write_text(
            json.dumps(session, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[{name}] complete: {len(result.get('trace_files', []))} trace files", flush=True)
        time.sleep(5)
    print(f"saved={run_dir}")


if __name__ == "__main__":
    main()
