"""Run a real GPU streaming benchmark against a vLLM OpenAI-compatible API."""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import requests

# Permit ``python scripts/gpu_benchmark.py`` from a fresh clone.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.common import PromptMode
from benchmarks.gpu_report import write_plots
from benchmarks.streaming import run_stream_case


SUMMARY_FIELDS = [
    "run", "concurrency", "requests", "warmup", "prompt_type", "prompt_mode", "max_tokens",
    "success", "failed", "error_rate", "wall_time", "throughput", "tokens_per_second",
    "avg_prompt_tokens", "avg_completion_tokens", "p50_latency", "p95_latency",
    "p50_ttft", "p95_ttft", "p50_tpot", "p95_tpot", "usage_complete", "first_error",
]


def _concurrencies(value: str) -> list[int]:
    result = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not result or any(item < 1 for item in result):
        raise argparse.ArgumentTypeError("concurrency must contain positive integers")
    return result


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _package_version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _command_snapshot(command: list[str]) -> dict[str, object]:
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=15, check=False
        )
        return {
            "command": command,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"command": command, "error": str(exc)}


def _gpu_snapshot() -> dict[str, object]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory",
        "--format=csv,noheader,nounits",
    ]
    snapshot = _command_snapshot(command)
    gpus: list[dict[str, object]] = []
    if snapshot.get("returncode") == 0 and isinstance(snapshot.get("stdout"), str):
        for line in snapshot["stdout"].splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 7:
                continue
            name, driver, total, used, free, gpu_util, mem_util = parts
            gpus.append(
                {
                    "name": name,
                    "driver_version": driver,
                    "memory_total_mib": int(total),
                    "memory_used_mib": int(used),
                    "memory_free_mib": int(free),
                    "gpu_utilization_percent": int(gpu_util),
                    "memory_utilization_percent": int(mem_util),
                }
            )
    snapshot["gpus"] = gpus
    return snapshot


def _environment_snapshot() -> dict[str, object]:
    snapshot: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            "vllm": _package_version("vllm"),
            "torch": _package_version("torch"),
            "requests": _package_version("requests"),
        },
        "nvidia_smi": _gpu_snapshot(),
    }
    try:
        import torch

        snapshot["torch_cuda"] = {
            "available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except (ImportError, RuntimeError) as exc:
        snapshot["torch_cuda"] = {"error": str(exc)}
    return snapshot


def _models_url(chat_completions_url: str) -> str:
    suffix = "/chat/completions"
    if not chat_completions_url.rstrip("/").endswith(suffix):
        raise ValueError("url must end with /v1/chat/completions")
    return chat_completions_url.rstrip("/").removesuffix(suffix) + "/models"


def _check_server(url: str, api_key: str | None, timeout: int) -> list[str]:
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
    response = requests.get(_models_url(url), headers=headers, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    models = payload.get("data", []) if isinstance(payload, dict) else []
    model_ids = [item.get("id") for item in models if isinstance(item, dict)]
    return [model_id for model_id in model_ids if isinstance(model_id, str) and model_id]


def _resolve_model_id(requested_model: str, served_models: list[str]) -> tuple[str, str]:
    """Resolve a host-path / container-path difference without hiding it.

    A vLLM server may expose a different model id than the path used to launch
    the benchmark. For example, a host model path can be mounted as `/models`
    inside a Kubernetes Pod. When exactly one model is served, use that id and
    record the rewrite. Multiple served models remain an explicit error.
    """
    if requested_model in served_models:
        return requested_model, "requested_model_matches_server"
    if len(served_models) == 1:
        return served_models[0], "rewritten_to_single_server_model"
    available = ", ".join(served_models) or "none"
    raise ValueError(
        f"requested model {requested_model!r} is not served; available model ids: {available}"
    )


def _mean(rows: list[dict], field: str) -> float:
    return sum(float(row[field]) for row in rows) / len(rows) if rows else 0.0


def _gpu_line(environment: dict[str, object]) -> str:
    nvidia_smi = environment.get("nvidia_smi", {})
    if not isinstance(nvidia_smi, dict):
        return "unavailable"
    gpus = nvidia_smi.get("gpus", [])
    if isinstance(gpus, list) and gpus:
        gpu = gpus[0]
        return (
            f"{gpu['name']}, driver {gpu['driver_version']}, "
            f"{gpu['memory_used_mib']} / {gpu['memory_total_mib']} MiB used"
        )
    return str(nvidia_smi.get("stdout", "unavailable"))


def _write_report(path: Path, metadata_dict: dict, rows: list[dict]) -> None:
    environment = metadata_dict.get("environment_after") or metadata_dict["environment"]
    grouped = {
        concurrency: [row for row in rows if row["concurrency"] == concurrency]
        for concurrency in sorted({row["concurrency"] for row in rows})
    }
    lines = [
        "# Real GPU Benchmark Report",
        "",
        "> This report is generated from a real OpenAI-compatible streaming endpoint, not the CPU simulation benchmark.",
        "",
        "## Workload",
        "",
        f"- Model: `{metadata_dict['model']}`",
        f"- Prompt: `{metadata_dict['prompt_type']}` / `{metadata_dict['prompt_mode']}`",
        f"- Requests per concurrency: `{metadata_dict['requests_per_level']}`",
        f"- Warmup per run: `{metadata_dict['warmup']}`",
        f"- Repetitions: `{metadata_dict['runs']}`",
        f"- Max completion tokens: `{metadata_dict['max_tokens']}`",
        "",
        "## Environment",
        "",
        f"- vLLM: `{environment['packages']['vllm']}`",
        f"- PyTorch: `{environment['packages']['torch']}`",
        f"- GPU: `{_gpu_line(environment)}`",
        f"- CUDA: `{environment['torch_cuda'].get('cuda_version')}`",
        "",
        "## Aggregate Results",
        "",
        "Each value is the arithmetic mean of the corresponding per-run result; P95 is computed within each run before averaging.",
        "",
        "| Concurrency | Success | Tokens/s | P95 TTFT (s) | P95 TPOT (s/token) | P95 E2E (s) | Error rate |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for concurrency, group in grouped.items():
        lines.append(
            "| {concurrency} | {success:.1f} | {tokens:.2f} | {ttft:.3f} | {tpot:.4f} | {latency:.3f} | {error:.2%} |".format(
                concurrency=concurrency,
                success=_mean(group, "success"),
                tokens=_mean(group, "tokens_per_second"),
                ttft=_mean(group, "p95_ttft"),
                tpot=_mean(group, "p95_tpot"),
                latency=_mean(group, "p95_latency"),
                error=_mean(group, "error_rate"),
            )
        )
    lines.extend(
        [
            "",
            "## Metric Definition",
            "",
            "- TTFT: request start to first non-empty generated content chunk.",
            "- TPOT: `(E2E - TTFT) / (completion_tokens - 1)` for requests with at least two output tokens.",
            "- E2E: request start to streaming completion.",
            "- Tokens/s: all successful completion tokens divided by the workload wall time.",
            "",
            "Raw per-run summaries are in `summary.csv`; request-level samples are in `requests.csv`.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure real vLLM streaming TTFT, TPOT, E2E P95 and throughput."
    )
    parser.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    parser.add_argument("--model", default=os.getenv("MODEL_NAME"))
    parser.add_argument("--prompt-type", choices=["short", "medium", "long", "mixed"], default="medium")
    parser.add_argument("--prompt-mode", choices=[mode.value for mode in PromptMode], default="unique")
    parser.add_argument("--concurrency", type=_concurrencies, default=[1, 2, 4])
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--output-dir", default="reports/gpu_benchmark")
    parser.add_argument("--experiment-variant", default="unspecified")
    parser.add_argument(
        "--server-args", default="", help="Server flags recorded in metadata; never executed."
    )
    parser.add_argument(
        "--api-key", default=os.getenv("VLLM_API_KEY", os.getenv("API_KEY", ""))
    )
    args = parser.parse_args()

    if not args.model:
        parser.error("--model is required (or set MODEL_NAME)")
    if args.requests < 1 or args.warmup < 0 or args.max_tokens < 1 or args.runs < 1:
        parser.error("requests and max-tokens must be positive; warmup cannot be negative")

    try:
        served_models = _check_server(args.url, args.api_key or None, min(args.timeout, 15))
        resolved_model, model_resolution = _resolve_model_id(args.model, served_models)
    except (requests.RequestException, ValueError) as exc:
        parser.error(f"vLLM endpoint preflight failed: {exc}")
    if resolved_model != args.model:
        print(
            "model_id_rewritten requested={requested!r} served={served!r} "
            "reason={reason}".format(
                requested=args.model, served=resolved_model, reason=model_resolution
            ),
            flush=True,
        )

    environment_before = _environment_snapshot()
    summaries: list[dict] = []
    records: list[dict] = []
    mode = PromptMode(args.prompt_mode)
    for run in range(1, args.runs + 1):
        for concurrency in args.concurrency:
            summary, case_records = run_stream_case(
                url=args.url, model=resolved_model, api_key=args.api_key or None,
                prompt_type=args.prompt_type, prompt_mode=mode, concurrency=concurrency,
                requests_count=args.requests, max_tokens=args.max_tokens,
                timeout=args.timeout, warmup=args.warmup,
            )
            summaries.append({"run": run, **summary})
            records.extend(
                {"run": run, "concurrency": concurrency, **record}
                for record in case_records
            )
            print(
                "run={run} concurrency={concurrency} success={success}/{requests} "
                "throughput={throughput:.2f} req/s tokens/s={tokens_per_second:.2f} "
                "p95_ttft={p95_ttft:.3f}s p95_tpot={p95_tpot:.4f}s "
                "p95_e2e={p95_latency:.3f}s".format(run=run, **summary),
                flush=True,
            )

    output_dir = Path(args.output_dir)
    _write_csv(output_dir / "summary.csv", summaries, SUMMARY_FIELDS)
    raw_fields = [
        "run", "concurrency", "request_id", "ok", "latency", "ttft", "tpot",
        "prompt_tokens", "completion_tokens", "chunks", "error",
    ]
    _write_csv(output_dir / "requests.csv", records, raw_fields)
    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "backend": "real OpenAI-compatible streaming endpoint",
        "experiment_variant": args.experiment_variant,
        "server_args": args.server_args,
        "url": args.url,
        "model": resolved_model,
        "requested_model": args.model,
        "served_models": served_models,
        "model_resolution": model_resolution,
        "prompt_type": args.prompt_type,
        "prompt_mode": args.prompt_mode,
        "concurrency": args.concurrency,
        "requests_per_level": args.requests,
        "warmup": args.warmup,
        "runs": args.runs,
        "max_tokens": args.max_tokens,
        "notes": "TPOT uses (E2E - TTFT) / (completion_tokens - 1).",
        "environment": environment_before,
        "environment_before": environment_before,
        "environment_after": _environment_snapshot(),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_report(output_dir / "REPORT.md", metadata, summaries)
    plot_paths = write_plots(summaries, records, output_dir)
    print("saved_plots=" + ", ".join(str(path) for path in plot_paths))
    print(f"saved_results={output_dir}")


if __name__ == "__main__":
    main()
