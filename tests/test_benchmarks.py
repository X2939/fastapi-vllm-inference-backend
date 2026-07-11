import csv
from pathlib import Path

import pytest
import yaml

from benchmarks.common import PromptMode, build_prompt, percentile, run_case
from benchmarks.report import write_csv, write_experiment_report
from experiments.runner import load_config, run_experiment


def test_percentile_interpolates_values():
    values = [1.0, 2.0, 3.0, 4.0]

    assert percentile(values, 0.50) == 2.5
    assert percentile(values, 0.95) == pytest.approx(3.85)


def test_build_prompt_shared_prefix_reuses_body():
    unique = build_prompt("medium", 1, PromptMode.UNIQUE)
    shared = build_prompt("medium", 1, PromptMode.SHARED_PREFIX)

    assert "请求编号" in unique
    assert "共享前缀" in shared
    assert "变体编号" in shared


def test_run_case_tracks_warmup_and_extended_metrics(monkeypatch):
    calls = []

    def fake_one_call(url, prompt, max_tokens, timeout, request_id):
        calls.append(request_id)
        if request_id < 0:
            return {
                "ok": True,
                "latency": 99.0,
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
                "error": "",
            }
        if request_id == 2:
            return {
                "ok": False,
                "latency": 0.3,
                "prompt_tokens": None,
                "completion_tokens": None,
                "total_tokens": None,
                "error": "boom",
            }
        return {
            "ok": True,
            "latency": 0.2 + request_id * 0.1,
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
            "error": "",
        }

    monkeypatch.setattr("benchmarks.common.one_call", fake_one_call)

    row = run_case(
        url="http://test",
        prompt_type="medium",
        concurrency=2,
        requests_count=4,
        max_tokens=16,
        timeout=1,
        warmup=2,
        prompt_mode=PromptMode.UNIQUE,
    )

    assert calls.count(-1) == 1
    assert calls.count(-2) == 1
    assert row["warmup"] == 2
    assert row["success"] == 3
    assert row["failed"] == 1
    assert row["error_rate"] == 0.25
    assert row["total_completion_tokens"] == 15
    assert row["prompt_mode"] == "unique"
    assert row["p99_latency"] >= row["p95_latency"]
    assert row["first_error"] == "boom"


def test_write_csv_includes_prompt_mode(tmp_path: Path):
    path = tmp_path / "bench.csv"
    row = {
        "concurrency": 1,
        "requests": 1,
        "warmup": 1,
        "prompt_type": "medium",
        "prompt_mode": "shared_prefix",
        "max_tokens": 128,
        "success": 1,
        "failed": 0,
        "error_rate": 0.0,
        "wall_time": 0.5,
        "avg_latency": 0.5,
        "min_latency": 0.5,
        "max_latency": 0.5,
        "p50_latency": 0.5,
        "p95_latency": 0.5,
        "p99_latency": 0.5,
        "throughput": 2.0,
        "total_prompt_tokens": 20,
        "avg_prompt_tokens": 20,
        "total_completion_tokens": 10,
        "avg_completion_tokens": 10,
        "tokens_per_second": 20.0,
        "tpot": 0.05,
        "first_error": "",
    }

    write_csv(path, [row])

    with path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))

    assert rows[0]["prompt_mode"] == "shared_prefix"
    assert rows[0]["total_prompt_tokens"] == "20"


def test_write_experiment_report_contains_hypothesis(tmp_path: Path):
    report_path = tmp_path / "report.md"
    meta = {
        "name": "exp_test",
        "topic": "scheduler",
        "hypothesis": "batching improves throughput",
        "url": "http://test/chat",
        "prompt_type": "medium",
        "prompt_mode": "unique",
        "concurrency_levels": [1, 2],
        "requests": 5,
        "warmup": 1,
        "max_tokens": 64,
        "csv_path": "results.csv",
    }
    rows = [
        {
            "concurrency": 1,
            "p50_latency": 1.0,
            "p95_latency": 1.1,
            "p99_latency": 1.2,
            "throughput": 0.9,
            "tokens_per_second": 100.0,
            "tpot": 0.01,
            "error_rate": 0.0,
        }
    ]

    write_experiment_report(report_path, meta, rows)
    content = report_path.read_text(encoding="utf-8")

    assert "exp_test" in content
    assert "batching improves throughput" in content
    assert "scheduler" in content


def test_load_config_reads_yaml(tmp_path: Path):
    config_path = tmp_path / "exp.yaml"
    config_path.write_text(
        "name: demo\n"
        "topic: baseline\n"
        "concurrency: [1, 2]\n"
        "requests: 3\n",
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert config["name"] == "demo"
    assert config["concurrency"] == [1, 2]


def test_run_experiment_writes_artifacts(tmp_path: Path, monkeypatch):
    captured = []

    def fake_run_case(**kwargs):
        captured.append(kwargs)
        return {
            "concurrency": kwargs["concurrency"],
            "requests": kwargs["requests_count"],
            "warmup": kwargs["warmup"],
            "prompt_type": kwargs["prompt_type"],
            "prompt_mode": kwargs["prompt_mode"].value,
            "max_tokens": kwargs["max_tokens"],
            "success": kwargs["requests_count"],
            "failed": 0,
            "error_rate": 0.0,
            "wall_time": 1.0,
            "avg_latency": 0.5,
            "min_latency": 0.4,
            "max_latency": 0.6,
            "p50_latency": 0.5,
            "p95_latency": 0.55,
            "p99_latency": 0.58,
            "throughput": 1.0,
            "total_prompt_tokens": 10,
            "avg_prompt_tokens": 10,
            "total_completion_tokens": 20,
            "avg_completion_tokens": 20,
            "tokens_per_second": 20.0,
            "tpot": 0.025,
            "first_error": "",
        }

    monkeypatch.setattr("experiments.runner.run_case", fake_run_case)

    config = {
        "name": "exp_mock",
        "topic": "scheduler",
        "hypothesis": "test",
        "url": "http://mock/chat",
        "prompt_type": "short",
        "prompt_mode": "unique",
        "concurrency": [1, 2],
        "requests": 2,
        "warmup": 1,
        "max_tokens": 32,
        "output_dir": str(tmp_path / "exp_mock"),
    }

    run_experiment(config)

    assert len(captured) == 2
    assert (tmp_path / "exp_mock" / "results.csv").exists()
    assert (tmp_path / "exp_mock" / "report.md").exists()
    assert (tmp_path / "exp_mock" / "meta.json").exists()
