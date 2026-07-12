from __future__ import annotations

from typing import Iterator

from benchmarks.common import PromptMode
from benchmarks.gpu_report import write_plots
from benchmarks.streaming import _sse_payload, run_stream_case, stream_one_call
from scripts.gpu_benchmark import _gpu_line, _models_url
from scripts.quality_smoke import _expected_ok, _parse_json


class _Response:
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self, **_kwargs) -> Iterator[str]:
        yield from self.lines


def _lines() -> list[str]:
    return [
        'data: {"choices":[{"delta":{"role":"assistant"}}]}',
        'data: {"choices":[{"delta":{"content":"hello"}}]}',
        'data: {"choices":[{"delta":{"content":" world"}}]}',
        'data: {"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":2}}',
        "data: [DONE]",
    ]


def test_sse_payload_ignores_transport_markers() -> None:
    assert _sse_payload("event: message") is None
    assert _sse_payload("data: [DONE]") is None
    assert _sse_payload('data: {"id":"chunk"}') == {"id": "chunk"}


def test_stream_one_call_collects_ttft_and_usage(monkeypatch) -> None:
    sent: dict = {}

    def fake_post(*_args, **kwargs):
        sent.update(kwargs)
        return _Response(_lines())

    monkeypatch.setattr(
        "benchmarks.streaming.requests.post", fake_post
    )
    result = stream_one_call(
        url="http://test/v1/chat/completions", model="test-model", prompt="hello",
        api_key="secret", max_tokens=8, timeout=1, request_id=1,
    )
    assert result["ok"] is True
    assert result["chunks"] == 2
    assert result["prompt_tokens"] == 7
    assert result["completion_tokens"] == 2
    assert result["ttft"] is not None
    assert result["tpot"] is not None
    assert sent["headers"] == {"Authorization": "Bearer secret"}


def test_models_url_uses_same_api_version() -> None:
    assert _models_url("http://host:8000/v1/chat/completions") == "http://host:8000/v1/models"


def test_gpu_line_includes_used_memory() -> None:
    environment = {
        "nvidia_smi": {
            "gpus": [
                {
                    "name": "Test GPU",
                    "driver_version": "999.99",
                    "memory_total_mib": 6144,
                    "memory_used_mib": 4096,
                }
            ]
        }
    }
    assert _gpu_line(environment) == "Test GPU, driver 999.99, 4096 / 6144 MiB used"


def test_quality_smoke_accepts_json_code_fences() -> None:
    ok, parsed = _parse_json('```json\n{"answer": "TTFT"}\n```')
    assert ok is True
    assert parsed == {"answer": "TTFT"}
    assert _expected_ok("Time To First Token", ["first", "token"]) is True


def test_gpu_report_writes_three_pngs(tmp_path) -> None:
    summaries = [
        {"concurrency": 1, "tokens_per_second": 10, "p95_latency": 1, "p95_ttft": 0.1, "p95_tpot": 0.02},
        {"concurrency": 2, "tokens_per_second": 15, "p95_latency": 1.4, "p95_ttft": 0.2, "p95_tpot": 0.03},
    ]
    requests = [
        {"concurrency": 1, "ok": True, "latency": 1.0},
        {"concurrency": 2, "ok": True, "latency": 1.4},
    ]
    paths = write_plots(summaries, requests, tmp_path)
    assert len(paths) == 3
    assert all(path.is_file() and path.stat().st_size > 0 for path in paths)


def test_run_stream_case_aggregates_request_metrics(monkeypatch) -> None:
    monkeypatch.setattr(
        "benchmarks.streaming.requests.post", lambda *_args, **_kwargs: _Response(_lines())
    )
    summary, records = run_stream_case(
        url="http://test/v1/chat/completions", model="test-model", prompt_type="short",
        prompt_mode=PromptMode.UNIQUE, concurrency=2, requests_count=3,
        max_tokens=8, timeout=1,
    )
    assert len(records) == 3
    assert summary["success"] == 3
    assert summary["usage_complete"] is True
    assert summary["p95_ttft"] >= 0
    assert summary["p95_tpot"] >= 0
