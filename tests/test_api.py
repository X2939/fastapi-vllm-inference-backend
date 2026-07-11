from fastapi.testclient import TestClient

from app.api import chat as chat_api
from app.main import app


client = TestClient(app)


class FakeVllmClient:
    def chat(self, req):
        return {
            "answer": f"echo: {req.message}",
            "elapsed": 0.01,
            "usage": {
                "prompt_tokens": 3,
                "completion_tokens": 4,
                "total_tokens": 7,
            },
        }

    def stream_chat(self, req):
        yield "hello"
        yield " world"


def test_health_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_health_returns_request_id_header():
    response = client.get("/health", headers={"X-Request-ID": "test-request-id"})

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == "test-request-id"


def test_request_log_contains_core_fields(caplog):
    with caplog.at_level("INFO", logger="app.request"):
        response = client.get("/health", headers={"X-Request-ID": "log-test-id"})

    assert response.status_code == 200
    log_text = caplog.text
    assert "request_id=log-test-id" in log_text
    assert "method=GET" in log_text
    assert "path=/health" in log_text
    assert "status_code=200" in log_text
    assert "latency_ms=" in log_text


def test_metrics_endpoint_exposes_http_metrics():
    client.get("/health")

    response = client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    body = response.text
    assert "vllm_demo_http_requests_total" in body
    assert "vllm_demo_http_request_duration_seconds_bucket" in body
    assert "vllm_demo_inference_requests_total" in body
    assert "vllm_demo_prompt_tokens_total" in body
    assert "vllm_demo_completion_tokens_total" in body
    assert 'method="GET"' in body
    assert 'path="/health"' in body
    assert 'status_code="200"' in body


def test_chat_records_inference_metrics(monkeypatch):
    monkeypatch.setattr(chat_api, "vllm_client", FakeVllmClient())

    client.post(
        "/chat",
        json={
            "message": "metrics test",
            "temperature": 0.1,
            "max_tokens": 32,
        },
    )

    response = client.get("/metrics")
    body = response.text
    assert 'vllm_demo_inference_requests_total{path="/chat",status="success"}' in body
    assert 'vllm_demo_prompt_tokens_total{path="/chat"} 3.0' in body
    assert 'vllm_demo_completion_tokens_total{path="/chat"} 4.0' in body


def test_chat_returns_model_response(monkeypatch):
    monkeypatch.setattr(chat_api, "vllm_client", FakeVllmClient())

    response = client.post(
        "/chat",
        json={
            "message": "explain kv cache",
            "temperature": 0.1,
            "max_tokens": 32,
        },
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "echo: explain kv cache",
        "elapsed": 0.01,
        "usage": {
            "prompt_tokens": 3,
            "completion_tokens": 4,
            "total_tokens": 7,
        },
    }


def test_chat_rejects_empty_message():
    response = client.post(
        "/chat",
        json={
            "message": "",
            "temperature": 0.7,
            "max_tokens": 128,
        },
    )

    assert response.status_code == 422


def test_stream_chat_returns_text(monkeypatch):
    monkeypatch.setattr(chat_api, "vllm_client", FakeVllmClient())

    response = client.post(
        "/chat/stream",
        json={
            "message": "stream",
            "temperature": 0.1,
            "max_tokens": 32,
        },
    )

    assert response.status_code == 200
    assert response.text == "hello world"
