from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest


REGISTRY = CollectorRegistry()

REQUEST_COUNT = Counter(
    "vllm_demo_http_requests_total",
    "Total HTTP requests handled by the FastAPI service.",
    ["method", "path", "status_code"],
    registry=REGISTRY,
)

REQUEST_LATENCY = Histogram(
    "vllm_demo_http_request_duration_seconds",
    "HTTP request latency in seconds.",
    ["method", "path", "status_code"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    registry=REGISTRY,
)

INFERENCE_REQUESTS = Counter(
    "vllm_demo_inference_requests_total",
    "Total inference requests completed at the application layer.",
    ["path", "status"],
    registry=REGISTRY,
)

PROMPT_TOKENS = Counter(
    "vllm_demo_prompt_tokens_total",
    "Total prompt tokens observed from vLLM usage metadata.",
    ["path"],
    registry=REGISTRY,
)

COMPLETION_TOKENS = Counter(
    "vllm_demo_completion_tokens_total",
    "Total completion tokens observed from vLLM usage metadata.",
    ["path"],
    registry=REGISTRY,
)

INFERENCE_LATENCY = Histogram(
    "vllm_demo_inference_latency_seconds",
    "End-to-end inference latency at the application layer.",
    ["path"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0),
    registry=REGISTRY,
)


def record_http_request(method: str, path: str, status_code: int, latency_seconds: float) -> None:
    labels = {
        "method": method,
        "path": path,
        "status_code": str(status_code),
    }
    REQUEST_COUNT.labels(**labels).inc()
    REQUEST_LATENCY.labels(**labels).observe(latency_seconds)


def record_inference(
    path: str,
    status: str,
    latency_seconds: float,
    prompt_tokens: int | None,
    completion_tokens: int | None,
) -> None:
    INFERENCE_REQUESTS.labels(path=path, status=status).inc()
    INFERENCE_LATENCY.labels(path=path).observe(latency_seconds)
    if prompt_tokens is not None:
        PROMPT_TOKENS.labels(path=path).inc(prompt_tokens)
    if completion_tokens is not None:
        COMPLETION_TOKENS.labels(path=path).inc(completion_tokens)


def render_metrics() -> bytes:
    return generate_latest(REGISTRY)
