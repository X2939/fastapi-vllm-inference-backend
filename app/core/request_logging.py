import logging
import time
from uuid import uuid4

from fastapi import FastAPI, Request, Response

from app.core.metrics import record_http_request


logger = logging.getLogger("app.request")


def setup_request_logging(app: FastAPI) -> None:
    @app.middleware("http")
    async def log_request(request: Request, call_next) -> Response:
        request_id = request.headers.get("X-Request-ID") or str(uuid4())
        start = time.perf_counter()
        status_code = 500

        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            latency_seconds = time.perf_counter() - start
            latency_ms = latency_seconds * 1000
            logger.info(
                "request_id=%s method=%s path=%s status_code=%s latency_ms=%.2f",
                request_id,
                request.method,
                request.url.path,
                status_code,
                latency_ms,
            )

            if "response" in locals():
                response.headers["X-Request-ID"] = request_id

            if request.url.path != "/metrics":
                record_http_request(
                    method=request.method,
                    path=request.url.path,
                    status_code=status_code,
                    latency_seconds=latency_seconds,
                )
