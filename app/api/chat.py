import logging
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.core.metrics import record_inference
from app.schemas.chat import ChatRequest, ChatResponse
from app.services.vllm_client import vllm_client


logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> dict:
    start = time.perf_counter()
    try:
        result = vllm_client.chat(req)
    except Exception as exc:
        record_inference(
            path="/chat",
            status="error",
            latency_seconds=time.perf_counter() - start,
            prompt_tokens=None,
            completion_tokens=None,
        )
        logger.exception("Chat request failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    usage = result.get("usage") or {}
    record_inference(
        path="/chat",
        status="success",
        latency_seconds=time.perf_counter() - start,
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
    )
    return result


@router.post("/chat/stream")
def chat_stream(req: ChatRequest) -> StreamingResponse:
    def generate():
        try:
            for delta in vllm_client.stream_chat(req):
                yield delta
        except Exception as exc:
            logger.exception("Streaming chat request failed")
            yield f"[ERROR] {str(exc)}"

    return StreamingResponse(generate(), media_type="text/plain;charset=utf-8")
