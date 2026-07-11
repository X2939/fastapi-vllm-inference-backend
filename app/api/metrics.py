from fastapi import APIRouter, Response

from app.core.metrics import render_metrics


router = APIRouter()


@router.get("/metrics")
def metrics() -> Response:
    return Response(render_metrics(), media_type="text/plain; version=0.0.4")
