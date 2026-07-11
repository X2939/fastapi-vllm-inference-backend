from fastapi import FastAPI

from app.api.chat import router as chat_router
from app.api.health import router as health_router
from app.api.metrics import router as metrics_router
from app.core.logging import configure_logging
from app.core.request_logging import setup_request_logging


configure_logging()

app = FastAPI(title="vLLM Demo Backend")
setup_request_logging(app)
app.include_router(health_router)
app.include_router(chat_router)
app.include_router(metrics_router)
