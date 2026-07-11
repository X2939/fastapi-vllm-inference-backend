import os
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Settings:
    vllm_base_url: str
    vllm_api_key: str
    model_name: str
    system_prompt: str


@lru_cache
def get_settings() -> Settings:
    return Settings(
        vllm_base_url=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        vllm_api_key=os.getenv("VLLM_API_KEY", "token-abc123"),
        model_name=os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct"),
        system_prompt=os.getenv("SYSTEM_PROMPT", "You are a concise assistant."),
    )
