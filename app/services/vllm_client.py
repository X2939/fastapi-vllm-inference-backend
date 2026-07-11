import time
from collections.abc import Iterator

from openai import OpenAI

from app.core.config import Settings, get_settings
from app.schemas.chat import ChatRequest


class VllmClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.client = OpenAI(
            base_url=self.settings.vllm_base_url,
            api_key=self.settings.vllm_api_key,
        )

    def _messages(self, user_message: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.settings.system_prompt},
            {"role": "user", "content": user_message},
        ]

    def chat(self, req: ChatRequest) -> dict:
        start = time.time()
        response = self.client.chat.completions.create(
            model=self.settings.model_name,
            messages=self._messages(req.message),
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )
        elapsed = time.time() - start
        usage = response.usage

        return {
            "answer": response.choices[0].message.content,
            "elapsed": round(elapsed, 2),
            "usage": {
                "prompt_tokens": usage.prompt_tokens if usage else None,
                "completion_tokens": usage.completion_tokens if usage else None,
                "total_tokens": usage.total_tokens if usage else None,
            },
        }

    def stream_chat(self, req: ChatRequest) -> Iterator[str]:
        stream = self.client.chat.completions.create(
            model=self.settings.model_name,
            messages=self._messages(req.message),
            temperature=req.temperature,
            max_tokens=req.max_tokens,
            stream=True,
        )

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


vllm_client = VllmClient()
