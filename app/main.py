import os
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from openai import OpenAI
from pydantic import BaseModel


VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "token-abc123")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")

client = OpenAI(base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY)

app = FastAPI(title="vLLM Demo Backend")


class ChatRequest(BaseModel):
    message: str
    temperature: float = 0.7
    max_tokens: int = 128


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    start = time.time()
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a concise assistant."},
                {"role": "user", "content": req.message},
            ],
            temperature=req.temperature,
            max_tokens=req.max_tokens,
        )

        elapsed = time.time() - start

        return {
            "answer": response.choices[0].message.content,
            "elapsed": round(elapsed, 2),
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                "completion_tokens": response.usage.completion_tokens if response.usage else None,
                "total_tokens": response.usage.total_tokens if response.usage else None,
            },
        }
    except Exception as e:
        print("CHAT ERROR:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    def generate():
        try:
            stream = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "You are a concise assistant."},
                    {"role": "user", "content": req.message},
                ],
                temperature=req.temperature,
                max_tokens=req.max_tokens,
                stream=True,
            )

            for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            yield f"[ERROR] {str(e)}"

    return StreamingResponse(generate(), media_type="text/plain;charset=utf-8")
