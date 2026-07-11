import os
import time
from openai import OpenAI

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000/v1")
API_KEY = os.getenv("API_KEY", "token-abc123")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

start = time.time()
response = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "system", "content": "You are a concise assistant."},
        {"role": "user", "content": "请用三句话解释什么是 attention。"}
    ],
    temperature=0.7,
    max_tokens=128,
)

elapsed = time.time() - start

print("=== response ===")
print(response.choices[0].message.content)
print()
print(f"elapsed={elapsed:.2f}s")
print("usage=", response.usage)
