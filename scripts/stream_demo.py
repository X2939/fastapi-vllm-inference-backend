import os
import time
from openai import OpenAI

BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000/v1")
API_KEY = os.getenv("API_KEY", "token-abc123")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

start = time.time()
first_token_time = None

stream = client.chat.completions.create(
    model=MODEL_NAME,
    messages=[
        {"role": "user", "content": "请解释什么是 KV Cache，并说明它为什么占显存。"}
    ],
    temperature=0.7,
    max_tokens=128,
    stream=True,
)

print("=== streaming response ===")
for chunk in stream:
    delta = chunk.choices[0].delta.content
    if not delta:
        continue

    if first_token_time is None:
        first_token_time = time.time() - start

    print(delta, end="", flush=True)#delta：要打印的字；end=""：不换行；flush=True：立刻显示，不缓存

total_time = time.time() - start
print("\n")
print(f"first_token_latency={first_token_time:.2f}s" if first_token_time else "first_token_latency=N/A")
print(f"total_time={total_time:.2f}s")
