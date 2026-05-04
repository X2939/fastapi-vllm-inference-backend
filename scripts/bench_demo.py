import time
from statistics import mean
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

URL = "http://127.0.0.1:9000/chat"
HEADERS = {"Content-Type": "application/json"}


def one_call(i: int) -> float:
    payload = {
        "message": f"请用一句话解释什么是 attention。请求编号 {i}",
        "temperature": 0.7,
        "max_tokens": 64,
    }

    start = time.time()
    response = requests.post(URL, headers=HEADERS, json=payload, timeout=60)#发送 POST 请求
    response.raise_for_status()
    _ = response.json()
    return time.time() - start


def serial_test(n: int = 5):#默认发 5 次请求,整体串行跑，一个一个完成
    times = []
    for i in range(n):
        t = one_call(i)
        times.append(t)

    print("=== Serial Test ===")
    print("times:", [round(t, 2) for t in times])
    print("avg:", round(mean(times), 2), "s")


def concurrent_test(n: int = 5, workers: int = 2):#默认：发 5 次请求，同时最多 2 个请求一起跑
    times = []
    wall_start = time.time()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(one_call, i) for i in range(n)]
        for future in as_completed(futures):#as_completed(futures) 表示哪个请求先完成，就先拿哪个结果
            times.append(future.result())

    wall_time = time.time() - wall_start

    print("=== Concurrent Test ===")
    print("workers:", workers)
    print("times:", [round(t, 2) for t in times])
    print("avg single request:", round(mean(times), 2), "s")
    print("total wall time:", round(wall_time, 2), "s")
    print("throughput:", round(n / wall_time, 2), "req/s")


if __name__ == "__main__":
    serial_test()
    print()
    concurrent_test()
