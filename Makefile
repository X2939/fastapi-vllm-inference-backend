PYTHON ?= python3
VLLM_HOST ?= 0.0.0.0
VLLM_PORT ?= 8000
API_HOST ?= 0.0.0.0
API_PORT ?= 9000
MODEL_PATH ?= /home/xxx/models/Qwen2.5-1.5B-Instruct
API_KEY ?= token-abc123

.PHONY: check test compile benchmark serve-api serve-vllm bench stream-bench \
	experiment-baseline experiment-scheduler experiment-kv-cache compose-up compose-down compose-logs

check: compile test

benchmark:
	$(PYTHON) -m benchmarks.runner

test:
	$(PYTHON) -m pytest

compile:
	$(PYTHON) -m compileall -q app attention benchmarks engine experiments scripts tests visualization

serve-api:
	uvicorn app.main:app --host $(API_HOST) --port $(API_PORT)

serve-vllm:
	vllm serve $(MODEL_PATH) \
		--host $(VLLM_HOST) \
		--port $(VLLM_PORT) \
		--api-key $(API_KEY) \
		--gpu-memory-utilization 0.65 \
		--max-model-len 2048

bench:
	$(PYTHON) scripts/inference_bench.py --concurrency 1,2,4 --requests 10

stream-bench:
	$(PYTHON) scripts/stream_bench.py

experiment-baseline:
	$(PYTHON) -m experiments.runner experiments/configs/exp001_baseline.yaml

experiment-scheduler:
	$(PYTHON) -m experiments.runner experiments/configs/exp002_scheduler_sweep.yaml

experiment-kv-cache:
	$(PYTHON) -m experiments.runner experiments/configs/exp003_kv_cache_prefix.yaml

compose-up:
	docker compose up --build -d

compose-down:
	docker compose down

compose-logs:
	docker compose logs -f
