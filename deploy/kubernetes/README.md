# Kubernetes GPU Deployment

This directory deploys the vLLM demo API, a single-GPU vLLM server, and
minimal Prometheus monitoring. It covers container resource management, GPU
resource delivery, service health checks, service discovery, and basic metrics.

The local K3s + WSL path has been run end-to-end on a single RTX 3060 GPU.
The exact environment and validation record are in
[`docs/kubernetes_gpu_validation.md`](../../docs/kubernetes_gpu_validation.md).
It is intentionally a single-node deployment, not a claim of multi-node
production-cluster operations.

## What It Covers

- `Namespace` and `ResourceQuota` for workload isolation.
- API `Deployment` with CPU/memory requests, limits, probes, and Prometheus
  scrape annotations.
- vLLM `Deployment` with `nvidia.com/gpu: 1`, model volume, readiness probe,
  and serving flags such as `--enable-prefix-caching`.
- `Service` objects for stable in-cluster access.
- Minimal Prometheus deployment that scrapes the API `/metrics` endpoint.

## Prerequisites

- A Kubernetes cluster with NVIDIA GPU nodes.
- NVIDIA driver, NVIDIA container runtime, and NVIDIA device plugin installed.
- A GPU node labeled with `accelerator=nvidia`, or adjust `nodeSelector`.
- The model mounted at `/models/Qwen2.5-1.5B-Instruct` on the GPU node, or
  replace the `hostPath` volume with PVC/object-storage based model delivery.
- A built API image named `vllm-api:latest`, or update `api.yaml`.

## Apply

```bash
kubectl apply -k deploy/kubernetes
```

### Local K3s + WSL GPU

The local GPU path uses K3s, NVIDIA Container Toolkit, the `nvidia`
`RuntimeClass`, and the cluster-level NVIDIA device plugin. It mounts models
already available at `/home/xxx/models` and applies smaller host-memory
requests suitable for the local node:

```bash
kubectl apply -k deploy/k3s
```

Verify GPU registration before checking vLLM:

```bash
kubectl describe node <node-name> | grep nvidia.com/gpu
kubectl apply -f deploy/kubernetes/k3s/gpu-smoke-test.yaml
kubectl logs job/nvidia-smi-smoke-test
```

For local clusters such as kind/minikube, build or load the API image first:

```bash
docker build -f Dockerfile.api -t vllm-api:latest .
```

## Check

```bash
kubectl get nodes
kubectl get pods -n vllm -o wide
kubectl describe pod -n vllm -l app.kubernetes.io/name=vllm-server
kubectl logs -n vllm deploy/vllm-server
kubectl port-forward -n vllm svc/vllm-api 9000:9000
curl http://127.0.0.1:9000/health
```

Prometheus:

```bash
kubectl port-forward -n vllm svc/prometheus 9090:9090
```

## Platform Mapping

| Platform topic | This project artifact |
|---|---|
| Container resource management | CPU/memory/GPU requests and limits |
| Compute resource delivery | `nvidia.com/gpu: 1`, model volume, vLLM server |
| Cluster stability | readiness/liveness probes, services, quota |
| Observability | `/metrics`, Prometheus scrape config |
| MLOps/MLSys bridge | repeatable serving deployment plus benchmark commands |

## Common Troubleshooting

Pod stays `Pending`:

```bash
kubectl describe pod -n vllm <pod-name>
kubectl describe node <gpu-node>
```

Likely causes: no GPU node, NVIDIA device plugin not ready, `nodeSelector`
does not match, model path missing, namespace quota exhausted.

Pod becomes `CrashLoopBackOff`:

```bash
kubectl logs -n vllm <pod-name> --previous
kubectl describe pod -n vllm <pod-name>
```

Likely causes: model path invalid, CUDA runtime mismatch, API key/config
missing, memory limit too low.

Low GPU utilization:

```bash
nvidia-smi
kubectl top pod -n vllm
```

Check request concurrency, prompt/output length, `max_num_seqs`,
`max_num_batched_tokens`, queueing, and whether the service is CPU/network bound.

High P95 latency:

- TTFT high: check queueing, long prompts, prefix-cache hit rate, and prefill
  contention.
- TPOT high: check decode batch shape, KV cache pressure, GPU memory bandwidth,
  and whether long prefills are interfering with decode.
- Error rate high: check OOM, timeout, readiness, and node pressure events.

## Scope Boundary

This repository validates a single-node GPU deployment. Production deployment
still needs multi-node scheduling, persistent model delivery, secure secret
management, autoscaling, centralized logging, and alerting.
