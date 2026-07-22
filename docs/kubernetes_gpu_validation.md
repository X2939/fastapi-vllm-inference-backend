# K3s GPU Deployment: Validation Record

This record documents an actual single-node Kubernetes GPU run of this
repository. It distinguishes verified behavior from deployment configuration.

## Environment

| Item | Value |
| --- | --- |
| Date | 2026-07-20 |
| Host | WSL 2 / Ubuntu |
| Kubernetes distribution | K3s v1.36.2+k3s1, one node |
| GPU | NVIDIA GeForce RTX 3060 Laptop GPU, 6 GiB |
| NVIDIA driver | 551.61 (CUDA compatibility shown by `nvidia-smi`: 12.4) |
| Container runtime | K3s containerd with NVIDIA Container Toolkit runtime |
| Device plugin | `nvcr.io/nvidia/k8s-device-plugin:v0.18.2` |
| vLLM image | `vllm/vllm-openai:v0.19.0` |
| Model mount | `/home/xxx/models` on the node → `/models` in the vLLM Pod |
| Model | `Qwen2.5-1.5B-Instruct` |

The version pin is deliberate. `vllm/vllm-openai:latest` resolved to vLLM
0.25.1 and failed on this driver because its CUDA/PyTorch requirement exceeded
the driver compatibility level. Pinning v0.19.0 started successfully.

## Applied Components

```bash
kubectl apply -k deploy/k3s
```

The overlay applies the namespace, quota, API Deployment/Service, vLLM
Deployment/Service, Prometheus, NVIDIA runtime class usage, NVIDIA device
plugin, the local model host path, and local memory limits.

## Verified Results

1. The device plugin registered `nvidia.com/gpu` with kubelet; the node exposed
   an allocatable GPU resource.
2. A Job requesting `nvidia.com/gpu: 1` ran `nvidia-smi` successfully in a Pod.
   The output reported the RTX 3060 and CUDA 12.4 driver compatibility.
3. The vLLM Pod reached `1/1 Running` and its `/v1/models` readiness probe
   returned HTTP 200.
4. vLLM logged a GPU KV cache capacity of 15,408 tokens and initialized CUDA
   graphs. This is GPU engine initialization, not a simulated engine path.
5. From the API Pod, a request to
   `http://vllm-server.vllm.svc.cluster.local:8000/v1/models` with the
   configured bearer token returned HTTP 200 and the mounted Qwen model.
6. A POST to the API Pod's `/chat` endpoint completed an inference request
   through the in-cluster vLLM Service. One observed response reported 45 total
   tokens and `elapsed: 4.1` seconds.

## Reproduce the Checks

```bash
kubectl get nodes
kubectl describe node <node-name>

kubectl apply -f deploy/kubernetes/k3s/gpu-smoke-test.yaml
kubectl logs job/nvidia-smi-smoke-test

kubectl get pods -n vllm -o wide
kubectl logs -n vllm deploy/vllm-server
kubectl exec -n vllm deploy/vllm-api -- \
  python -c "import urllib.request; r=urllib.request.Request('http://vllm-server.vllm.svc.cluster.local:8000/v1/models', headers={'Authorization':'Bearer token-abc123'}); print(urllib.request.urlopen(r).status)"
```

The final command uses the local example key. Replace it with a real secret for
any non-local deployment. An unauthenticated request returns HTTP 401, which
confirms that the vLLM API key is being enforced.

## Scope and Limitations

- One local node and one GPU; no multi-node scheduling, autoscaling, or
  cross-node communication measurement.
- `hostPath` model delivery is convenient locally but is not a portable
  production model-distribution mechanism.
- The local NVIDIA driver constrains container CUDA/PyTorch versions; retain a
  tested image tag rather than using `latest`.
- The WSL log reports `pin_memory=False`; these measurements should not be
  presented as a production Linux GPU performance baseline.
