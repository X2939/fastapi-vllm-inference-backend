# K3s GPU prerequisites

This directory contains the cluster-level NVIDIA device plugin required by the
single-node K3s deployment path. Apply it before the application manifests:

```bash
kubectl apply -f deploy/kubernetes/k3s/nvidia-device-plugin.yaml
kubectl rollout status -n kube-system daemonset/nvidia-device-plugin-daemonset
kubectl describe node <node-name> | grep nvidia.com/gpu
```

The plugin runs with `runtimeClassName: nvidia`, so its own container can load
NVML and advertise the GPU as `nvidia.com/gpu`. The application vLLM Pod must
also request this resource and use the same runtime class.
