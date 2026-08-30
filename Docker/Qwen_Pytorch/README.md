# Qwen PyTorch runtime

PyTorch-only GPU runtime for Twin2Attr and Qwen3-Reranker. Model weights and
the Twin2Attr source tree are supplied separately; ONNX Runtime and TensorRT
are intentionally absent.

Build from the repository root. Keeping `Docker/Qwen_Pytorch` as the build
context prevents local datasets and model weights from being uploaded to the
Docker daemon:

```bash
docker build \
  --platform linux/amd64 \
  --tag twin2attr-qwen-pytorch:torch2.13-cu126 \
  Docker/Qwen_Pytorch
```

Verify the GPU runtime on a Linux NVIDIA host:

```bash
docker run --rm --gpus all \
  twin2attr-qwen-pytorch:torch2.13-cu126 \
  python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

The corresponding packaged solution must select the PyTorch backend:

```json
{
  "predictor": "transformer",
  "backend": "pytorch",
  "dtype": "bf16"
}
```
