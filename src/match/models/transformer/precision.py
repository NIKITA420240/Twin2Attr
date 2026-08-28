"""Shared inference precision validation and PyTorch dtype conversion."""

from __future__ import annotations

from contextlib import nullcontext

import torch


def normalize_inference_dtype(dtype: str) -> str:
    normalized = dtype.lower()
    if normalized not in {"float32", "float16", "bfloat16"}:
        raise ValueError("dtype must be one of: float32, float16, bfloat16")
    return normalized


def torch_inference_dtype(dtype: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[normalize_inference_dtype(dtype)]


def autocast_context(device: torch.device, dtype: str):
    normalized = normalize_inference_dtype(dtype)
    if normalized == "float32":
        return nullcontext()
    if device.type == "cuda":
        if normalized == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("the selected CUDA device does not support bfloat16")
        return torch.autocast(
            device_type="cuda",
            dtype=torch_inference_dtype(normalized),
        )
    if device.type == "cpu" and normalized == "bfloat16":
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    raise RuntimeError(f"dtype={normalized!r} is not supported on {device.type}")


__all__ = [
    "autocast_context",
    "normalize_inference_dtype",
    "torch_inference_dtype",
]
