"""Weight-only FP8 storage with explicit ONNX DequantizeLinear nodes."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max


class _DequantizeFp8(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        values: torch.Tensor,
        scale: torch.Tensor,
        axis: int,
    ) -> torch.Tensor:
        del ctx
        shape = [1] * values.ndim
        shape[axis] = scale.numel()
        return values.to(dtype=scale.dtype) * scale.reshape(shape)

    @staticmethod
    def symbolic(graph, values, scale, axis):
        return graph.op("DequantizeLinear", values, scale, axis_i=int(axis))


def _quantize_rows(
    weight: torch.Tensor,
    *,
    chunk_elements: int = 8 * 1024 * 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize rows without materialising a full FP32 copy of a large weight."""
    source = weight.detach().to(device="cpu")
    rows = source.reshape(source.shape[0], -1)
    rows_per_chunk = max(1, chunk_elements // rows.shape[1])
    quantized_rows = torch.empty(rows.shape, dtype=FP8_DTYPE, device="cpu")
    scales = torch.empty(rows.shape[0], dtype=torch.float16, device="cpu")
    minimum_scale = torch.finfo(torch.float16).tiny
    for start in range(0, rows.shape[0], rows_per_chunk):
        stop = min(start + rows_per_chunk, rows.shape[0])
        values = rows[start:stop].to(dtype=torch.float32)
        scale = (values.abs().amax(dim=1) / FP8_MAX).clamp_min(minimum_scale)
        quantized_rows[start:stop] = (
            values / scale.unsqueeze(1)
        ).clamp(-FP8_MAX, FP8_MAX).to(dtype=FP8_DTYPE)
        scales[start:stop] = scale.to(dtype=torch.float16)
    return quantized_rows.reshape(source.shape), scales


class Fp8WeightLinear(nn.Module):
    def __init__(self, source: nn.Linear) -> None:
        super().__init__()
        weight, scale = _quantize_rows(source.weight)
        self.register_buffer("weight", weight)
        self.register_buffer("weight_scale", scale)
        if source.bias is None:
            self.register_buffer("bias", None)
        else:
            self.register_buffer("bias", source.bias.detach().to(dtype=torch.float16))
        self.in_features = source.in_features
        self.out_features = source.out_features

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = _DequantizeFp8.apply(self.weight, self.weight_scale, 0)
        bias = None if self.bias is None else self.bias.to(dtype=inputs.dtype)
        return F.linear(inputs, weight.to(dtype=inputs.dtype), bias)


class Fp8WeightEmbedding(nn.Module):
    def __init__(self, source: nn.Embedding) -> None:
        super().__init__()
        weight, scale = _quantize_rows(source.weight)
        self.register_buffer("weight", weight)
        self.register_buffer("weight_scale", scale)
        self.num_embeddings = source.num_embeddings
        self.embedding_dim = source.embedding_dim
        self.padding_idx = source.padding_idx
        self.max_norm = source.max_norm
        self.norm_type = source.norm_type
        self.scale_grad_by_freq = source.scale_grad_by_freq
        self.sparse = source.sparse

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = _DequantizeFp8.apply(self.weight, self.weight_scale, 0)
        return F.embedding(
            inputs,
            weight,
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


@dataclass(frozen=True, slots=True)
class Fp8QuantizationResult:
    linear_layers: int
    embedding_layers: int
    fp8_weight_bytes: int
    scale_bytes: int


def quantize_module_weights_to_fp8(module: nn.Module) -> Fp8QuantizationResult:
    """Replace Linear/Embedding children recursively with FP8-storage modules."""
    linear_layers = 0
    embedding_layers = 0
    fp8_weight_bytes = 0
    scale_bytes = 0

    def replace(parent: nn.Module) -> None:
        nonlocal linear_layers, embedding_layers, fp8_weight_bytes, scale_bytes
        for name, child in list(parent.named_children()):
            replacement: nn.Module | None = None
            if isinstance(child, nn.Linear):
                replacement = Fp8WeightLinear(child)
                linear_layers += 1
            elif isinstance(child, nn.Embedding):
                replacement = Fp8WeightEmbedding(child)
                embedding_layers += 1
            if replacement is None:
                replace(child)
                continue
            setattr(parent, name, replacement)
            weight = replacement.weight
            scale = replacement.weight_scale
            fp8_weight_bytes += weight.numel() * weight.element_size()
            scale_bytes += scale.numel() * scale.element_size()

    replace(module)
    return Fp8QuantizationResult(
        linear_layers=linear_layers,
        embedding_layers=embedding_layers,
        fp8_weight_bytes=fp8_weight_bytes,
        scale_bytes=scale_bytes,
    )


__all__ = [
    "FP8_DTYPE",
    "Fp8QuantizationResult",
    "Fp8WeightEmbedding",
    "Fp8WeightLinear",
    "quantize_module_weights_to_fp8",
]
