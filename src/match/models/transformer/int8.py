"""Weight-only INT8 storage with explicit ONNX DequantizeLinear nodes."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


INT8_MAX = 127.0


class _DequantizeInt8(torch.autograd.Function):
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
    """Symmetrically quantize rows without a full-size FP32 temporary."""
    source = weight.detach().to(device="cpu")
    rows = source.reshape(source.shape[0], -1)
    rows_per_chunk = max(1, chunk_elements // rows.shape[1])
    quantized_rows = torch.empty(rows.shape, dtype=torch.int8, device="cpu")
    scales = torch.empty(rows.shape[0], dtype=torch.float16, device="cpu")
    minimum_scale = torch.finfo(torch.float16).tiny
    for start in range(0, rows.shape[0], rows_per_chunk):
        stop = min(start + rows_per_chunk, rows.shape[0])
        values = rows[start:stop].to(dtype=torch.float32)
        scale = (values.abs().amax(dim=1) / INT8_MAX).clamp_min(minimum_scale)
        quantized_rows[start:stop] = (
            values / scale.unsqueeze(1)
        ).round().clamp(-INT8_MAX, INT8_MAX).to(dtype=torch.int8)
        scales[start:stop] = scale.to(dtype=torch.float16)
    return quantized_rows.reshape(source.shape), scales


class Int8WeightLinear(nn.Module):
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

    @classmethod
    def empty_like(cls, source: nn.Linear) -> Int8WeightLinear:
        """Create a state-dict-compatible shell without reading source weights."""
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        device = source.weight.device
        instance.register_buffer(
            "weight",
            torch.empty(source.weight.shape, dtype=torch.int8, device=device),
        )
        instance.register_buffer(
            "weight_scale",
            torch.empty(source.out_features, dtype=torch.float16, device=device),
        )
        if source.bias is None:
            instance.register_buffer("bias", None)
        else:
            instance.register_buffer(
                "bias",
                torch.empty(source.bias.shape, dtype=torch.float16, device=device),
            )
        instance.in_features = source.in_features
        instance.out_features = source.out_features
        return instance

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = _DequantizeInt8.apply(self.weight, self.weight_scale, 0)
        bias = None if self.bias is None else self.bias.to(dtype=inputs.dtype)
        return F.linear(inputs, weight.to(dtype=inputs.dtype), bias)


class Int8WeightEmbedding(nn.Module):
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

    @classmethod
    def empty_like(cls, source: nn.Embedding) -> Int8WeightEmbedding:
        """Create a state-dict-compatible shell without reading source weights."""
        instance = cls.__new__(cls)
        nn.Module.__init__(instance)
        device = source.weight.device
        instance.register_buffer(
            "weight",
            torch.empty(source.weight.shape, dtype=torch.int8, device=device),
        )
        instance.register_buffer(
            "weight_scale",
            torch.empty(source.num_embeddings, dtype=torch.float16, device=device),
        )
        instance.num_embeddings = source.num_embeddings
        instance.embedding_dim = source.embedding_dim
        instance.padding_idx = source.padding_idx
        instance.max_norm = source.max_norm
        instance.norm_type = source.norm_type
        instance.scale_grad_by_freq = source.scale_grad_by_freq
        instance.sparse = source.sparse
        return instance

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        weight = _DequantizeInt8.apply(self.weight, self.weight_scale, 0)
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
class Int8QuantizationResult:
    linear_layers: int
    embedding_layers: int
    int8_weight_bytes: int
    scale_bytes: int


def quantize_module_weights_to_int8(module: nn.Module) -> Int8QuantizationResult:
    """Replace Linear/Embedding children recursively with INT8-storage modules."""
    linear_layers = 0
    embedding_layers = 0
    int8_weight_bytes = 0
    scale_bytes = 0

    def replace(parent: nn.Module) -> None:
        nonlocal linear_layers, embedding_layers, int8_weight_bytes, scale_bytes
        for name, child in list(parent.named_children()):
            replacement: nn.Module | None = None
            if isinstance(child, nn.Linear):
                replacement = Int8WeightLinear(child)
                linear_layers += 1
            elif isinstance(child, nn.Embedding):
                replacement = Int8WeightEmbedding(child)
                embedding_layers += 1
            if replacement is None:
                replace(child)
                continue
            setattr(parent, name, replacement)
            weight = replacement.weight
            scale = replacement.weight_scale
            int8_weight_bytes += weight.numel() * weight.element_size()
            scale_bytes += scale.numel() * scale.element_size()

    replace(module)
    return Int8QuantizationResult(
        linear_layers=linear_layers,
        embedding_layers=embedding_layers,
        int8_weight_bytes=int8_weight_bytes,
        scale_bytes=scale_bytes,
    )


def replace_module_weights_with_int8_shells(module: nn.Module) -> Int8QuantizationResult:
    """Replace meta-device Linear/Embedding layers with empty INT8 shells."""
    linear_layers = 0
    embedding_layers = 0
    int8_weight_bytes = 0
    scale_bytes = 0

    def replace(parent: nn.Module) -> None:
        nonlocal linear_layers, embedding_layers, int8_weight_bytes, scale_bytes
        for name, child in list(parent.named_children()):
            replacement: nn.Module | None = None
            if isinstance(child, nn.Linear):
                replacement = Int8WeightLinear.empty_like(child)
                linear_layers += 1
            elif isinstance(child, nn.Embedding):
                replacement = Int8WeightEmbedding.empty_like(child)
                embedding_layers += 1
            if replacement is None:
                replace(child)
                continue
            setattr(parent, name, replacement)
            weight = replacement.weight
            scale = replacement.weight_scale
            int8_weight_bytes += weight.numel() * weight.element_size()
            scale_bytes += scale.numel() * scale.element_size()

    replace(module)
    return Int8QuantizationResult(
        linear_layers=linear_layers,
        embedding_layers=embedding_layers,
        int8_weight_bytes=int8_weight_bytes,
        scale_bytes=scale_bytes,
    )


__all__ = [
    "INT8_MAX",
    "Int8QuantizationResult",
    "Int8WeightEmbedding",
    "Int8WeightLinear",
    "quantize_module_weights_to_int8",
    "replace_module_weights_with_int8_shells",
]
