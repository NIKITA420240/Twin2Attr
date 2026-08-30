"""Resolve the concrete Transformer runtime before dependency setup."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

CONCRETE_BACKENDS = frozenset({"pytorch", "onnxruntime", "tensorrt"})


def resolve_effective_backend(
    solution: Mapping[str, Any],
    pair_count: int,
) -> str:
    """Return one concrete backend for the complete inference invocation."""
    if pair_count < 0:
        raise ValueError("pair_count must not be negative")
    backend = str(solution.get("backend", "pytorch")).lower()
    if backend in CONCRETE_BACKENDS:
        return backend
    if backend != "adaptive":
        raise ValueError(
            "solution backend must be pytorch, onnxruntime, tensorrt or adaptive"
        )
    threshold = int(solution.get("adaptive_pair_threshold", 10_000))
    if threshold < 1:
        raise ValueError("adaptive_pair_threshold must be positive")
    return "onnxruntime" if pair_count <= threshold else "tensorrt"


__all__ = ["CONCRETE_BACKENDS", "resolve_effective_backend"]
