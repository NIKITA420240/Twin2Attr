"""Typed execution contract shared by Transformer inference backends."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
import torch


class TransformerExecutorOutOfMemoryError(RuntimeError):
    """A backend-independent signal used for adaptive batch-size retries."""


@runtime_checkable
class TransformerExecutor(Protocol):
    """Execute already tokenized Transformer batches."""

    @property
    def device(self) -> torch.device:
        ...

    @property
    def output_dim(self) -> int:
        ...

    @property
    def max_length(self) -> int | None:
        ...

    @property
    def use_field_tokens(self) -> bool:
        ...

    @property
    def max_attribute_value_chars(self) -> int | None:
        ...

    @property
    def max_attribute_value_tokens(self) -> int | None:
        ...

    def predict_logits(
        self,
        batch: dict[str, torch.Tensor],
        *,
        non_blocking: bool,
    ) -> np.ndarray:
        ...

    def encode_cls(
        self,
        batch: dict[str, torch.Tensor],
        *,
        non_blocking: bool,
    ) -> np.ndarray:
        ...

    def validate(self, *, classifier: bool, encoder: bool) -> None:
        ...

    def clear_cache(self) -> None:
        ...


__all__ = ["TransformerExecutor", "TransformerExecutorOutOfMemoryError"]
