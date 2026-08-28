"""PyTorch implementation of the Transformer execution contract."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from loguru import logger
from transformers import PreTrainedModel

from ...pair_encoding import DEFAULT_MAX_ATTRIBUTE_VALUE_TOKENS
from .executor import TransformerExecutorOutOfMemoryError


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


class CompiledForward:
    """Use a compiled callable while retaining a safe eager fallback."""

    def __init__(
        self,
        eager: Any,
        *,
        name: str,
        mode: str,
        dynamic: bool,
    ) -> None:
        self._eager = eager
        self._name = name
        try:
            self._compiled = torch.compile(
                eager,
                mode=mode,
                dynamic=dynamic,
            )
        except Exception as error:
            self._compiled = None
            logger.warning(
                "torch.compile setup failed for {}; using eager mode: {}",
                name,
                error,
            )

    def __call__(self, **inputs: torch.Tensor) -> Any:
        if self._compiled is not None:
            try:
                return self._compiled(**inputs)
            except torch.cuda.OutOfMemoryError:
                raise
            except Exception as error:
                self._compiled = None
                logger.warning(
                    "torch.compile execution failed for {}; using eager mode: {}",
                    self._name,
                    error,
                )
        return self._eager(**inputs)


@dataclass(slots=True)
class PyTorchTransformerExecutor:
    model: PreTrainedModel
    dtype: str = "float32"
    compile_enabled: bool = False
    compile_mode: str = "reduce-overhead"
    compile_dynamic: bool = True
    forward_model: Any | None = None
    backbone_model: Any | None = None
    _compiled_model: Any | None = field(init=False, default=None, repr=False)
    _compiled_backbone: Any | None = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        self.dtype = normalize_inference_dtype(self.dtype)
        if self.compile_mode not in {
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        }:
            raise ValueError("unsupported torch.compile mode")
        self.model.eval()
        if self.compile_enabled:
            self._compiled_model = CompiledForward(
                self.model,
                name="Transformer classifier",
                mode=self.compile_mode,
                dynamic=self.compile_dynamic,
            )
            self._compiled_backbone = CompiledForward(
                self.model.base_model,
                name="Transformer backbone",
                mode=self.compile_mode,
                dynamic=self.compile_dynamic,
            )

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    @property
    def output_dim(self) -> int:
        return int(self.model.config.hidden_size)

    @property
    def max_length(self) -> int | None:
        value = getattr(self.model.config, "match_max_length", None)
        return None if value is None else int(value)

    @property
    def use_field_tokens(self) -> bool:
        return bool(getattr(self.model.config, "match_use_field_tokens", True))

    @property
    def max_attribute_value_chars(self) -> int | None:
        return getattr(self.model.config, "match_max_attribute_value_chars", None)

    @property
    def max_attribute_value_tokens(self) -> int | None:
        return getattr(
            self.model.config,
            "match_max_attribute_value_tokens",
            DEFAULT_MAX_ATTRIBUTE_VALUE_TOKENS,
        )

    def _inputs(
        self,
        batch: dict[str, torch.Tensor],
        *,
        non_blocking: bool,
    ) -> dict[str, torch.Tensor]:
        return {
            name: value.to(self.device, non_blocking=non_blocking)
            for name, value in batch.items()
        }

    def predict_logits(
        self,
        batch: dict[str, torch.Tensor],
        *,
        non_blocking: bool,
    ) -> np.ndarray:
        try:
            inputs = self._inputs(batch, non_blocking=non_blocking)
            with torch.inference_mode(), autocast_context(self.device, self.dtype):
                forward = self.forward_model or self._compiled_model or self.model
                logits = forward(**inputs).logits
            if logits.ndim != 2 or logits.shape[1] != 2:
                raise RuntimeError(
                    "Transformer classifier must return [batch, 2] logits"
                )
            return logits.float().cpu().numpy()
        except torch.cuda.OutOfMemoryError as error:
            raise TransformerExecutorOutOfMemoryError from error

    def encode_cls(
        self,
        batch: dict[str, torch.Tensor],
        *,
        non_blocking: bool,
    ) -> np.ndarray:
        try:
            inputs = self._inputs(batch, non_blocking=non_blocking)
            with torch.inference_mode(), autocast_context(self.device, self.dtype):
                backbone = (
                    self.backbone_model
                    or self._compiled_backbone
                    or self.model.base_model
                )
                hidden_state = backbone(
                    **inputs,
                    return_dict=True,
                ).last_hidden_state
            return hidden_state[:, 0, :].float().cpu().numpy()
        except torch.cuda.OutOfMemoryError as error:
            raise TransformerExecutorOutOfMemoryError from error

    def prepare(self, *, classifier: bool, encoder: bool) -> None:
        del classifier, encoder

    def warmup(self, *, classifier: bool, encoder: bool) -> None:
        # PyTorch receives real tokenized inputs in TransformerPredictor. Keeping
        # this hook explicit avoids manufacturing tokenizer-dependent dummy data.
        del classifier, encoder

    def clear_cache(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.empty_cache()


__all__ = [
    "CompiledForward",
    "PyTorchTransformerExecutor",
    "autocast_context",
    "normalize_inference_dtype",
    "torch_inference_dtype",
]
