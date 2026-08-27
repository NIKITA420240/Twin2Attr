"""Backend-independent Transformer pair prediction."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from ...pair_encoding import infer_pair_max_length
from ...prepare_data import PreparedPair
from ..contracts import PredictionBatch
from .batching import (
    TransformerBatchingSettings,
    inference_dataset,
    inference_loader,
    length_bucket_order,
    restore_original_order,
)
from .executor import TransformerExecutor, TransformerExecutorOutOfMemoryError
from .head import PoolingSequenceClassifier, PoolingSequenceClassifierConfig
from .pytorch_executor import (
    CompiledForward,
    PyTorchTransformerExecutor,
    normalize_inference_dtype,
    torch_inference_dtype,
)

# Backward-compatible private aliases used by older internal callers/tests.
_CompiledForward = CompiledForward
_length_bucket_order = length_bucket_order
_restore_original_order = restore_original_order


def _resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_trained_classifier(
    model_dir: str | Path,
    *,
    device: str | torch.device | None = None,
    dtype: str = "float32",
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    target_device = _resolve_device(device)
    target_dtype = torch_inference_dtype(dtype)
    if (
        target_device.type == "cuda"
        and target_dtype == torch.bfloat16
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("the selected CUDA device does not support bfloat16")
    if target_device.type == "mps" and target_dtype == torch.bfloat16:
        raise RuntimeError("bfloat16 inference is not supported on MPS")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    config_path = Path(model_dir) / "config.json"
    config_values = json.loads(config_path.read_text(encoding="utf-8"))
    if config_values.get("model_type") == PoolingSequenceClassifierConfig.model_type:
        model = PoolingSequenceClassifier.from_pretrained(model_dir)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device=target_device, dtype=target_dtype).eval()
    return tokenizer, model


@dataclass(slots=True)
class TransformerPredictor:
    tokenizer: PreTrainedTokenizerBase
    executor: TransformerExecutor
    batch_size: int = 64
    num_workers: int = 0
    prefetch_factor: int = 2
    pin_memory: bool = True
    non_blocking_transfer: bool = True
    length_bucketing: bool = False
    padding_length_buckets: tuple[int, ...] | None = None
    batch_fields: bool = False
    field_chunk_size: int = 16_384
    _batching: TransformerBatchingSettings = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.executor, TransformerExecutor):
            raise TypeError("executor must implement TransformerExecutor")
        self._batching = TransformerBatchingSettings(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
            non_blocking_transfer=self.non_blocking_transfer,
            length_bucketing=self.length_bucketing,
            padding_length_buckets=self.padding_length_buckets,
            batch_fields=self.batch_fields,
            field_chunk_size=self.field_chunk_size,
        )

    @classmethod
    def load(
        cls,
        model_directory: str | Path,
        *,
        batch_size: int = 64,
        dtype: str = "float32",
        num_workers: int = 0,
        prefetch_factor: int = 2,
        pin_memory: bool = True,
        non_blocking_transfer: bool = True,
        length_bucketing: bool = False,
        padding_length_buckets: tuple[int, ...] | None = None,
        batch_fields: bool = False,
        field_chunk_size: int = 16_384,
        compile_enabled: bool = False,
        compile_mode: str = "reduce-overhead",
        compile_dynamic: bool = True,
        device: str | None = None,
    ) -> TransformerPredictor:
        tokenizer, model = load_trained_classifier(
            model_directory,
            device=device,
            dtype=dtype,
        )
        executor = PyTorchTransformerExecutor(
            model,
            dtype=dtype,
            compile_enabled=compile_enabled,
            compile_mode=compile_mode,
            compile_dynamic=compile_dynamic,
        )
        return cls(
            tokenizer=tokenizer,
            executor=executor,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            non_blocking_transfer=non_blocking_transfer,
            length_bucketing=length_bucketing,
            padding_length_buckets=padding_length_buckets,
            batch_fields=batch_fields,
            field_chunk_size=field_chunk_size,
        )

    @property
    def output_dim(self) -> int:
        return self.executor.output_dim

    def _resolved_max_length(
        self,
        pairs: Sequence[PreparedPair],
        max_length: int | None,
    ) -> int:
        resolved = max_length if max_length is not None else self.executor.max_length
        if resolved is not None:
            return int(resolved)
        return infer_pair_max_length(
            self.tokenizer,
            pairs,
            use_field_tokens=self.executor.use_field_tokens,
            max_attribute_value_chars=self.executor.max_attribute_value_chars,
            max_attribute_value_tokens=self.executor.max_attribute_value_tokens,
        )

    def _execute_pairs(
        self,
        pairs: Sequence[PreparedPair],
        *,
        operation: Callable[..., np.ndarray],
        output_width: int,
        max_length: int | None = None,
    ) -> np.ndarray:
        if not pairs:
            return np.empty((0, output_width), dtype=np.float32)
        collator = self._batching.collator(
            self.tokenizer,
            max_length=self._resolved_max_length(pairs, max_length),
            use_field_tokens=self.executor.use_field_tokens,
            max_attribute_value_chars=self.executor.max_attribute_value_chars,
            max_attribute_value_tokens=self.executor.max_attribute_value_tokens,
        )
        dataset, order = inference_dataset(
            pairs,
            length_bucketing=self._batching.length_bucketing,
            max_attribute_value_chars=self.executor.max_attribute_value_chars,
            max_attribute_value_tokens=self.executor.max_attribute_value_tokens,
        )
        current_batch_size = self._batching.batch_size
        while True:
            try:
                loader, use_pinned_memory = inference_loader(
                    dataset,
                    collator,
                    batch_size=current_batch_size,
                    device=self.executor.device,
                    num_workers=self._batching.num_workers,
                    prefetch_factor=self._batching.prefetch_factor,
                    pin_memory=self._batching.pin_memory,
                )
                chunks = [
                    operation(
                        batch,
                        non_blocking=(
                            self._batching.non_blocking_transfer
                            and use_pinned_memory
                        ),
                    )
                    for batch in loader
                ]
                values = np.concatenate(chunks).astype(np.float32, copy=False)
                if values.ndim != 2 or values.shape[1] != output_width:
                    raise RuntimeError(
                        f"Transformer executor returned shape {values.shape}, "
                        f"expected [batch, {output_width}]"
                    )
                return restore_original_order(values, order)
            except TransformerExecutorOutOfMemoryError:
                if current_batch_size == 1:
                    raise
                current_batch_size = max(1, current_batch_size // 2)
                self.executor.clear_cache()
                logger.warning(
                    "Transformer executor ran out of memory; retrying with "
                    "batch_size={}",
                    current_batch_size,
                )

    def predict_pair_logits(
        self,
        pairs: Sequence[PreparedPair],
        *,
        max_length: int | None = None,
    ) -> np.ndarray:
        return self._execute_pairs(
            pairs,
            operation=self.executor.predict_logits,
            output_width=2,
            max_length=max_length,
        )

    def encode_pairs(
        self,
        pairs: Sequence[PreparedPair],
        *,
        max_length: int | None = None,
    ) -> np.ndarray:
        return self._execute_pairs(
            pairs,
            operation=self.executor.encode_cls,
            output_width=self.output_dim,
            max_length=max_length,
        )

    def encode(self, batch: PredictionBatch) -> np.ndarray:
        return self.encode_pairs(batch.prepared_pairs())

    def predict_proba(self, batch: PredictionBatch) -> np.ndarray:
        logits = self.predict_pair_logits(batch.prepared_pairs())
        if not len(logits):
            return np.empty(0, dtype=np.float32)
        shifted = logits - logits.max(axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        return (exponentials[:, 1] / exponentials.sum(axis=1)).astype(
            np.float32,
            copy=False,
        )

    def predict_logit_margin(self, batch: PredictionBatch) -> np.ndarray:
        logits = self.predict_pair_logits(batch.prepared_pairs())
        if not len(logits):
            return np.empty(0, dtype=np.float32)
        return (logits[:, 1] - logits[:, 0]).astype(np.float32, copy=False)


def _pytorch_predictor(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    *,
    batch_size: int,
    dtype: str,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
    non_blocking_transfer: bool,
    length_bucketing: bool,
    padding_length_buckets: tuple[int, ...] | None,
    batch_fields: bool,
    field_chunk_size: int,
    forward_model: Any | None = None,
    backbone_model: Any | None = None,
) -> TransformerPredictor:
    return TransformerPredictor(
        tokenizer,
        PyTorchTransformerExecutor(
            model,
            dtype=normalize_inference_dtype(dtype),
            forward_model=forward_model,
            backbone_model=backbone_model,
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        non_blocking_transfer=non_blocking_transfer,
        length_bucketing=length_bucketing,
        padding_length_buckets=padding_length_buckets,
        batch_fields=batch_fields,
        field_chunk_size=field_chunk_size,
    )


def predict_pair_logits(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    pairs: Sequence[PreparedPair],
    *,
    batch_size: int = 64,
    max_length: int | None = None,
    dtype: str = "float32",
    num_workers: int = 0,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
    non_blocking_transfer: bool = True,
    length_bucketing: bool = False,
    padding_length_buckets: tuple[int, ...] | None = None,
    batch_fields: bool = False,
    field_chunk_size: int = 16_384,
    _forward_model: Any | None = None,
) -> np.ndarray:
    """Return two classifier logits for every prepared pair."""
    return _pytorch_predictor(
        model,
        tokenizer,
        batch_size=batch_size,
        dtype=dtype,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        non_blocking_transfer=non_blocking_transfer,
        length_bucketing=length_bucketing,
        padding_length_buckets=padding_length_buckets,
        batch_fields=batch_fields,
        field_chunk_size=field_chunk_size,
        forward_model=_forward_model,
    ).predict_pair_logits(pairs, max_length=max_length)


def predict_match_probabilities(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    pairs: Sequence[PreparedPair],
    **kwargs: Any,
) -> np.ndarray:
    logits = predict_pair_logits(model, tokenizer, pairs, **kwargs)
    if not len(logits):
        return np.empty(0, dtype=np.float32)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return (exponentials[:, 1] / exponentials.sum(axis=1)).astype(
        np.float32,
        copy=False,
    )


def predict_logit_margins(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    pairs: Sequence[PreparedPair],
    **kwargs: Any,
) -> np.ndarray:
    """Return ``match_logit - different_logit`` for stacking."""
    logits = predict_pair_logits(model, tokenizer, pairs, **kwargs)
    if not len(logits):
        return np.empty(0, dtype=np.float32)
    return (logits[:, 1] - logits[:, 0]).astype(np.float32, copy=False)


def encode_pair_cls(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    pairs: Sequence[PreparedPair],
    *,
    batch_size: int = 64,
    max_length: int | None = None,
    dtype: str = "float32",
    num_workers: int = 0,
    prefetch_factor: int = 2,
    pin_memory: bool = True,
    non_blocking_transfer: bool = True,
    length_bucketing: bool = False,
    padding_length_buckets: tuple[int, ...] | None = None,
    batch_fields: bool = False,
    field_chunk_size: int = 16_384,
    _backbone_model: Any | None = None,
) -> np.ndarray:
    return _pytorch_predictor(
        model,
        tokenizer,
        batch_size=batch_size,
        dtype=dtype,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        non_blocking_transfer=non_blocking_transfer,
        length_bucketing=length_bucketing,
        padding_length_buckets=padding_length_buckets,
        batch_fields=batch_fields,
        field_chunk_size=field_chunk_size,
        backbone_model=_backbone_model,
    ).encode_pairs(pairs, max_length=max_length)


__all__ = [
    "TransformerPredictor",
    "encode_pair_cls",
    "load_trained_classifier",
    "predict_logit_margins",
    "predict_match_probabilities",
    "predict_pair_logits",
]
