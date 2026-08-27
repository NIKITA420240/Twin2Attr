"""Transformer loading, pair encoding and probability prediction."""

from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from ...pair_encoding import (
    DEFAULT_MAX_ATTRIBUTE_VALUE_TOKENS,
    PairEncodingCollator,
    PreparedPairDataset,
    infer_pair_max_length,
)
from ...prepare_data import PreparedCard, PreparedPair
from ..contracts import PredictionBatch
from .head import PoolingSequenceClassifier, PoolingSequenceClassifierConfig


def _resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _normalize_inference_dtype(dtype: str) -> str:
    normalized = dtype.lower()
    if normalized not in {"float32", "float16", "bfloat16"}:
        raise ValueError(
            "dtype must be one of: float32, float16, bfloat16"
        )
    return normalized


def _torch_inference_dtype(dtype: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[_normalize_inference_dtype(dtype)]


def _autocast_context(device: torch.device, dtype: str):
    normalized = _normalize_inference_dtype(dtype)
    if normalized == "float32":
        return nullcontext()
    if device.type == "cuda":
        if normalized == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("the selected CUDA device does not support bfloat16")
        return torch.autocast(
            device_type="cuda",
            dtype=_torch_inference_dtype(normalized),
        )
    if device.type == "cpu" and normalized == "bfloat16":
        return torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    raise RuntimeError(f"dtype={normalized!r} is not supported on {device.type}")


class _CompiledForward:
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


def load_trained_classifier(
    model_dir: str | Path,
    *,
    device: str | torch.device | None = None,
    dtype: str = "float32",
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    target_device = _resolve_device(device)
    target_dtype = _torch_inference_dtype(dtype)
    if target_device.type == "cuda" and target_dtype == torch.bfloat16:
        if not torch.cuda.is_bf16_supported():
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


def _encoding_settings(
    model: PreTrainedModel,
) -> tuple[bool, int | None, int | None]:
    return (
        bool(getattr(model.config, "match_use_field_tokens", True)),
        getattr(model.config, "match_max_attribute_value_chars", None),
        getattr(
            model.config,
            "match_max_attribute_value_tokens",
            DEFAULT_MAX_ATTRIBUTE_VALUE_TOKENS,
        ),
    )


def _inference_settings(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    pairs: Sequence[PreparedPair],
    max_length: int | None,
) -> tuple[int, bool, int | None, int | None]:
    (
        use_field_tokens,
        max_attribute_value_chars,
        max_attribute_value_tokens,
    ) = _encoding_settings(model)
    resolved_length = (
        max_length
        if max_length is not None
        else getattr(model.config, "match_max_length", None)
    )
    if resolved_length is None:
        resolved_length = infer_pair_max_length(
            tokenizer,
            pairs,
            use_field_tokens=use_field_tokens,
            max_attribute_value_chars=max_attribute_value_chars,
            max_attribute_value_tokens=max_attribute_value_tokens,
        )
    return (
        resolved_length,
        use_field_tokens,
        max_attribute_value_chars,
        max_attribute_value_tokens,
    )


def _estimated_card_length(
    card: PreparedCard,
    max_attribute_value_chars: int | None,
    max_attribute_value_tokens: int | None,
) -> int:
    """Estimate token demand cheaply, without running the tokenizer twice."""
    value_character_limit = max_attribute_value_chars
    if max_attribute_value_tokens is not None:
        token_estimate_limit = max_attribute_value_tokens * 4
        value_character_limit = (
            token_estimate_limit
            if value_character_limit is None
            else min(value_character_limit, token_estimate_limit)
        )
    demand = len(card.name) + len(card.category)
    for key, value in card.attributes:
        value_length = len(value)
        if value_character_limit is not None:
            value_length = min(value_length, value_character_limit)
        demand += len(key) + value_length
    return demand


def _length_bucket_order(
    pairs: Sequence[PreparedPair],
    max_attribute_value_tokens: int | None,
    max_attribute_value_chars: int | None = None,
) -> np.ndarray:
    """Return sorted-position to original-position indices for bucketing."""
    estimates = np.fromiter(
        (
            _estimated_card_length(
                pair.left,
                max_attribute_value_chars,
                max_attribute_value_tokens,
            )
            + _estimated_card_length(
                pair.right,
                max_attribute_value_chars,
                max_attribute_value_tokens,
            )
            for pair in pairs
        ),
        dtype=np.int64,
        count=len(pairs),
    )
    return np.argsort(estimates, kind="stable")


class _OrderedPreparedPairDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[PreparedPair],
        order: np.ndarray,
    ) -> None:
        self._pairs = pairs
        self._order = order

    def __len__(self) -> int:
        return len(self._order)

    def __getitem__(self, index: int) -> PreparedPair:
        return self._pairs[int(self._order[index])]


def _inference_dataset(
    pairs: Sequence[PreparedPair],
    *,
    length_bucketing: bool,
    max_attribute_value_chars: int | None,
    max_attribute_value_tokens: int | None,
) -> tuple[Dataset, np.ndarray | None]:
    if not length_bucketing or len(pairs) < 2:
        return PreparedPairDataset(pairs), None
    order = _length_bucket_order(
        pairs,
        max_attribute_value_tokens,
        max_attribute_value_chars,
    )
    return _OrderedPreparedPairDataset(pairs, order), order


def _inference_loader(
    dataset: Dataset,
    collator: PairEncodingCollator,
    *,
    batch_size: int,
    device: torch.device,
    num_workers: int,
    prefetch_factor: int,
    pin_memory: bool,
) -> tuple[DataLoader, bool]:
    use_pinned_memory = pin_memory and device.type == "cuda"
    kwargs = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "collate_fn": collator,
        "num_workers": num_workers,
        "pin_memory": use_pinned_memory,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = prefetch_factor
    return DataLoader(**kwargs), use_pinned_memory


def _restore_original_order(
    values: np.ndarray,
    order: np.ndarray | None,
) -> np.ndarray:
    if order is None:
        return values
    restored = np.empty_like(values)
    restored[order] = values
    return restored


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
    _forward_model: Any | None = None,
) -> np.ndarray:
    """Return the two classifier logits for every prepared pair."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must not be negative")
    if prefetch_factor < 1:
        raise ValueError("prefetch_factor must be positive")
    if not pairs:
        return np.empty((0, 2), dtype=np.float32)
    max_length, use_field_tokens, char_limit, value_limit = _inference_settings(
        model,
        tokenizer,
        pairs,
        max_length,
    )
    collator = PairEncodingCollator(
        tokenizer,
        max_length,
        use_field_tokens=use_field_tokens,
        max_attribute_value_chars=char_limit,
        max_attribute_value_tokens=value_limit,
        include_labels=False,
        padding_length_buckets=(
            padding_length_buckets if length_bucketing else None
        ),
    )
    device = next(model.parameters()).device
    dataset, order = _inference_dataset(
        pairs,
        length_bucketing=length_bucketing,
        max_attribute_value_chars=char_limit,
        max_attribute_value_tokens=value_limit,
    )
    current_batch_size = batch_size
    while True:
        try:
            loader, use_pinned_memory = _inference_loader(
                dataset,
                collator,
                batch_size=current_batch_size,
                device=device,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                pin_memory=pin_memory,
            )
            chunks: list[np.ndarray] = []
            model.eval()
            with torch.inference_mode():
                for batch in loader:
                    inputs = {
                        key: value.to(
                            device,
                            non_blocking=(
                                non_blocking_transfer and use_pinned_memory
                            ),
                        )
                        for key, value in batch.items()
                    }
                    with _autocast_context(device, dtype):
                        forward_model = (
                            model if _forward_model is None else _forward_model
                        )
                        logits = forward_model(**inputs).logits
                    if logits.ndim != 2 or logits.shape[1] != 2:
                        raise RuntimeError(
                            "stacking requires a binary Transformer classifier"
                        )
                    chunks.append(logits.float().cpu().numpy())
            logits = np.concatenate(chunks).astype(np.float32, copy=False)
            return _restore_original_order(logits, order)
        except torch.cuda.OutOfMemoryError:
            if current_batch_size == 1:
                raise
            current_batch_size = max(1, current_batch_size // 2)
            torch.cuda.empty_cache()


def predict_match_probabilities(
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
    _forward_model: Any | None = None,
) -> np.ndarray:
    logits = predict_pair_logits(
        model,
        tokenizer,
        pairs,
        batch_size=batch_size,
        max_length=max_length,
        dtype=dtype,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        non_blocking_transfer=non_blocking_transfer,
        length_bucketing=length_bucketing,
        padding_length_buckets=padding_length_buckets,
        _forward_model=_forward_model,
    )
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
    _forward_model: Any | None = None,
) -> np.ndarray:
    """Return ``match_logit - different_logit`` for stacking."""
    logits = predict_pair_logits(
        model,
        tokenizer,
        pairs,
        batch_size=batch_size,
        max_length=max_length,
        dtype=dtype,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
        non_blocking_transfer=non_blocking_transfer,
        length_bucketing=length_bucketing,
        padding_length_buckets=padding_length_buckets,
        _forward_model=_forward_model,
    )
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
    _backbone_model: Any | None = None,
) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if num_workers < 0:
        raise ValueError("num_workers must not be negative")
    if prefetch_factor < 1:
        raise ValueError("prefetch_factor must be positive")
    if not pairs:
        return np.empty((0, model.config.hidden_size), dtype=np.float32)
    max_length, use_field_tokens, char_limit, value_limit = _inference_settings(
        model,
        tokenizer,
        pairs,
        max_length,
    )
    collator = PairEncodingCollator(
        tokenizer,
        max_length,
        use_field_tokens=use_field_tokens,
        max_attribute_value_chars=char_limit,
        max_attribute_value_tokens=value_limit,
        include_labels=False,
        padding_length_buckets=(
            padding_length_buckets if length_bucketing else None
        ),
    )
    device = next(model.parameters()).device
    dataset, order = _inference_dataset(
        pairs,
        length_bucketing=length_bucketing,
        max_attribute_value_chars=char_limit,
        max_attribute_value_tokens=value_limit,
    )
    current_batch_size = batch_size
    while True:
        try:
            loader, use_pinned_memory = _inference_loader(
                dataset,
                collator,
                batch_size=current_batch_size,
                device=device,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                pin_memory=pin_memory,
            )
            chunks: list[np.ndarray] = []
            model.eval()
            with torch.inference_mode():
                for batch in loader:
                    inputs = {
                        key: value.to(
                            device,
                            non_blocking=(
                                non_blocking_transfer and use_pinned_memory
                            ),
                        )
                        for key, value in batch.items()
                    }
                    with _autocast_context(device, dtype):
                        backbone_model = (
                            model.base_model
                            if _backbone_model is None
                            else _backbone_model
                        )
                        hidden_state = backbone_model(
                            **inputs,
                            return_dict=True,
                        ).last_hidden_state
                    chunks.append(hidden_state[:, 0, :].float().cpu().numpy())
            embeddings = np.concatenate(chunks).astype(np.float32, copy=False)
            return _restore_original_order(embeddings, order)
        except torch.cuda.OutOfMemoryError:
            if current_batch_size == 1:
                raise
            current_batch_size = max(1, current_batch_size // 2)
            torch.cuda.empty_cache()


@dataclass(slots=True)
class TransformerPredictor:
    tokenizer: PreTrainedTokenizerBase
    model: PreTrainedModel
    batch_size: int = 64
    dtype: str = "float32"
    num_workers: int = 0
    prefetch_factor: int = 2
    pin_memory: bool = True
    non_blocking_transfer: bool = True
    length_bucketing: bool = False
    padding_length_buckets: tuple[int, ...] | None = None
    compile_enabled: bool = False
    compile_mode: str = "reduce-overhead"
    compile_dynamic: bool = True
    _compiled_model: Any | None = field(init=False, default=None, repr=False)
    _compiled_backbone: Any | None = field(
        init=False,
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must not be negative")
        if self.prefetch_factor < 1:
            raise ValueError("prefetch_factor must be positive")
        self.dtype = _normalize_inference_dtype(self.dtype)
        if self.compile_mode not in {
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        }:
            raise ValueError("unsupported torch.compile mode")
        if self.compile_enabled:
            self._compiled_model = _CompiledForward(
                self.model,
                name="Transformer classifier",
                mode=self.compile_mode,
                dynamic=self.compile_dynamic,
            )
            self._compiled_backbone = _CompiledForward(
                self.model.base_model,
                name="Transformer backbone",
                mode=self.compile_mode,
                dynamic=self.compile_dynamic,
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
        return cls(
            tokenizer=tokenizer,
            model=model,
            batch_size=batch_size,
            dtype=dtype,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            non_blocking_transfer=non_blocking_transfer,
            length_bucketing=length_bucketing,
            padding_length_buckets=padding_length_buckets,
            compile_enabled=compile_enabled,
            compile_mode=compile_mode,
            compile_dynamic=compile_dynamic,
        )

    @property
    def output_dim(self) -> int:
        return int(self.model.config.hidden_size)

    def encode(self, batch: PredictionBatch) -> np.ndarray:
        return encode_pair_cls(
            self.model,
            self.tokenizer,
            batch.prepared_pairs(),
            batch_size=self.batch_size,
            dtype=self.dtype,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
            non_blocking_transfer=self.non_blocking_transfer,
            length_bucketing=self.length_bucketing,
            padding_length_buckets=self.padding_length_buckets,
            _backbone_model=self._compiled_backbone,
        )

    def predict_proba(self, batch: PredictionBatch) -> np.ndarray:
        return predict_match_probabilities(
            self.model,
            self.tokenizer,
            batch.prepared_pairs(),
            batch_size=self.batch_size,
            dtype=self.dtype,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
            non_blocking_transfer=self.non_blocking_transfer,
            length_bucketing=self.length_bucketing,
            padding_length_buckets=self.padding_length_buckets,
            _forward_model=self._compiled_model,
        )

    def predict_logit_margin(self, batch: PredictionBatch) -> np.ndarray:
        return predict_logit_margins(
            self.model,
            self.tokenizer,
            batch.prepared_pairs(),
            batch_size=self.batch_size,
            dtype=self.dtype,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
            non_blocking_transfer=self.non_blocking_transfer,
            length_bucketing=self.length_bucketing,
            padding_length_buckets=self.padding_length_buckets,
            _forward_model=self._compiled_model,
        )


__all__ = [
    "TransformerPredictor",
    "encode_pair_cls",
    "load_trained_classifier",
    "predict_logit_margins",
    "predict_pair_logits",
    "predict_match_probabilities",
]
