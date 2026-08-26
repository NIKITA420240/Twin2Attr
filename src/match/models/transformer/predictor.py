"""Transformer loading, pair encoding and probability prediction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader
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
from ...prepare_data import PreparedPair
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


def load_trained_classifier(
    model_dir: str | Path,
    *,
    device: str | torch.device | None = None,
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    target_device = _resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    config_path = Path(model_dir) / "config.json"
    config_values = json.loads(config_path.read_text(encoding="utf-8"))
    if config_values.get("model_type") == PoolingSequenceClassifierConfig.model_type:
        model = PoolingSequenceClassifier.from_pretrained(model_dir)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(target_device).eval()
    return tokenizer, model


def _encoding_settings(model: PreTrainedModel) -> tuple[bool, int | None]:
    return (
        bool(getattr(model.config, "match_use_field_tokens", True)),
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
) -> tuple[int, bool, int | None]:
    use_field_tokens, max_attribute_value_tokens = _encoding_settings(model)
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
            max_attribute_value_tokens=max_attribute_value_tokens,
        )
    return resolved_length, use_field_tokens, max_attribute_value_tokens


def predict_pair_logits(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    pairs: Sequence[PreparedPair],
    *,
    batch_size: int = 64,
    max_length: int | None = None,
) -> np.ndarray:
    """Return the two classifier logits for every prepared pair."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not pairs:
        return np.empty((0, 2), dtype=np.float32)
    max_length, use_field_tokens, value_limit = _inference_settings(
        model,
        tokenizer,
        pairs,
        max_length,
    )
    collator = PairEncodingCollator(
        tokenizer,
        max_length,
        use_field_tokens=use_field_tokens,
        max_attribute_value_tokens=value_limit,
        include_labels=False,
    )
    device = next(model.parameters()).device
    current_batch_size = batch_size
    while True:
        try:
            loader = DataLoader(
                PreparedPairDataset(pairs),
                batch_size=current_batch_size,
                shuffle=False,
                collate_fn=collator,
            )
            chunks: list[np.ndarray] = []
            model.eval()
            with torch.inference_mode():
                for batch in loader:
                    inputs = {key: value.to(device) for key, value in batch.items()}
                    logits = model(**inputs).logits
                    if logits.ndim != 2 or logits.shape[1] != 2:
                        raise RuntimeError(
                            "stacking requires a binary Transformer classifier"
                        )
                    chunks.append(logits.float().cpu().numpy())
            return np.concatenate(chunks).astype(np.float32, copy=False)
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
) -> np.ndarray:
    logits = predict_pair_logits(
        model,
        tokenizer,
        pairs,
        batch_size=batch_size,
        max_length=max_length,
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
) -> np.ndarray:
    """Return ``match_logit - different_logit`` for stacking."""
    logits = predict_pair_logits(
        model,
        tokenizer,
        pairs,
        batch_size=batch_size,
        max_length=max_length,
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
) -> np.ndarray:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not pairs:
        return np.empty((0, model.config.hidden_size), dtype=np.float32)
    max_length, use_field_tokens, value_limit = _inference_settings(
        model,
        tokenizer,
        pairs,
        max_length,
    )
    collator = PairEncodingCollator(
        tokenizer,
        max_length,
        use_field_tokens=use_field_tokens,
        max_attribute_value_tokens=value_limit,
        include_labels=False,
    )
    device = next(model.parameters()).device
    current_batch_size = batch_size
    while True:
        try:
            loader = DataLoader(
                PreparedPairDataset(pairs),
                batch_size=current_batch_size,
                shuffle=False,
                collate_fn=collator,
            )
            chunks: list[np.ndarray] = []
            model.eval()
            with torch.inference_mode():
                for batch in loader:
                    inputs = {key: value.to(device) for key, value in batch.items()}
                    hidden_state = model.base_model(
                        **inputs,
                        return_dict=True,
                    ).last_hidden_state
                    chunks.append(hidden_state[:, 0, :].float().cpu().numpy())
            return np.concatenate(chunks).astype(np.float32, copy=False)
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

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")

    @classmethod
    def load(
        cls,
        model_directory: str | Path,
        *,
        batch_size: int = 64,
        device: str | None = None,
    ) -> TransformerPredictor:
        tokenizer, model = load_trained_classifier(model_directory, device=device)
        return cls(tokenizer=tokenizer, model=model, batch_size=batch_size)

    @property
    def output_dim(self) -> int:
        return int(self.model.config.hidden_size)

    def encode(self, batch: PredictionBatch) -> np.ndarray:
        return encode_pair_cls(
            self.model,
            self.tokenizer,
            batch.prepared_pairs(),
            batch_size=self.batch_size,
        )

    def predict_proba(self, batch: PredictionBatch) -> np.ndarray:
        return predict_match_probabilities(
            self.model,
            self.tokenizer,
            batch.prepared_pairs(),
            batch_size=self.batch_size,
        )

    def predict_logit_margin(self, batch: PredictionBatch) -> np.ndarray:
        return predict_logit_margins(
            self.model,
            self.tokenizer,
            batch.prepared_pairs(),
            batch_size=self.batch_size,
        )


__all__ = [
    "TransformerPredictor",
    "encode_pair_cls",
    "load_trained_classifier",
    "predict_logit_margins",
    "predict_pair_logits",
    "predict_match_probabilities",
]
