"""Adapters exposing concrete Twin2Attr models through common capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import joblib
import numpy as np

from .contracts import PairEncoder, PredictionBatch

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

    from ..fusion import FusionClassifier
    from ..maxpooling import MaxPoolingModel


@dataclass(slots=True)
class TransformerPredictor:
    """Transformer adapter supporting direct prediction and CLS encoding."""

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
        from ..transformer import load_trained_classifier

        tokenizer, model = load_trained_classifier(model_directory, device=device)
        return cls(tokenizer=tokenizer, model=model, batch_size=batch_size)

    @property
    def output_dim(self) -> int:
        return int(self.model.config.hidden_size)

    def encode(self, batch: PredictionBatch) -> np.ndarray:
        from ..transformer import encode_pair_cls

        return encode_pair_cls(
            self.model,
            self.tokenizer,
            batch.prepared_pairs(),
            batch_size=self.batch_size,
        )

    def predict_proba(self, batch: PredictionBatch) -> np.ndarray:
        from ..transformer import predict_match_probabilities

        return predict_match_probabilities(
            self.model,
            self.tokenizer,
            batch.prepared_pairs(),
            batch_size=self.batch_size,
        )


@dataclass(slots=True)
class MaxPoolingPredictor:
    """Max-pooling adapter supporting raw encoding and MLP prediction."""

    model: MaxPoolingModel
    batch_size: int = 512
    device: str | None = None

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")

    @classmethod
    def load(
        cls,
        model_path: str | Path,
        *,
        batch_size: int = 512,
        device: str | None = None,
    ) -> MaxPoolingPredictor:
        from ..maxpooling import MaxPoolingModel

        model = joblib.load(model_path)
        if not isinstance(model, MaxPoolingModel):
            raise TypeError("model_path does not contain a MaxPoolingModel")
        return cls(model=model, batch_size=batch_size, device=device)

    @property
    def output_dim(self) -> int:
        return 2 * int(self.model.vector_size)

    def encode(self, batch: PredictionBatch) -> np.ndarray:
        from ..maxpooling import encode_attribute_pairs

        return encode_attribute_pairs(
            batch.items,
            batch.matches,
            self.model,
            attributes_column=batch.attributes_column,
        )

    def predict_proba(self, batch: PredictionBatch) -> np.ndarray:
        from ..maxpooling import predict_maxpooling_probabilities

        return predict_maxpooling_probabilities(
            batch.items,
            batch.matches,
            self.model,
            attributes_column=batch.attributes_column,
            batch_size=self.batch_size,
            device=self.device,
        )


@dataclass(slots=True)
class FusionPredictor:
    """Predict from Transformer and max-pooling features by composition."""

    model: FusionClassifier
    transformer: PairEncoder
    maxpooling: PairEncoder
    batch_size: int = 512

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")

    @classmethod
    def load(
        cls,
        fusion_path: str | Path,
        *,
        transformer: PairEncoder,
        maxpooling: PairEncoder,
        batch_size: int = 512,
        device: str | None = None,
    ) -> FusionPredictor:
        from ..fusion import load_fusion_classifier

        model = load_fusion_classifier(fusion_path, device=device)
        return cls(
            model=model,
            transformer=transformer,
            maxpooling=maxpooling,
            batch_size=batch_size,
        )

    def predict_proba(self, batch: PredictionBatch) -> np.ndarray:
        from ..fusion import predict_fusion_probabilities

        return predict_fusion_probabilities(
            self.model,
            self.transformer.encode(batch),
            self.maxpooling.encode(batch),
            batch_size=self.batch_size,
        )


__all__ = [
    "FusionPredictor",
    "MaxPoolingPredictor",
    "TransformerPredictor",
]
