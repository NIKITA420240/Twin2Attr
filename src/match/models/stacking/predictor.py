"""Inference adapter for Transformer/CatBoost stacking."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..boosting.features import BoostingFeatureBuilder
from ..contracts import PredictionBatch
from ..transformer.predictor import TransformerPredictor
from .features import build_stacking_features
from .serialization import load_stacking_model


class StackingPredictor:
    def __init__(
        self,
        model,
        feature_names: list[str],
        *,
        transformer: TransformerPredictor,
        thread_count: int = -1,
        feature_builder: BoostingFeatureBuilder | None = None,
    ) -> None:
        self.model = model
        self.feature_names = tuple(feature_names)
        self.transformer = transformer
        self.thread_count = thread_count
        self.feature_builder = feature_builder or BoostingFeatureBuilder()

    @classmethod
    def load(
        cls,
        directory: str | Path,
        *,
        transformer: TransformerPredictor,
        thread_count: int = -1,
    ) -> "StackingPredictor":
        model, manifest = load_stacking_model(directory)
        return cls(
            model,
            [str(name) for name in manifest["feature_names"]],
            transformer=transformer,
            thread_count=thread_count,
        )

    def predict_proba(self, batch: PredictionBatch) -> np.ndarray:
        if batch.matches.height == 0:
            return np.empty(0, dtype=np.float32)
        margins = self.transformer.predict_logit_margin(batch)
        features = build_stacking_features(
            batch,
            margins,
            structured_builder=self.feature_builder,
        )
        actual_names = tuple(str(name) for name in features.columns)
        if actual_names != self.feature_names:
            raise ValueError(
                "stacking feature schema mismatch: artifact and generated columns differ"
            )
        probabilities = np.asarray(
            self.model.predict_proba(
                features,
                thread_count=self.thread_count,
            )[:, 1],
            dtype=np.float32,
        )
        if probabilities.shape != (batch.matches.height,) or not np.isfinite(
            probabilities
        ).all():
            raise RuntimeError("stacking predictor returned invalid probabilities")
        return probabilities


__all__ = ["StackingPredictor"]
