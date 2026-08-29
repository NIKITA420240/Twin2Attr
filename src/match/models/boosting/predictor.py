"""Inference adapter for the fast CatBoost matcher."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ..contracts import PredictionBatch
from .features import BoostingFeatureBuilder
from .serialization import load_boosting_model


class BoostingPredictor:
    def __init__(
        self,
        model: Any,
        feature_names: list[str],
        *,
        thread_count: int = -1,
        feature_builder: BoostingFeatureBuilder | None = None,
    ) -> None:
        self.model = model
        self.feature_names = tuple(feature_names)
        self.thread_count = thread_count
        self.feature_builder = feature_builder or BoostingFeatureBuilder()

    @classmethod
    def load(
        cls, directory: str | Path, *, thread_count: int = -1
    ) -> BoostingPredictor:
        model, manifest = load_boosting_model(directory)
        builder = BoostingFeatureBuilder.from_feature_options(
            manifest.get("feature_options")
        )
        return cls(
            model,
            [str(name) for name in manifest["feature_names"]],
            thread_count=thread_count,
            feature_builder=builder,
        )

    def predict_proba(self, batch: PredictionBatch) -> np.ndarray:
        if batch.matches.height == 0:
            return np.empty(0, dtype=np.float32)
        features = self.feature_builder.transform(batch)
        actual_names = tuple(str(name) for name in features.columns)
        if actual_names != self.feature_names:
            raise ValueError(
                "boosting feature schema mismatch: artifact and generated columns differ"
            )
        probabilities = np.asarray(
            self.model.predict_proba(features, thread_count=self.thread_count)[:, 1],
            dtype=np.float32,
        )
        if (
            probabilities.shape != (batch.matches.height,)
            or not np.isfinite(probabilities).all()
        ):
            raise RuntimeError("boosting predictor returned invalid probabilities")
        return probabilities


__all__ = ["BoostingPredictor"]
