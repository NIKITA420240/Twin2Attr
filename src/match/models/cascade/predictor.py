"""Confidence-based routing from a fast predictor to a main predictor."""

from __future__ import annotations

import numpy as np

from ..contracts import MatchPredictor, PredictionBatch


class CascadePredictor:
    def __init__(
        self,
        fast_model: MatchPredictor,
        main_model: MatchPredictor,
        *,
        negative_threshold: float,
        positive_threshold: float,
    ) -> None:
        if not 0.0 <= negative_threshold < positive_threshold <= 1.0:
            raise ValueError(
                "cascade thresholds must satisfy 0 <= negative < positive <= 1"
            )
        self.fast_model = fast_model
        self.main_model = main_model
        self.negative_threshold = negative_threshold
        self.positive_threshold = positive_threshold

    def predict_proba(self, batch: PredictionBatch) -> np.ndarray:
        fast_probabilities = np.asarray(
            self.fast_model.predict_proba(batch),
            dtype=np.float32,
        )
        if fast_probabilities.shape != (batch.matches.height,):
            raise RuntimeError("cascade fast model returned an invalid result shape")
        uncertain = np.flatnonzero(
            (fast_probabilities > self.negative_threshold)
            & (fast_probabilities < self.positive_threshold)
        )
        if uncertain.size == 0:
            return fast_probabilities

        main_probabilities = np.asarray(
            self.main_model.predict_proba(batch.take_indices(uncertain.tolist())),
            dtype=np.float32,
        )
        if main_probabilities.shape != (uncertain.size,):
            raise RuntimeError("cascade main model returned an invalid result shape")
        result = fast_probabilities.copy()
        result[uncertain] = main_probabilities
        if not np.isfinite(result).all():
            raise RuntimeError("cascade predictor returned non-finite probabilities")
        return result


__all__ = ["CascadePredictor"]
