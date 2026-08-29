"""Feature composition for Transformer/CatBoost stacking."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..boosting.features import BoostingFeatureBuilder
from ..contracts import PredictionBatch

TRANSFORMER_LOGIT_MARGIN = "transformer_logit_margin"


def build_stacking_features(
    batch: PredictionBatch,
    logit_margins: np.ndarray,
    *,
    structured_builder: BoostingFeatureBuilder | None = None,
) -> pd.DataFrame:
    """Append one out-of-sample Transformer margin to structured features."""
    margins = np.asarray(logit_margins, dtype=np.float32)
    if margins.shape != (batch.matches.height,):
        raise ValueError("Transformer margins must contain one value per pair")
    if not np.isfinite(margins).all():
        raise ValueError("Transformer margins must be finite")
    builder = structured_builder or BoostingFeatureBuilder()
    features = builder.transform(batch)
    features[TRANSFORMER_LOGIT_MARGIN] = margins
    return features


__all__ = ["TRANSFORMER_LOGIT_MARGIN", "build_stacking_features"]
