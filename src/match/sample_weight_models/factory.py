"""Construction of configured sample-weight models."""

from __future__ import annotations

from ..config import SampleWeightModelSettings
from .constant import ConstantSampleWeightModel
from .contracts import SampleWeightModel
from .transitivity import TransitivitySampleWeightModel


def build_sample_weight_model(
    settings: SampleWeightModelSettings,
) -> SampleWeightModel:
    if not settings.enabled or settings.type == "constant":
        return ConstantSampleWeightModel()
    if settings.type == "transitivity":
        return TransitivitySampleWeightModel(
            penalty_strength=settings.penalty_strength,
            min_weight_multiplier=settings.min_weight_multiplier,
            min_comparable_neighbors=settings.min_comparable_neighbors,
            confidence_weighted_violations=(
                settings.confidence_weighted_violations
            ),
        )
    raise ValueError(f"unsupported sample weight model: {settings.type!r}")


__all__ = ["build_sample_weight_model"]
