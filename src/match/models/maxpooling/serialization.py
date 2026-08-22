"""Persistence of a fitted max-pooling pipeline."""

from __future__ import annotations

from pathlib import Path

import joblib
from loguru import logger

from .model import MaxPoolingModel


def save_maxpooling_model(
    model: MaxPoolingModel,
    model_path: str | Path,
) -> Path:
    target_path = Path(model_path).expanduser().resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target_path)
    logger.info("Saved max-pooling model to {!s}", target_path)
    return target_path


def load_maxpooling_model(model_path: str | Path) -> MaxPoolingModel:
    model = joblib.load(Path(model_path).expanduser().resolve())
    if not isinstance(model, MaxPoolingModel):
        raise TypeError("model_path does not contain a MaxPoolingModel")
    return model


__all__ = ["load_maxpooling_model", "save_maxpooling_model"]
