"""Persistence and feature-contract validation for stacking CatBoost."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..boosting.features import CATEGORICAL_FEATURES, FEATURE_SCHEMA_VERSION
from .features import TRANSFORMER_LOGIT_MARGIN

MODEL_FILENAME = "model.cbm"
MANIFEST_FILENAME = "manifest.json"
FORMAT_VERSION = 1


def save_stacking_model(
    model: Any,
    directory: str | Path,
    feature_names: list[str],
) -> Path:
    output = Path(directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output / MODEL_FILENAME))
    manifest = {
        "format_version": FORMAT_VERSION,
        "base_model": "transformer",
        "stacking_model": "boosting",
        "base_model_output": "logit_margin",
        "structured_feature_schema_version": FEATURE_SCHEMA_VERSION,
        "structured_feature_count": len(feature_names) - 1,
        "total_feature_count": len(feature_names),
        "feature_names": feature_names,
        "categorical_features": list(CATEGORICAL_FEATURES),
        "transformer_feature": TRANSFORMER_LOGIT_MARGIN,
        "positive_class": "match",
    }
    (output / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def load_stacking_model(directory: str | Path) -> tuple[Any, dict[str, Any]]:
    source = Path(directory).expanduser().resolve()
    manifest_path = source / MANIFEST_FILENAME
    model_path = source / MODEL_FILENAME
    if not manifest_path.is_file() or not model_path.is_file():
        raise FileNotFoundError(
            f"stacking artifact must contain {MODEL_FILENAME} and "
            f"{MANIFEST_FILENAME}: {source}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "format_version": FORMAT_VERSION,
        "base_model": "transformer",
        "stacking_model": "boosting",
        "base_model_output": "logit_margin",
        "structured_feature_schema_version": FEATURE_SCHEMA_VERSION,
        "transformer_feature": TRANSFORMER_LOGIT_MARGIN,
        "positive_class": "match",
    }
    for name, value in expected.items():
        if manifest.get(name) != value:
            raise ValueError(f"invalid stacking manifest value for {name!r}")
    feature_names = manifest.get("feature_names")
    if not isinstance(feature_names, list) or not feature_names:
        raise ValueError("stacking manifest feature_names must be a non-empty list")
    if manifest.get("total_feature_count") != len(feature_names):
        raise ValueError("stacking manifest feature count does not match feature_names")

    from catboost import CatBoostClassifier

    model = CatBoostClassifier()
    model.load_model(str(model_path))
    return model, manifest


__all__ = ["load_stacking_model", "save_stacking_model"]
