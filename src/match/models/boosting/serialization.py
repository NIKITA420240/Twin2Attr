"""Persistence for a CatBoost model and its feature contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .features import CATEGORICAL_FEATURES, FEATURE_SCHEMA_VERSION

MODEL_FILENAME = "model.cbm"
MANIFEST_FILENAME = "manifest.json"


def save_boosting_model(
    model: Any, directory: str | Path, feature_names: list[str]
) -> Path:
    output = Path(directory).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    model.save_model(str(output / MODEL_FILENAME))
    manifest = {
        "format_version": 1,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_names": feature_names,
        "categorical_features": list(CATEGORICAL_FEATURES),
        "positive_class": "match",
        "postprocessing": "none",
    }
    (output / MANIFEST_FILENAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return output


def load_boosting_model(directory: str | Path) -> tuple[Any, dict[str, Any]]:
    source = Path(directory).expanduser().resolve()
    manifest_path = source / MANIFEST_FILENAME
    model_path = source / MODEL_FILENAME
    if not manifest_path.is_file() or not model_path.is_file():
        raise FileNotFoundError(
            f"boosting artifact must contain {MODEL_FILENAME} and {MANIFEST_FILENAME}: {source}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1:
        raise ValueError("unsupported boosting artifact format")
    if manifest.get("feature_schema_version") != FEATURE_SCHEMA_VERSION:
        raise ValueError("boosting feature schema version does not match the code")
    if manifest.get("positive_class") != "match":
        raise ValueError("boosting artifact positive class must be 'match'")

    from catboost import CatBoostClassifier

    model = CatBoostClassifier()
    model.load_model(str(model_path))
    return model, manifest


__all__ = ["load_boosting_model", "save_boosting_model"]
