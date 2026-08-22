"""Translate inference-manifest feature values into typed settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from ..config import FeatureSettings, NerSettings, PhysicalFeatureSettings


def _path(value: Any, *, root: Path, name: str) -> Path | None:
    if value is None or not str(value).strip():
        return None
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else root / path


def feature_settings_from_manifest(
    values: Mapping[str, Any],
    solution_root: Path,
) -> FeatureSettings:
    ner_values = values.get("ner")
    ner = ner_values if isinstance(ner_values, Mapping) else {}
    physical_values = values.get("physical")
    physical = physical_values if isinstance(physical_values, Mapping) else {}
    return FeatureSettings(
        ner=NerSettings(
            enabled=bool(ner.get("enabled", False)),
            provider=None if ner.get("provider") is None else str(ner["provider"]),
            model_dir=_path(
                ner.get("model_dir"),
                root=solution_root,
                name="features.ner.model_dir",
            ),
            cluster_centers_path=_path(
                ner.get("cluster_centers_path"),
                root=solution_root,
                name="features.ner.cluster_centers_path",
            ),
            source_column=str(ner.get("source_column", "name")),
            output_column=str(ner.get("output_column", "ner_attributes")),
            enriched_column=str(
                ner.get("enriched_column", "enriched_attributes")
            ),
            merge_policy=str(ner.get("merge_policy", "missing_only")),
            batch_size=int(ner.get("batch_size", 512)),
            max_length=int(ner.get("max_length", 100)),
            use_amp=bool(ner.get("use_amp", True)),
            semantic_cleanup=bool(ner.get("semantic_cleanup", True)),
        ),
        physical=PhysicalFeatureSettings(
            enabled=bool(physical.get("enabled", False)),
            source_column=str(physical.get("source_column", "name")),
            output_column=str(
                physical.get("output_column", "physical_attributes")
            ),
            enriched_column=str(
                physical.get("enriched_column", "feature_attributes")
            ),
            merge_policy=str(physical.get("merge_policy", "missing_only")),
            normalize_units=bool(physical.get("normalize_units", True)),
            n_jobs=int(physical.get("n_jobs", 1)),
            chunk_size=int(physical.get("chunk_size", 10_000)),
        ),
    )


__all__ = ["feature_settings_from_manifest"]
