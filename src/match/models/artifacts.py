"""Training artifact descriptions and inference-manifest persistence."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from ..config import AppConfig


@dataclass(frozen=True, slots=True)
class TrainingArtifacts:
    """Paths and metrics produced by one selected training strategy."""

    predictor: str
    transformer_dir: Path | None = None
    maxpooling_path: Path | None = None
    fusion_path: Path | None = None
    boosting_dir: Path | None = None
    resolved_config_path: Path | None = None
    solution_path: Path | None = None
    metrics: tuple[tuple[str, float], ...] = ()

    def __post_init__(self) -> None:
        if self.predictor == "transformer" and self.transformer_dir is None:
            raise ValueError("transformer artifacts require transformer_dir")
        if self.predictor == "maxpooling" and self.maxpooling_path is None:
            raise ValueError("maxpooling artifacts require maxpooling_path")
        if self.predictor == "fusion" and (
            self.transformer_dir is None
            or self.maxpooling_path is None
            or self.fusion_path is None
        ):
            raise ValueError("fusion artifacts require all three model paths")
        if self.predictor == "boosting" and self.boosting_dir is None:
            raise ValueError("boosting artifacts require boosting_dir")
        if self.predictor == "cascade" and (
            self.transformer_dir is None or self.boosting_dir is None
        ):
            raise ValueError("cascade artifacts require boosting and transformer paths")
        if self.predictor not in {
            "transformer",
            "maxpooling",
            "fusion",
            "boosting",
            "cascade",
        }:
            raise ValueError(f"unsupported artifacts predictor: {self.predictor!r}")


def _manifest_path(path: Path, *, root: Path) -> str:
    return os.path.relpath(path, root)


def build_solution_manifest(
    config: AppConfig,
    artifacts: TrainingArtifacts,
    *,
    map_path: Callable[[Path], str],
) -> dict[str, object]:
    """Build an inference manifest with paths mapped for its destination."""
    solution: dict[str, object] = {"predictor": artifacts.predictor}

    if artifacts.transformer_dir is not None:
        solution["model_directory"] = map_path(artifacts.transformer_dir)
        solution["batch_size"] = config.models_parameters.transformer.batch_size
    if artifacts.maxpooling_path is not None:
        solution["maxpooling_path"] = map_path(artifacts.maxpooling_path)
        solution["maxpooling_batch_size"] = (
            config.models_parameters.maxpooling.batch_size
        )
    if artifacts.fusion_path is not None:
        solution["fusion_path"] = map_path(artifacts.fusion_path)
        solution["fusion_batch_size"] = config.models_parameters.fusion.batch_size
    if artifacts.boosting_dir is not None:
        solution["boosting_directory"] = map_path(artifacts.boosting_dir)
        solution["boosting_thread_count"] = (
            config.models_parameters.boosting.thread_count
        )
    if artifacts.predictor == "cascade":
        cascade = config.models_parameters.cascade
        solution["fast_model"] = cascade.fast_model
        solution["main_model"] = cascade.main_model
        solution["negative_threshold"] = cascade.negative_threshold
        solution["positive_threshold"] = cascade.positive_threshold
    normalization = config.features.normalization
    normalization_solution: dict[str, object] = {
        "enabled": normalization.enabled,
        "output_column": normalization.output_column,
    }
    if normalization.enabled:
        normalization_solution.update(
            {
                "synonyms_path": map_path(normalization.synonyms_path),
                "unique_attributes_path": map_path(
                    normalization.unique_attributes_path
                ),
                "n_jobs": normalization.n_jobs,
                "chunk_size": normalization.chunk_size,
            }
        )
    features_solution: dict[str, object] = {
        "execution_order": list(config.features.execution_order),
        "normalization": normalization_solution,
    }
    if config.features.ner.enabled:
        ner = config.features.ner
        if ner.model_dir is None:
            raise ValueError("enabled NER requires model_dir in solution manifest")
        ner_solution: dict[str, object] = {
            "enabled": True,
            "provider": ner.provider,
            "model_dir": map_path(ner.model_dir),
            "source_column": ner.source_column,
            "output_column": ner.output_column,
            "enriched_column": ner.enriched_column,
            "merge_policy": ner.merge_policy,
            "batch_size": ner.batch_size,
            "max_length": ner.max_length,
            "use_amp": ner.use_amp,
            "semantic_cleanup": ner.semantic_cleanup,
        }
        if ner.cluster_centers_path is not None:
            ner_solution["cluster_centers_path"] = map_path(
                ner.cluster_centers_path
            )
        features_solution["ner"] = ner_solution
    if config.features.physical.enabled:
        physical = config.features.physical
        features_solution["physical"] = {
            "enabled": True,
            "source_column": physical.source_column,
            "output_column": physical.output_column,
            "enriched_column": physical.enriched_column,
            "merge_policy": physical.merge_policy,
            "normalize_units": physical.normalize_units,
            "n_jobs": physical.n_jobs,
            "chunk_size": physical.chunk_size,
        }
    if features_solution:
        solution["features"] = features_solution

    return solution


def save_solution_manifest(
    config: AppConfig,
    artifacts: TrainingArtifacts,
) -> Path:
    """Write an inference manifest matching the produced artifact set."""
    output_path = config.artifacts.solution_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    root = output_path.parent
    solution = build_solution_manifest(
        config,
        artifacts,
        map_path=lambda path: _manifest_path(path, root=root),
    )

    output_path.write_text(
        json.dumps(solution, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("Saved solution manifest to {!s}", output_path)
    return output_path


__all__ = [
    "TrainingArtifacts",
    "build_solution_manifest",
    "save_solution_manifest",
]
