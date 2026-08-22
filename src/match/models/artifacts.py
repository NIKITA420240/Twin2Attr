"""Training artifact descriptions and inference-manifest persistence."""

from __future__ import annotations

import json
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
        if self.predictor not in {"transformer", "maxpooling", "fusion"}:
            raise ValueError(f"unsupported artifacts predictor: {self.predictor!r}")


def _manifest_path(path: Path, *, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def save_solution_manifest(
    config: AppConfig,
    artifacts: TrainingArtifacts,
) -> Path:
    """Write an inference manifest matching the produced artifact set."""
    output_path = config.artifacts.solution_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    root = output_path.parent
    solution: dict[str, object] = {"predictor": artifacts.predictor}

    if artifacts.transformer_dir is not None:
        solution["model_directory"] = _manifest_path(
            artifacts.transformer_dir,
            root=root,
        )
        solution["batch_size"] = config.models_parameters.transformer.batch_size
    if artifacts.maxpooling_path is not None:
        solution["maxpooling_path"] = _manifest_path(
            artifacts.maxpooling_path,
            root=root,
        )
        solution["maxpooling_batch_size"] = (
            config.models_parameters.maxpooling.batch_size
        )
    if artifacts.fusion_path is not None:
        solution["fusion_path"] = _manifest_path(artifacts.fusion_path, root=root)
        solution["fusion_batch_size"] = config.models_parameters.fusion.batch_size
    normalization_solution: dict[str, object] = {
        "enabled": config.normalization.enabled,
        "source_column": config.normalization.source_column,
        "output_column": config.normalization.output_column,
    }
    if config.normalization.enabled:
        normalization_solution.update(
            {
                "synonyms_path": _manifest_path(
                    config.normalization.synonyms_path,
                    root=root,
                ),
                "unique_attributes_path": _manifest_path(
                    config.normalization.unique_attributes_path,
                    root=root,
                ),
                "n_jobs": config.normalization.n_jobs,
                "chunk_size": config.normalization.chunk_size,
            }
        )
    solution["normalization"] = normalization_solution
    features_solution: dict[str, object] = {}
    if config.features.ner.enabled:
        ner = config.features.ner
        if ner.model_dir is None:
            raise ValueError("enabled NER requires model_dir in solution manifest")
        ner_solution: dict[str, object] = {
            "enabled": True,
            "provider": ner.provider,
            "model_dir": _manifest_path(ner.model_dir, root=root),
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
            ner_solution["cluster_centers_path"] = _manifest_path(
                ner.cluster_centers_path,
                root=root,
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

    output_path.write_text(
        json.dumps(solution, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("Saved solution manifest to {!s}", output_path)
    return output_path


__all__ = ["TrainingArtifacts", "save_solution_manifest"]
