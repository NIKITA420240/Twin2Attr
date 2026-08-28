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
    stacking_dir: Path | None = None
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
        if self.predictor == "stacking" and (
            self.transformer_dir is None or self.stacking_dir is None
        ):
            raise ValueError("stacking artifacts require transformer and stacking paths")
        if self.predictor not in {
            "transformer",
            "maxpooling",
            "fusion",
            "boosting",
            "cascade",
            "stacking",
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
    solution["augmentation_model"] = config.inference.augmentation_model
    if config.inference.augmentation_model == "attribute_shuffle":
        augmentation = config.augmentation_models.attribute_shuffle
        solution["augmentation_models"] = {
            "attribute_shuffle": {
                "shuffled_copies": augmentation.shuffled_copies,
                "keep_original": augmentation.keep_original,
                "seed": augmentation.seed,
                "shuffle_cards_independently": (
                    augmentation.shuffle_cards_independently
                ),
                "skip_oversized": augmentation.skip_oversized,
            }
        }
    solution["data_postprocessing_model"] = (
        config.inference.data_postprocessing_model
    )
    if config.inference.data_postprocessing_model == "attribute_sort":
        postprocessing = config.data_postprocessing_models.attribute_sort
        solution["data_postprocessing_models"] = {
            "attribute_sort": {
                "priorities_path": map_path(postprocessing.priorities_path),
            }
        }

    if artifacts.transformer_dir is not None:
        solution["model_directory"] = map_path(artifacts.transformer_dir)
        solution["transformer_profile"] = (
            config.model_description.transformer.profile
        )
        solution["transformer_head"] = {
            "type": config.model_description.transformer.head.type,
        }
        solution["backend"] = config.inference.transformer.backend
        solution["batch_size"] = config.inference.transformer.batch_size
        solution["dtype"] = config.inference.transformer.dtype
        solution["pair_encoding"] = {
            "max_length": config.inference.transformer.max_length,
            "max_attribute_value_chars": (
                config.inference.transformer.max_attribute_value_chars
            ),
            "max_attribute_value_tokens": (
                config.inference.transformer.max_attribute_value_tokens
            ),
        }
        solution["num_workers"] = config.inference.transformer.num_workers
        solution["prefetch_factor"] = (
            config.inference.transformer.prefetch_factor
        )
        solution["pin_memory"] = config.inference.transformer.pin_memory
        solution["non_blocking_transfer"] = (
            config.inference.transformer.non_blocking_transfer
        )
        batch_fields = (
            config.model_description.transformer.tokenizer.batch_fields
        )
        solution["tokenizer"] = {
            "batch_fields": {
                "enabled": batch_fields.enabled,
                "chunk_size": batch_fields.chunk_size,
            }
        }
        length_bucketing = config.inference.transformer.length_bucketing
        solution["length_bucketing"] = {
            "enabled": length_bucketing.enabled,
            "padding_length_buckets": (
                None
                if length_bucketing.padding_length_buckets is None
                else list(length_bucketing.padding_length_buckets)
            ),
        }
        solution["torch_compile"] = {
            "enabled": config.inference.transformer.torch_compile.enabled,
            "mode": config.inference.transformer.torch_compile.mode,
            "dynamic": config.inference.transformer.torch_compile.dynamic,
        }
        onnxruntime = config.inference.transformer.onnxruntime
        solution["onnxruntime"] = {
            "provider": onnxruntime.provider,
            "device_id": onnxruntime.device_id,
            "io_binding": onnxruntime.io_binding,
            "graph_optimization": onnxruntime.graph_optimization,
            "fallback_to_pytorch": onnxruntime.fallback_to_pytorch,
            "tensorrt": {
                "engine_cache": {
                    "enabled": onnxruntime.tensorrt.engine_cache.enabled,
                    "path": onnxruntime.tensorrt.engine_cache.path,
                },
                "timing_cache": {
                    "enabled": onnxruntime.tensorrt.timing_cache.enabled,
                    "path": onnxruntime.tensorrt.timing_cache.path,
                },
                "profiles": {
                    "min_batch_size": (
                        onnxruntime.tensorrt.profiles.min_batch_size
                    ),
                    "opt_batch_size": (
                        onnxruntime.tensorrt.profiles.opt_batch_size
                    ),
                    "max_batch_size": (
                        onnxruntime.tensorrt.profiles.max_batch_size
                    ),
                    "sequence_lengths": list(
                        onnxruntime.tensorrt.profiles.sequence_lengths
                    ),
                },
            },
        }
        native_tensorrt = config.inference.transformer.tensorrt
        solution["tensorrt"] = {
            "device_id": native_tensorrt.device_id,
            "workspace_size_gb": native_tensorrt.workspace_size_gb,
            "builder_optimization_level": (
                native_tensorrt.builder_optimization_level
            ),
            "fallback_to_onnxruntime": (
                native_tensorrt.fallback_to_onnxruntime
            ),
            "engine_cache": {
                "enabled": native_tensorrt.engine_cache.enabled,
                "path": native_tensorrt.engine_cache.path,
            },
            "timing_cache": {
                "enabled": native_tensorrt.timing_cache.enabled,
                "path": native_tensorrt.timing_cache.path,
            },
            "profiles": {
                "min_batch_size": native_tensorrt.profiles.min_batch_size,
                "opt_batch_size": native_tensorrt.profiles.opt_batch_size,
                "max_batch_size": native_tensorrt.profiles.max_batch_size,
                "sequence_lengths": list(
                    native_tensorrt.profiles.sequence_lengths
                ),
            },
        }
        onnx_export = config.model_description.transformer.export.onnx
        solution["onnx_artifacts"] = {
            "enabled": onnx_export.enabled,
            "precision": onnx_export.precision,
            "opset": onnx_export.opset,
            "classifier_path": (
                "onnx/classifier.onnx"
                if onnx_export.export_classifier
                else None
            ),
            "encoder_path": (
                "onnx/encoder.onnx" if onnx_export.export_encoder else None
            ),
            "dynamic_batch": onnx_export.dynamic_batch,
            "dynamic_sequence_length": onnx_export.dynamic_sequence_length,
        }
    if artifacts.maxpooling_path is not None:
        solution["maxpooling_path"] = map_path(artifacts.maxpooling_path)
        solution["maxpooling_batch_size"] = (
            config.model_description.maxpooling.batch_size
        )
    if artifacts.fusion_path is not None:
        solution["fusion_path"] = map_path(artifacts.fusion_path)
        solution["fusion_batch_size"] = config.model_description.fusion.batch_size
    if artifacts.boosting_dir is not None:
        solution["boosting_directory"] = map_path(artifacts.boosting_dir)
        solution["boosting_thread_count"] = (
            config.model_description.boosting.thread_count
        )
    if artifacts.stacking_dir is not None:
        solution["stacking_directory"] = map_path(artifacts.stacking_dir)
        solution["stacking_thread_count"] = (
            config.model_description.boosting.thread_count
        )
    if artifacts.predictor == "cascade":
        cascade = config.model_description.cascade
        solution["fast_model"] = cascade.fast_model
        solution["main_model"] = cascade.main_model
        solution["negative_threshold"] = cascade.negative_threshold
        solution["positive_threshold"] = cascade.positive_threshold
    if artifacts.predictor == "stacking":
        stacking = config.model_description.stacking
        solution["base_model"] = stacking.base_model
        solution["stacking_model"] = stacking.stacking_model
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
    output_path = config.training.solution_path
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
