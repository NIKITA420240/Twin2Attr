"""Typed application configuration and the OmegaConf boundary adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .models.transformer.profile import (
    PROMPTED_BINARY_RERANKER_PROFILE,
    SEQUENCE_CLASSIFIER_PROFILE,
    is_prompted_profile,
    normalize_profile,
    validate_profile_head,
)
from .pair_features import TypedAttributeOptions
from .paths import resolve_project_path

FEATURE_PROVIDER_NAMES = ("normalization", "ner", "physical")

if TYPE_CHECKING:
    from omegaconf import DictConfig


@dataclass(frozen=True, slots=True)
class DatasetSplitterSettings:
    splitter_type: str
    score_type: str
    total_votes: int | None
    negative_threshold: float
    positive_threshold: float
    uncertain_action: str
    target_mode: str = "hard"

    def __post_init__(self) -> None:
        if self.splitter_type != "binary":
            raise ValueError("splitter_type currently must be 'binary'")
        if self.score_type not in {"label", "votes", "probability"}:
            raise ValueError(
                "score_type must be one of: label, votes, probability"
            )
        if self.target_mode not in {"hard", "soft"}:
            raise ValueError("target_mode must be one of: hard, soft")
        if self.score_type == "label" and self.target_mode != "hard":
            raise ValueError("label splitter currently requires target_mode='hard'")
        if self.score_type == "votes":
            if self.total_votes is None or self.total_votes < 1:
                raise ValueError("votes splitter requires positive total_votes")
            if not (
                self.negative_threshold.is_integer()
                and self.positive_threshold.is_integer()
            ):
                raise ValueError("vote thresholds must contain whole vote counts")
            if not (
                0
                <= self.negative_threshold
                < self.positive_threshold
                <= self.total_votes
            ):
                raise ValueError(
                    "vote thresholds must satisfy 0 <= negative < positive "
                    "<= total_votes"
                )
        elif self.score_type == "probability":
            if self.total_votes is not None:
                raise ValueError(
                    "probability splitter must not define total_votes"
                )
            if not (
                0.0
                <= self.negative_threshold
                <= self.positive_threshold
                <= 1.0
            ):
                raise ValueError(
                    "probability thresholds must satisfy 0 <= negative <= "
                    "positive <= 1"
                )
        elif self.total_votes is not None:
            raise ValueError("label splitter must not define total_votes")
        if self.uncertain_action not in {"drop", "keep"}:
            raise ValueError("uncertain_action must be one of: drop, keep")


@dataclass(frozen=True, slots=True)
class ConfidenceWeightingSettings:
    enabled: bool = False
    method: str = "distance_from_midpoint"
    min_weight_multiplier: float = 0.2
    power: float = 1.0

    def __post_init__(self) -> None:
        if self.method != "distance_from_midpoint":
            raise ValueError(
                "confidence_weighting.method currently must be "
                "'distance_from_midpoint'"
            )
        if not 0.0 < self.min_weight_multiplier <= 1.0:
            raise ValueError(
                "confidence_weighting.min_weight_multiplier must be in (0, 1]"
            )
        if self.power <= 0.0:
            raise ValueError("confidence_weighting.power must be positive")


@dataclass(frozen=True, slots=True)
class SampleWeightModelSettings:
    type: str = "constant"
    enabled: bool = False
    penalty_strength: float = 1.0
    min_weight_multiplier: float = 0.25
    min_comparable_neighbors: int = 2
    confidence_weighted_violations: bool = True

    def __post_init__(self) -> None:
        if self.type not in {"constant", "transitivity"}:
            raise ValueError(
                "dataset source weight_model.type must be constant or transitivity"
            )
        if self.penalty_strength < 0.0:
            raise ValueError("weight_model.penalty_strength must not be negative")
        if not 0.0 < self.min_weight_multiplier <= 1.0:
            raise ValueError(
                "weight_model.min_weight_multiplier must be in (0, 1]"
            )
        if self.min_comparable_neighbors < 1:
            raise ValueError(
                "weight_model.min_comparable_neighbors must be positive"
            )


@dataclass(frozen=True, slots=True)
class DatasetSourceSettings:
    name: str
    matches: Path
    weight: float
    max_rows: int | None
    sampling_strategy: str
    splitter: DatasetSplitterSettings
    weight_model: SampleWeightModelSettings = SampleWeightModelSettings()
    confidence_weighting: ConfidenceWeightingSettings = (
        ConfidenceWeightingSettings()
    )
    confidence_power: float = 2.0
    target_column: str = "target"
    items: Path | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("dataset source name must not be empty")
        if not self.target_column.strip():
            raise ValueError("dataset source target_column must not be empty")
        if self.weight <= 0.0:
            raise ValueError("dataset source weight must be positive")
        if self.max_rows is not None and self.max_rows < 1:
            raise ValueError("dataset source max_rows must be positive or null")
        if self.sampling_strategy not in {
            "random",
            "category_target_balanced",
            "category_target_confidence_weighted",
            "category_target_confidence_priority",
        }:
            raise ValueError(
                "sampling_strategy must be one of: random, "
                "category_target_balanced, category_target_confidence_weighted, "
                "category_target_confidence_priority"
            )
        if self.confidence_power <= 0.0:
            raise ValueError("confidence_power must be positive")


@dataclass(frozen=True, slots=True)
class DatasetOverlapResolutionSettings:
    enabled: bool = False
    source_priority: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if len(set(self.source_priority)) != len(self.source_priority):
            raise ValueError(
                "mix_dataset.overlap_resolution.source_priority must be unique"
            )
        if any(not name.strip() for name in self.source_priority):
            raise ValueError(
                "mix_dataset.overlap_resolution.source_priority names "
                "must not be empty"
            )


@dataclass(frozen=True, slots=True)
class BaseDatasetSettings:
    items: Path
    matches: Path
    validation_fraction: float
    stacking_train_fraction: float
    leakage_scope: str
    candidate_splits: int
    seed: int

    def __post_init__(self) -> None:
        _validate_data_split_settings(
            self.validation_fraction,
            self.leakage_scope,
            self.candidate_splits,
        )
        if not 0.0 < self.stacking_train_fraction < 1.0:
            raise ValueError("stacking_train_fraction must be between zero and one")
        if self.validation_fraction + self.stacking_train_fraction >= 1.0:
            raise ValueError(
                "validation_fraction + stacking_train_fraction must be less than one"
            )


@dataclass(frozen=True, slots=True)
class MixedDatasetSettings:
    items: Path
    sources: tuple[DatasetSourceSettings, ...]
    validation_source: str
    validation_fraction: float
    leakage_scope: str
    candidate_splits: int
    seed: int
    overlap_resolution: DatasetOverlapResolutionSettings = (
        DatasetOverlapResolutionSettings()
    )

    def __post_init__(self) -> None:
        _validate_data_split_settings(
            self.validation_fraction,
            self.leakage_scope,
            self.candidate_splits,
        )
        names = [source.name for source in self.sources]
        if len(self.sources) < 2:
            raise ValueError("mix_dataset requires at least two sources")
        if len(names) != len(set(names)):
            raise ValueError("mix_dataset source names must be unique")
        if self.validation_source not in names:
            raise ValueError("validation_source must name a configured source")
        if self.overlap_resolution.enabled:
            priority = self.overlap_resolution.source_priority
            if set(priority) != set(names) or len(priority) != len(names):
                raise ValueError(
                    "enabled mix_dataset.overlap_resolution.source_priority "
                    "must contain every configured source exactly once"
                )


@dataclass(frozen=True, slots=True)
class DataModelDescriptionSettings:
    base_dataset: BaseDatasetSettings
    mix_dataset: MixedDatasetSettings
    mix_dataset_hard_negative: MixedDatasetSettings
    mix_dataset_codex: MixedDatasetSettings
    mix_dataset_neural_review: MixedDatasetSettings


def _validate_data_split_settings(
    validation_fraction: float,
    leakage_scope: str,
    candidate_splits: int,
) -> None:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    if leakage_scope not in {"none", "pair", "item"}:
        raise ValueError("leakage_scope must be one of: none, pair, item")
    if candidate_splits < 1:
        raise ValueError("candidate_splits must be positive")


@dataclass(frozen=True, slots=True)
class TrainingSettings:
    model: str
    data_model: str
    augmentation_model: str | None
    data_postprocessing_model: str | None
    resolved_config_path: Path
    solution_path: Path

    def __post_init__(self) -> None:
        if self.augmentation_model not in {
            None,
            "attribute_shuffle",
            "attribute_word_dropout",
        }:
            raise ValueError(
                "training.augmentation_model must be null, 'attribute_shuffle', "
                "or 'attribute_word_dropout'"
            )
        if self.data_postprocessing_model not in {None, "attribute_sort"}:
            raise ValueError(
                "training.data_postprocessing_model must be null or "
                "'attribute_sort'"
            )
        if self.model not in {
            "transformer",
            "maxpooling",
            "fusion",
            "boosting",
            "stacking",
        }:
            raise ValueError(
                "training.model must be one of: transformer, maxpooling, fusion, "
                "boosting, stacking"
            )
        if self.data_model not in {
            "base_dataset",
            "mix_dataset",
            "mix_dataset_hard_negative",
            "mix_dataset_codex",
            "mix_dataset_neural_review",
        }:
            raise ValueError(
                "training.data_model must be one of: base_dataset, mix_dataset, "
                "mix_dataset_hard_negative, mix_dataset_codex, "
                "mix_dataset_neural_review"
            )
        if self.model == "stacking" and self.data_model != "base_dataset":
            raise ValueError("stacking training currently requires base_dataset")


@dataclass(frozen=True, slots=True)
class TorchCompileSettings:
    enabled: bool = False
    mode: str = "reduce-overhead"
    dynamic: bool = True

    def __post_init__(self) -> None:
        if self.mode not in {
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        }:
            raise ValueError(
                "torch_compile.mode must be one of: "
                "default, reduce-overhead, max-autotune, "
                "max-autotune-no-cudagraphs"
            )


@dataclass(frozen=True, slots=True)
class AttentionSettings:
    """Attention kernel selected when a PyTorch Transformer is constructed."""

    implementation: str = "auto"

    def __post_init__(self) -> None:
        normalized = self.implementation.lower()
        if normalized not in {"auto", "eager", "sdpa"}:
            raise ValueError(
                "attention.implementation must be one of: auto, eager, sdpa"
            )
        object.__setattr__(self, "implementation", normalized)


@dataclass(frozen=True, slots=True)
class TrainingLengthBucketingSettings:
    enabled: bool = True
    mega_batch_multiplier: int = 50
    padding_length_buckets: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if self.mega_batch_multiplier < 1:
            raise ValueError(
                "transformer.training_runtime.length_bucketing."
                "mega_batch_multiplier must be positive"
            )
        buckets = self.padding_length_buckets
        if buckets is None:
            return
        if not buckets or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in buckets
        ):
            raise ValueError(
                "transformer.training_runtime.length_bucketing."
                "padding_length_buckets must contain positive integers"
            )
        if tuple(sorted(set(buckets))) != buckets:
            raise ValueError(
                "transformer.training_runtime.length_bucketing."
                "padding_length_buckets must be strictly increasing"
            )


@dataclass(frozen=True, slots=True)
class TrainingDataLoaderSettings:
    num_workers: int = 4
    prefetch_factor: int = 2
    persistent_workers: bool = True
    pin_memory: bool = True
    non_blocking_transfer: bool = True

    def __post_init__(self) -> None:
        if self.num_workers < 0:
            raise ValueError(
                "transformer.training_runtime.dataloader.num_workers "
                "must not be negative"
            )
        if self.prefetch_factor < 1:
            raise ValueError(
                "transformer.training_runtime.dataloader.prefetch_factor "
                "must be positive"
            )
        if self.num_workers == 0 and self.persistent_workers:
            raise ValueError(
                "transformer.training_runtime.dataloader.persistent_workers "
                "requires num_workers > 0"
            )


@dataclass(frozen=True, slots=True)
class TrainingPerformanceLoggingSettings:
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class TrainingTokenCacheSettings:
    enabled: bool = False
    directory: Path = Path(".cache/tokenized_pairs")
    build_chunk_size: int = 4_096

    def __post_init__(self) -> None:
        if self.build_chunk_size < 1:
            raise ValueError(
                "transformer.training_runtime.token_cache.build_chunk_size "
                "must be positive"
            )


@dataclass(frozen=True, slots=True)
class TransformerTrainingRuntimeSettings:
    attention: AttentionSettings = AttentionSettings()
    length_bucketing: TrainingLengthBucketingSettings = (
        TrainingLengthBucketingSettings()
    )
    dataloader: TrainingDataLoaderSettings = TrainingDataLoaderSettings()
    performance_logging: TrainingPerformanceLoggingSettings = (
        TrainingPerformanceLoggingSettings()
    )
    token_cache: TrainingTokenCacheSettings = TrainingTokenCacheSettings()
    torch_compile: TorchCompileSettings = TorchCompileSettings()


@dataclass(frozen=True, slots=True)
class TensorRTEngineCacheSettings:
    enabled: bool = True
    path: str = "onnx/trt_cache"

    def __post_init__(self) -> None:
        if self.enabled and not self.path.strip():
            raise ValueError(
                "TensorRT engine cache path must not be empty when enabled"
            )


@dataclass(frozen=True, slots=True)
class TensorRTTimingCacheSettings:
    enabled: bool = True
    path: str | None = None

    def __post_init__(self) -> None:
        if self.path is not None and not self.path.strip():
            raise ValueError(
                "TensorRT timing cache path must be null or non-empty"
            )


@dataclass(frozen=True, slots=True)
class TensorRTProfileSettings:
    min_batch_size: int = 1
    opt_batch_size: int = 2048
    max_batch_size: int = 2048
    min_sequence_length: int = 64
    opt_sequence_length: int = 96
    max_sequence_length: int = 128

    def __post_init__(self) -> None:
        batches = (
            self.min_batch_size,
            self.opt_batch_size,
            self.max_batch_size,
        )
        if any(isinstance(value, bool) or value < 1 for value in batches):
            raise ValueError(
                "TensorRT profile batch sizes must be positive integers"
            )
        if not (
            self.min_batch_size
            <= self.opt_batch_size
            <= self.max_batch_size
        ):
            raise ValueError(
                "TensorRT profile batch sizes must satisfy min <= opt <= max"
            )
        lengths = self.sequence_lengths
        if any(isinstance(value, bool) or value < 1 for value in lengths):
            raise ValueError(
                "TensorRT profile sequence lengths must be positive integers"
            )
        if not lengths[0] <= lengths[1] <= lengths[2]:
            raise ValueError(
                "TensorRT profile sequence lengths must satisfy min <= opt <= max"
            )

    @property
    def sequence_lengths(self) -> tuple[int, int, int]:
        return (
            self.min_sequence_length,
            self.opt_sequence_length,
            self.max_sequence_length,
        )


@dataclass(frozen=True, slots=True)
class OrtTensorRTProviderSettings:
    engine_cache: TensorRTEngineCacheSettings = TensorRTEngineCacheSettings()
    timing_cache: TensorRTTimingCacheSettings = TensorRTTimingCacheSettings()
    profiles: TensorRTProfileSettings = TensorRTProfileSettings()


@dataclass(frozen=True, slots=True)
class OnnxRuntimeSettings:
    provider: str = "cuda"
    device_id: int = 0
    io_binding: bool = True
    graph_optimization: str = "all"
    fallback_to_pytorch: bool = True
    tensorrt: OrtTensorRTProviderSettings = OrtTensorRTProviderSettings()

    def __post_init__(self) -> None:
        if isinstance(self.device_id, bool) or self.device_id < 0:
            raise ValueError(
                "inference.transformer.onnxruntime.device_id must be a "
                "non-negative integer"
            )
        if self.provider not in {"cpu", "cuda", "tensorrt"}:
            raise ValueError(
                "inference.transformer.onnxruntime.provider must be one of: "
                "cpu, cuda, tensorrt"
            )
        if self.graph_optimization not in {
            "disabled",
            "basic",
            "extended",
            "all",
        }:
            raise ValueError(
                "inference.transformer.onnxruntime.graph_optimization must be "
                "one of: disabled, basic, extended, all"
            )


@dataclass(frozen=True, slots=True)
class NativeTensorRTSettings:
    device_id: int = 0
    workspace_size_gb: float = 8.0
    builder_optimization_level: int = 3
    fallback_to_onnxruntime: bool = True
    engine_cache: TensorRTEngineCacheSettings = TensorRTEngineCacheSettings(
        path="onnx/native_trt_cache"
    )
    timing_cache: TensorRTTimingCacheSettings = TensorRTTimingCacheSettings()
    profiles: TensorRTProfileSettings = TensorRTProfileSettings(
        opt_batch_size=512,
        max_batch_size=512,
        min_sequence_length=1,
        opt_sequence_length=256,
        max_sequence_length=472,
    )

    def __post_init__(self) -> None:
        if isinstance(self.device_id, bool) or self.device_id < 0:
            raise ValueError("inference.transformer.tensorrt.device_id is invalid")
        if self.workspace_size_gb <= 0:
            raise ValueError(
                "inference.transformer.tensorrt.workspace_size_gb must be positive"
            )
        if self.builder_optimization_level not in range(6):
            raise ValueError(
                "inference.transformer.tensorrt.builder_optimization_level "
                "must be 0..5"
            )


@dataclass(frozen=True, slots=True)
class LengthBucketingSettings:
    enabled: bool = True
    padding_length_buckets: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        buckets = self.padding_length_buckets
        if buckets is None:
            return
        if not buckets or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 1
            for value in buckets
        ):
            raise ValueError(
                "inference.transformer.length_bucketing."
                "padding_length_buckets must contain positive integers"
            )
        if tuple(sorted(set(buckets))) != buckets:
            raise ValueError(
                "inference.transformer.length_bucketing."
                "padding_length_buckets must be strictly increasing"
            )


@dataclass(frozen=True, slots=True)
class TransformerInferenceSettings:
    batch_size: int
    dtype: str
    max_length: int | None = None
    max_attribute_value_chars: int | None = None
    max_attribute_value_tokens: int | None = None
    backend: str = "pytorch"
    num_workers: int = 0
    prefetch_factor: int = 2
    pin_memory: bool = True
    non_blocking_transfer: bool = True
    attention: AttentionSettings = AttentionSettings()
    length_bucketing: LengthBucketingSettings = LengthBucketingSettings()
    torch_compile: TorchCompileSettings = TorchCompileSettings()
    onnxruntime: OnnxRuntimeSettings = OnnxRuntimeSettings()
    tensorrt: NativeTensorRTSettings = NativeTensorRTSettings()

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("inference.transformer.batch_size must be positive")
        for name, value in (
            ("max_length", self.max_length),
            ("max_attribute_value_chars", self.max_attribute_value_chars),
            ("max_attribute_value_tokens", self.max_attribute_value_tokens),
        ):
            if value is not None and value < 1:
                raise ValueError(
                    f"inference.transformer.{name} must be positive or null"
                )
        if self.num_workers < 0:
            raise ValueError(
                "inference.transformer.num_workers must not be negative"
            )
        if self.prefetch_factor < 1:
            raise ValueError(
                "inference.transformer.prefetch_factor must be positive"
            )
        if self.dtype not in {"float32", "float16", "bfloat16"}:
            raise ValueError(
                "inference.transformer.dtype must be one of: float32, float16, "
                "bfloat16"
            )
        if self.backend not in {"pytorch", "onnxruntime", "tensorrt"}:
            raise ValueError(
                "inference.transformer.backend must be one of: pytorch, "
                "onnxruntime, tensorrt"
            )


@dataclass(frozen=True, slots=True)
class InferenceSettings:
    model: str
    augmentation_model: str | None
    data_postprocessing_model: str | None
    solution_path: Path
    transformer: TransformerInferenceSettings

    def __post_init__(self) -> None:
        if self.augmentation_model not in {None, "attribute_shuffle"}:
            raise ValueError(
                "inference.augmentation_model must be null or 'attribute_shuffle'"
            )
        if self.data_postprocessing_model not in {None, "attribute_sort"}:
            raise ValueError(
                "inference.data_postprocessing_model must be null or "
                "'attribute_sort'"
            )
        if self.model not in {
            "transformer",
            "maxpooling",
            "fusion",
            "boosting",
            "cascade",
            "stacking",
        }:
            raise ValueError(
                "inference.model must be one of: transformer, maxpooling, fusion, "
                "boosting, cascade, stacking"
            )


@dataclass(frozen=True, slots=True)
class AnalysisSettings:
    analysis_model: str
    data_model: str
    augmentation_model: str | None
    data_postprocessing_model: str | None

    def __post_init__(self) -> None:
        if self.augmentation_model not in {None, "attribute_shuffle"}:
            raise ValueError(
                "analysis.augmentation_model must be null or 'attribute_shuffle'"
            )
        if self.data_postprocessing_model not in {None, "attribute_sort"}:
            raise ValueError(
                "analysis.data_postprocessing_model must be null or "
                "'attribute_sort'"
            )
        if self.analysis_model not in {
            "attribute_importance",
            "backend_benchmark",
        }:
            raise ValueError(
                "analysis.analysis_model must be one of: attribute_importance, "
                "backend_benchmark"
            )
        if self.data_model not in {
            "base_dataset",
            "mix_dataset",
            "mix_dataset_hard_negative",
            "mix_dataset_codex",
            "mix_dataset_neural_review",
        }:
            raise ValueError(
                "analysis.data_model must be one of: base_dataset, mix_dataset, "
                "mix_dataset_hard_negative, mix_dataset_codex, "
                "mix_dataset_neural_review"
            )


@dataclass(frozen=True, slots=True)
class LlmLabelingSettings:
    base_url: str
    model: str
    token_env: str
    temperature: float
    verify_ssl: bool
    request_batch_size: int
    max_concurrency: int
    min_concurrency: int
    max_attempts: int
    max_rounds: int
    retry_base_seconds: float
    max_prompt_chars: int

    def __post_init__(self) -> None:
        for name, value in (
            ("base_url", self.base_url),
            ("model", self.model),
            ("token_env", self.token_env),
        ):
            if not value.strip():
                raise ValueError(f"labeling.llm.{name} must not be empty")
        if self.temperature < 0.0:
            raise ValueError("labeling.llm.temperature must not be negative")
        if self.request_batch_size < 1:
            raise ValueError("labeling.llm.request_batch_size must be positive")
        if self.max_concurrency < 1:
            raise ValueError("labeling.llm.max_concurrency must be positive")
        if not 1 <= self.min_concurrency <= self.max_concurrency:
            raise ValueError(
                "labeling.llm.min_concurrency must be between 1 and "
                "max_concurrency"
            )
        if self.max_attempts < 1:
            raise ValueError("labeling.llm.max_attempts must be positive")
        if self.max_rounds < 1:
            raise ValueError("labeling.llm.max_rounds must be positive")
        if self.retry_base_seconds < 0.0:
            raise ValueError(
                "labeling.llm.retry_base_seconds must not be negative"
            )
        if self.max_prompt_chars < 1:
            raise ValueError("labeling.llm.max_prompt_chars must be positive")


@dataclass(frozen=True, slots=True)
class DatasetLabelingSettings:
    source_matches_path: Path
    items_path: Path
    output_path: Path
    score_column: str
    lower_p: float
    upper_p: float
    sample_size: int
    seed: int
    checkpoint_every_batches: int
    llm: LlmLabelingSettings

    def __post_init__(self) -> None:
        if not self.score_column.strip():
            raise ValueError("labeling.score_column must not be empty")
        if not 0.0 <= self.lower_p <= self.upper_p <= 1.0:
            raise ValueError(
                "labeling thresholds must satisfy 0 <= lower_p <= upper_p <= 1"
            )
        if self.sample_size < 1:
            raise ValueError("labeling.sample_size must be positive")
        if self.checkpoint_every_batches < 1:
            raise ValueError(
                "labeling.checkpoint_every_batches must be positive"
            )


@dataclass(frozen=True, slots=True)
class NormalizationSettings:
    enabled: bool
    output_column: str
    synonyms_path: Path
    unique_attributes_path: Path
    n_jobs: int
    chunk_size: int


@dataclass(frozen=True, slots=True)
class PairEncodingSettings:
    use_field_tokens: bool
    max_attribute_value_chars: int | None
    max_attribute_value_tokens: int | None
    max_length: int | None
    quantile: float
    sample_size: int
    hard_cap: int

    def __post_init__(self) -> None:
        if (
            self.max_attribute_value_chars is not None
            and self.max_attribute_value_chars < 1
        ):
            raise ValueError(
                "model_description.transformer.pair_encoding."
                "max_attribute_value_chars must be positive or null"
            )


@dataclass(frozen=True, slots=True)
class SpecialTokenInitializationSettings:
    """Semantic initialization settings for the pair field tokens."""

    enabled: bool = False
    key_seed_texts: tuple[str, ...] = ()
    value_seed_texts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.enabled:
            return
        for texts, name in (
            (self.key_seed_texts, "key"),
            (self.value_seed_texts, "value"),
        ):
            if not texts or any(not value.strip() for value in texts):
                raise ValueError(
                    "model_description.transformer.special_token_initialization."
                    f"{name}_seed_texts must contain non-empty strings"
                )


@dataclass(frozen=True, slots=True)
class FastDevValidationSettings:
    """Small fixed validation sample used only for frequent diagnostics."""

    enabled: bool = False
    max_rows: int = 20_000
    every_n_optimizer_steps: int = 1_000

    def __post_init__(self) -> None:
        if self.max_rows < 1:
            raise ValueError(
                "model_description.transformer.validation.fast_dev.max_rows "
                "must be positive"
            )
        if self.every_n_optimizer_steps < 1:
            raise ValueError(
                "model_description.transformer.validation.fast_dev."
                "every_n_optimizer_steps must be positive"
            )


@dataclass(frozen=True, slots=True)
class TransformerValidationSettings:
    fast_dev: FastDevValidationSettings = FastDevValidationSettings()


@dataclass(frozen=True, slots=True)
class BatchFieldsSettings:
    enabled: bool = False
    chunk_size: int = 16_384

    def __post_init__(self) -> None:
        if self.chunk_size < 1:
            raise ValueError(
                "model_description.transformer.tokenizer.batch_fields."
                "chunk_size must be positive"
            )


@dataclass(frozen=True, slots=True)
class TransformerTokenizerSettings:
    batch_fields: BatchFieldsSettings = BatchFieldsSettings()


@dataclass(frozen=True, slots=True)
class OnnxExportSettings:
    enabled: bool = False
    opset: int = 18
    precision: str = "float16"
    dynamic_batch: bool = True
    dynamic_sequence_length: bool = True
    export_classifier: bool = True
    export_encoder: bool = True

    def __post_init__(self) -> None:
        if self.opset < 14:
            raise ValueError(
                "model_description.transformer.export.onnx.opset must be at "
                "least 14"
            )
        if self.precision not in {"float32", "float16"}:
            raise ValueError(
                "model_description.transformer.export.onnx.precision must be "
                "float32 or float16"
            )
        if self.enabled and not (self.export_classifier or self.export_encoder):
            raise ValueError(
                "enabled ONNX export requires classifier or encoder output"
            )


@dataclass(frozen=True, slots=True)
class TransformerExportSettings:
    onnx: OnnxExportSettings = OnnxExportSettings()


@dataclass(frozen=True, slots=True)
class TransformerHeadParameters:
    type: str = "default"
    poolings: tuple[str, ...] = ("cls",)
    mlp_hidden_dims: tuple[int, ...] = ()
    dropout: float = 0.1
    attention_hidden_dim: int | None = None
    attention_num_heads: int = 1
    native_logit_weight: float = 1.0
    attention_logit_weight: float = 0.0
    train_logit_weights: bool = True
    typed_hidden_dims: tuple[int, ...] = (128, 64)
    typed_layer_norm: bool = True

    def __post_init__(self) -> None:
        normalized_type = self.type.strip().lower()
        supported_types = {
            "default",
            "pooling",
            "hybrid",
            "native",
            "attention_pooling",
            "typed_attribute_fusion",
            "gated_residual_fusion",
        }
        if normalized_type not in supported_types:
            raise ValueError(
                "transformer.head.type must be one of: default, pooling, hybrid, "
                "native, attention_pooling, typed_attribute_fusion, "
                "gated_residual_fusion"
            )
        normalized_poolings = tuple(
            pooling.strip().lower() for pooling in self.poolings
        )
        if not normalized_poolings:
            raise ValueError("transformer.head.poolings must not be empty")
        unknown = set(normalized_poolings) - {"cls", "mean", "max", "attention"}
        if unknown:
            raise ValueError(
                "unsupported transformer.head.poolings: "
                + ", ".join(sorted(unknown))
            )
        if len(set(normalized_poolings)) != len(normalized_poolings):
            raise ValueError("transformer.head.poolings must be unique")
        if any(dimension < 1 for dimension in self.mlp_hidden_dims):
            raise ValueError("transformer.head.mlp_hidden_dims must be positive")
        if any(dimension < 1 for dimension in self.typed_hidden_dims):
            raise ValueError("transformer.head.typed_hidden_dims must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("transformer.head.dropout must be in [0, 1)")
        if self.attention_hidden_dim is not None and self.attention_hidden_dim < 1:
            raise ValueError(
                "transformer.head.attention_hidden_dim must be positive or null"
            )
        if self.attention_num_heads < 1:
            raise ValueError(
                "transformer.head.attention_num_heads must be positive"
            )
        if self.native_logit_weight < 0.0 or self.attention_logit_weight < 0.0:
            raise ValueError("transformer.head logit weights must be non-negative")
        if self.native_logit_weight == 0.0 and self.attention_logit_weight == 0.0:
            raise ValueError("at least one transformer.head logit weight must be positive")
        object.__setattr__(self, "type", normalized_type)
        object.__setattr__(
            self,
            "poolings",
            normalized_poolings,
        )


@dataclass(frozen=True, slots=True)
class TransformerParameters:
    profile: str
    pretrained_model_path: str
    artifact_dir: Path
    tokenizer: TransformerTokenizerSettings
    pair_encoding: PairEncodingSettings
    export: TransformerExportSettings
    max_epochs: int
    hpo_trials: int
    learning_rate: float
    weight_decay: float
    hpo_learning_rate_min: float
    hpo_learning_rate_max: float
    hpo_weight_decay_min: float
    hpo_weight_decay_max: float
    train_batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    warmup_ratio: float
    max_grad_norm: float
    early_stopping_patience: int
    auto_find_batch_size: bool
    embeddings_learning_rate: float | None = None
    train_new_token_embeddings_only: bool = False
    train_last_n_layers: int | None = None
    special_token_initialization: SpecialTokenInitializationSettings = (
        SpecialTokenInitializationSettings()
    )
    validation: TransformerValidationSettings = TransformerValidationSettings()
    lr_scheduler_type: str = "linear"
    head_learning_rate: float | None = None
    layerwise_lr_decay: float = 1.0
    training_runtime: TransformerTrainingRuntimeSettings = (
        TransformerTrainingRuntimeSettings()
    )
    head: TransformerHeadParameters = TransformerHeadParameters()

    def __post_init__(self) -> None:
        normalized_profile = normalize_profile(self.profile)
        validate_profile_head(normalized_profile, self.head.type)
        if is_prompted_profile(normalized_profile):
            if self.head.type == "attention_pooling" and "cls" in self.head.poolings:
                raise ValueError(
                    f"{PROMPTED_BINARY_RERANKER_PROFILE} attention_pooling cannot "
                    "use cls pooling"
                )
            if self.pair_encoding.use_field_tokens:
                raise ValueError(
                    f"{PROMPTED_BINARY_RERANKER_PROFILE} requires pair_encoding."
                    "use_field_tokens=false to preserve pretrained token semantics"
                )
        object.__setattr__(self, "profile", normalized_profile)
        if not self.pretrained_model_path.strip():
            raise ValueError("transformer.pretrained_model_path must not be empty")
        if self.max_epochs < 1 or self.hpo_trials < 1:
            raise ValueError("transformer epoch and HPO counts must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("transformer optimizer parameters are invalid")
        if (
            self.embeddings_learning_rate is not None
            and self.embeddings_learning_rate <= 0.0
        ):
            raise ValueError("transformer optimizer parameters are invalid")
        if self.head_learning_rate is not None and self.head_learning_rate <= 0.0:
            raise ValueError("transformer optimizer parameters are invalid")
        if not 0.0 < self.layerwise_lr_decay <= 1.0:
            raise ValueError("transformer optimizer parameters are invalid")
        if self.train_last_n_layers is not None and self.train_last_n_layers < 1:
            raise ValueError("transformer optimizer parameters are invalid")
        if (
            self.special_token_initialization.enabled
            and not self.pair_encoding.use_field_tokens
        ):
            raise ValueError(
                "special token initialization requires pair_encoding.use_field_tokens"
            )
        if self.lr_scheduler_type not in {"linear", "cosine"}:
            raise ValueError(
                "transformer.lr_scheduler_type must be 'linear' or 'cosine'"
            )
        if not 0.0 < self.hpo_learning_rate_min < self.hpo_learning_rate_max:
            raise ValueError("transformer HPO learning-rate bounds are invalid")
        if not 0.0 <= self.hpo_weight_decay_min < self.hpo_weight_decay_max:
            raise ValueError("transformer HPO weight-decay bounds are invalid")
        if min(
            self.train_batch_size,
            self.eval_batch_size,
            self.gradient_accumulation_steps,
            self.early_stopping_patience,
        ) < 1:
            raise ValueError("transformer batch and patience values must be positive")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("transformer.warmup_ratio must be in [0, 1)")
        if self.max_grad_norm <= 0.0:
            raise ValueError("transformer.max_grad_norm must be positive")


@dataclass(frozen=True, slots=True)
class MaxPoolingParameters:
    artifact_path: Path
    vector_size: int
    window: int
    min_count: int
    workers: int
    fasttext_epochs: int
    classifier_epochs: int
    batch_size: int
    patience: int
    dropout: float
    learning_rate: float
    weight_decay: float


@dataclass(frozen=True, slots=True)
class FusionParameters:
    artifact_path: Path
    embedding_batch_size: int
    hidden_dim: int
    dropout: float
    batch_size: int
    max_epochs: int
    patience: int
    learning_rate: float
    weight_decay: float


@dataclass(frozen=True, slots=True)
class BoostingParameters:
    artifact_dir: Path
    iterations: int
    depth: int
    learning_rate: float
    loss_function: str
    early_stopping_rounds: int
    thread_count: int

    def __post_init__(self) -> None:
        if self.iterations < 1 or self.depth < 1:
            raise ValueError("boosting iterations and depth must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("boosting.learning_rate must be positive")
        if self.early_stopping_rounds < 1 or self.thread_count == 0:
            raise ValueError(
                "boosting early_stopping_rounds must be positive and "
                "thread_count must not be zero"
            )


@dataclass(frozen=True, slots=True)
class CascadeParameters:
    fast_model: str
    main_model: str
    negative_threshold: float
    positive_threshold: float

    def __post_init__(self) -> None:
        if self.fast_model != "boosting":
            raise ValueError("cascade.fast_model currently must be 'boosting'")
        if self.main_model != "transformer":
            raise ValueError("cascade.main_model currently must be 'transformer'")
        if not 0.0 <= self.negative_threshold < self.positive_threshold <= 1.0:
            raise ValueError(
                "cascade thresholds must satisfy 0 <= negative < positive <= 1"
            )


@dataclass(frozen=True, slots=True)
class StackingParameters:
    artifact_dir: Path
    base_model: str
    stacking_model: str

    def __post_init__(self) -> None:
        if self.base_model != "transformer":
            raise ValueError("stacking.base_model currently must be 'transformer'")
        if self.stacking_model != "boosting":
            raise ValueError("stacking.stacking_model currently must be 'boosting'")


@dataclass(frozen=True, slots=True)
class ModelDescriptionSettings:
    transformer: TransformerParameters
    maxpooling: MaxPoolingParameters
    fusion: FusionParameters
    boosting: BoostingParameters
    cascade: CascadeParameters
    stacking: StackingParameters


@dataclass(frozen=True, slots=True)
class AttributeImportanceAnalysisSettings:
    model: str
    output_dir: Path
    sample_size: int | None
    group_by_category: bool
    min_occurrences: int
    score_type: str
    output_file: str
    metadata_file: str

    def __post_init__(self) -> None:
        if self.model != "transformer":
            raise ValueError(
                "attribute_importance.model currently must be 'transformer'"
            )
        if self.sample_size is not None and self.sample_size < 1:
            raise ValueError(
                "attribute_importance.sample_size must be positive or null"
            )
        if self.min_occurrences < 1:
            raise ValueError(
                "attribute_importance.min_occurrences must be positive"
            )
        if self.score_type != "normalized_mean_attention":
            raise ValueError(
                "attribute_importance.score_type currently must be "
                "'normalized_mean_attention'"
            )
        for name, value in (
            ("output_file", self.output_file),
            ("metadata_file", self.metadata_file),
        ):
            path = Path(value)
            if not value.strip() or path.is_absolute() or path.name != value:
                raise ValueError(
                    f"attribute_importance.{name} must be a relative file name"
                )


@dataclass(frozen=True, slots=True)
class BackendBenchmarkSettings:
    tests_path: Path
    selected_tests: tuple[str, ...] | None
    sample_size: int | None
    warmup_batches: int
    measured_runs: int
    reference_test: str
    compare_predictions: bool
    output_dir: Path

    def __post_init__(self) -> None:
        if self.sample_size is not None and self.sample_size < 1:
            raise ValueError(
                "backend_benchmark.sample_size must be positive or null"
            )
        if self.warmup_batches < 0:
            raise ValueError(
                "backend_benchmark.warmup_batches must not be negative"
            )
        if self.measured_runs < 1:
            raise ValueError(
                "backend_benchmark.measured_runs must be positive"
            )
        if not self.reference_test.strip():
            raise ValueError(
                "backend_benchmark.reference_test must not be empty"
            )
        if self.selected_tests is not None:
            if not self.selected_tests:
                raise ValueError(
                    "backend_benchmark.selected_tests must not be empty; "
                    "use null to select every test"
                )
            if len(set(self.selected_tests)) != len(self.selected_tests):
                raise ValueError(
                    "backend_benchmark.selected_tests must be unique"
                )


@dataclass(frozen=True, slots=True)
class BenchmarkJobSettings:
    name: str
    type: str
    enabled: bool
    tests_path: Path
    selected_tests: tuple[str, ...] | None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("benchmark job name must not be empty")
        if self.type not in {"inference", "training_quality"}:
            raise ValueError(
                "benchmark job type must be inference or training_quality"
            )
        if self.selected_tests is not None:
            if not self.selected_tests:
                raise ValueError(
                    "benchmark job selected_tests must not be empty; use null"
                )
            if len(set(self.selected_tests)) != len(self.selected_tests):
                raise ValueError("benchmark job selected_tests must be unique")


@dataclass(frozen=True, slots=True)
class BenchmarkSettings:
    base_config_path: Path
    output_dir: Path
    fail_fast: bool
    jobs: tuple[BenchmarkJobSettings, ...]

    def __post_init__(self) -> None:
        if not self.jobs:
            raise ValueError("benchmark.jobs must not be empty")
        names = [job.name for job in self.jobs]
        if len(names) != len(set(names)):
            raise ValueError("benchmark job names must be unique")


@dataclass(frozen=True, slots=True)
class AnalysisModelsSettings:
    attribute_importance: AttributeImportanceAnalysisSettings
    backend_benchmark: BackendBenchmarkSettings | None = None


@dataclass(frozen=True, slots=True)
class AttributeShuffleSettings:
    shuffled_copies: int
    keep_original: bool
    seed: int
    shuffle_cards_independently: bool
    skip_oversized: bool

    def __post_init__(self) -> None:
        if self.shuffled_copies < 1:
            raise ValueError("attribute_shuffle.shuffled_copies must be positive")


@dataclass(frozen=True, slots=True)
class AttributeWordDropoutSettings:
    pair_probability: float
    attribute_dropout_probability: float
    word_dropout_probability: float
    keyboard_typo_probability: float
    word_shuffle_probability: float
    seed: int

    def __post_init__(self) -> None:
        for name, probability in (
            ("pair_probability", self.pair_probability),
            ("attribute_dropout_probability", self.attribute_dropout_probability),
            ("word_dropout_probability", self.word_dropout_probability),
            ("keyboard_typo_probability", self.keyboard_typo_probability),
            ("word_shuffle_probability", self.word_shuffle_probability),
        ):
            if not 0.0 <= probability <= 1.0:
                raise ValueError(
                    f"attribute_word_dropout.{name} must be in [0, 1]"
                )


@dataclass(frozen=True, slots=True)
class AugmentationModelsSettings:
    attribute_shuffle: AttributeShuffleSettings
    attribute_word_dropout: AttributeWordDropoutSettings


@dataclass(frozen=True, slots=True)
class AttributeSortSettings:
    priorities_path: Path


@dataclass(frozen=True, slots=True)
class DataPostprocessingModelsSettings:
    attribute_sort: AttributeSortSettings


@dataclass(frozen=True, slots=True)
class NerSettings:
    enabled: bool
    provider: str | None
    model_dir: Path | None
    cluster_centers_path: Path | None
    source_column: str
    output_column: str
    enriched_column: str
    merge_policy: str
    batch_size: int
    max_length: int
    use_amp: bool
    semantic_cleanup: bool

    def __post_init__(self) -> None:
        if self.provider not in {None, "word_ner"}:
            raise ValueError("features.ner.provider must be 'word_ner' or null")
        if self.merge_policy != "missing_only":
            raise ValueError("features.ner.merge_policy must be 'missing_only'")
        if not self.source_column.strip():
            raise ValueError("features.ner.source_column must not be empty")
        if not self.output_column.strip() or not self.enriched_column.strip():
            raise ValueError("features.ner output columns must not be empty")
        if self.output_column == self.enriched_column:
            raise ValueError("NER output and enriched columns must be different")
        if self.batch_size < 1 or self.max_length < 8:
            raise ValueError("NER batch_size must be positive and max_length at least 8")
        if self.enabled:
            if self.provider != "word_ner":
                raise ValueError("enabled NER requires provider='word_ner'")
            if self.model_dir is None:
                raise ValueError("enabled NER requires features.ner.model_dir")
            if self.semantic_cleanup and self.cluster_centers_path is None:
                raise ValueError(
                    "NER semantic cleanup requires cluster_centers_path"
                )


@dataclass(frozen=True, slots=True)
class PhysicalFeatureSettings:
    enabled: bool
    source_column: str
    output_column: str
    enriched_column: str
    merge_policy: str
    normalize_units: bool
    n_jobs: int
    chunk_size: int

    def __post_init__(self) -> None:
        if not self.source_column.strip():
            raise ValueError("features.physical.source_column must not be empty")
        if not self.output_column.strip() or not self.enriched_column.strip():
            raise ValueError("features.physical output columns must not be empty")
        if self.output_column == self.enriched_column:
            raise ValueError("physical output and enriched columns must be different")
        if self.merge_policy != "missing_only":
            raise ValueError(
                "features.physical.merge_policy must be 'missing_only'"
            )
        if self.n_jobs == 0 or self.chunk_size < 1:
            raise ValueError(
                "features.physical n_jobs must not be zero and chunk_size positive"
            )


@dataclass(frozen=True, slots=True)
class FeatureSettings:
    execution_order: tuple[str, ...]
    normalization: NormalizationSettings
    ner: NerSettings
    physical: PhysicalFeatureSettings

    def __post_init__(self) -> None:
        normalized = tuple(name.strip().lower() for name in self.execution_order)
        if len(set(normalized)) != len(normalized):
            raise ValueError("features.execution_order must not contain duplicates")
        expected = set(FEATURE_PROVIDER_NAMES)
        actual = set(normalized)
        if actual != expected:
            details = []
            missing = expected - actual
            unknown = actual - expected
            if missing:
                details.append("missing: " + ", ".join(sorted(missing)))
            if unknown:
                details.append("unknown: " + ", ".join(sorted(unknown)))
            raise ValueError(
                "features.execution_order must contain every feature exactly once"
                + (" (" + "; ".join(details) + ")" if details else "")
            )
        object.__setattr__(self, "execution_order", normalized)


@dataclass(frozen=True, slots=True)
class PairFeatureSettings:
    """Features that can only be computed after two cards are paired."""

    typed_attributes: TypedAttributeOptions = TypedAttributeOptions()


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    device: str | None
    seed: int


@dataclass(frozen=True, slots=True)
class LoggingSettings:
    level: str
    file: Path | None
    rotation: str


@dataclass(frozen=True, slots=True)
class SubmissionSettings:
    output_path: Path


@dataclass(frozen=True, slots=True)
class AppConfig:
    training: TrainingSettings
    data_model_description: DataModelDescriptionSettings
    inference: InferenceSettings
    analysis: AnalysisSettings
    labeling: DatasetLabelingSettings
    analysis_models: AnalysisModelsSettings
    augmentation_models: AugmentationModelsSettings
    data_postprocessing_models: DataPostprocessingModelsSettings
    model_description: ModelDescriptionSettings
    features: FeatureSettings
    pair_features: PairFeatureSettings
    runtime: RuntimeSettings
    logging: LoggingSettings
    submission: SubmissionSettings
    benchmark: BenchmarkSettings | None


if TYPE_CHECKING:
    ConfigSource = AppConfig | DictConfig | Mapping[str, Any]
else:
    ConfigSource = Any


def _section(values: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    section = values.get(name)
    if not isinstance(section, Mapping):
        raise ValueError(f"config section {name!r} must be a mapping")
    return section


def _required(values: Mapping[str, Any], name: str) -> Any:
    value = values.get(name)
    if value is None:
        raise ValueError(f"config value {name!r} is required")
    return value


def _path(value: Any, name: str) -> Path:
    if value is None or not str(value).strip():
        raise ValueError(f"config value {name!r} must contain a path")
    return resolve_project_path(str(value))


def _optional_path(value: Any) -> Path | None:
    if value is None or not str(value).strip():
        return None
    return resolve_project_path(str(value))


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _sequence(value: Any, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"config field {name!r} must be a sequence")
    return value


def _string_tuple(value: Any, name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"config value {name!r} must be a sequence of strings")
    texts = tuple(value)
    if any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError(f"config value {name!r} must contain non-empty strings")
    return texts


def _int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"config value {name!r} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"config value {name!r} must be an integer") from error


def _bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    raise ValueError(f"config value {name!r} must be a boolean")


def _tensorrt_shared_settings(
    value: Mapping[str, Any],
    *,
    prefix: str,
    default_engine_cache_path: str,
    default_opt_batch_size: int,
    default_max_batch_size: int,
    default_sequence_lengths: tuple[int, ...],
) -> tuple[
    TensorRTEngineCacheSettings,
    TensorRTTimingCacheSettings,
    TensorRTProfileSettings,
]:
    engine_cache = value.get("engine_cache", {})
    timing_cache = value.get("timing_cache", {})
    profiles = value.get("profiles", {})
    for name, section in (
        ("engine_cache", engine_cache),
        ("timing_cache", timing_cache),
        ("profiles", profiles),
    ):
        if not isinstance(section, Mapping):
            raise ValueError(
                f"config section '{prefix}.{name}' must be a mapping"
            )
    lengths_value = profiles.get("sequence_lengths", default_sequence_lengths)
    if not isinstance(lengths_value, Sequence) or isinstance(
        lengths_value,
        (str, bytes),
    ):
        raise ValueError(
            f"config field '{prefix}.profiles.sequence_lengths' must be a sequence"
        )
    if not lengths_value:
        raise ValueError(
            f"config field '{prefix}.profiles.sequence_lengths' must not be empty"
        )
    timing_path = timing_cache.get("path")
    return (
        TensorRTEngineCacheSettings(
            enabled=_bool(
                engine_cache.get("enabled", True),
                f"{prefix}.engine_cache.enabled",
            ),
            path=str(engine_cache.get("path", default_engine_cache_path)),
        ),
        TensorRTTimingCacheSettings(
            enabled=_bool(
                timing_cache.get("enabled", True),
                f"{prefix}.timing_cache.enabled",
            ),
            path=None if timing_path is None else str(timing_path),
        ),
        TensorRTProfileSettings(
            min_batch_size=_int(
                profiles.get("min_batch_size", 1),
                f"{prefix}.profiles.min_batch_size",
            ),
            opt_batch_size=_int(
                profiles.get("opt_batch_size", default_opt_batch_size),
                f"{prefix}.profiles.opt_batch_size",
            ),
            max_batch_size=_int(
                profiles.get("max_batch_size", default_max_batch_size),
                f"{prefix}.profiles.max_batch_size",
            ),
            min_sequence_length=int(lengths_value[0]),
            opt_sequence_length=int(lengths_value[len(lengths_value) // 2]),
            max_sequence_length=int(lengths_value[-1]),
        ),
    )


def _ort_tensorrt_settings(
    onnxruntime: Mapping[str, Any],
) -> OrtTensorRTProviderSettings:
    value = onnxruntime.get("tensorrt", {})
    if not isinstance(value, Mapping):
        raise ValueError(
            "config section 'inference.transformer.onnxruntime.tensorrt' "
            "must be a mapping"
        )
    engine_cache, timing_cache, profiles = _tensorrt_shared_settings(
        value,
        prefix="inference.transformer.onnxruntime.tensorrt",
        default_engine_cache_path="onnx/trt_cache",
        default_opt_batch_size=2048,
        default_max_batch_size=2048,
        default_sequence_lengths=(64, 96, 128),
    )
    return OrtTensorRTProviderSettings(
        engine_cache=engine_cache,
        timing_cache=timing_cache,
        profiles=profiles,
    )


def _native_tensorrt_settings(value: Mapping[str, Any]) -> NativeTensorRTSettings:
    engine_cache, timing_cache, profiles = _tensorrt_shared_settings(
        value,
        prefix="inference.transformer.tensorrt",
        default_engine_cache_path="onnx/native_trt_cache",
        default_opt_batch_size=512,
        default_max_batch_size=512,
        default_sequence_lengths=(1, 256, 472),
    )
    return NativeTensorRTSettings(
        device_id=_int(
            value.get("device_id", 0),
            "inference.transformer.tensorrt.device_id",
        ),
        workspace_size_gb=float(value.get("workspace_size_gb", 8.0)),
        builder_optimization_level=_int(
            value.get("builder_optimization_level", 3),
            "inference.transformer.tensorrt.builder_optimization_level",
        ),
        fallback_to_onnxruntime=_bool(
            value.get("fallback_to_onnxruntime", True),
            "inference.transformer.tensorrt.fallback_to_onnxruntime",
        ),
        engine_cache=engine_cache,
        timing_cache=timing_cache,
        profiles=profiles,
    )


def _dataset_splitter(
    values: Mapping[str, Any],
) -> DatasetSplitterSettings:
    score_type = str(values.get("score_type", "label"))
    total_votes = _optional_int(values.get("total_votes"))
    return DatasetSplitterSettings(
        splitter_type=str(values.get("splitter_type", "binary")),
        score_type=score_type,
        total_votes=total_votes,
        negative_threshold=float(values.get("negative_threshold", 0)),
        positive_threshold=float(values.get("positive_threshold", 1)),
        uncertain_action=str(values.get("uncertain_action", "drop")),
        target_mode=str(values.get("target_mode", "hard")),
    )


def _dataset_source(
    name: str,
    values: Mapping[str, Any],
    *,
    dataset_name: str,
) -> DatasetSourceSettings:
    prefix = f"data_model_description.{dataset_name}.sources.{name}"
    splitter = values.get("splitter", {})
    if not isinstance(splitter, Mapping):
        raise ValueError(f"{prefix}.splitter must be a mapping")
    weight_model = values.get("weight_model", {})
    if not isinstance(weight_model, Mapping):
        raise ValueError(f"{prefix}.weight_model must be a mapping")
    confidence_weighting = values.get("confidence_weighting", {})
    if not isinstance(confidence_weighting, Mapping):
        raise ValueError(f"{prefix}.confidence_weighting must be a mapping")
    return DatasetSourceSettings(
        name=name,
        matches=_path(_required(values, "matches"), f"{prefix}.matches"),
        weight=float(values.get("weight", 1.0)),
        max_rows=_optional_int(values.get("max_rows")),
        sampling_strategy=str(values.get("sampling_strategy", "random")),
        splitter=_dataset_splitter(splitter),
        weight_model=SampleWeightModelSettings(
            type=str(weight_model.get("type", "constant")),
            enabled=_bool(
                weight_model.get("enabled", False),
                f"{prefix}.weight_model.enabled",
            ),
            penalty_strength=float(weight_model.get("penalty_strength", 1.0)),
            min_weight_multiplier=float(
                weight_model.get("min_weight_multiplier", 0.25)
            ),
            min_comparable_neighbors=_int(
                weight_model.get("min_comparable_neighbors", 2),
                f"{prefix}.weight_model.min_comparable_neighbors",
            ),
            confidence_weighted_violations=_bool(
                weight_model.get("confidence_weighted_violations", True),
                f"{prefix}.weight_model.confidence_weighted_violations",
            ),
        ),
        confidence_weighting=ConfidenceWeightingSettings(
            enabled=_bool(
                confidence_weighting.get("enabled", False),
                f"{prefix}.confidence_weighting.enabled",
            ),
            method=str(
                confidence_weighting.get(
                    "method",
                    "distance_from_midpoint",
                )
            ),
            min_weight_multiplier=float(
                confidence_weighting.get("min_weight_multiplier", 0.2)
            ),
            power=float(confidence_weighting.get("power", 1.0)),
        ),
        confidence_power=float(values.get("confidence_power", 2.0)),
        target_column=str(values.get("target_column", "target")),
        items=(
            None
            if values.get("items") is None
            else _path(values["items"], f"{prefix}.items")
        ),
    )


def _dataset_sources(
    values: Any,
    *,
    dataset_name: str,
) -> tuple[DatasetSourceSettings, ...]:
    if isinstance(values, Mapping):
        entries = values.items()
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        normalized: list[tuple[str, Mapping[str, Any]]] = []
        for value in values:
            if not isinstance(value, Mapping):
                raise ValueError(f"{dataset_name} sources must contain mappings")
            normalized.append((str(_required(value, "name")), value))
        entries = normalized
    else:
        raise ValueError(f"{dataset_name} sources must be a mapping or sequence")
    sources: list[DatasetSourceSettings] = []
    for name, source in entries:
        if not isinstance(source, Mapping):
            raise ValueError(f"{dataset_name} source {name!r} must be a mapping")
        sources.append(
            _dataset_source(
                str(name),
                source,
                dataset_name=dataset_name,
            )
        )
    return tuple(sources)


def _mixed_dataset_settings(
    values: Mapping[str, Any],
    *,
    dataset_name: str,
    runtime_seed: int,
) -> MixedDatasetSettings:
    prefix = f"data_model_description.{dataset_name}"
    overlap = values.get("overlap_resolution", {})
    if not isinstance(overlap, Mapping):
        raise ValueError(
            f"config section '{prefix}.overlap_resolution' must be a mapping"
        )
    return MixedDatasetSettings(
        items=_path(_required(values, "items"), f"{prefix}.items"),
        sources=_dataset_sources(
            _required(values, "sources"),
            dataset_name=dataset_name,
        ),
        validation_source=str(_required(values, "validation_source")),
        validation_fraction=float(_required(values, "validation_fraction")),
        leakage_scope=str(_required(values, "leakage_scope")),
        candidate_splits=int(values.get("candidate_splits", 128)),
        seed=int(values.get("seed", runtime_seed)),
        overlap_resolution=DatasetOverlapResolutionSettings(
            enabled=_bool(
                overlap.get("enabled", False),
                f"{prefix}.overlap_resolution.enabled",
            ),
            source_priority=tuple(
                str(name)
                for name in _sequence(
                    overlap.get("source_priority", []),
                    f"{prefix}.overlap_resolution.source_priority",
                )
            ),
        ),
    )


def _benchmark_jobs(values: Any) -> tuple[BenchmarkJobSettings, ...]:
    if isinstance(values, Mapping):
        entries = values.items()
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        normalized: list[tuple[str, Mapping[str, Any]]] = []
        for value in values:
            if not isinstance(value, Mapping):
                raise ValueError("benchmark jobs must contain mappings")
            normalized.append((str(_required(value, "name")), value))
        entries = normalized
    else:
        raise ValueError(
            "config section 'benchmark.jobs' must be a mapping or sequence"
        )
    jobs: list[BenchmarkJobSettings] = []
    for raw_name, raw_job in entries:
        name = str(raw_name)
        if not isinstance(raw_job, Mapping):
            raise ValueError(f"benchmark job {name!r} must be a mapping")
        selected_value = raw_job.get("selected_tests")
        selected = (
            None
            if selected_value is None
            else tuple(
                str(test_name)
                for test_name in _sequence(
                    selected_value,
                    f"benchmark.jobs.{name}.selected_tests",
                )
            )
        )
        jobs.append(
            BenchmarkJobSettings(
                name=name,
                type=str(_required(raw_job, "type")).lower(),
                enabled=_bool(
                    raw_job.get("enabled", True),
                    f"benchmark.jobs.{name}.enabled",
                ),
                tests_path=_path(
                    _required(raw_job, "tests_path"),
                    f"benchmark.jobs.{name}.tests_path",
                ),
                selected_tests=selected,
            )
        )
    return tuple(jobs)


def load_app_config(config: ConfigSource) -> AppConfig:
    """Resolve OmegaConf values once and return immutable typed settings."""
    from omegaconf import DictConfig, OmegaConf

    if isinstance(config, AppConfig):
        return config
    omega = config if isinstance(config, DictConfig) else OmegaConf.create(config)
    resolved = OmegaConf.to_container(omega, resolve=True)
    if not isinstance(resolved, Mapping):
        raise ValueError("application config must be a mapping")

    training = _section(resolved, "training")
    data_model_description = _section(resolved, "data_model_description")
    base_dataset = _section(data_model_description, "base_dataset")
    mix_dataset = _section(data_model_description, "mix_dataset")
    mix_dataset_hard_negative = _section(
        data_model_description,
        "mix_dataset_hard_negative",
    )
    mix_dataset_codex = _section(data_model_description, "mix_dataset_codex")
    mix_dataset_neural_review = _section(
        data_model_description,
        "mix_dataset_neural_review",
    )
    inference_value = resolved.get("inference", training)
    if not isinstance(inference_value, Mapping):
        raise ValueError("config section 'inference' must be a mapping")
    inference = inference_value
    inference_transformer = _section(inference, "transformer")
    length_bucketing_value = inference_transformer.get("length_bucketing", True)
    if isinstance(length_bucketing_value, Mapping):
        padding_length_buckets_value = length_bucketing_value.get(
            "padding_length_buckets"
        )
        if padding_length_buckets_value is None:
            padding_length_buckets = None
        elif isinstance(padding_length_buckets_value, Sequence) and not isinstance(
            padding_length_buckets_value, (str, bytes)
        ):
            padding_length_buckets = tuple(padding_length_buckets_value)
        else:
            raise ValueError(
                "config field 'inference.transformer.length_bucketing."
                "padding_length_buckets' must be a sequence or null"
            )
        length_bucketing = LengthBucketingSettings(
            enabled=_bool(
                length_bucketing_value.get("enabled", True),
                "inference.transformer.length_bucketing.enabled",
            ),
            padding_length_buckets=padding_length_buckets,
        )
    else:
        # Backward compatibility with manifests/configs that used a boolean.
        length_bucketing = LengthBucketingSettings(
            enabled=_bool(
                length_bucketing_value,
                "inference.transformer.length_bucketing",
            )
        )
    torch_compile_value = inference_transformer.get("torch_compile", {})
    if not isinstance(torch_compile_value, Mapping):
        raise ValueError(
            "config section 'inference.transformer.torch_compile' must be a mapping"
        )
    inference_attention_value = inference_transformer.get("attention", {})
    if not isinstance(inference_attention_value, Mapping):
        raise ValueError(
            "config section 'inference.transformer.attention' must be a mapping"
        )
    onnxruntime_value = inference_transformer.get("onnxruntime", {})
    if not isinstance(onnxruntime_value, Mapping):
        raise ValueError(
            "config section 'inference.transformer.onnxruntime' must be a mapping"
        )
    tensorrt_value = inference_transformer.get("tensorrt", {})
    if not isinstance(tensorrt_value, Mapping):
        raise ValueError(
            "config section 'inference.transformer.tensorrt' must be a mapping"
        )
    analysis = _section(resolved, "analysis")
    labeling = _section(resolved, "labeling")
    labeling_llm = _section(labeling, "llm")
    analysis_models = _section(resolved, "analysis_models")
    attribute_importance = _section(
        analysis_models,
        "attribute_importance",
    )
    backend_benchmark_value = analysis_models.get("backend_benchmark")
    if backend_benchmark_value is not None and not isinstance(
        backend_benchmark_value,
        Mapping,
    ):
        raise ValueError(
            "config section 'analysis_models.backend_benchmark' must be a mapping"
        )
    augmentation_models = _section(resolved, "augmentation_models")
    attribute_shuffle = _section(augmentation_models, "attribute_shuffle")
    attribute_word_dropout = _section(
        augmentation_models,
        "attribute_word_dropout",
    )
    data_postprocessing_models = _section(
        resolved,
        "data_postprocessing_models",
    )
    attribute_sort = _section(data_postprocessing_models, "attribute_sort")
    model_description = _section(resolved, "model_description")
    transformer = _section(model_description, "transformer")
    transformer_profile = str(
        transformer.get("profile", SEQUENCE_CLASSIFIER_PROFILE)
    ).strip().lower()
    tokenizer_value = transformer.get("tokenizer", {})
    if not isinstance(tokenizer_value, Mapping):
        raise ValueError("config section 'transformer.tokenizer' must be a mapping")
    batch_fields_value = tokenizer_value.get("batch_fields", {})
    if not isinstance(batch_fields_value, Mapping):
        raise ValueError(
            "config section 'transformer.tokenizer.batch_fields' must be a mapping"
        )
    transformer_export_value = transformer.get("export", {})
    if not isinstance(transformer_export_value, Mapping):
        raise ValueError("config section 'transformer.export' must be a mapping")
    onnx_export_value = transformer_export_value.get("onnx", {})
    if not isinstance(onnx_export_value, Mapping):
        raise ValueError(
            "config section 'transformer.export.onnx' must be a mapping"
        )
    encoding = _section(transformer, "pair_encoding")
    special_token_initialization_value = transformer.get(
        "special_token_initialization", {}
    )
    if not isinstance(special_token_initialization_value, Mapping):
        raise ValueError(
            "config section 'transformer.special_token_initialization' must be a mapping"
        )
    validation_value = transformer.get("validation", {})
    if not isinstance(validation_value, Mapping):
        raise ValueError("config section 'transformer.validation' must be a mapping")
    fast_dev_validation_value = validation_value.get("fast_dev", {})
    if not isinstance(fast_dev_validation_value, Mapping):
        raise ValueError(
            "config section 'transformer.validation.fast_dev' must be a mapping"
        )
    transformer_head_value = transformer.get("head", {})
    if not isinstance(transformer_head_value, Mapping):
        raise ValueError("config section 'transformer.head' must be a mapping")
    transformer_head = transformer_head_value
    training_runtime_value = transformer.get("training_runtime", {})
    if not isinstance(training_runtime_value, Mapping):
        raise ValueError(
            "config section 'transformer.training_runtime' must be a mapping"
        )
    training_attention_value = training_runtime_value.get("attention", {})
    if not isinstance(training_attention_value, Mapping):
        raise ValueError(
            "config section 'transformer.training_runtime.attention' "
            "must be a mapping"
        )
    training_length_bucketing_value = training_runtime_value.get(
        "length_bucketing", {}
    )
    if not isinstance(training_length_bucketing_value, Mapping):
        raise ValueError(
            "config section 'transformer.training_runtime.length_bucketing' "
            "must be a mapping"
        )
    training_padding_buckets_value = training_length_bucketing_value.get(
        "padding_length_buckets"
    )
    if training_padding_buckets_value is None:
        training_padding_buckets = None
    elif isinstance(training_padding_buckets_value, Sequence) and not isinstance(
        training_padding_buckets_value, (str, bytes)
    ):
        training_padding_buckets = tuple(training_padding_buckets_value)
    else:
        raise ValueError(
            "config field 'transformer.training_runtime.length_bucketing."
            "padding_length_buckets' must be a sequence or null"
        )
    training_dataloader_value = training_runtime_value.get("dataloader", {})
    if not isinstance(training_dataloader_value, Mapping):
        raise ValueError(
            "config section 'transformer.training_runtime.dataloader' "
            "must be a mapping"
        )
    performance_logging_value = training_runtime_value.get(
        "performance_logging", {}
    )
    if not isinstance(performance_logging_value, Mapping):
        raise ValueError(
            "config section 'transformer.training_runtime.performance_logging' "
            "must be a mapping"
        )
    training_token_cache_value = training_runtime_value.get("token_cache", {})
    if not isinstance(training_token_cache_value, Mapping):
        raise ValueError(
            "config section 'transformer.training_runtime.token_cache' "
            "must be a mapping"
        )
    training_torch_compile_value = training_runtime_value.get("torch_compile", {})
    if not isinstance(training_torch_compile_value, Mapping):
        raise ValueError(
            "config section 'transformer.training_runtime.torch_compile' "
            "must be a mapping"
        )
    maxpooling = _section(model_description, "maxpooling")
    fusion = _section(model_description, "fusion")
    boosting = _section(model_description, "boosting")
    cascade = _section(model_description, "cascade")
    stacking = _section(model_description, "stacking")
    features = _section(resolved, "features")
    normalization_value = features.get(
        "normalization",
        resolved.get("normalization"),
    )
    if not isinstance(normalization_value, Mapping):
        raise ValueError("config section 'features.normalization' must be a mapping")
    normalization = normalization_value
    execution_order = tuple(
        str(name)
        for name in features.get("execution_order", FEATURE_PROVIDER_NAMES)
    )
    ner = _section(features, "ner")
    physical = _section(features, "physical")
    pair_features_value = resolved.get("pair_features", {})
    if not isinstance(pair_features_value, Mapping):
        raise ValueError("config section 'pair_features' must be a mapping")
    typed_attributes_value = pair_features_value.get("typed_attributes", {})
    if not isinstance(typed_attributes_value, Mapping):
        raise ValueError(
            "config section 'pair_features.typed_attributes' must be a mapping"
        )
    typed_attribute_types = typed_attributes_value.get("types", {})
    if not isinstance(typed_attribute_types, Mapping):
        raise ValueError(
            "config section 'pair_features.typed_attributes.types' must be a mapping"
        )
    runtime = _section(resolved, "runtime")
    logging = _section(resolved, "logging")
    submission_value = resolved.get("submission", {})
    if not isinstance(submission_value, Mapping):
        raise ValueError("config section 'submission' must be a mapping")
    submission = submission_value
    benchmark_value = resolved.get("benchmark")
    if benchmark_value is not None and not isinstance(benchmark_value, Mapping):
        raise ValueError("config section 'benchmark' must be a mapping")

    return AppConfig(
        training=TrainingSettings(
            model=str(_required(training, "model")),
            data_model=str(_required(training, "data_model")),
            augmentation_model=(
                None
                if training.get("augmentation_model") is None
                else str(training["augmentation_model"])
            ),
            data_postprocessing_model=(
                None
                if training.get("data_postprocessing_model") is None
                else str(training["data_postprocessing_model"])
            ),
            resolved_config_path=_path(
                _required(training, "resolved_config_path"),
                "training.resolved_config_path",
            ),
            solution_path=_path(
                _required(training, "solution_path"),
                "training.solution_path",
            ),
        ),
        data_model_description=DataModelDescriptionSettings(
            base_dataset=BaseDatasetSettings(
                items=_path(
                    _required(base_dataset, "items"),
                    "data_model_description.base_dataset.items",
                ),
                matches=_path(
                    _required(base_dataset, "matches"),
                    "data_model_description.base_dataset.matches",
                ),
                validation_fraction=float(
                    _required(base_dataset, "validation_fraction")
                ),
                stacking_train_fraction=float(
                    _required(base_dataset, "stacking_train_fraction")
                ),
                leakage_scope=str(_required(base_dataset, "leakage_scope")),
                candidate_splits=int(
                    base_dataset.get("candidate_splits", 128)
                ),
                seed=int(base_dataset.get("seed", _required(runtime, "seed"))),
            ),
            mix_dataset=_mixed_dataset_settings(
                mix_dataset,
                dataset_name="mix_dataset",
                runtime_seed=int(_required(runtime, "seed")),
            ),
            mix_dataset_hard_negative=_mixed_dataset_settings(
                mix_dataset_hard_negative,
                dataset_name="mix_dataset_hard_negative",
                runtime_seed=int(_required(runtime, "seed")),
            ),
            mix_dataset_codex=_mixed_dataset_settings(
                mix_dataset_codex,
                dataset_name="mix_dataset_codex",
                runtime_seed=int(_required(runtime, "seed")),
            ),
            mix_dataset_neural_review=_mixed_dataset_settings(
                mix_dataset_neural_review,
                dataset_name="mix_dataset_neural_review",
                runtime_seed=int(_required(runtime, "seed")),
            ),
        ),
        inference=InferenceSettings(
            model=str(_required(inference, "model")),
            augmentation_model=(
                None
                if inference.get("augmentation_model") is None
                else str(inference["augmentation_model"])
            ),
            data_postprocessing_model=(
                None
                if inference.get("data_postprocessing_model") is None
                else str(inference["data_postprocessing_model"])
            ),
            solution_path=_path(
                _required(inference, "solution_path"),
                "inference.solution_path",
            ),
            transformer=TransformerInferenceSettings(
                batch_size=int(_required(inference_transformer, "batch_size")),
                dtype=str(_required(inference_transformer, "dtype")).lower(),
                max_length=_optional_int(
                    inference_transformer.get("max_length")
                ),
                max_attribute_value_chars=_optional_int(
                    inference_transformer.get("max_attribute_value_chars")
                ),
                max_attribute_value_tokens=_optional_int(
                    inference_transformer.get("max_attribute_value_tokens")
                ),
                backend=str(
                    inference_transformer.get("backend", "pytorch")
                ).lower(),
                num_workers=int(inference_transformer.get("num_workers", 0)),
                prefetch_factor=int(
                    inference_transformer.get("prefetch_factor", 2)
                ),
                pin_memory=_bool(
                    inference_transformer.get("pin_memory", True),
                    "inference.transformer.pin_memory",
                ),
                non_blocking_transfer=_bool(
                    inference_transformer.get("non_blocking_transfer", True),
                    "inference.transformer.non_blocking_transfer",
                ),
                attention=AttentionSettings(
                    implementation=str(
                        inference_attention_value.get("implementation", "auto")
                    )
                ),
                length_bucketing=length_bucketing,
                torch_compile=TorchCompileSettings(
                    enabled=_bool(
                        torch_compile_value.get("enabled", False),
                        "inference.transformer.torch_compile.enabled",
                    ),
                    mode=str(
                        torch_compile_value.get("mode", "reduce-overhead")
                    ),
                    dynamic=_bool(
                        torch_compile_value.get("dynamic", True),
                        "inference.transformer.torch_compile.dynamic",
                    ),
                ),
                onnxruntime=OnnxRuntimeSettings(
                    provider=str(
                        onnxruntime_value.get("provider", "cuda")
                    ).lower(),
                    device_id=_int(
                        onnxruntime_value.get("device_id", 0),
                        "inference.transformer.onnxruntime.device_id",
                    ),
                    io_binding=_bool(
                        onnxruntime_value.get("io_binding", True),
                        "inference.transformer.onnxruntime.io_binding",
                    ),
                    graph_optimization=str(
                        onnxruntime_value.get("graph_optimization", "all")
                    ).lower(),
                    fallback_to_pytorch=_bool(
                        onnxruntime_value.get("fallback_to_pytorch", True),
                        "inference.transformer.onnxruntime.fallback_to_pytorch",
                    ),
                    tensorrt=_ort_tensorrt_settings(onnxruntime_value),
                ),
                tensorrt=_native_tensorrt_settings(tensorrt_value),
            ),
        ),
        analysis=AnalysisSettings(
            analysis_model=str(_required(analysis, "analysis_model")),
            data_model=str(_required(analysis, "data_model")),
            augmentation_model=(
                None
                if analysis.get("augmentation_model") is None
                else str(analysis["augmentation_model"])
            ),
            data_postprocessing_model=(
                None
                if analysis.get("data_postprocessing_model") is None
                else str(analysis["data_postprocessing_model"])
            ),
        ),
        labeling=DatasetLabelingSettings(
            source_matches_path=_path(
                _required(labeling, "source_matches_path"),
                "labeling.source_matches_path",
            ),
            items_path=_path(
                _required(labeling, "items_path"),
                "labeling.items_path",
            ),
            output_path=_path(
                _required(labeling, "output_path"),
                "labeling.output_path",
            ),
            score_column=str(_required(labeling, "score_column")),
            lower_p=float(_required(labeling, "lower_p")),
            upper_p=float(_required(labeling, "upper_p")),
            sample_size=int(_required(labeling, "sample_size")),
            seed=int(labeling.get("seed", _required(runtime, "seed"))),
            checkpoint_every_batches=int(
                _required(labeling, "checkpoint_every_batches")
            ),
            llm=LlmLabelingSettings(
                base_url=str(_required(labeling_llm, "base_url")),
                model=str(_required(labeling_llm, "model")),
                token_env=str(_required(labeling_llm, "token_env")),
                temperature=float(labeling_llm.get("temperature", 0.0)),
                verify_ssl=_bool(
                    labeling_llm.get("verify_ssl", True),
                    "labeling.llm.verify_ssl",
                ),
                request_batch_size=int(
                    _required(labeling_llm, "request_batch_size")
                ),
                max_concurrency=int(
                    _required(labeling_llm, "max_concurrency")
                ),
                min_concurrency=int(
                    _required(labeling_llm, "min_concurrency")
                ),
                max_attempts=int(_required(labeling_llm, "max_attempts")),
                max_rounds=int(_required(labeling_llm, "max_rounds")),
                retry_base_seconds=float(
                    _required(labeling_llm, "retry_base_seconds")
                ),
                max_prompt_chars=int(
                    _required(labeling_llm, "max_prompt_chars")
                ),
            ),
        ),
        analysis_models=AnalysisModelsSettings(
            attribute_importance=AttributeImportanceAnalysisSettings(
                model=str(_required(attribute_importance, "model")),
                output_dir=_path(
                    _required(attribute_importance, "output_dir"),
                    "analysis_models.attribute_importance.output_dir",
                ),
                sample_size=_optional_int(attribute_importance.get("sample_size")),
                group_by_category=_bool(
                    _required(attribute_importance, "group_by_category"),
                    "analysis_models.attribute_importance.group_by_category",
                ),
                min_occurrences=int(
                    _required(attribute_importance, "min_occurrences")
                ),
                score_type=str(_required(attribute_importance, "score_type")),
                output_file=str(_required(attribute_importance, "output_file")),
                metadata_file=str(
                    _required(attribute_importance, "metadata_file")
                ),
            ),
            backend_benchmark=(
                None
                if backend_benchmark_value is None
                else BackendBenchmarkSettings(
                    tests_path=_path(
                        _required(backend_benchmark_value, "tests_path"),
                        "analysis_models.backend_benchmark.tests_path",
                    ),
                    selected_tests=(
                        None
                        if backend_benchmark_value.get("selected_tests") is None
                        else tuple(
                            str(name)
                            for name in _sequence(
                                backend_benchmark_value.get("selected_tests"),
                                "analysis_models.backend_benchmark.selected_tests",
                            )
                        )
                    ),
                    sample_size=_optional_int(
                        backend_benchmark_value.get("sample_size")
                    ),
                    warmup_batches=int(
                        _required(backend_benchmark_value, "warmup_batches")
                    ),
                    measured_runs=int(
                        _required(backend_benchmark_value, "measured_runs")
                    ),
                    reference_test=str(
                        _required(backend_benchmark_value, "reference_test")
                    ),
                    compare_predictions=_bool(
                        _required(
                            backend_benchmark_value,
                            "compare_predictions",
                        ),
                        "analysis_models.backend_benchmark.compare_predictions",
                    ),
                    output_dir=_path(
                        _required(backend_benchmark_value, "output_dir"),
                        "analysis_models.backend_benchmark.output_dir",
                    ),
                )
            ),
        ),
        augmentation_models=AugmentationModelsSettings(
            attribute_shuffle=AttributeShuffleSettings(
                shuffled_copies=int(
                    _required(attribute_shuffle, "shuffled_copies")
                ),
                keep_original=_bool(
                    _required(attribute_shuffle, "keep_original"),
                    "augmentation_models.attribute_shuffle.keep_original",
                ),
                seed=int(_required(attribute_shuffle, "seed")),
                shuffle_cards_independently=_bool(
                    _required(attribute_shuffle, "shuffle_cards_independently"),
                    "augmentation_models.attribute_shuffle."
                    "shuffle_cards_independently",
                ),
                skip_oversized=_bool(
                    _required(attribute_shuffle, "skip_oversized"),
                    "augmentation_models.attribute_shuffle.skip_oversized",
                ),
            ),
            attribute_word_dropout=AttributeWordDropoutSettings(
                pair_probability=float(
                    _required(attribute_word_dropout, "pair_probability")
                ),
                attribute_dropout_probability=float(
                    _required(
                        attribute_word_dropout,
                        "attribute_dropout_probability",
                    )
                ),
                word_dropout_probability=float(
                    _required(attribute_word_dropout, "word_dropout_probability")
                ),
                keyboard_typo_probability=float(
                    _required(attribute_word_dropout, "keyboard_typo_probability")
                ),
                word_shuffle_probability=float(
                    _required(attribute_word_dropout, "word_shuffle_probability")
                ),
                seed=int(_required(attribute_word_dropout, "seed")),
            ),
        ),
        data_postprocessing_models=DataPostprocessingModelsSettings(
            attribute_sort=AttributeSortSettings(
                priorities_path=_path(
                    _required(attribute_sort, "priorities_path"),
                    "data_postprocessing_models.attribute_sort.priorities_path",
                ),
            ),
        ),
        model_description=ModelDescriptionSettings(
            transformer=TransformerParameters(
                profile=transformer_profile,
                pretrained_model_path=str(
                    _required(transformer, "pretrained_model_path")
                ),
                artifact_dir=_path(
                    _required(transformer, "artifact_dir"),
                    "model_description.transformer.artifact_dir",
                ),
                tokenizer=TransformerTokenizerSettings(
                    batch_fields=BatchFieldsSettings(
                        enabled=_bool(
                            batch_fields_value.get("enabled", False),
                            "model_description.transformer.tokenizer."
                            "batch_fields.enabled",
                        ),
                        chunk_size=int(
                            batch_fields_value.get("chunk_size", 16_384)
                        ),
                    )
                ),
                export=TransformerExportSettings(
                    onnx=OnnxExportSettings(
                        enabled=_bool(
                            onnx_export_value.get("enabled", False),
                            "model_description.transformer.export.onnx.enabled",
                        ),
                        opset=int(onnx_export_value.get("opset", 18)),
                        precision=str(
                            onnx_export_value.get("precision", "float16")
                        ).lower(),
                        dynamic_batch=_bool(
                            onnx_export_value.get("dynamic_batch", True),
                            "model_description.transformer.export.onnx."
                            "dynamic_batch",
                        ),
                        dynamic_sequence_length=_bool(
                            onnx_export_value.get(
                                "dynamic_sequence_length", True
                            ),
                            "model_description.transformer.export.onnx."
                            "dynamic_sequence_length",
                        ),
                        export_classifier=_bool(
                            onnx_export_value.get("export_classifier", True),
                            "model_description.transformer.export.onnx."
                            "export_classifier",
                        ),
                        export_encoder=_bool(
                            onnx_export_value.get("export_encoder", True),
                            "model_description.transformer.export.onnx."
                            "export_encoder",
                        ),
                    )
                ),
                pair_encoding=PairEncodingSettings(
                    use_field_tokens=_bool(
                        _required(encoding, "use_field_tokens"),
                        "model_description.transformer.pair_encoding.use_field_tokens",
                    ),
                    max_attribute_value_chars=_optional_int(
                        encoding.get("max_attribute_value_chars")
                    ),
                    max_attribute_value_tokens=_optional_int(
                        encoding.get("max_attribute_value_tokens")
                    ),
                    max_length=_optional_int(encoding.get("max_length")),
                    quantile=float(_required(encoding, "quantile")),
                    sample_size=int(_required(encoding, "sample_size")),
                    hard_cap=int(_required(encoding, "hard_cap")),
                ),
                max_epochs=int(_required(transformer, "max_epochs")),
                hpo_trials=int(_required(transformer, "hpo_trials")),
                learning_rate=float(_required(transformer, "learning_rate")),
                embeddings_learning_rate=(
                    None
                    if transformer.get("embeddings_learning_rate") is None
                    else float(transformer["embeddings_learning_rate"])
                ),
                train_new_token_embeddings_only=_bool(
                    transformer.get("train_new_token_embeddings_only", False),
                    "model_description.transformer.train_new_token_embeddings_only",
                ),
                train_last_n_layers=_optional_int(
                    transformer.get("train_last_n_layers")
                ),
                special_token_initialization=SpecialTokenInitializationSettings(
                    enabled=_bool(
                        special_token_initialization_value.get("enabled", False),
                        "model_description.transformer.special_token_initialization."
                        "enabled",
                    ),
                    key_seed_texts=_string_tuple(
                        special_token_initialization_value.get(
                            "key_seed_texts", ()
                        ),
                        "model_description.transformer.special_token_initialization."
                        "key_seed_texts",
                    ),
                    value_seed_texts=_string_tuple(
                        special_token_initialization_value.get(
                            "value_seed_texts", ()
                        ),
                        "model_description.transformer.special_token_initialization."
                        "value_seed_texts",
                    ),
                ),
                validation=TransformerValidationSettings(
                    fast_dev=FastDevValidationSettings(
                        enabled=_bool(
                            fast_dev_validation_value.get("enabled", False),
                            "model_description.transformer.validation.fast_dev."
                            "enabled",
                        ),
                        max_rows=int(
                            fast_dev_validation_value.get("max_rows", 20_000)
                        ),
                        every_n_optimizer_steps=int(
                            fast_dev_validation_value.get(
                                "every_n_optimizer_steps", 1_000
                            )
                        ),
                    )
                ),
                lr_scheduler_type=str(
                    transformer.get("lr_scheduler_type", "linear")
                ),
                head_learning_rate=(
                    None
                    if transformer.get("head_learning_rate") is None
                    else float(transformer["head_learning_rate"])
                ),
                layerwise_lr_decay=float(
                    transformer.get("layerwise_lr_decay", 1.0)
                ),
                training_runtime=TransformerTrainingRuntimeSettings(
                    attention=AttentionSettings(
                        implementation=str(
                            training_attention_value.get(
                                "implementation", "auto"
                            )
                        )
                    ),
                    length_bucketing=TrainingLengthBucketingSettings(
                        enabled=_bool(
                            training_length_bucketing_value.get("enabled", True),
                            "model_description.transformer.training_runtime."
                            "length_bucketing.enabled",
                        ),
                        mega_batch_multiplier=int(
                            training_length_bucketing_value.get(
                                "mega_batch_multiplier", 50
                            )
                        ),
                        padding_length_buckets=training_padding_buckets,
                    ),
                    dataloader=TrainingDataLoaderSettings(
                        num_workers=int(
                            training_dataloader_value.get("num_workers", 4)
                        ),
                        prefetch_factor=int(
                            training_dataloader_value.get("prefetch_factor", 2)
                        ),
                        persistent_workers=_bool(
                            training_dataloader_value.get(
                                "persistent_workers", True
                            ),
                            "model_description.transformer.training_runtime."
                            "dataloader.persistent_workers",
                        ),
                        pin_memory=_bool(
                            training_dataloader_value.get("pin_memory", True),
                            "model_description.transformer.training_runtime."
                            "dataloader.pin_memory",
                        ),
                        non_blocking_transfer=_bool(
                            training_dataloader_value.get(
                                "non_blocking_transfer", True
                            ),
                            "model_description.transformer.training_runtime."
                            "dataloader.non_blocking_transfer",
                        ),
                    ),
                    performance_logging=TrainingPerformanceLoggingSettings(
                        enabled=_bool(
                            performance_logging_value.get("enabled", True),
                            "model_description.transformer.training_runtime."
                            "performance_logging.enabled",
                        )
                    ),
                    token_cache=TrainingTokenCacheSettings(
                        enabled=_bool(
                            training_token_cache_value.get("enabled", False),
                            "model_description.transformer.training_runtime."
                            "token_cache.enabled",
                        ),
                        directory=_path(
                            training_token_cache_value.get(
                                "directory", ".cache/tokenized_pairs"
                            ),
                            "model_description.transformer.training_runtime."
                            "token_cache.directory",
                        ),
                        build_chunk_size=int(
                            training_token_cache_value.get(
                                "build_chunk_size", 4_096
                            )
                        ),
                    ),
                    torch_compile=TorchCompileSettings(
                        enabled=_bool(
                            training_torch_compile_value.get("enabled", False),
                            "model_description.transformer.training_runtime."
                            "torch_compile.enabled",
                        ),
                        mode=str(
                            training_torch_compile_value.get(
                                "mode", "reduce-overhead"
                            )
                        ),
                        dynamic=_bool(
                            training_torch_compile_value.get("dynamic", True),
                            "model_description.transformer.training_runtime."
                            "torch_compile.dynamic",
                        ),
                    ),
                ),
                weight_decay=float(_required(transformer, "weight_decay")),
                hpo_learning_rate_min=float(
                    _required(transformer, "hpo_learning_rate_min")
                ),
                hpo_learning_rate_max=float(
                    _required(transformer, "hpo_learning_rate_max")
                ),
                hpo_weight_decay_min=float(
                    _required(transformer, "hpo_weight_decay_min")
                ),
                hpo_weight_decay_max=float(
                    _required(transformer, "hpo_weight_decay_max")
                ),
                train_batch_size=int(
                    _required(transformer, "train_batch_size")
                ),
                eval_batch_size=int(
                    _required(transformer, "eval_batch_size")
                ),
                gradient_accumulation_steps=int(
                    _required(transformer, "gradient_accumulation_steps")
                ),
                warmup_ratio=float(_required(transformer, "warmup_ratio")),
                max_grad_norm=float(_required(transformer, "max_grad_norm")),
                early_stopping_patience=int(
                    _required(transformer, "early_stopping_patience")
                ),
                auto_find_batch_size=_bool(
                    _required(transformer, "auto_find_batch_size"),
                    "model_description.transformer.auto_find_batch_size",
                ),
                head=TransformerHeadParameters(
                    type=str(
                        transformer_head.get(
                            "type",
                            (
                                "native"
                                if is_prompted_profile(transformer_profile)
                                else "default"
                            ),
                        )
                    ),
                    poolings=tuple(
                        str(value)
                        for value in transformer_head.get("poolings", ("cls",))
                    ),
                    mlp_hidden_dims=tuple(
                        int(value)
                        for value in transformer_head.get("mlp_hidden_dims", ())
                    ),
                    dropout=float(transformer_head.get("dropout", 0.1)),
                    attention_hidden_dim=_optional_int(
                        transformer_head.get("attention_hidden_dim")
                    ),
                     attention_num_heads=int(
                         transformer_head.get("attention_num_heads", 1)
                     ),
                     native_logit_weight=float(
                         transformer_head.get("native_logit_weight", 1.0)
                     ),
                     attention_logit_weight=float(
                         transformer_head.get("attention_logit_weight", 0.0)
                     ),
                     train_logit_weights=_bool(
                         transformer_head.get("train_logit_weights", True),
                         "model_description.transformer.head.train_logit_weights",
                     ),
                     typed_hidden_dims=tuple(
                         int(value)
                         for value in transformer_head.get(
                             "typed_hidden_dims", (128, 64)
                         )
                     ),
                     typed_layer_norm=_bool(
                         transformer_head.get("typed_layer_norm", True),
                         "model_description.transformer.head.typed_layer_norm",
                     ),
                ),
            ),
            maxpooling=MaxPoolingParameters(
                artifact_path=_path(
                    _required(maxpooling, "artifact_path"),
                    "model_description.maxpooling.artifact_path",
                ),
                vector_size=int(_required(maxpooling, "vector_size")),
                window=int(_required(maxpooling, "window")),
                min_count=int(_required(maxpooling, "min_count")),
                workers=int(_required(maxpooling, "workers")),
                fasttext_epochs=int(_required(maxpooling, "fasttext_epochs")),
                classifier_epochs=int(
                    _required(maxpooling, "classifier_epochs")
                ),
                batch_size=int(_required(maxpooling, "batch_size")),
                patience=int(_required(maxpooling, "patience")),
                dropout=float(_required(maxpooling, "dropout")),
                learning_rate=float(_required(maxpooling, "learning_rate")),
                weight_decay=float(_required(maxpooling, "weight_decay")),
            ),
            fusion=FusionParameters(
                artifact_path=_path(
                    _required(fusion, "artifact_path"),
                    "model_description.fusion.artifact_path",
                ),
                embedding_batch_size=int(
                    _required(fusion, "embedding_batch_size")
                ),
                hidden_dim=int(_required(fusion, "hidden_dim")),
                dropout=float(_required(fusion, "dropout")),
                batch_size=int(_required(fusion, "batch_size")),
                max_epochs=int(_required(fusion, "max_epochs")),
                patience=int(_required(fusion, "patience")),
                learning_rate=float(_required(fusion, "learning_rate")),
                weight_decay=float(_required(fusion, "weight_decay")),
            ),
            boosting=BoostingParameters(
                artifact_dir=_path(
                    _required(boosting, "artifact_dir"),
                    "model_description.boosting.artifact_dir",
                ),
                iterations=int(_required(boosting, "iterations")),
                depth=int(_required(boosting, "depth")),
                learning_rate=float(_required(boosting, "learning_rate")),
                loss_function=str(_required(boosting, "loss_function")),
                early_stopping_rounds=int(
                    _required(boosting, "early_stopping_rounds")
                ),
                thread_count=int(_required(boosting, "thread_count")),
            ),
            cascade=CascadeParameters(
                fast_model=str(_required(cascade, "fast_model")),
                main_model=str(_required(cascade, "main_model")),
                negative_threshold=float(
                    _required(cascade, "negative_threshold")
                ),
                positive_threshold=float(
                    _required(cascade, "positive_threshold")
                ),
            ),
            stacking=StackingParameters(
                artifact_dir=_path(
                    _required(stacking, "artifact_dir"),
                    "model_description.stacking.artifact_dir",
                ),
                base_model=str(_required(stacking, "base_model")),
                stacking_model=str(_required(stacking, "stacking_model")),
            ),
        ),
        features=FeatureSettings(
            execution_order=execution_order,
            normalization=NormalizationSettings(
                enabled=_bool(
                    normalization.get("enabled", False),
                    "features.normalization.enabled",
                ),
                output_column=str(_required(normalization, "output_column")),
                synonyms_path=_path(
                    _required(normalization, "synonyms_path"),
                    "features.normalization.synonyms_path",
                ),
                unique_attributes_path=_path(
                    _required(normalization, "unique_attributes_path"),
                    "features.normalization.unique_attributes_path",
                ),
                n_jobs=int(_required(normalization, "n_jobs")),
                chunk_size=int(_required(normalization, "chunk_size")),
            ),
            ner=NerSettings(
                enabled=_bool(
                    ner.get("enabled", False),
                    "features.ner.enabled",
                ),
                provider=None if ner.get("provider") is None else str(ner["provider"]),
                model_dir=_optional_path(ner.get("model_dir")),
                cluster_centers_path=_optional_path(
                    ner.get("cluster_centers_path")
                ),
                source_column=str(ner.get("source_column", "name")),
                output_column=str(ner.get("output_column", "ner_attributes")),
                enriched_column=str(
                    ner.get("enriched_column", "enriched_attributes")
                ),
                merge_policy=str(ner.get("merge_policy", "missing_only")),
                batch_size=int(ner.get("batch_size", 512)),
                max_length=int(ner.get("max_length", 100)),
                use_amp=_bool(
                    ner.get("use_amp", True),
                    "features.ner.use_amp",
                ),
                semantic_cleanup=_bool(
                    ner.get("semantic_cleanup", True),
                    "features.ner.semantic_cleanup",
                ),
            ),
            physical=PhysicalFeatureSettings(
                enabled=_bool(
                    physical.get("enabled", False),
                    "features.physical.enabled",
                ),
                source_column=str(physical.get("source_column", "name")),
                output_column=str(
                    physical.get("output_column", "physical_attributes")
                ),
                enriched_column=str(
                    physical.get("enriched_column", "feature_attributes")
                ),
                merge_policy=str(
                    physical.get("merge_policy", "missing_only")
                ),
                normalize_units=_bool(
                    physical.get("normalize_units", True),
                    "features.physical.normalize_units",
                ),
                n_jobs=int(physical.get("n_jobs", 1)),
                chunk_size=int(physical.get("chunk_size", 10_000)),
            ),
        ),
        pair_features=PairFeatureSettings(
            typed_attributes=TypedAttributeOptions(
                enabled=_bool(
                    typed_attributes_value.get("enabled", False),
                    "pair_features.typed_attributes.enabled",
                ),
                detector=str(typed_attributes_value.get("detector", "rules")),
                preserve_semantic_type_when_missing=_bool(
                    typed_attributes_value.get(
                        "preserve_semantic_type_when_missing", True
                    ),
                    "pair_features.typed_attributes."
                    "preserve_semantic_type_when_missing",
                ),
                symmetric=_bool(
                    typed_attributes_value.get("symmetric", True),
                    "pair_features.typed_attributes.symmetric",
                ),
                code=_bool(
                    typed_attribute_types.get("code", True),
                    "pair_features.typed_attributes.types.code",
                ),
                physical=_bool(
                    typed_attribute_types.get("physical", True),
                    "pair_features.typed_attributes.types.physical",
                ),
                numeric=_bool(
                    typed_attribute_types.get("numeric", True),
                    "pair_features.typed_attributes.types.numeric",
                ),
                set=_bool(
                    typed_attribute_types.get("set", True),
                    "pair_features.typed_attributes.types.set",
                ),
                text=_bool(
                    typed_attribute_types.get("text", True),
                    "pair_features.typed_attributes.types.text",
                ),
            )
        ),
        runtime=RuntimeSettings(
            device=None if runtime.get("device") is None else str(runtime["device"]),
            seed=int(_required(runtime, "seed")),
        ),
        logging=LoggingSettings(
            level=str(_required(logging, "level")),
            file=_optional_path(logging.get("file")),
            rotation=str(_required(logging, "rotation")),
        ),
        submission=SubmissionSettings(
            output_path=_path(
                submission.get(
                    "output_path",
                    "dist/twin2attr_submission.zip",
                ),
                "submission.output_path",
            ),
        ),
        benchmark=(
            None
            if benchmark_value is None
            else BenchmarkSettings(
                base_config_path=_path(
                    _required(benchmark_value, "base_config_path"),
                    "benchmark.base_config_path",
                ),
                output_dir=_path(
                    _required(benchmark_value, "output_dir"),
                    "benchmark.output_dir",
                ),
                fail_fast=_bool(
                    benchmark_value.get("fail_fast", False),
                    "benchmark.fail_fast",
                ),
                jobs=_benchmark_jobs(_required(benchmark_value, "jobs")),
            )
        ),
    )


def load_app_config_file(
    path: str | Path,
    overrides: Sequence[str] = (),
) -> AppConfig:
    """Load YAML plus dot-list overrides at the application boundary."""
    from omegaconf import OmegaConf

    config_path = resolve_project_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Pipeline config does not exist: {config_path}")
    config = OmegaConf.load(config_path)
    clean_overrides = [value for value in overrides if value != "--"]
    if clean_overrides:
        config = OmegaConf.merge(
            config,
            OmegaConf.from_dotlist(clean_overrides),
        )
    return load_app_config(config)


def load_benchmark_config_file(
    path: str | Path,
    overrides: Sequence[str] = (),
) -> AppConfig:
    """Load lightweight benchmark orchestration plus its full base config."""
    from omegaconf import OmegaConf

    orchestration_path = resolve_project_path(path)
    if not orchestration_path.is_file():
        raise FileNotFoundError(
            f"Benchmark config does not exist: {orchestration_path}"
        )
    orchestration = OmegaConf.load(orchestration_path)
    benchmark = orchestration.get("benchmark")
    if benchmark is None or not hasattr(benchmark, "get"):
        raise ValueError("benchmark config must contain a 'benchmark' mapping")
    raw_base_path = benchmark.get("base_config_path")
    if raw_base_path is None or not str(raw_base_path).strip():
        raise ValueError("benchmark.base_config_path is required")
    base_path = resolve_project_path(str(raw_base_path))
    if not base_path.is_file():
        raise FileNotFoundError(
            f"Benchmark base config does not exist: {base_path}"
        )
    base = OmegaConf.load(base_path)
    combined = OmegaConf.merge(base, orchestration)
    clean_overrides = [value for value in overrides if value != "--"]
    if clean_overrides:
        combined = OmegaConf.merge(
            combined,
            OmegaConf.from_dotlist(clean_overrides),
        )
    return load_app_config(combined)


def _serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, TypedAttributeOptions):
        return value.to_dict()
    if isinstance(value, TensorRTProfileSettings):
        return {
            "min_batch_size": value.min_batch_size,
            "opt_batch_size": value.opt_batch_size,
            "max_batch_size": value.max_batch_size,
            "sequence_lengths": list(value.sequence_lengths),
        }
    if isinstance(value, MixedDatasetSettings):
        serialized = {
            field.name: _serializable(getattr(value, field.name))
            for field in fields(value)
            if field.name != "sources"
        }
        serialized["sources"] = {
            source.name: {
                field.name: _serializable(getattr(source, field.name))
                for field in fields(source)
                if field.name != "name"
            }
            for source in value.sources
        }
        return serialized
    if isinstance(value, BenchmarkSettings):
        return {
            "base_config_path": _serializable(value.base_config_path),
            "output_dir": _serializable(value.output_dir),
            "fail_fast": value.fail_fast,
            "jobs": {
                job.name: {
                    field.name: _serializable(getattr(job, field.name))
                    for field in fields(job)
                    if field.name != "name"
                }
                for job in value.jobs
            },
        }
    if is_dataclass(value):
        return {
            field.name: _serializable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        return {key: _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    return value


def save_app_config(config: AppConfig, path: Path) -> None:
    """Persist the resolved typed configuration as YAML."""
    from omegaconf import OmegaConf

    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(_serializable(config)), path)


def app_config_to_mapping(config: AppConfig) -> dict[str, Any]:
    """Return a mutable, YAML-compatible representation of a typed config."""
    serialized = _serializable(config)
    if not isinstance(serialized, dict):
        raise TypeError("serialized application config must be a mapping")
    return serialized


__all__ = [
    "FEATURE_PROVIDER_NAMES",
    "AnalysisModelsSettings",
    "AnalysisSettings",
    "AttentionSettings",
    "AppConfig",
    "AttributeImportanceAnalysisSettings",
    "BackendBenchmarkSettings",
    "BenchmarkJobSettings",
    "BenchmarkSettings",
    "AttributeShuffleSettings",
    "AttributeSortSettings",
    "AttributeWordDropoutSettings",
    "AugmentationModelsSettings",
    "BaseDatasetSettings",
    "BatchFieldsSettings",
    "BoostingParameters",
    "CascadeParameters",
    "ConfigSource",
    "DataModelDescriptionSettings",
    "DatasetLabelingSettings",
    "DatasetSourceSettings",
    "DatasetOverlapResolutionSettings",
    "DatasetSplitterSettings",
    "DataPostprocessingModelsSettings",
    "FeatureSettings",
    "FusionParameters",
    "InferenceSettings",
    "LoggingSettings",
    "LengthBucketingSettings",
    "LlmLabelingSettings",
    "MaxPoolingParameters",
    "MixedDatasetSettings",
    "ModelDescriptionSettings",
    "NerSettings",
    "NormalizationSettings",
    "NativeTensorRTSettings",
    "OnnxExportSettings",
    "OnnxRuntimeSettings",
    "PairEncodingSettings",
    "PairFeatureSettings",
    "PhysicalFeatureSettings",
    "RuntimeSettings",
    "SampleWeightModelSettings",
    "SpecialTokenInitializationSettings",
    "SubmissionSettings",
    "TensorRTEngineCacheSettings",
    "TensorRTProfileSettings",
    "OrtTensorRTProviderSettings",
    "TensorRTTimingCacheSettings",
    "TrainingSettings",
    "TorchCompileSettings",
    "TrainingDataLoaderSettings",
    "TrainingLengthBucketingSettings",
    "TrainingPerformanceLoggingSettings",
    "TransformerTrainingRuntimeSettings",
    "TransformerExportSettings",
    "TransformerHeadParameters",
    "TransformerInferenceSettings",
    "TransformerParameters",
    "TransformerTokenizerSettings",
    "load_app_config",
    "load_benchmark_config_file",
    "load_app_config_file",
    "app_config_to_mapping",
    "save_app_config",
]
