"""Typed application configuration and the OmegaConf boundary adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

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

    def __post_init__(self) -> None:
        if self.splitter_type != "binary":
            raise ValueError("splitter_type currently must be 'binary'")
        if self.score_type not in {"label", "votes"}:
            raise ValueError("score_type must be one of: label, votes")
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
        elif self.total_votes is not None:
            raise ValueError("label splitter must not define total_votes")
        if self.uncertain_action != "drop":
            raise ValueError("uncertain_action currently must be 'drop'")


@dataclass(frozen=True, slots=True)
class DatasetSourceSettings:
    name: str
    matches: Path
    weight: float
    max_rows: int | None
    sampling_strategy: str
    splitter: DatasetSplitterSettings

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("dataset source name must not be empty")
        if self.weight <= 0.0:
            raise ValueError("dataset source weight must be positive")
        if self.max_rows is not None and self.max_rows < 1:
            raise ValueError("dataset source max_rows must be positive or null")
        if self.sampling_strategy not in {"random", "category_target_balanced"}:
            raise ValueError(
                "sampling_strategy must be random or category_target_balanced"
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


@dataclass(frozen=True, slots=True)
class DataModelDescriptionSettings:
    base_dataset: BaseDatasetSettings
    mix_dataset: MixedDatasetSettings


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
        if self.data_model not in {"base_dataset", "mix_dataset"}:
            raise ValueError(
                "training.data_model must be one of: base_dataset, mix_dataset"
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
                "inference.transformer.torch_compile.mode must be one of: "
                "default, reduce-overhead, max-autotune, "
                "max-autotune-no-cudagraphs"
            )


@dataclass(frozen=True, slots=True)
class OnnxRuntimeSettings:
    provider: str = "cuda"
    device_id: int = 0
    io_binding: bool = True
    graph_optimization: str = "all"
    fallback_to_pytorch: bool = True

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
    backend: str = "pytorch"
    num_workers: int = 0
    prefetch_factor: int = 2
    pin_memory: bool = True
    non_blocking_transfer: bool = True
    length_bucketing: LengthBucketingSettings = LengthBucketingSettings()
    torch_compile: TorchCompileSettings = TorchCompileSettings()
    onnxruntime: OnnxRuntimeSettings = OnnxRuntimeSettings()

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("inference.transformer.batch_size must be positive")
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
        if self.backend not in {"pytorch", "onnxruntime"}:
            raise ValueError(
                "inference.transformer.backend must be one of: pytorch, "
                "onnxruntime"
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
        if self.analysis_model != "attribute_importance":
            raise ValueError(
                "analysis.analysis_model currently must be 'attribute_importance'"
            )
        if self.data_model not in {"base_dataset", "mix_dataset"}:
            raise ValueError(
                "analysis.data_model must be one of: base_dataset, mix_dataset"
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

    def __post_init__(self) -> None:
        normalized_type = self.type.strip().lower()
        if normalized_type not in {"default", "pooling"}:
            raise ValueError("transformer.head.type must be 'default' or 'pooling'")
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
        object.__setattr__(self, "type", normalized_type)
        object.__setattr__(
            self,
            "poolings",
            normalized_poolings,
        )


@dataclass(frozen=True, slots=True)
class TransformerParameters:
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
    lr_scheduler_type: str = "linear"
    head_learning_rate: float | None = None
    layerwise_lr_decay: float = 1.0
    head: TransformerHeadParameters = TransformerHeadParameters()

    def __post_init__(self) -> None:
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
class AnalysisModelsSettings:
    attribute_importance: AttributeImportanceAnalysisSettings


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
    analysis_models: AnalysisModelsSettings
    augmentation_models: AugmentationModelsSettings
    data_postprocessing_models: DataPostprocessingModelsSettings
    model_description: ModelDescriptionSettings
    features: FeatureSettings
    runtime: RuntimeSettings
    logging: LoggingSettings
    submission: SubmissionSettings


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
    )


def _dataset_source(
    name: str,
    values: Mapping[str, Any],
) -> DatasetSourceSettings:
    splitter = values.get("splitter", {})
    if not isinstance(splitter, Mapping):
        raise ValueError(
            f"data_model_description.mix_dataset.sources.{name}.splitter "
            "must be a mapping"
        )
    prefix = f"data_model_description.mix_dataset.sources.{name}"
    return DatasetSourceSettings(
        name=name,
        matches=_path(_required(values, "matches"), f"{prefix}.matches"),
        weight=float(values.get("weight", 1.0)),
        max_rows=_optional_int(values.get("max_rows")),
        sampling_strategy=str(values.get("sampling_strategy", "random")),
        splitter=_dataset_splitter(splitter),
    )


def _dataset_sources(values: Any) -> tuple[DatasetSourceSettings, ...]:
    if isinstance(values, Mapping):
        entries = values.items()
    elif isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        normalized: list[tuple[str, Mapping[str, Any]]] = []
        for value in values:
            if not isinstance(value, Mapping):
                raise ValueError("mix_dataset sources must contain mappings")
            normalized.append((str(_required(value, "name")), value))
        entries = normalized
    else:
        raise ValueError("mix_dataset sources must be a mapping or sequence")
    sources: list[DatasetSourceSettings] = []
    for name, source in entries:
        if not isinstance(source, Mapping):
            raise ValueError(f"mix_dataset source {name!r} must be a mapping")
        sources.append(_dataset_source(str(name), source))
    return tuple(sources)


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
    mix_sources = _required(mix_dataset, "sources")
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
    onnxruntime_value = inference_transformer.get("onnxruntime", {})
    if not isinstance(onnxruntime_value, Mapping):
        raise ValueError(
            "config section 'inference.transformer.onnxruntime' must be a mapping"
        )
    analysis = _section(resolved, "analysis")
    analysis_models = _section(resolved, "analysis_models")
    attribute_importance = _section(
        analysis_models,
        "attribute_importance",
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
    transformer_head_value = transformer.get("head", {})
    if not isinstance(transformer_head_value, Mapping):
        raise ValueError("config section 'transformer.head' must be a mapping")
    transformer_head = transformer_head_value
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
    runtime = _section(resolved, "runtime")
    logging = _section(resolved, "logging")
    submission_value = resolved.get("submission", {})
    if not isinstance(submission_value, Mapping):
        raise ValueError("config section 'submission' must be a mapping")
    submission = submission_value

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
            mix_dataset=MixedDatasetSettings(
                items=_path(
                    _required(mix_dataset, "items"),
                    "data_model_description.mix_dataset.items",
                ),
                sources=_dataset_sources(mix_sources),
                validation_source=str(
                    _required(mix_dataset, "validation_source")
                ),
                validation_fraction=float(
                    _required(mix_dataset, "validation_fraction")
                ),
                leakage_scope=str(_required(mix_dataset, "leakage_scope")),
                candidate_splits=int(mix_dataset.get("candidate_splits", 128)),
                seed=int(mix_dataset.get("seed", _required(runtime, "seed"))),
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
                ),
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
                    type=str(transformer_head.get("type", "default")),
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


def _serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
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


__all__ = [
    "FEATURE_PROVIDER_NAMES",
    "AnalysisModelsSettings",
    "AnalysisSettings",
    "AppConfig",
    "AttributeImportanceAnalysisSettings",
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
    "DatasetSourceSettings",
    "DatasetSplitterSettings",
    "DataPostprocessingModelsSettings",
    "FeatureSettings",
    "FusionParameters",
    "InferenceSettings",
    "LoggingSettings",
    "LengthBucketingSettings",
    "MaxPoolingParameters",
    "MixedDatasetSettings",
    "ModelDescriptionSettings",
    "NerSettings",
    "NormalizationSettings",
    "OnnxExportSettings",
    "OnnxRuntimeSettings",
    "PairEncodingSettings",
    "PhysicalFeatureSettings",
    "RuntimeSettings",
    "SubmissionSettings",
    "TrainingSettings",
    "TorchCompileSettings",
    "TransformerExportSettings",
    "TransformerHeadParameters",
    "TransformerInferenceSettings",
    "TransformerParameters",
    "TransformerTokenizerSettings",
    "load_app_config",
    "load_app_config_file",
    "save_app_config",
]
