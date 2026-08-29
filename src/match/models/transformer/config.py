"""Configuration and result types for Transformer training."""

from dataclasses import dataclass
from pathlib import Path

from ...pair_encoding import DEFAULT_MAX_ATTRIBUTE_VALUE_TOKENS
from .head import PoolingHeadConfig
from .profile import (
    PROMPTED_BINARY_RERANKER_PROFILE,
    SEQUENCE_CLASSIFIER_PROFILE,
    is_prompted_profile,
    validate_profile_head,
)


@dataclass(frozen=True)
class SequenceClassifierConfig:
    model_path: str
    profile: str = SEQUENCE_CLASSIFIER_PROFILE
    max_epochs: int = 5
    hpo_trials: int = 10
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    hpo_learning_rate_min: float = 1e-6
    hpo_learning_rate_max: float = 5e-5
    hpo_weight_decay_min: float = 0.0
    hpo_weight_decay_max: float = 0.1
    train_batch_size: int = 64
    eval_batch_size: int = 64
    gradient_accumulation_steps: int = 1
    warmup_ratio: float = 0.06
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 2
    auto_find_batch_size: bool = True
    seed: int = 42
    use_field_tokens: bool = True
    special_token_initialization_enabled: bool = False
    special_token_key_seed_texts: tuple[str, ...] = ()
    special_token_value_seed_texts: tuple[str, ...] = ()
    fast_dev_validation_enabled: bool = False
    fast_dev_validation_max_rows: int = 20_000
    fast_dev_validation_every_n_optimizer_steps: int = 1_000
    batch_fields: bool = False
    field_chunk_size: int = 16_384
    max_attribute_value_chars: int | None = None
    max_attribute_value_tokens: int | None = DEFAULT_MAX_ATTRIBUTE_VALUE_TOKENS
    max_length: int | None = None
    max_length_quantile: float = 0.95
    max_length_sample_size: int = 10_000
    max_length_hard_cap: int = 512
    head_type: str = "default"
    head_config: PoolingHeadConfig = PoolingHeadConfig()
    embeddings_learning_rate: float | None = None
    train_new_token_embeddings_only: bool = False
    train_last_n_layers: int | None = None
    lr_scheduler_type: str = "linear"
    head_learning_rate: float | None = None
    layerwise_lr_decay: float = 1.0
    train_length_bucketing: bool = True
    train_mega_batch_multiplier: int = 50
    train_padding_length_buckets: tuple[int, ...] | None = None
    dataloader_num_workers: int = 0
    dataloader_prefetch_factor: int = 2
    dataloader_persistent_workers: bool = False
    dataloader_pin_memory: bool = True
    non_blocking_transfer: bool = True
    performance_logging: bool = True
    torch_compile: bool = False
    torch_compile_mode: str = "reduce-overhead"
    onnx_export_enabled: bool = False
    onnx_opset: int = 18
    onnx_precision: str = "float16"
    onnx_dynamic_batch: bool = True
    onnx_dynamic_sequence_length: bool = True
    onnx_export_classifier: bool = True
    onnx_export_encoder: bool = True

    def __post_init__(self) -> None:
        if not self.model_path.strip():
            raise ValueError("model_path must not be empty")
        if self.max_epochs < 1:
            raise ValueError("max_epochs must be positive")
        if self.hpo_trials < 1:
            raise ValueError("hpo_trials must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if (
            self.embeddings_learning_rate is not None
            and self.embeddings_learning_rate <= 0.0
        ):
            raise ValueError("embeddings_learning_rate must be positive or None")
        if self.head_learning_rate is not None and self.head_learning_rate <= 0.0:
            raise ValueError("head_learning_rate must be positive or None")
        if not 0.0 < self.layerwise_lr_decay <= 1.0:
            raise ValueError("layerwise_lr_decay must be in (0, 1]")
        if self.weight_decay < 0.0:
            raise ValueError("weight_decay must not be negative")
        if not 0.0 < self.hpo_learning_rate_min < self.hpo_learning_rate_max:
            raise ValueError(
                "HPO learning-rate bounds must be positive and increasing"
            )
        if not 0.0 <= self.hpo_weight_decay_min < self.hpo_weight_decay_max:
            raise ValueError(
                "HPO weight-decay bounds must be non-negative and increasing"
            )
        if self.train_batch_size < 1 or self.eval_batch_size < 1:
            raise ValueError("train and eval batch sizes must be positive")
        if self.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("warmup_ratio must be in [0, 1)")
        if self.max_grad_norm <= 0.0:
            raise ValueError("max_grad_norm must be positive")
        if self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be positive")
        if self.fast_dev_validation_max_rows < 1:
            raise ValueError("fast_dev_validation_max_rows must be positive")
        if self.fast_dev_validation_every_n_optimizer_steps < 1:
            raise ValueError(
                "fast_dev_validation_every_n_optimizer_steps must be positive"
            )
        if self.field_chunk_size < 1:
            raise ValueError("field_chunk_size must be positive")
        if self.train_mega_batch_multiplier < 1:
            raise ValueError("train_mega_batch_multiplier must be positive")
        if self.dataloader_num_workers < 0:
            raise ValueError("dataloader_num_workers must not be negative")
        if self.dataloader_prefetch_factor < 1:
            raise ValueError("dataloader_prefetch_factor must be positive")
        if self.dataloader_num_workers == 0 and self.dataloader_persistent_workers:
            raise ValueError(
                "dataloader_persistent_workers requires dataloader_num_workers > 0"
            )
        buckets = self.train_padding_length_buckets
        if buckets is not None:
            if not buckets or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in buckets
            ):
                raise ValueError(
                    "train_padding_length_buckets must contain positive integers"
                )
            if tuple(sorted(set(buckets))) != buckets:
                raise ValueError(
                    "train_padding_length_buckets must be strictly increasing"
                )
            if self.max_length is not None and buckets[-1] != self.max_length:
                raise ValueError(
                    "the final train padding bucket must equal max_length"
                )
        if self.torch_compile_mode not in {
            "default",
            "reduce-overhead",
            "max-autotune",
            "max-autotune-no-cudagraphs",
        }:
            raise ValueError("unsupported torch_compile_mode")
        if (
            self.max_attribute_value_chars is not None
            and self.max_attribute_value_chars < 1
        ):
            raise ValueError("max_attribute_value_chars must be positive or None")
        if (
            self.max_attribute_value_tokens is not None
            and self.max_attribute_value_tokens < 1
        ):
            raise ValueError("max_attribute_value_tokens must be positive or None")
        if self.max_length is not None and self.max_length < 8:
            raise ValueError("max_length must be at least 8 or None")
        if not 0.0 < self.max_length_quantile <= 1.0:
            raise ValueError("max_length_quantile must be in (0, 1]")
        if self.max_length_sample_size < 1:
            raise ValueError("max_length_sample_size must be positive")
        if self.max_length_hard_cap < 8:
            raise ValueError("max_length_hard_cap must be at least 8")
        validate_profile_head(self.profile, self.head_type)
        if (
            is_prompted_profile(self.profile)
            and self.head_type == "attention_pooling"
            and "cls" in self.head_config.poolings
        ):
            raise ValueError("Nemotron attention pooling cannot use cls")
        if is_prompted_profile(self.profile) and self.use_field_tokens:
            raise ValueError(
                f"{PROMPTED_BINARY_RERANKER_PROFILE} requires use_field_tokens=False"
            )
        if self.train_new_token_embeddings_only and not self.use_field_tokens:
            raise ValueError(
                "train_new_token_embeddings_only requires use_field_tokens"
            )
        if self.special_token_initialization_enabled:
            if not self.use_field_tokens:
                raise ValueError(
                    "special_token_initialization requires use_field_tokens"
                )
            for seed_texts, name in (
                (self.special_token_key_seed_texts, "key"),
                (self.special_token_value_seed_texts, "value"),
            ):
                if not seed_texts or any(not value.strip() for value in seed_texts):
                    raise ValueError(
                        f"special_token_{name}_seed_texts must be non-empty"
                    )
        if self.train_last_n_layers is not None and self.train_last_n_layers < 1:
            raise ValueError("train_last_n_layers must be positive or None")
        if self.lr_scheduler_type not in {"linear", "cosine"}:
            raise ValueError("lr_scheduler_type must be 'linear' or 'cosine'")
        if self.onnx_opset < 14:
            raise ValueError("onnx_opset must be at least 14")
        if self.onnx_precision not in {"float32", "float16"}:
            raise ValueError("onnx_precision must be float32 or float16")
        if self.onnx_export_enabled and not (
            self.onnx_export_classifier or self.onnx_export_encoder
        ):
            raise ValueError("enabled ONNX export requires at least one graph")

    @property
    def resolved_embeddings_learning_rate(self) -> float:
        return self.embeddings_learning_rate or self.learning_rate

    @property
    def resolved_head_learning_rate(self) -> float:
        return self.head_learning_rate or self.learning_rate


@dataclass(frozen=True)
class ResolvedTrainingConfig:
    max_epochs: int
    max_length: int
    train_batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    embeddings_learning_rate: float
    train_new_token_embeddings_only: bool
    train_last_n_layers: int | None
    head_learning_rate: float
    layerwise_lr_decay: float
    weight_decay: float
    use_field_tokens: bool
    special_token_initialization_enabled: bool
    special_token_key_seed_texts: tuple[str, ...]
    special_token_value_seed_texts: tuple[str, ...]
    fast_dev_validation_enabled: bool
    fast_dev_validation_max_rows: int
    fast_dev_validation_every_n_optimizer_steps: int
    batch_fields: bool
    field_chunk_size: int
    max_attribute_value_chars: int | None
    max_attribute_value_tokens: int | None
    warmup_ratio: float
    lr_scheduler_type: str
    gradient_clip_norm: float
    early_stopping_patience: int
    auto_find_batch_size: bool
    train_length_bucketing: bool
    train_mega_batch_multiplier: int
    train_padding_length_buckets: tuple[int, ...] | None
    dataloader_num_workers: int
    dataloader_prefetch_factor: int
    dataloader_persistent_workers: bool
    dataloader_pin_memory: bool
    non_blocking_transfer: bool
    performance_logging: bool
    torch_compile: bool
    torch_compile_mode: str


@dataclass(frozen=True)
class TrainingResult:
    model_dir: Path
    validation_macro_pr_auc: float
    best_hyperparameters: dict[str, float]
    resolved_config: ResolvedTrainingConfig


__all__ = [
    "ResolvedTrainingConfig",
    "SequenceClassifierConfig",
    "TrainingResult",
]
