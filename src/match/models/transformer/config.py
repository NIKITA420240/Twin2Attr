"""Configuration and result types for Transformer training."""

from dataclasses import dataclass
from pathlib import Path

from ...pair_encoding import DEFAULT_MAX_ATTRIBUTE_VALUE_TOKENS


@dataclass(frozen=True)
class SequenceClassifierConfig:
    model_path: str
    max_epochs: int = 5
    hpo_trials: int = 10
    seed: int = 42
    use_field_tokens: bool = True
    max_attribute_value_tokens: int | None = DEFAULT_MAX_ATTRIBUTE_VALUE_TOKENS
    max_length: int | None = None
    max_length_quantile: float = 0.95
    max_length_sample_size: int = 10_000
    max_length_hard_cap: int = 512

    def __post_init__(self) -> None:
        if not self.model_path.strip():
            raise ValueError("model_path must not be empty")
        if self.max_epochs < 1:
            raise ValueError("max_epochs must be positive")
        if self.hpo_trials < 1:
            raise ValueError("hpo_trials must be positive")
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


@dataclass(frozen=True)
class ResolvedTrainingConfig:
    max_length: int
    train_batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    use_field_tokens: bool
    max_attribute_value_tokens: int | None
    warmup_ratio: float = 0.06
    gradient_clip_norm: float = 1.0


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
