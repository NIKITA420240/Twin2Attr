"""Transformer training and inference for prepared product-card pairs.

Reading parquet files and structuring cards belong to ``prepare_data``;
field serialization, token budgets and collation belong to ``pair_encoding``.
This module owns model optimization, evaluation, persistence and inference.

Geometrically, a cross-encoder maps a pair of cards to one point ``h`` in a
learned representation space. The classification head learns a separating
hyperplane, while fine-tuning is also free to move the points themselves::

    before fine-tuning              after fine-tuning

      x   o  x                         x x x | o o o
        o   x                                |
      x   o                           decision boundary

Here ``x`` denotes different products and ``o`` denotes matching products.
"""

from __future__ import annotations

import inspect
import json
import optuna
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any, Sequence

from loguru import logger

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    EvalPrediction,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)

from .pair_encoding import (
    DEFAULT_MAX_ATTRIBUTE_VALUE_TOKENS,
    PairEncodingCollator,
    PreparedPairDataset,
    add_pair_special_tokens,
    infer_pair_max_length,
)
from .prepare_data import PreparedPair

__all__ = [
    "ResolvedTrainingConfig",
    "SequenceClassifierConfig",
    "TrainingResult",
    "compute_class_weights",
    "compute_macro_pr_auc",
    "compute_pr_auc",
    "encode_pair_cls",
    "load_trained_classifier",
    "predict_match_probabilities",
    "train_sequence_classifier",
]


@dataclass(frozen=True)
class SequenceClassifierConfig:
    """Minimal user-controlled fine-tuning configuration.

    Parameters
    ----------
    model_path:
        Hugging Face model id or path to a local checkpoint.
    max_epochs:
        Maximum number of final-training epochs. Early stopping may finish the
        run sooner.
    hpo_trials:
        Number of Optuna trials used to maximize validation macro PR-AUC. A
        value of one disables the search and uses conservative defaults.
    seed:
        Random seed shared by sampling, training and hyperparameter search.
    use_field_tokens:
        Whether pair encoding uses the added ``[KEY]`` and ``[VAL]`` tokens.
    max_attribute_value_tokens:
        Per-attribute value limit. ``None`` disables individual truncation.
    max_length:
        Explicit pair length. ``None`` infers it from the training data.
    max_length_quantile, max_length_sample_size, max_length_hard_cap:
        Parameters used only when ``max_length`` is inferred.
    """

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
        if self.max_attribute_value_tokens is not None and self.max_attribute_value_tokens < 1:
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
    """Concrete parameters selected from data, hardware and HPO."""

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
    """Summary of a completed fine-tuning run."""

    model_dir: Path
    validation_macro_pr_auc: float
    best_hyperparameters: dict[str, float]
    resolved_config: ResolvedTrainingConfig


class WeightedSequenceTrainer(Trainer):
    """Trainer using balanced cross-entropy for imbalanced match labels."""

    def __init__(self, *args: Any, class_weights: torch.Tensor, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.detach().to(dtype=torch.float32)

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """Compute weighted binary cross-entropy over two logits."""
        del num_items_in_batch
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        loss = F.cross_entropy(outputs.logits, labels, weight=self.class_weights.to(outputs.logits.device))
        return (loss, outputs) if return_outputs else loss


def compute_class_weights(labels: Sequence[int]) -> torch.Tensor:
    """Return inverse-frequency weights normalized to mean sample weight one."""
    label_array = np.asarray(labels, dtype=np.int64)
    if label_array.ndim != 1 or label_array.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional sequence")
    if not set(np.unique(label_array)).issubset({0, 1}):
        raise ValueError("labels must contain only 0 and 1")

    counts = np.bincount(label_array, minlength=2)
    if np.any(counts == 0):
        raise ValueError("both classes must be present in training labels")
    weights = label_array.size / (2.0 * counts.astype(np.float64))
    return torch.tensor(weights, dtype=torch.float32)


def compute_pr_auc(eval_prediction: EvalPrediction | tuple[Any, Any]) -> dict[str, float]:
    """Compute threshold-free Average Precision from positive-class scores."""
    if isinstance(eval_prediction, EvalPrediction):
        logits = eval_prediction.predictions
        labels = eval_prediction.label_ids
    else:
        logits, labels = eval_prediction
    if isinstance(logits, tuple):
        logits = logits[0]
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if logits.ndim != 2 or logits.shape[1] != 2:
        raise ValueError("expected logits with shape (n_samples, 2)")

    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    probabilities = exponentiated[:, 1] / exponentiated.sum(axis=1)
    return {"pr_auc": float(average_precision_score(labels, probabilities))}

def compute_macro_pr_auc(
    eval_prediction: EvalPrediction | tuple[Any, Any],
    categories: Sequence[str],
) -> dict[str, float]:
    if isinstance(eval_prediction, EvalPrediction):
        logits = eval_prediction.predictions
        labels = eval_prediction.label_ids
    else:
        logits, labels = eval_prediction

    if isinstance(logits, tuple):
        logits = logits[0]

    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    categories = np.asarray(categories)

    shifted = logits - logits.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    probabilities = exponentiated[:, 1] / exponentiated.sum(axis=1)

    category_scores = [
        average_precision_score(
            labels[categories == category],
            probabilities[categories == category],
        )
        for category in np.unique(categories)
    ]

    return {"macro_pr_auc": float(np.mean(category_scores))}

def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _model_init(
    model_path: str,
    tokenizer: PreTrainedTokenizerBase,
    *,
    use_field_tokens: bool,
):
    def initialize_model(trial: Any | None = None) -> PreTrainedModel:
        del trial
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=2,
            id2label={0: "different", 1: "match"},
            label2id={"different": 0, "match": 1},
            ignore_mismatched_sizes=True,
        )
        if use_field_tokens:
            add_pair_special_tokens(tokenizer, model)
        return model

    return initialize_model


def _training_arguments(
    output_dir: Path,
    config: SequenceClassifierConfig,
    *,
    learning_rate: float,
    weight_decay: float,
) -> TrainingArguments:
    kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "num_train_epochs": config.max_epochs,
        "per_device_train_batch_size": 1024,
        "per_device_eval_batch_size": 2048,
        "auto_find_batch_size": True,
        "gradient_accumulation_steps": 1,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "max_grad_norm": 1.0,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 100,
        "logging_first_step": True,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_pr_auc",
        "greater_is_better": True,
        "save_total_limit": 1,
        "remove_unused_columns": False,
        "report_to": "none",
        "seed": config.seed,
        "data_seed": config.seed,
        "fp16": torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        "bf16": torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    }

    # Transformers 4.x used ``evaluation_strategy``; 5.x uses
    # ``eval_strategy``. Supporting both keeps the declared >=4.44 range valid.
    parameters = inspect.signature(TrainingArguments).parameters
    if "warmup_ratio" in parameters:
        kwargs["warmup_ratio"] = 0.06
    else:
        # Transformers 5 accepts a float below one as a ratio in
        # ``warmup_steps``.
        kwargs["warmup_steps"] = 0.06
    if "eval_strategy" not in parameters:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    return TrainingArguments(**kwargs)


def _hp_space(trial: Any) -> dict[str, float]:
    return {
        "learning_rate": trial.suggest_float(
            "learning_rate",
            1e-6,
            5e-5,
            log=True,
        ),
        "weight_decay": trial.suggest_float(
            "weight_decay",
            0.0,
            0.1,
        ),
    }


def _validate_validation_labels(labels: Sequence[int]) -> None:
    unique_labels = set(int(label) for label in labels)
    if unique_labels != {0, 1}:
        raise ValueError("validation labels must contain both classes for meaningful PR-AUC")


def _labels_from_pairs(
    pairs: Sequence[PreparedPair],
    *,
    split_name: str,
) -> list[int]:
    if not pairs:
        raise ValueError(f"{split_name}_pairs must not be empty")

    labels: list[int] = []
    for pair in pairs:
        if pair.label not in (0, 1):
            raise ValueError(f"{split_name}_pairs must all have binary labels")
        labels.append(int(pair.label))
    return labels


def train_sequence_classifier(
    train_pairs: Sequence[PreparedPair],
    validation_pairs: Sequence[PreparedPair],
    config: SequenceClassifierConfig,
    *,
    output_dir: str | Path,
) -> TrainingResult:
    """Fine-tune a cross-encoder and select the best model by macro PR-AUC.

    Labels and validation categories are read directly from ``PreparedPair``.
    Input length is inferred from structured training pairs, the largest
    feasible batch is found by ``Trainer``, and Optuna selects learning rate
    and weight decay.

    Train and validation pairs must be split before this call. In product
    matching, group-aware splitting by product id is preferred so the same
    product cannot leak across the two subsets.
    """
    train_labels = _labels_from_pairs(train_pairs, split_name="train")
    validation_labels = _labels_from_pairs(validation_pairs, split_name="validation")
    validation_categories = [pair.category for pair in validation_pairs]

    logger.info("Training sequence classifier")
    logger.info("Training pairs: {}", len(train_pairs))
    logger.info("Validation pairs: {}", len(validation_pairs))
    _validate_validation_labels(validation_labels)

    train_counts = np.bincount(np.asarray(train_labels, dtype=np.int64), minlength=2)
    logger.info(
        "Training labels: different={}, match={}",
        int(train_counts[0]),
        int(train_counts[1]),
    )

    class_weights = compute_class_weights(train_labels)
    logger.info("Class weights [different, match]: {}", class_weights.tolist())
    logger.info("Use [KEY]/[VAL] field tokens: {}", config.use_field_tokens)
    logger.info("Max tokens per attribute value: {}", config.max_attribute_value_tokens)

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    if config.use_field_tokens:
        add_pair_special_tokens(tokenizer)
    max_length = config.max_length
    if max_length is None:
        max_length = infer_pair_max_length(
            tokenizer,
            train_pairs,
            quantile=config.max_length_quantile,
            sample_size=config.max_length_sample_size,
            hard_cap=config.max_length_hard_cap,
            use_field_tokens=config.use_field_tokens,
            max_attribute_value_tokens=config.max_attribute_value_tokens,
        )
    logger.info("Max input length: {}", max_length)

    train_dataset = PreparedPairDataset(train_pairs)
    validation_dataset = PreparedPairDataset(validation_pairs)
    collator = PairEncodingCollator(
        tokenizer,
        max_length,
        use_field_tokens=config.use_field_tokens,
        max_attribute_value_tokens=config.max_attribute_value_tokens,
    )
    model_init = _model_init(
        config.model_path,
        tokenizer,
        use_field_tokens=config.use_field_tokens,
    )

    best_hyperparameters = {"learning_rate": 2e-5, "weight_decay": 0.01}
    validation_metric = partial(compute_macro_pr_auc, categories=validation_categories)
    if config.hpo_trials > 1:
        logger.info("Starting Optuna search: trials={}", config.hpo_trials)
        hpo_arguments = _training_arguments(output_path / ".hpo", config, **best_hyperparameters)
        hpo_trainer = WeightedSequenceTrainer(
            args=hpo_arguments,
            model_init=model_init,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            data_collator=collator,
            compute_metrics=validation_metric,
            class_weights=class_weights,
        )
        best_run = hpo_trainer.hyperparameter_search(
            backend="optuna",
            direction="maximize",
            hp_space=_hp_space,
            compute_objective=lambda metrics: metrics["eval_macro_pr_auc"],
            n_trials=config.hpo_trials,
        )
        best_hyperparameters.update(
            {
                key: float(value)
                for key, value in best_run.hyperparameters.items()
                if key in best_hyperparameters
            }
        )
        logger.info("Optuna completed")
    logger.info("Best hyperparameters: {}", best_hyperparameters)

    training_arguments = _training_arguments(output_path, config, **best_hyperparameters)
    trainer = WeightedSequenceTrainer(
        args=training_arguments,
        model_init=model_init,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=collator,
        compute_metrics=validation_metric,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        class_weights=class_weights,
    )
    trainer.train()
    logger.info("Training completed")
    metrics = trainer.evaluate()
    logger.info("Validation metrics: {}", metrics)
    trainer.model.config.match_max_length = max_length
    trainer.model.config.match_use_field_tokens = config.use_field_tokens
    trainer.model.config.match_max_attribute_value_tokens = config.max_attribute_value_tokens
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(output_path)

    actual_train_batch_size = int(getattr(trainer, "_train_batch_size", training_arguments.per_device_train_batch_size))
    resolved_config = ResolvedTrainingConfig(
        max_length=max_length,
        train_batch_size=actual_train_batch_size,
        eval_batch_size=training_arguments.per_device_eval_batch_size,
        gradient_accumulation_steps=training_arguments.gradient_accumulation_steps,
        learning_rate=best_hyperparameters["learning_rate"],
        weight_decay=best_hyperparameters["weight_decay"],
        use_field_tokens=config.use_field_tokens,
        max_attribute_value_tokens=config.max_attribute_value_tokens,
    )
    metadata = {
        "validation_macro_pr_auc": float(metrics["eval_macro_pr_auc"]),
        "best_hyperparameters": best_hyperparameters,
        "resolved_config": asdict(resolved_config),
    }
    logger.info("Saving metadata: {}", metadata)
    (output_path / "training_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Model and metadata saved to {!s}", output_path)
    return TrainingResult(
        model_dir=output_path,
        validation_macro_pr_auc=float(metrics["eval_macro_pr_auc"]),
        best_hyperparameters=best_hyperparameters,
        resolved_config=resolved_config,
    )


def load_trained_classifier(
    model_dir: str | Path,
    *,
    device: str | torch.device | None = None,
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Load a saved tokenizer and sequence classifier for inference."""
    target_device = torch.device(device) if device is not None else _resolve_device()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(target_device).eval()
    return tokenizer, model


def _saved_pair_encoding_settings(model: PreTrainedModel) -> tuple[bool, int | None]:
    return (
        bool(getattr(model.config, "match_use_field_tokens", True)),
        getattr(
            model.config,
            "match_max_attribute_value_tokens",
            DEFAULT_MAX_ATTRIBUTE_VALUE_TOKENS,
        ),
    )


def predict_match_probabilities(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    pairs: Sequence[PreparedPair],
    *,
    batch_size: int = 64,
    max_length: int | None = None,
) -> np.ndarray:
    """Return threshold-free positive-class probabilities for prepared pairs.

    Examples
    --------
    >>> probabilities = predict_match_probabilities(model, tokenizer, pairs)
    >>> probabilities.shape
    (len(pairs),)

    The returned scores can be passed directly to
    ``sklearn.metrics.average_precision_score``; no decision threshold is used.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not pairs:
        return np.empty(0, dtype=np.float32)
    use_field_tokens, max_attribute_value_tokens = _saved_pair_encoding_settings(model)
    if max_length is None:
        max_length = getattr(model.config, "match_max_length", None)
    if max_length is None:
        max_length = infer_pair_max_length(
            tokenizer,
            pairs,
            use_field_tokens=use_field_tokens,
            max_attribute_value_tokens=max_attribute_value_tokens,
        )

    dataset = PreparedPairDataset(pairs)
    collator = PairEncodingCollator(
        tokenizer,
        max_length,
        use_field_tokens=use_field_tokens,
        max_attribute_value_tokens=max_attribute_value_tokens,
        include_labels=False,
    )
    device = next(model.parameters()).device
    current_batch_size = batch_size

    while True:
        try:
            dataloader = DataLoader(
                dataset,
                batch_size=current_batch_size,
                shuffle=False,
                collate_fn=collator,
            )
            chunks: list[np.ndarray] = []
            model.eval()
            with torch.inference_mode():
                for batch in dataloader:
                    batch = {key: value.to(device) for key, value in batch.items()}
                    logits = model(**batch).logits
                    chunks.append(logits.softmax(dim=-1)[:, 1].float().cpu().numpy())
            return np.concatenate(chunks).astype(np.float32, copy=False)
        except torch.cuda.OutOfMemoryError:
            if current_batch_size == 1:
                raise
            current_batch_size = max(1, current_batch_size // 2)
            torch.cuda.empty_cache()


def encode_pair_cls(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    pairs: Sequence[PreparedPair],
    *,
    batch_size: int = 64,
    max_length: int | None = None,
) -> np.ndarray:
    """Return final-layer CLS embeddings for prepared product pairs.

    Examples
    --------
    >>> embeddings = encode_pair_cls(model, tokenizer, pairs)
    >>> embeddings.shape
    (len(pairs), model.config.hidden_size)
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not pairs:
        return np.empty((0, model.config.hidden_size), dtype=np.float32)
    use_field_tokens, max_attribute_value_tokens = _saved_pair_encoding_settings(model)
    if max_length is None:
        max_length = getattr(model.config, "match_max_length", None)
    if max_length is None:
        max_length = infer_pair_max_length(
            tokenizer,
            pairs,
            use_field_tokens=use_field_tokens,
            max_attribute_value_tokens=max_attribute_value_tokens,
        )

    dataset = PreparedPairDataset(pairs)
    collator = PairEncodingCollator(
        tokenizer,
        max_length,
        use_field_tokens=use_field_tokens,
        max_attribute_value_tokens=max_attribute_value_tokens,
        include_labels=False,
    )
    device = next(model.parameters()).device
    current_batch_size = batch_size
    while True:
        try:
            dataloader = DataLoader(
                dataset,
                batch_size=current_batch_size,
                shuffle=False,
                collate_fn=collator,
            )
            chunks: list[np.ndarray] = []
            model.eval()
            with torch.inference_mode():
                for batch in dataloader:
                    batch = {key: value.to(device) for key, value in batch.items()}
                    embeddings = model.base_model(**batch, return_dict=True).last_hidden_state[:, 0, :]
                    chunks.append(embeddings.float().cpu().numpy())
            return np.concatenate(chunks).astype(np.float32, copy=False)
        except torch.cuda.OutOfMemoryError:
            if current_batch_size == 1:
                raise
            current_batch_size = max(1, current_batch_size // 2)
            torch.cuda.empty_cache()
