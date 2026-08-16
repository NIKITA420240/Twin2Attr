"""End-to-end transformer training for product-card matching.

The module deliberately accepts already prepared text pairs. Reading parquet
files, joining product ids and formatting product text belong to the data
pipeline, while this module owns tokenization, optimization and inference.

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

import importlib.util
import inspect
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.utils.data import DataLoader, Dataset
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

__all__ = [
    "PairDataCollator",
    "ResolvedTrainingConfig",
    "SequenceClassifierConfig",
    "SequencePairDataset",
    "TrainingResult",
    "compute_class_weights",
    "compute_pr_auc",
    "infer_max_length",
    "load_trained_classifier",
    "predict_match_probabilities",
    "train_sequence_classifier",
]

TextPair = tuple[str, str]


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
        Number of Optuna trials used to maximize validation PR-AUC. A value of
        one disables the search and uses conservative defaults.
    seed:
        Random seed shared by sampling, training and hyperparameter search.

    Notes
    -----
    Sequence length and the largest feasible training batch are inferred
    automatically. Learning rate and weight decay are selected by Optuna when
    ``hpo_trials > 1``. Warmup and gradient clipping are internal stability
    policy rather than public hyperparameters.
    """

    model_path: str
    max_epochs: int = 5
    hpo_trials: int = 10
    seed: int = 42

    def __post_init__(self) -> None:
        if not self.model_path.strip():
            raise ValueError("model_path must not be empty")
        if self.max_epochs < 1:
            raise ValueError("max_epochs must be positive")
        if self.hpo_trials < 1:
            raise ValueError("hpo_trials must be positive")


@dataclass(frozen=True)
class ResolvedTrainingConfig:
    """Concrete parameters selected from data, hardware and HPO."""

    max_length: int
    train_batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float = 0.06
    gradient_clip_norm: float = 1.0


@dataclass(frozen=True)
class TrainingResult:
    """Summary of a completed fine-tuning run."""

    model_dir: Path
    validation_pr_auc: float
    best_hyperparameters: dict[str, float]
    resolved_config: ResolvedTrainingConfig


class SequencePairDataset(Dataset):
    """Dataset of text pairs and optional binary match labels.

    Parameters
    ----------
    text_pairs:
        Pairs in ``(first_card_text, second_card_text)`` order.
    labels:
        Optional labels where 0 means different products and 1 means a match.

    Examples
    --------
    >>> pairs = [("black office chair", "черное офисное кресло")]
    >>> dataset = SequencePairDataset(pairs, labels=[1])
    >>> dataset[0]["label"]
    1
    """

    def __init__(self, text_pairs: Sequence[TextPair], labels: Sequence[int] | None = None) -> None:
        self._pairs = [(str(left), str(right)) for left, right in text_pairs]
        self._labels = None if labels is None else [int(label) for label in labels]

        if not self._pairs:
            raise ValueError("text_pairs must not be empty")
        if self._labels is not None and len(self._pairs) != len(self._labels):
            raise ValueError("text_pairs and labels must have equal lengths")
        if self._labels is not None and not set(self._labels).issubset({0, 1}):
            raise ValueError("labels must contain only 0 and 1")

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> dict[str, Any]:
        left, right = self._pairs[index]
        item: dict[str, Any] = {"text1": left, "text2": right}
        if self._labels is not None:
            item["label"] = self._labels[index]
        return item


class PairDataCollator:
    """Dynamically tokenize and pad a batch of text pairs."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase, max_length: int) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = self.tokenizer(
            text=[item["text1"] for item in items],
            text_pair=[item["text2"] for item in items],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        if "label" in items[0]:
            batch["labels"] = torch.tensor([item["label"] for item in items], dtype=torch.long)
        return batch


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


def infer_max_length(
    tokenizer: PreTrainedTokenizerBase,
    text_pairs: Sequence[TextPair],
    *,
    quantile: float = 0.95,
    sample_size: int = 10_000,
    hard_cap: int = 512,
) -> int:
    """Infer a token limit covering most pairs without wasting computation.

    The selected length is the requested empirical quantile, rounded up to a
    multiple of eight and capped by both the tokenizer limit and ``hard_cap``.

    Examples
    --------
    If 95% of sampled pairs contain at most 233 tokens, the result is 240.
    """
    if not text_pairs:
        raise ValueError("text_pairs must not be empty")
    if not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must be in (0, 1]")
    if sample_size < 1 or hard_cap < 8:
        raise ValueError("sample_size must be positive and hard_cap at least 8")

    if len(text_pairs) > sample_size:
        indices = np.linspace(0, len(text_pairs) - 1, sample_size, dtype=int)
        sample = [text_pairs[index] for index in indices]
    else:
        sample = text_pairs

    encoded = tokenizer(
        text=[str(pair[0]) for pair in sample],
        text_pair=[str(pair[1]) for pair in sample],
        add_special_tokens=True,
        padding=False,
        truncation=False,
    )
    lengths = np.fromiter((len(token_ids) for token_ids in encoded["input_ids"]), dtype=np.int32)
    selected = int(np.quantile(lengths, quantile, method="higher"))
    selected = max(8, int(math.ceil(selected / 8) * 8))

    model_limit = getattr(tokenizer, "model_max_length", hard_cap)
    if not isinstance(model_limit, int) or model_limit <= 0 or model_limit > 100_000:
        model_limit = hard_cap
    return min(selected, model_limit, hard_cap)


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


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _model_init(model_path: str):
    def initialize_model(trial: Any | None = None) -> PreTrainedModel:
        del trial
        return AutoModelForSequenceClassification.from_pretrained(
            model_path,
            num_labels=2,
            id2label={0: "different", 1: "match"},
            label2id={"different": 0, "match": 1},
            ignore_mismatched_sizes=True,
        )

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
        "per_device_train_batch_size": 64,
        "per_device_eval_batch_size": 64,
        "auto_find_batch_size": True,
        "gradient_accumulation_steps": 1,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "max_grad_norm": 1.0,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "pr_auc",
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
        raise ValueError(
            "validation labels must contain both classes for meaningful PR-AUC"
        )


def train_sequence_classifier(
    train_pairs: Sequence[TextPair],
    train_labels: Sequence[int],
    validation_pairs: Sequence[TextPair],
    validation_labels: Sequence[int],
    config: SequenceClassifierConfig,
    *,
    output_dir: str | Path,
) -> TrainingResult:
    """Fine-tune a cross-encoder and select the best model by PR-AUC.

    Only the checkpoint and compute budget are user-controlled. Input length is
    inferred from the training distribution, the largest feasible batch is
    found by ``Trainer``, and Optuna selects learning rate and weight decay.

    Train and validation pairs must be split before this call. In product
    matching, group-aware splitting by product id is preferred so the same
    product cannot leak across the two subsets.
    """
    if len(train_pairs) != len(train_labels):
        raise ValueError("train_pairs and train_labels must have equal lengths")
    if len(validation_pairs) != len(validation_labels):
        raise ValueError("validation_pairs and validation_labels must have equal lengths")
    _validate_validation_labels(validation_labels)
    class_weights = compute_class_weights(train_labels)

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(config.model_path)
    max_length = infer_max_length(tokenizer, train_pairs)

    train_dataset = SequencePairDataset(train_pairs, train_labels)
    validation_dataset = SequencePairDataset(validation_pairs, validation_labels)
    collator = PairDataCollator(tokenizer, max_length)
    model_init = _model_init(config.model_path)

    best_hyperparameters = {"learning_rate": 2e-5, "weight_decay": 0.01}

    if config.hpo_trials > 1:
        if importlib.util.find_spec("optuna") is None:
            raise ImportError("Optuna is required when hpo_trials > 1; install project dependencies")
        hpo_arguments = _training_arguments(output_path / ".hpo", config, **best_hyperparameters)
        hpo_trainer = WeightedSequenceTrainer(
            args=hpo_arguments,
            model_init=model_init,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            data_collator=collator,
            compute_metrics=compute_pr_auc,
            class_weights=class_weights,
        )
        best_run = hpo_trainer.hyperparameter_search(
            backend="optuna",
            direction="maximize",
            hp_space=_hp_space,
            compute_objective=lambda metrics: metrics["eval_pr_auc"],
            n_trials=config.hpo_trials,
        )
        best_hyperparameters.update(
            {
                key: float(value)
                for key, value in best_run.hyperparameters.items()
                if key in best_hyperparameters
            }
        )

    training_arguments = _training_arguments(output_path, config, **best_hyperparameters)
    trainer = WeightedSequenceTrainer(
        args=training_arguments,
        model_init=model_init,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=collator,
        compute_metrics=compute_pr_auc,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        class_weights=class_weights,
    )
    trainer.train()
    metrics = trainer.evaluate()
    trainer.model.config.match_max_length = max_length
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
    )
    metadata = {"validation_pr_auc": float(metrics["eval_pr_auc"]), "best_hyperparameters": best_hyperparameters, "resolved_config": asdict(resolved_config)}
    (output_path / "training_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return TrainingResult(
        model_dir=output_path,
        validation_pr_auc=float(metrics["eval_pr_auc"]),
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


def predict_match_probabilities(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    text_pairs: Sequence[TextPair],
    *,
    batch_size: int = 64,
    max_length: int | None = None,
) -> np.ndarray:
    """Return threshold-free positive-class probabilities for text pairs.

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
    if not text_pairs:
        return np.empty(0, dtype=np.float32)
    if max_length is None:
        max_length = getattr(model.config, "match_max_length", None)
    if max_length is None:
        max_length = infer_max_length(tokenizer, text_pairs)

    dataset = SequencePairDataset(text_pairs)
    collator = PairDataCollator(tokenizer, max_length)
    device = next(model.parameters()).device
    current_batch_size = batch_size

    while True:
        try:
            dataloader = DataLoader(dataset, batch_size=current_batch_size, shuffle=False, collate_fn=collator)
            chunks: list[np.ndarray] = []
            model.eval()
            with torch.inference_mode():
                for batch in dataloader:
                    batch = {key: value.to(device) for key, value in batch.items()}
                    logits = model(**batch).logits
                    chunks.append(
                        logits.softmax(dim=-1)[:, 1].float().cpu().numpy()
                    )
            return np.concatenate(chunks).astype(np.float32, copy=False)
        except torch.cuda.OutOfMemoryError:
            if current_batch_size == 1:
                raise
            current_batch_size = max(1, current_batch_size // 2)
            torch.cuda.empty_cache()
