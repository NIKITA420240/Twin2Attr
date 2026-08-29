"""Fine-tuning and HPO for the Transformer cross-encoder."""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from loguru import logger
from transformers import AutoTokenizer, EarlyStoppingCallback, TrainingArguments

from ...augmentations import AttributeWordDropoutAugmenter
from ...config import AppConfig
from ...data import TrainingData
from ...pair_encoding import (
    PairEncodingCollator,
    PreparedPairDataset,
    add_pair_special_tokens,
    infer_pair_max_length,
    pair_special_token_ids,
)
from ...prepare_data import PreparedPair
from ..artifacts import TrainingArtifacts
from .config import ResolvedTrainingConfig, SequenceClassifierConfig, TrainingResult
from .head import PoolingHeadConfig
from .metrics import compute_class_weights, compute_macro_pr_auc
from .construction import model_factory
from .model import WeightedSequenceTrainer
from .profile import (
    TransformerArtifactContract,
    TransformerRuntimeContract,
    is_prompted_profile,
)
from .optimizer import LearningRateMultipliers


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
        "per_device_train_batch_size": config.train_batch_size,
        "per_device_eval_batch_size": config.eval_batch_size,
        "auto_find_batch_size": config.auto_find_batch_size,
        "gradient_accumulation_steps": config.gradient_accumulation_steps,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "optim": "adamw_torch",
        "lr_scheduler_type": config.lr_scheduler_type,
        "max_grad_norm": config.max_grad_norm,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": 100,
        "logging_first_step": True,
        "disable_tqdm": True,
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
    parameters = inspect.signature(TrainingArguments).parameters
    if "warmup_ratio" in parameters:
        kwargs["warmup_ratio"] = config.warmup_ratio
    else:
        kwargs["warmup_steps"] = 0
    if "eval_strategy" not in parameters:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")
    return TrainingArguments(**kwargs)


def _hp_space(
    trial: Any,
    config: SequenceClassifierConfig,
) -> dict[str, float]:
    return {
        "learning_rate": trial.suggest_float(
            "learning_rate",
            config.hpo_learning_rate_min,
            config.hpo_learning_rate_max,
            log=True,
        ),
        "weight_decay": trial.suggest_float(
            "weight_decay",
            config.hpo_weight_decay_min,
            config.hpo_weight_decay_max,
        ),
    }


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
    train_pair_transform: Callable[[PreparedPair], PreparedPair] | None = None,
) -> TrainingResult:
    train_labels = _labels_from_pairs(train_pairs, split_name="train")
    validation_labels = _labels_from_pairs(
        validation_pairs,
        split_name="validation",
    )
    if set(validation_labels) != {0, 1}:
        raise ValueError(
            "validation labels must contain both classes for meaningful PR-AUC"
        )
    validation_categories = [pair.category for pair in validation_pairs]

    logger.info("Training sequence classifier")
    logger.info("Training pairs: {}", len(train_pairs))
    logger.info("Validation pairs: {}", len(validation_pairs))
    train_counts = np.bincount(np.asarray(train_labels, dtype=np.int64), minlength=2)
    logger.info(
        "Training labels: different={}, match={}",
        int(train_counts[0]),
        int(train_counts[1]),
    )
    class_weights = compute_class_weights(
        [
            pair.training_target
            if pair.training_target is not None
            else float(pair.label)
            for pair in train_pairs
        ],
        [pair.sample_weight for pair in train_pairs],
    )
    logger.info("Class weights [different, match]: {}", class_weights.tolist())

    output_path = Path(output_dir).resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_path,
        trust_remote_code=is_prompted_profile(config.profile),
    )
    if config.use_field_tokens:
        add_pair_special_tokens(tokenizer)
        if config.train_new_token_embeddings_only:
            logger.info(
                "Word-embedding updates restricted to field token ids: {}",
                pair_special_token_ids(tokenizer),
            )
    max_length = config.max_length
    if max_length is None:
        max_length = infer_pair_max_length(
            tokenizer,
            train_pairs,
            quantile=config.max_length_quantile,
            sample_size=config.max_length_sample_size,
            hard_cap=config.max_length_hard_cap,
            use_field_tokens=config.use_field_tokens,
            max_attribute_value_chars=config.max_attribute_value_chars,
            max_attribute_value_tokens=config.max_attribute_value_tokens,
            profile=config.profile,
        )
    logger.info("Max input length: {}", max_length)

    train_dataset = PreparedPairDataset(
        train_pairs,
        transform=train_pair_transform,
    )
    validation_dataset = PreparedPairDataset(validation_pairs)
    collator = PairEncodingCollator(
        tokenizer,
        max_length,
        use_field_tokens=config.use_field_tokens,
        max_attribute_value_chars=config.max_attribute_value_chars,
        max_attribute_value_tokens=config.max_attribute_value_tokens,
        batch_fields=config.batch_fields,
        field_chunk_size=config.field_chunk_size,
        profile=config.profile,
    )
    initialize_model = model_factory(
        config.model_path,
        tokenizer,
        use_field_tokens=config.use_field_tokens,
        train_new_token_embeddings_only=config.train_new_token_embeddings_only,
        train_last_n_layers=config.train_last_n_layers,
        head_type=config.head_type,
        head_config=config.head_config,
        profile=config.profile,
    )
    best_hyperparameters = {
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
    }
    learning_rate_multipliers = LearningRateMultipliers.from_learning_rates(
        embeddings_lr=config.resolved_embeddings_learning_rate,
        backbone_lr=config.learning_rate,
        head_lr=config.resolved_head_learning_rate,
        layerwise_decay=config.layerwise_lr_decay,
    )
    validation_metric = partial(
        compute_macro_pr_auc,
        categories=validation_categories,
    )
    if config.hpo_trials > 1:
        hpo_trainer = WeightedSequenceTrainer(
            args=_training_arguments(
                output_path / ".hpo",
                config,
                **best_hyperparameters,
            ),
            model_init=initialize_model,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            data_collator=collator,
            compute_metrics=validation_metric,
            class_weights=class_weights,
            learning_rate_multipliers=learning_rate_multipliers,
        )
        best_run = hpo_trainer.hyperparameter_search(
            backend="optuna",
            direction="maximize",
            hp_space=partial(_hp_space, config=config),
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

    training_arguments = _training_arguments(
        output_path,
        config,
        **best_hyperparameters,
    )
    trainer = WeightedSequenceTrainer(
        args=training_arguments,
        model_init=initialize_model,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=collator,
        compute_metrics=validation_metric,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=config.early_stopping_patience
            )
        ],
        class_weights=class_weights,
        learning_rate_multipliers=learning_rate_multipliers,
    )
    trainer.train()
    metrics = trainer.evaluate()
    output_contract = TransformerArtifactContract.for_training(
        profile=config.profile,
        head_type=config.head_type,
        num_logits=int(trainer.model.config.num_labels),
    )
    output_contract.apply_to(trainer.model.config)
    TransformerRuntimeContract(
        output=output_contract,
        hidden_size=int(trainer.model.config.hidden_size),
        max_length=max_length,
        use_field_tokens=config.use_field_tokens,
        max_attribute_value_chars=config.max_attribute_value_chars,
        max_attribute_value_tokens=config.max_attribute_value_tokens,
    ).apply_encoding_to(trainer.model.config)
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(output_path)
    if config.onnx_export_enabled:
        from .onnx_export import export_transformer_to_onnx

        export_transformer_to_onnx(
            output_path,
            opset=config.onnx_opset,
            precision=config.onnx_precision,
            dynamic_batch=config.onnx_dynamic_batch,
            dynamic_sequence_length=config.onnx_dynamic_sequence_length,
            export_classifier=config.onnx_export_classifier,
            export_encoder=config.onnx_export_encoder,
        )

    actual_batch_size = int(
        getattr(
            trainer,
            "_train_batch_size",
            training_arguments.per_device_train_batch_size,
        )
    )
    resolved_config = ResolvedTrainingConfig(
        max_epochs=config.max_epochs,
        max_length=max_length,
        train_batch_size=actual_batch_size,
        eval_batch_size=training_arguments.per_device_eval_batch_size,
        gradient_accumulation_steps=training_arguments.gradient_accumulation_steps,
        learning_rate=best_hyperparameters["learning_rate"],
        embeddings_learning_rate=(
            best_hyperparameters["learning_rate"]
            * learning_rate_multipliers.embeddings
        ),
        train_new_token_embeddings_only=config.train_new_token_embeddings_only,
        train_last_n_layers=config.train_last_n_layers,
        head_learning_rate=(
            best_hyperparameters["learning_rate"] * learning_rate_multipliers.head
        ),
        layerwise_lr_decay=learning_rate_multipliers.layerwise_decay,
        weight_decay=best_hyperparameters["weight_decay"],
        use_field_tokens=config.use_field_tokens,
        batch_fields=config.batch_fields,
        field_chunk_size=config.field_chunk_size,
        max_attribute_value_chars=config.max_attribute_value_chars,
        max_attribute_value_tokens=config.max_attribute_value_tokens,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        gradient_clip_norm=config.max_grad_norm,
        early_stopping_patience=config.early_stopping_patience,
        auto_find_batch_size=config.auto_find_batch_size,
    )
    metadata = {
        "profile": config.profile,
        "head_type": config.head_type,
        "head_config": config.head_config.to_dict(),
        "num_logits": int(trainer.model.config.num_labels),
        "probability_transform": (
            "sigmoid" if int(trainer.model.config.num_labels) == 1 else "softmax"
        ),
        "validation_macro_pr_auc": float(metrics["eval_macro_pr_auc"]),
        "best_hyperparameters": best_hyperparameters,
        "resolved_config": asdict(resolved_config),
    }
    (output_path / "training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Model and metadata saved to {!s}", output_path)
    return TrainingResult(
        model_dir=output_path,
        validation_macro_pr_auc=float(metrics["eval_macro_pr_auc"]),
        best_hyperparameters=best_hyperparameters,
        resolved_config=resolved_config,
    )


def _sequence_config(config: AppConfig) -> SequenceClassifierConfig:
    parameters = config.model_description.transformer
    encoding = parameters.pair_encoding
    batch_fields = parameters.tokenizer.batch_fields
    onnx_export = parameters.export.onnx
    return SequenceClassifierConfig(
        model_path=parameters.pretrained_model_path,
        profile=parameters.profile,
        max_epochs=parameters.max_epochs,
        hpo_trials=parameters.hpo_trials,
        learning_rate=parameters.learning_rate,
        embeddings_learning_rate=parameters.embeddings_learning_rate,
        train_new_token_embeddings_only=parameters.train_new_token_embeddings_only,
        train_last_n_layers=parameters.train_last_n_layers,
        lr_scheduler_type=parameters.lr_scheduler_type,
        head_learning_rate=parameters.head_learning_rate,
        layerwise_lr_decay=parameters.layerwise_lr_decay,
        weight_decay=parameters.weight_decay,
        hpo_learning_rate_min=parameters.hpo_learning_rate_min,
        hpo_learning_rate_max=parameters.hpo_learning_rate_max,
        hpo_weight_decay_min=parameters.hpo_weight_decay_min,
        hpo_weight_decay_max=parameters.hpo_weight_decay_max,
        train_batch_size=parameters.train_batch_size,
        eval_batch_size=parameters.eval_batch_size,
        gradient_accumulation_steps=parameters.gradient_accumulation_steps,
        warmup_ratio=parameters.warmup_ratio,
        max_grad_norm=parameters.max_grad_norm,
        early_stopping_patience=parameters.early_stopping_patience,
        auto_find_batch_size=parameters.auto_find_batch_size,
        seed=config.runtime.seed,
        use_field_tokens=encoding.use_field_tokens,
        batch_fields=batch_fields.enabled,
        field_chunk_size=batch_fields.chunk_size,
        max_attribute_value_chars=encoding.max_attribute_value_chars,
        max_attribute_value_tokens=encoding.max_attribute_value_tokens,
        max_length=encoding.max_length,
        max_length_quantile=encoding.quantile,
        max_length_sample_size=encoding.sample_size,
        max_length_hard_cap=encoding.hard_cap,
        head_type=parameters.head.type,
        head_config=PoolingHeadConfig(
            poolings=parameters.head.poolings,
            mlp_hidden_dims=parameters.head.mlp_hidden_dims,
            dropout=parameters.head.dropout,
            attention_hidden_dim=parameters.head.attention_hidden_dim,
            attention_num_heads=parameters.head.attention_num_heads,
        ),
        onnx_export_enabled=onnx_export.enabled,
        onnx_opset=onnx_export.opset,
        onnx_precision=onnx_export.precision,
        onnx_dynamic_batch=onnx_export.dynamic_batch,
        onnx_dynamic_sequence_length=onnx_export.dynamic_sequence_length,
        onnx_export_classifier=onnx_export.export_classifier,
        onnx_export_encoder=onnx_export.export_encoder,
    )


@dataclass(frozen=True, slots=True)
class TransformerTrainer:
    config: AppConfig

    def train(self, data: TrainingData) -> TrainingArtifacts:
        train_pair_transform = None
        if self.config.training.augmentation_model == "attribute_word_dropout":
            settings = self.config.augmentation_models.attribute_word_dropout
            train_pair_transform = AttributeWordDropoutAugmenter(settings)
            logger.info(
                "Dynamic training augmentation enabled: pair={}, "
                "attribute_dropout={}, word_dropout={}, keyboard_typo={}, "
                "word_shuffle={}",
                settings.pair_probability,
                settings.attribute_dropout_probability,
                settings.word_dropout_probability,
                settings.keyboard_typo_probability,
                settings.word_shuffle_probability,
            )
        result = train_sequence_classifier(
            data.train_pairs,
            data.validation_pairs,
            _sequence_config(self.config),
            output_dir=self.config.model_description.transformer.artifact_dir,
            train_pair_transform=train_pair_transform,
        )
        return TrainingArtifacts(
            predictor="transformer",
            transformer_dir=result.model_dir,
            metrics=((
                "transformer.validation_macro_pr_auc",
                result.validation_macro_pr_auc,
            ),),
        )


__all__ = ["TransformerTrainer", "train_sequence_classifier"]
