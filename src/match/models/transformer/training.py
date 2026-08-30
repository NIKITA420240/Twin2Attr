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
from .batching import estimated_pair_lengths
from .head import PoolingHeadConfig
from .metrics import compute_class_weights, compute_macro_pr_auc
from .construction import model_factory
from .model import WeightedSequenceTrainer
from .profile import (
    TransformerArtifactContract,
    TransformerRuntimeContract,
    head_uses_typed_features,
    is_prompted_profile,
)
from .optimizer import LearningRateMultipliers
from .train_runtime import (
    EpochPerformanceCallback,
    EpochPerformanceTracker,
    stratified_sample_indices,
)
from .token_cache import load_or_build_sharded_token_cache
from .typed_fusion import (
    TYPED_FUSION_SCHEMA_VERSION,
    TypedFeatureDataset,
    TypedPairEncodingCollator,
    build_typed_feature_matrix,
)


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
        "dataloader_num_workers": config.dataloader_num_workers,
        "dataloader_pin_memory": config.dataloader_pin_memory,
    }
    parameters = inspect.signature(TrainingArguments).parameters
    if config.dataloader_num_workers > 0:
        if "dataloader_prefetch_factor" in parameters:
            kwargs["dataloader_prefetch_factor"] = (
                config.dataloader_prefetch_factor
            )
        if "dataloader_persistent_workers" in parameters:
            kwargs["dataloader_persistent_workers"] = (
                config.dataloader_persistent_workers
            )
    if "torch_compile" in parameters:
        kwargs["torch_compile"] = config.torch_compile
    if config.torch_compile and "torch_compile_mode" in parameters:
        kwargs["torch_compile_mode"] = config.torch_compile_mode
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


def _prepare_pair_datasets(
    train_pairs: Sequence[PreparedPair],
    validation_pairs: Sequence[PreparedPair],
    tokenizer: Any,
    collator: PairEncodingCollator,
    config: SequenceClassifierConfig,
    *,
    train_pair_transform: Callable[[PreparedPair], PreparedPair] | None,
    train_cache_shards: Sequence[object] | None = None,
    validation_cache_shards: Sequence[object] | None = None,
) -> tuple[Any, Any]:
    train_dataset: Any = PreparedPairDataset(
        train_pairs,
        transform=train_pair_transform,
    )
    validation_dataset: Any = PreparedPairDataset(validation_pairs)
    if not config.token_cache_enabled:
        return train_dataset, validation_dataset

    validation_dataset = load_or_build_sharded_token_cache(
        validation_pairs,
        (
            validation_cache_shards
            if validation_cache_shards is not None
            else ("validation",) * len(validation_pairs)
        ),
        tokenizer,
        collator,
        config.token_cache_directory,
        split_name="validation",
        chunk_size=config.token_cache_build_chunk_size,
    )
    if train_pair_transform is None:
        train_dataset = load_or_build_sharded_token_cache(
            train_pairs,
            (
                train_cache_shards
                if train_cache_shards is not None
                else ("train",) * len(train_pairs)
            ),
            tokenizer,
            collator,
            config.token_cache_directory,
            split_name="train",
            chunk_size=config.token_cache_build_chunk_size,
        )
    else:
        logger.info(
            "Transformer train token cache bypassed because dynamic "
            "augmentation is enabled"
        )
    return train_dataset, validation_dataset


def train_sequence_classifier(
    train_pairs: Sequence[PreparedPair],
    validation_pairs: Sequence[PreparedPair],
    config: SequenceClassifierConfig,
    *,
    output_dir: str | Path,
    train_pair_transform: Callable[[PreparedPair], PreparedPair] | None = None,
    train_cache_shards: Sequence[object] | None = None,
    validation_cache_shards: Sequence[object] | None = None,
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
    train_pair_lengths = None
    if config.train_length_bucketing:
        train_pair_lengths = estimated_pair_lengths(
            train_pairs,
            max_attribute_value_tokens=config.max_attribute_value_tokens,
            max_attribute_value_chars=config.max_attribute_value_chars,
        )
    logger.info(
        "Train runtime: length_bucketing={}, mega_batch_multiplier={}, "
        "padding_length_buckets={}, workers={}, prefetch_factor={}, "
        "persistent_workers={}, pin_memory={}, non_blocking_transfer={}, "
        "attention={}, torch_compile={}",
        config.train_length_bucketing,
        config.train_mega_batch_multiplier,
        config.train_padding_length_buckets,
        config.dataloader_num_workers,
        config.dataloader_prefetch_factor,
        config.dataloader_persistent_workers,
        config.dataloader_pin_memory,
        config.non_blocking_transfer,
        config.attention_implementation,
        config.torch_compile,
    )

    base_collator = PairEncodingCollator(
        tokenizer,
        max_length,
        use_field_tokens=config.use_field_tokens,
        max_attribute_value_chars=config.max_attribute_value_chars,
        max_attribute_value_tokens=config.max_attribute_value_tokens,
        batch_fields=config.batch_fields,
        field_chunk_size=config.field_chunk_size,
        padding_length_buckets=config.train_padding_length_buckets,
        profile=config.profile,
    )
    train_dataset, validation_dataset = _prepare_pair_datasets(
        train_pairs,
        validation_pairs,
        tokenizer,
        base_collator,
        config,
        train_pair_transform=train_pair_transform,
        train_cache_shards=train_cache_shards,
        validation_cache_shards=validation_cache_shards,
    )
    typed_feature_names: tuple[str, ...] = ()
    validation_typed_features = None
    uses_typed_fusion = head_uses_typed_features(config.head_type)
    if uses_typed_fusion:
        logger.info("Precomputing typed fusion features")
        typed_feature_names, train_typed_features = build_typed_feature_matrix(
            train_pairs,
            config.typed_attribute_options,
        )
        validation_names, validation_typed_features = build_typed_feature_matrix(
            validation_pairs,
            config.typed_attribute_options,
        )
        if validation_names != typed_feature_names:
            raise RuntimeError("train and validation typed schemas differ")
        train_dataset = TypedFeatureDataset(train_dataset, train_typed_features)
        validation_dataset = TypedFeatureDataset(
            validation_dataset,
            validation_typed_features,
        )
        collator = TypedPairEncodingCollator(
            base_collator,
            options=config.typed_attribute_options,
            feature_names=typed_feature_names,
        )
        logger.info(
            "Typed fusion schema: version={}, features={}",
            TYPED_FUSION_SCHEMA_VERSION,
            len(typed_feature_names),
        )
    else:
        collator = base_collator
    initialize_model = model_factory(
        config.model_path,
        tokenizer,
        use_field_tokens=config.use_field_tokens,
        special_token_initialization_enabled=(
            config.special_token_initialization_enabled
        ),
        special_token_key_seed_texts=config.special_token_key_seed_texts,
        special_token_value_seed_texts=config.special_token_value_seed_texts,
        train_new_token_embeddings_only=config.train_new_token_embeddings_only,
        train_last_n_layers=config.train_last_n_layers,
        head_type=config.head_type,
        head_config=config.head_config,
        typed_feature_count=len(typed_feature_names),
        profile=config.profile,
        attention_implementation=config.attention_implementation,
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
    fast_dev_dataset = None
    fast_dev_metric = None
    if config.fast_dev_validation_enabled:
        fast_dev_indices = stratified_sample_indices(
            validation_labels,
            max_rows=config.fast_dev_validation_max_rows,
            seed=config.seed,
        )
        fast_dev_pairs = [validation_pairs[index] for index in fast_dev_indices]
        fast_dev_dataset = (
            load_or_build_sharded_token_cache(
                fast_dev_pairs,
                (
                    [validation_cache_shards[index] for index in fast_dev_indices]
                    if validation_cache_shards is not None
                    else ("validation",) * len(fast_dev_pairs)
                ),
                tokenizer,
                base_collator,
                config.token_cache_directory,
                split_name="fast-dev",
                chunk_size=config.token_cache_build_chunk_size,
            )
            if config.token_cache_enabled
            else PreparedPairDataset(fast_dev_pairs)
        )
        if uses_typed_fusion:
            if validation_typed_features is None:
                raise RuntimeError("typed validation features are unavailable")
            fast_dev_dataset = TypedFeatureDataset(
                fast_dev_dataset,
                validation_typed_features[fast_dev_indices],
            )
        fast_dev_metric = partial(
            compute_macro_pr_auc,
            categories=[pair.category for pair in fast_dev_pairs],
        )
        logger.info(
            "Fast-dev validation: rows={}, requested_rows={}, every_steps={}",
            len(fast_dev_pairs),
            config.fast_dev_validation_max_rows,
            config.fast_dev_validation_every_n_optimizer_steps,
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
            train_pair_lengths=train_pair_lengths,
            length_bucketing=config.train_length_bucketing,
            mega_batch_multiplier=config.train_mega_batch_multiplier,
            non_blocking_transfer=config.non_blocking_transfer,
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
    performance_tracker = EpochPerformanceTracker(
        enabled=config.performance_logging
    )
    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=config.early_stopping_patience
        )
    ]
    if config.performance_logging:
        callbacks.append(EpochPerformanceCallback(performance_tracker))
    trainer = WeightedSequenceTrainer(
        args=training_arguments,
        model_init=initialize_model,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=collator,
        compute_metrics=validation_metric,
        callbacks=callbacks,
        class_weights=class_weights,
        learning_rate_multipliers=learning_rate_multipliers,
        train_pair_lengths=train_pair_lengths,
        length_bucketing=config.train_length_bucketing,
        mega_batch_multiplier=config.train_mega_batch_multiplier,
        non_blocking_transfer=config.non_blocking_transfer,
        performance_tracker=performance_tracker,
        fast_dev_dataset=fast_dev_dataset,
        fast_dev_compute_metrics=fast_dev_metric,
        fast_dev_every_n_optimizer_steps=(
            config.fast_dev_validation_every_n_optimizer_steps
            if fast_dev_dataset is not None
            else None
        ),
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
        typed_attribute_options=(
            config.typed_attribute_options if uses_typed_fusion else None
        ),
        typed_feature_names=typed_feature_names,
        typed_feature_schema_version=(
            TYPED_FUSION_SCHEMA_VERSION if uses_typed_fusion else None
        ),
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
        special_token_initialization_enabled=(
            config.special_token_initialization_enabled
        ),
        special_token_key_seed_texts=config.special_token_key_seed_texts,
        special_token_value_seed_texts=config.special_token_value_seed_texts,
        fast_dev_validation_enabled=config.fast_dev_validation_enabled,
        fast_dev_validation_max_rows=config.fast_dev_validation_max_rows,
        fast_dev_validation_every_n_optimizer_steps=(
            config.fast_dev_validation_every_n_optimizer_steps
        ),
        batch_fields=config.batch_fields,
        field_chunk_size=config.field_chunk_size,
        max_attribute_value_chars=config.max_attribute_value_chars,
        max_attribute_value_tokens=config.max_attribute_value_tokens,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        gradient_clip_norm=config.max_grad_norm,
        early_stopping_patience=config.early_stopping_patience,
        auto_find_batch_size=config.auto_find_batch_size,
        train_length_bucketing=config.train_length_bucketing,
        train_mega_batch_multiplier=config.train_mega_batch_multiplier,
        train_padding_length_buckets=config.train_padding_length_buckets,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_prefetch_factor=config.dataloader_prefetch_factor,
        dataloader_persistent_workers=config.dataloader_persistent_workers,
        dataloader_pin_memory=config.dataloader_pin_memory,
        non_blocking_transfer=config.non_blocking_transfer,
        performance_logging=config.performance_logging,
        token_cache_enabled=config.token_cache_enabled,
        token_cache_directory=str(config.token_cache_directory),
        token_cache_build_chunk_size=config.token_cache_build_chunk_size,
        torch_compile=config.torch_compile,
        torch_compile_mode=config.torch_compile_mode,
    )
    metadata = {
        "profile": config.profile,
        "head_type": config.head_type,
        "head_config": config.head_config.to_dict(),
        "typed_fusion": (
            {
                "schema_version": TYPED_FUSION_SCHEMA_VERSION,
                "feature_names": list(typed_feature_names),
                "options": config.typed_attribute_options.to_dict(),
            }
            if uses_typed_fusion
            else None
        ),
        "num_logits": int(trainer.model.config.num_labels),
        "probability_transform": (
            "sigmoid" if int(trainer.model.config.num_labels) == 1 else "softmax"
        ),
        "validation_macro_pr_auc": float(metrics["eval_macro_pr_auc"]),
        "best_hyperparameters": best_hyperparameters,
        "resolved_config": asdict(resolved_config),
        "performance_history": performance_tracker.history,
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
    runtime = parameters.training_runtime
    return SequenceClassifierConfig(
        model_path=parameters.pretrained_model_path,
        profile=parameters.profile,
        max_epochs=parameters.max_epochs,
        hpo_trials=parameters.hpo_trials,
        learning_rate=parameters.learning_rate,
        embeddings_learning_rate=parameters.embeddings_learning_rate,
        train_new_token_embeddings_only=parameters.train_new_token_embeddings_only,
        train_last_n_layers=parameters.train_last_n_layers,
        special_token_initialization_enabled=(
            parameters.special_token_initialization.enabled
        ),
        special_token_key_seed_texts=(
            parameters.special_token_initialization.key_seed_texts
        ),
        special_token_value_seed_texts=(
            parameters.special_token_initialization.value_seed_texts
        ),
        fast_dev_validation_enabled=parameters.validation.fast_dev.enabled,
        fast_dev_validation_max_rows=parameters.validation.fast_dev.max_rows,
        fast_dev_validation_every_n_optimizer_steps=(
            parameters.validation.fast_dev.every_n_optimizer_steps
        ),
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
        train_length_bucketing=runtime.length_bucketing.enabled,
        train_mega_batch_multiplier=(
            runtime.length_bucketing.mega_batch_multiplier
        ),
        train_padding_length_buckets=(
            runtime.length_bucketing.padding_length_buckets
        ),
        dataloader_num_workers=runtime.dataloader.num_workers,
        dataloader_prefetch_factor=runtime.dataloader.prefetch_factor,
        dataloader_persistent_workers=runtime.dataloader.persistent_workers,
        dataloader_pin_memory=runtime.dataloader.pin_memory,
        non_blocking_transfer=runtime.dataloader.non_blocking_transfer,
        performance_logging=runtime.performance_logging.enabled,
        token_cache_enabled=runtime.token_cache.enabled,
        token_cache_directory=runtime.token_cache.directory,
        token_cache_build_chunk_size=runtime.token_cache.build_chunk_size,
        attention_implementation=runtime.attention.implementation,
        torch_compile=runtime.torch_compile.enabled,
        torch_compile_mode=runtime.torch_compile.mode,
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
            native_logit_weight=parameters.head.native_logit_weight,
            attention_logit_weight=parameters.head.attention_logit_weight,
            train_logit_weights=parameters.head.train_logit_weights,
            typed_hidden_dims=parameters.head.typed_hidden_dims,
            typed_layer_norm=parameters.head.typed_layer_norm,
        ),
        typed_attribute_options=config.pair_features.typed_attributes,
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
            train_cache_shards=(
                data.train_matches.get_column("data_source").to_list()
                if "data_source" in data.train_matches.columns
                else None
            ),
            validation_cache_shards=(
                data.validation_matches.get_column("data_source").to_list()
                if "data_source" in data.validation_matches.columns
                else None
            ),
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
