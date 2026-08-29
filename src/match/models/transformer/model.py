"""Transformer model construction and weighted Hugging Face trainer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from time import perf_counter
from typing import Any

import torch
import torch.nn.functional as F
from loguru import logger
from transformers import (
    AutoModelForSequenceClassification,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
)

from ...pair_encoding import (
    add_pair_special_tokens,
    initialize_pair_special_token_embeddings,
    pair_special_token_ids,
)
from .head import HybridSequenceClassifier, PoolingHeadConfig, PoolingSequenceClassifier
from .optimizer import (
    LearningRateMultipliers,
    build_transformer_optimizer,
    freeze_backbone_except_last_layers,
    restrict_word_embedding_updates,
)
from .train_runtime import EpochPerformanceTracker, LengthAwareSampler


class WeightedSequenceTrainer(Trainer):
    def __init__(
        self,
        *args: Any,
        class_weights: torch.Tensor,
        learning_rate_multipliers: LearningRateMultipliers | None = None,
        train_pair_lengths: Sequence[int] | None = None,
        length_bucketing: bool = False,
        mega_batch_multiplier: int = 50,
        non_blocking_transfer: bool = False,
        performance_tracker: EpochPerformanceTracker | None = None,
        fast_dev_dataset: Any | None = None,
        fast_dev_compute_metrics: Any | None = None,
        fast_dev_every_n_optimizer_steps: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.learning_rate_multipliers = (
            learning_rate_multipliers or LearningRateMultipliers()
        )
        self.train_pair_lengths = (
            None if train_pair_lengths is None else tuple(train_pair_lengths)
        )
        self.length_bucketing = bool(length_bucketing)
        self.mega_batch_multiplier = int(mega_batch_multiplier)
        self.non_blocking_transfer = bool(non_blocking_transfer)
        self.performance_tracker = performance_tracker
        if fast_dev_dataset is None:
            if fast_dev_compute_metrics is not None or fast_dev_every_n_optimizer_steps is not None:
                raise ValueError(
                    "fast-dev metric and interval require fast_dev_dataset"
                )
        elif fast_dev_compute_metrics is None or fast_dev_every_n_optimizer_steps is None:
            raise ValueError(
                "fast_dev_dataset requires a metric and evaluation interval"
            )
        elif fast_dev_every_n_optimizer_steps < 1:
            raise ValueError("fast-dev evaluation interval must be positive")
        self.fast_dev_dataset = fast_dev_dataset
        self.fast_dev_compute_metrics = fast_dev_compute_metrics
        self.fast_dev_every_n_optimizer_steps = fast_dev_every_n_optimizer_steps
        self._last_fast_dev_evaluation_step = 0
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.detach().to(dtype=torch.float32)

    def _get_train_sampler(self, train_dataset=None):
        dataset = self.train_dataset if train_dataset is None else train_dataset
        if not self.length_bucketing:
            return super()._get_train_sampler(dataset)
        if self.train_pair_lengths is None:
            raise RuntimeError("length bucketing requires train_pair_lengths")
        if dataset is None or len(dataset) != len(self.train_pair_lengths):
            raise RuntimeError(
                "train_pair_lengths must align with the active train dataset"
            )
        return LengthAwareSampler(
            self.train_pair_lengths,
            batch_size=int(self._train_batch_size),
            mega_batch_multiplier=self.mega_batch_multiplier,
            seed=int(self.args.data_seed or self.args.seed),
        )

    def _prepare_input(self, data: Any) -> Any:
        if not self.non_blocking_transfer:
            return super()._prepare_input(data)
        if isinstance(data, Mapping):
            return type(data)(
                {
                    key: self._prepare_input(value)
                    for key, value in data.items()
                }
            )
        if isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(value) for value in data)
        if isinstance(data, torch.Tensor):
            kwargs: dict[str, Any] = {
                "device": self.args.device,
                "non_blocking": True,
            }
            if self.is_deepspeed_enabled and (
                torch.is_floating_point(data) or torch.is_complex(data)
            ):
                kwargs["dtype"] = (
                    self.accelerator.state.deepspeed_plugin.hf_ds_config.dtype()
                )
            return data.to(**kwargs)
        return data

    def create_optimizer(self) -> torch.optim.Optimizer:
        if self.optimizer is None:
            self.optimizer = build_transformer_optimizer(
                self.model,
                backbone_lr=float(self.args.learning_rate),
                multipliers=self.learning_rate_multipliers,
                weight_decay=float(self.args.weight_decay),
            )
        return self.optimizer

    def _run_fast_dev_evaluation(self) -> None:
        """Evaluate diagnostics without affecting callbacks or model selection."""

        if self.fast_dev_dataset is None or self.fast_dev_compute_metrics is None:
            return
        started_at = perf_counter()
        original_compute_metrics = self.compute_metrics
        self.compute_metrics = self.fast_dev_compute_metrics
        try:
            output = self.evaluation_loop(
                self.get_eval_dataloader(self.fast_dev_dataset),
                description="Fast development evaluation",
                prediction_loss_only=None,
                ignore_keys=None,
                metric_key_prefix="fast_dev",
            )
        finally:
            self.compute_metrics = original_compute_metrics
        elapsed = max(perf_counter() - started_at, 1e-9)
        metrics = dict(output.metrics)
        metrics["fast_dev_runtime_seconds"] = elapsed
        metrics["fast_dev_examples_per_second"] = output.num_samples / elapsed
        self.log(metrics)
        logger.info(
            "Fast-dev validation: step={}, rows={}, macro_pr_auc={:.6f}, "
            "seconds={:.1f}",
            self.state.global_step,
            output.num_samples,
            float(metrics.get("fast_dev_macro_pr_auc", float("nan"))),
            elapsed,
        )

    def _maybe_log_save_evaluate(
        self,
        tr_loss: torch.Tensor,
        grad_norm: torch.Tensor | float | None,
        model: PreTrainedModel,
        trial: Any,
        epoch: float,
        ignore_keys_for_eval: list[str] | None,
        start_time: float,
        learning_rate: float | None = None,
    ) -> None:
        super()._maybe_log_save_evaluate(
            tr_loss,
            grad_norm,
            model,
            trial,
            epoch,
            ignore_keys_for_eval,
            start_time,
            learning_rate,
        )
        interval = self.fast_dev_every_n_optimizer_steps
        if (
            self.fast_dev_dataset is None
            or interval is None
            or self.state.global_step == 0
            or self.state.global_step == self._last_fast_dev_evaluation_step
            or self.state.global_step % interval != 0
        ):
            return
        self._run_fast_dev_evaluation()
        self._last_fast_dev_evaluation_step = self.state.global_step

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        del num_items_in_batch
        if model.training and self.performance_tracker is not None:
            self.performance_tracker.observe(inputs.get("attention_mask"))
        labels = inputs.pop("labels")
        sample_weights = inputs.pop("sample_weights")
        outputs = model(**inputs)
        losses = F.cross_entropy(
            outputs.logits,
            labels,
            weight=self.class_weights.to(outputs.logits.device),
            reduction="none",
        )
        weights = sample_weights.to(outputs.logits.device)
        loss = torch.sum(losses * weights) / torch.sum(weights)
        return (loss, outputs) if return_outputs else loss


def model_factory(
    model_path: str,
    tokenizer: PreTrainedTokenizerBase,
    *,
    use_field_tokens: bool,
    special_token_initialization_enabled: bool = False,
    special_token_key_seed_texts: Sequence[str] = (),
    special_token_value_seed_texts: Sequence[str] = (),
    train_new_token_embeddings_only: bool = False,
    train_last_n_layers: int | None = None,
    head_type: str = "default",
    head_config: PoolingHeadConfig | None = None,
):
    def initialize_model(trial: Any | None = None) -> PreTrainedModel:
        del trial
        if head_type == "hybrid":
            model = HybridSequenceClassifier.from_backbone_pretrained(
                model_path,
                head_config=head_config or PoolingHeadConfig(poolings=("attention",)),
                id2label={0: "different", 1: "match"},
                label2id={"different": 0, "match": 1},
            )
        elif head_type == "pooling":
            model = PoolingSequenceClassifier.from_backbone_pretrained(
                model_path,
                head_config=head_config or PoolingHeadConfig(),
                num_labels=2,
                id2label={0: "different", 1: "match"},
                label2id={"different": 0, "match": 1},
            )
        elif head_type == "default":
            model = AutoModelForSequenceClassification.from_pretrained(
                model_path,
                num_labels=2,
                id2label={0: "different", 1: "match"},
                label2id={"different": 0, "match": 1},
                ignore_mismatched_sizes=True,
            )
        else:
            raise ValueError("head_type must be 'default', 'pooling' or 'hybrid'")
        if use_field_tokens:
            add_pair_special_tokens(tokenizer, model)
            if special_token_initialization_enabled:
                initialize_pair_special_token_embeddings(
                    tokenizer,
                    model,
                    key_seed_texts=special_token_key_seed_texts,
                    value_seed_texts=special_token_value_seed_texts,
                )
            if train_new_token_embeddings_only:
                restrict_word_embedding_updates(
                    model,
                    pair_special_token_ids(tokenizer),
                )
        if train_last_n_layers is not None:
            freeze_backbone_except_last_layers(
                model,
                train_last_n_layers,
                train_input_word_embeddings=train_new_token_embeddings_only,
            )
        return model

    return initialize_model


__all__ = ["WeightedSequenceTrainer", "model_factory"]
