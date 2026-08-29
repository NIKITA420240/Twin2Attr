"""Transformer model construction and weighted Hugging Face trainer."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from time import perf_counter
from typing import Any

import torch
from loguru import logger
from transformers import PreTrainedModel, Trainer

from .construction import model_factory
from .objective import weighted_classification_loss
from .optimizer import (
    LearningRateMultipliers,
    build_transformer_optimizer,
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
            original_eval_dataset = self.eval_dataset
            try:
                self.eval_dataset = {"fast_dev": self.fast_dev_dataset}
                fast_dev_dataloader = self.get_eval_dataloader("fast_dev")
            finally:
                self.eval_dataset = original_eval_dataset
            output = self.evaluation_loop(
                fast_dev_dataloader,
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
        performance_tracker = getattr(self, "performance_tracker", None)
        if model.training and performance_tracker is not None:
            performance_tracker.observe(inputs.get("attention_mask"))
        model_inputs = dict(inputs)
        labels = model_inputs.pop("labels")
        sample_weights = model_inputs.pop("sample_weights")
        outputs = model(**model_inputs)
        loss = weighted_classification_loss(
            outputs.logits,
            labels,
            sample_weights,
            self.class_weights,
        )
        return (loss, outputs) if return_outputs else loss


__all__ = ["WeightedSequenceTrainer", "model_factory"]
