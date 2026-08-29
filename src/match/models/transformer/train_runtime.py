"""Train-time batching, transfer and low-noise throughput telemetry."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np
import torch
from loguru import logger
from torch.utils.data import Sampler
from transformers import TrainerCallback


class LengthAwareSampler(Sampler[int]):
    """Shuffle globally and sort only inside randomized length megabatches."""

    def __init__(
        self,
        lengths: Sequence[int] | np.ndarray,
        *,
        batch_size: int,
        mega_batch_multiplier: int,
        seed: int,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if mega_batch_multiplier < 1:
            raise ValueError("mega_batch_multiplier must be positive")
        values = np.asarray(lengths, dtype=np.int64)
        if values.ndim != 1 or values.size == 0:
            raise ValueError("lengths must be a non-empty one-dimensional sequence")
        if np.any(values < 1):
            raise ValueError("length estimates must be positive")
        self.lengths = values
        self.batch_size = int(batch_size)
        self.mega_batch_multiplier = int(mega_batch_multiplier)
        self.seed = int(seed)
        self.epoch = 0

    def __len__(self) -> int:
        return int(self.lengths.size)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        shuffled = torch.randperm(len(self), generator=generator).tolist()
        mega_batch_size = self.batch_size * self.mega_batch_multiplier
        batches: list[list[int]] = []
        for offset in range(0, len(shuffled), mega_batch_size):
            mega_batch = shuffled[offset : offset + mega_batch_size]
            mega_batch.sort(key=lambda index: int(self.lengths[index]))
            batches.extend(
                mega_batch[start : start + self.batch_size]
                for start in range(0, len(mega_batch), self.batch_size)
            )
        batch_order = torch.randperm(len(batches), generator=generator).tolist()
        for batch_index in batch_order:
            yield from batches[batch_index]


def stratified_sample_indices(
    labels: Sequence[int],
    *,
    max_rows: int,
    seed: int,
) -> tuple[int, ...]:
    """Select a deterministic class-stratified diagnostic sample."""

    values = np.asarray(labels, dtype=np.int64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("labels must be a non-empty one-dimensional sequence")
    if not set(np.unique(values)).issubset({0, 1}):
        raise ValueError("labels must contain only 0 and 1")
    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    if max_rows >= values.size:
        return tuple(range(values.size))

    available = {label: np.flatnonzero(values == label) for label in (0, 1)}
    quotas = {
        0: int(round(max_rows * available[0].size / values.size)),
        1: 0,
    }
    quotas[0] = min(quotas[0], available[0].size)
    quotas[1] = min(max_rows - quotas[0], available[1].size)
    remaining = max_rows - quotas[0] - quotas[1]
    for label in (0, 1):
        if remaining <= 0:
            break
        capacity = available[label].size - quotas[label]
        addition = min(remaining, capacity)
        quotas[label] += addition
        remaining -= addition

    generator = np.random.default_rng(seed)
    selected = [
        generator.choice(available[label], size=quotas[label], replace=False)
        for label in (0, 1)
        if quotas[label]
    ]
    indices = np.concatenate(selected)
    generator.shuffle(indices)
    return tuple(int(index) for index in indices)


@dataclass(slots=True)
class EpochPerformanceTracker:
    """Aggregate train throughput and padding metrics once per epoch."""

    enabled: bool = True
    history: list[dict[str, float | int]] = field(default_factory=list)
    _started_at: float | None = None
    _examples: int = 0
    _real_tokens: torch.Tensor | None = None
    _padded_tokens: int = 0

    def start_epoch(self) -> None:
        if not self.enabled:
            return
        self._started_at = perf_counter()
        self._examples = 0
        self._real_tokens = None
        self._padded_tokens = 0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def observe(self, attention_mask: torch.Tensor | None) -> None:
        if not self.enabled or self._started_at is None or attention_mask is None:
            return
        self._examples += int(attention_mask.shape[0])
        real_tokens = attention_mask.detach().sum()
        if self._real_tokens is None:
            self._real_tokens = real_tokens
        else:
            self._real_tokens = self._real_tokens + real_tokens
        self._padded_tokens += int(attention_mask.numel())

    def finish_epoch(self, epoch: int) -> dict[str, float | int] | None:
        if not self.enabled or self._started_at is None:
            return None
        elapsed = max(perf_counter() - self._started_at, 1e-9)
        real_tokens = (
            int(self._real_tokens.item()) if self._real_tokens is not None else 0
        )
        padded = max(self._padded_tokens, 1)
        record: dict[str, float | int] = {
            "epoch": int(epoch),
            "train_seconds": float(elapsed),
            "examples": self._examples,
            "real_tokens": real_tokens,
            "padded_tokens": self._padded_tokens,
            "padding_efficiency": real_tokens / padded,
            "examples_per_second": self._examples / elapsed,
            "real_tokens_per_second": real_tokens / elapsed,
            "padded_tokens_per_second": self._padded_tokens / elapsed,
        }
        if torch.cuda.is_available():
            gib = 1024**3
            record["peak_cuda_memory_gib"] = (
                torch.cuda.max_memory_allocated() / gib
            )
            record["peak_cuda_reserved_gib"] = torch.cuda.max_memory_reserved() / gib
        self.history.append(record)
        self._started_at = None
        return record


class EpochPerformanceCallback(TrainerCallback):
    def __init__(self, tracker: EpochPerformanceTracker) -> None:
        self.tracker = tracker

    def on_epoch_begin(self, args, state, control, **kwargs):
        del args, state, kwargs
        self.tracker.start_epoch()
        return control

    def on_epoch_end(self, args, state, control, **kwargs):
        del args, kwargs
        epoch = len(self.tracker.history) + 1
        record = self.tracker.finish_epoch(epoch)
        if record is not None:
            logger.info(
                "Train epoch performance: epoch={}, seconds={:.1f}, "
                "examples_per_second={:.1f}, real_tokens_per_second={:.0f}, "
                "padded_tokens_per_second={:.0f}, "
                "padding_efficiency={:.3f}, peak_cuda_memory_gib={}",
                record["epoch"],
                record["train_seconds"],
                record["examples_per_second"],
                record["real_tokens_per_second"],
                record["padded_tokens_per_second"],
                record["padding_efficiency"],
                (
                    f'{record["peak_cuda_memory_gib"]:.2f}'
                    if "peak_cuda_memory_gib" in record
                    else "n/a"
                ),
            )
        return control


__all__ = [
    "EpochPerformanceCallback",
    "EpochPerformanceTracker",
    "LengthAwareSampler",
    "stratified_sample_indices",
]
