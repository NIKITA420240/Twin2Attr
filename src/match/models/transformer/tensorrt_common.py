"""Shared TensorRT profile and error types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence


class TensorRTUnavailableError(RuntimeError):
    """TensorRT or the requested CUDA device is unavailable."""


class TensorRTInitializationError(RuntimeError):
    """A TensorRT-backed executor could not be prepared or warmed up."""


ProfileSelector = Literal["min", "opt", "max"]


@dataclass(frozen=True, slots=True)
class TensorRTProfile:
    """One optimization profile shared by ORT and native TensorRT."""

    min_batch_size: int = 1
    opt_batch_size: int = 512
    max_batch_size: int = 512
    min_sequence_length: int = 1
    opt_sequence_length: int = 256
    max_sequence_length: int = 472
    input_names: tuple[str, ...] = ("input_ids", "attention_mask")

    def __post_init__(self) -> None:
        batches = self.batch_sizes
        lengths = self.sequence_lengths
        if any(
            isinstance(value, bool) or value < 1
            for value in (*batches, *lengths)
        ):
            raise ValueError("TensorRT profile dimensions must be positive integers")
        if not batches[0] <= batches[1] <= batches[2]:
            raise ValueError("TensorRT batch profile must satisfy min <= opt <= max")
        if not lengths[0] <= lengths[1] <= lengths[2]:
            raise ValueError("TensorRT sequence profile must satisfy min <= opt <= max")
        if not self.input_names or any(
            not name.strip() for name in self.input_names
        ):
            raise ValueError("TensorRT profile input names must not be empty")

    @classmethod
    def from_sequence_lengths(
        cls,
        sequence_lengths: Sequence[int],
        *,
        min_batch_size: int,
        opt_batch_size: int,
        max_batch_size: int,
        input_names: tuple[str, ...] = ("input_ids", "attention_mask"),
    ) -> TensorRTProfile:
        values = tuple(sequence_lengths)
        if not values or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in values
        ):
            raise ValueError("TensorRT sequence lengths must be positive integers")
        if tuple(sorted(values)) != values:
            raise ValueError("TensorRT sequence lengths must be non-decreasing")
        return cls(
            min_batch_size=min_batch_size,
            opt_batch_size=opt_batch_size,
            max_batch_size=max_batch_size,
            min_sequence_length=values[0],
            opt_sequence_length=values[len(values) // 2],
            max_sequence_length=values[-1],
            input_names=input_names,
        )

    @property
    def batch_sizes(self) -> tuple[int, int, int]:
        return self.min_batch_size, self.opt_batch_size, self.max_batch_size

    @property
    def sequence_lengths(self) -> tuple[int, int, int]:
        return (
            self.min_sequence_length,
            self.opt_sequence_length,
            self.max_sequence_length,
        )

    def dimensions(self, selector: ProfileSelector) -> tuple[int, int]:
        index = {"min": 0, "opt": 1, "max": 2}[selector]
        return self.batch_sizes[index], self.sequence_lengths[index]

    def resolve_shape(
        self,
        shape: tuple[int, ...],
        selector: ProfileSelector,
    ) -> tuple[int, ...]:
        batch_size, sequence_length = self.dimensions(selector)
        values = list(shape)
        if values and values[0] == -1:
            values[0] = batch_size
        if len(values) > 1 and values[1] == -1:
            values[1] = sequence_length
        if any(value == -1 for value in values):
            raise TensorRTInitializationError(
                f"unsupported dynamic TensorRT input shape: {shape}"
            )
        return tuple(values)


__all__ = [
    "ProfileSelector",
    "TensorRTInitializationError",
    "TensorRTProfile",
    "TensorRTUnavailableError",
]
