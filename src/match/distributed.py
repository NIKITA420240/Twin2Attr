"""Small, launcher-agnostic helpers for distributed training workflows."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import TransformerDistributedSettings


def _environment_integer(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {value!r}") from error


@dataclass(frozen=True, slots=True)
class DistributedProcess:
    world_size: int
    rank: int
    local_rank: int

    def __post_init__(self) -> None:
        if self.world_size < 1:
            raise ValueError("WORLD_SIZE must be positive")
        if not 0 <= self.rank < self.world_size:
            raise ValueError("RANK must be in [0, WORLD_SIZE)")
        if self.local_rank < 0:
            raise ValueError("LOCAL_RANK must not be negative")

    @property
    def distributed(self) -> bool:
        return self.world_size > 1

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def current_process() -> DistributedProcess:
    """Read the process topology supplied by ``torchrun``."""
    return DistributedProcess(
        world_size=_environment_integer("WORLD_SIZE", 1),
        rank=_environment_integer("RANK", 0),
        local_rank=_environment_integer("LOCAL_RANK", 0),
    )


def validate_distributed_launch(
    settings: "TransformerDistributedSettings",
) -> DistributedProcess:
    """Ensure the config and launcher agree before expensive data loading."""
    process = current_process()
    if not settings.enabled:
        if process.distributed:
            raise RuntimeError(
                "torchrun started multiple processes, but transformer.distributed."
                "enabled=false"
            )
        return process
    if process.world_size != settings.expected_world_size:
        raise RuntimeError(
            "distributed world-size mismatch: configured "
            f"expected_world_size={settings.expected_world_size}, launcher supplied "
            f"WORLD_SIZE={process.world_size}"
        )
    return process


def wait_for_everyone(process: DistributedProcess | None = None) -> None:
    """Synchronize initialized ranks after rank-zero filesystem writes."""
    selected = process or current_process()
    if not selected.distributed:
        return
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError(
            "distributed process group is not initialized; launch training with "
            "torchrun"
        )
    dist.barrier()


__all__ = [
    "DistributedProcess",
    "current_process",
    "validate_distributed_launch",
    "wait_for_everyone",
]
