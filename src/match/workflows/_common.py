"""Shared helpers for explicit application workflows."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from loguru import logger

from ..config import AppConfig


def normalization_enabled(config: AppConfig) -> bool:
    return config.features.normalization.enabled


@contextmanager
def workflow_logging(config: AppConfig, *, workflow_name: str) -> Iterator[None]:
    """Configure the optional file sink for one workflow invocation."""
    log_sink: int | None = None
    if config.logging.file:
        log_path = config.logging.file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_sink = logger.add(
            log_path,
            level=config.logging.level,
            rotation=config.logging.rotation,
            enqueue=True,
        )
    try:
        logger.info(
            "Starting Twin2Attr workflow: name={}, normalization={}, "
            "training_model={}, ner={}",
            workflow_name,
            normalization_enabled(config),
            config.training.model,
            config.features.ner.enabled,
        )
        yield
    finally:
        if log_sink is not None:
            logger.remove(log_sink)
