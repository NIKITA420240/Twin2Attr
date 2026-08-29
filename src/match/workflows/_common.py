"""Shared helpers for explicit train and inspect workflows."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator

from loguru import logger

from ..config import AppConfig
from ..models.transformer.profile import is_prompted_profile

if TYPE_CHECKING:
    from ..prepare_data import PreparedPair


def normalization_enabled(config: AppConfig) -> bool:
    return config.features.normalization.enabled


def resolve_max_length(
    config: AppConfig,
    pairs: list[PreparedPair],
) -> int:
    from transformers import AutoTokenizer

    from ..pair_encoding import add_pair_special_tokens, infer_pair_max_length

    encoding = config.model_description.transformer.pair_encoding
    if encoding.max_length is not None:
        return encoding.max_length

    tokenizer = AutoTokenizer.from_pretrained(
        config.model_description.transformer.pretrained_model_path,
        trust_remote_code=is_prompted_profile(
            config.model_description.transformer.profile
        ),
    )
    if encoding.use_field_tokens:
        add_pair_special_tokens(tokenizer)
    max_length = infer_pair_max_length(
        tokenizer,
        pairs,
        quantile=encoding.quantile,
        sample_size=encoding.sample_size,
        hard_cap=encoding.hard_cap,
        use_field_tokens=encoding.use_field_tokens,
        max_attribute_value_chars=encoding.max_attribute_value_chars,
        max_attribute_value_tokens=encoding.max_attribute_value_tokens,
        profile=config.model_description.transformer.profile,
    )
    logger.info("Pair encoding resolved max_length={}", max_length)
    return max_length


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
