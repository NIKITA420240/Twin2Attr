"""Aggregate Transformer attention-pooling weights by raw attribute name."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
from loguru import logger
from transformers import PreTrainedTokenizerBase

from ..config import AppConfig, AttributeImportanceAnalysisSettings
from ..models.transformer.head import PoolingSequenceClassifier
from ..models.transformer.predictor import load_trained_classifier
from ..pair_encoding import encode_prepared_pair_with_attributes
from ..prepare_data import PreparedPair


@dataclass(frozen=True, slots=True)
class AttributeImportanceResult:
    output_path: Path
    metadata_path: Path
    sample_rows: int
    attribute_rows: int


@dataclass(frozen=True, slots=True)
class _Observation:
    category: str
    attribute: str
    score: float | None
    token_count: int
    present_sections: int
    included_sections: int


@dataclass(slots=True)
class _Statistics:
    occurrences: int = 0
    scored_occurrences: int = 0
    score_sum: float = 0.0
    token_count_sum: int = 0
    present_sections: int = 0
    included_sections: int = 0

    def add(self, observation: _Observation) -> None:
        self.occurrences += 1
        self.present_sections += observation.present_sections
        self.included_sections += observation.included_sections
        if observation.score is not None:
            self.scored_occurrences += 1
            self.score_sum += observation.score
            self.token_count_sum += observation.token_count

    @property
    def raw_importance(self) -> float | None:
        if not self.scored_occurrences:
            return None
        return self.score_sum / self.scored_occurrences


def _model_inputs(
    tokenizer: PreTrainedTokenizerBase,
    encoded,
) -> dict[str, torch.Tensor]:
    return tokenizer.pad(
        [value.inputs for value in encoded],
        padding=True,
        return_tensors="pt",
    )


def _batch_observations(
    model: PoolingSequenceClassifier,
    tokenizer: PreTrainedTokenizerBase,
    pairs: Sequence[PreparedPair],
    *,
    max_length: int,
    use_field_tokens: bool,
    max_attribute_value_tokens: int | None,
) -> list[_Observation]:
    encoded = [
        encode_prepared_pair_with_attributes(
            tokenizer,
            pair,
            max_length=max_length,
            use_field_tokens=use_field_tokens,
            max_attribute_value_tokens=max_attribute_value_tokens,
        )
        for pair in pairs
    ]
    inputs = _model_inputs(tokenizer, encoded)
    device = next(model.parameters()).device
    model_inputs = {name: value.to(device) for name, value in inputs.items()}
    attention_mask = model_inputs["attention_mask"]
    backbone_inputs = dict(model_inputs)
    backbone_inputs.pop("attention_mask")
    with torch.inference_mode():
        outputs = model.backbone(
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=False,
            output_attentions=False,
            **backbone_inputs,
        )
        weights = model.head.attention_weights(
            outputs.last_hidden_state,
            attention_mask,
        ).float().mean(dim=-1).cpu().numpy()
    valid_lengths = inputs["attention_mask"].sum(dim=1).tolist()
    observations: list[_Observation] = []
    for row_index, (pair, encoded_pair) in enumerate(
        zip(pairs, encoded, strict=True)
    ):
        valid_length = int(valid_lengths[row_index])
        token_attributes = encoded_pair.token_attributes[:valid_length]
        positions: dict[str, list[int]] = defaultdict(list)
        for token_index, attribute in enumerate(token_attributes):
            if attribute is not None:
                positions[attribute].append(token_index)
        for attribute, present_sections in (
            encoded_pair.present_attribute_sections.items()
        ):
            indices = positions.get(attribute, [])
            score = None
            if indices:
                mean_attention = float(weights[row_index, indices].mean())
                score = float(valid_length * mean_attention)
                if not np.isfinite(score):
                    raise RuntimeError("attribute attention score is not finite")
            observations.append(
                _Observation(
                    category=pair.category,
                    attribute=attribute,
                    score=score,
                    token_count=len(indices),
                    present_sections=present_sections,
                    included_sections=encoded_pair.included_attribute_sections.get(
                        attribute,
                        0,
                    ),
                )
            )
    return observations


def _collect_observations(
    model: PoolingSequenceClassifier,
    tokenizer: PreTrainedTokenizerBase,
    pairs: Sequence[PreparedPair],
    *,
    batch_size: int,
    max_length: int,
    use_field_tokens: bool,
    max_attribute_value_tokens: int | None,
) -> list[_Observation]:
    observations: list[_Observation] = []
    current_batch_size = batch_size
    offset = 0
    while offset < len(pairs):
        batch = pairs[offset : offset + current_batch_size]
        try:
            observations.extend(
                _batch_observations(
                    model,
                    tokenizer,
                    batch,
                    max_length=max_length,
                    use_field_tokens=use_field_tokens,
                    max_attribute_value_tokens=max_attribute_value_tokens,
                )
            )
            offset += len(batch)
        except torch.cuda.OutOfMemoryError:
            if current_batch_size == 1:
                raise
            current_batch_size = max(1, current_batch_size // 2)
            torch.cuda.empty_cache()
            logger.warning(
                "Attribute analysis reduced batch_size to {} after CUDA OOM",
                current_batch_size,
            )
    return observations


def _aggregate(
    observations: Sequence[_Observation],
    *,
    group_by_category: bool,
    min_occurrences: int,
) -> pl.DataFrame:
    global_statistics: dict[str, _Statistics] = defaultdict(_Statistics)
    category_statistics: dict[tuple[str, str], _Statistics] = defaultdict(
        _Statistics
    )
    for observation in observations:
        global_statistics[observation.attribute].add(observation)
        if group_by_category:
            category_statistics[(observation.category, observation.attribute)].add(
                observation
            )

    rows: list[dict[str, Any]] = []

    def append_row(
        scope: str,
        category: str | None,
        attribute: str,
        statistics: _Statistics,
    ) -> None:
        raw = statistics.raw_importance
        source = scope
        importance = raw
        if (
            scope == "category"
            and statistics.scored_occurrences < min_occurrences
        ):
            fallback = global_statistics[attribute].raw_importance
            importance = fallback
            source = "global"
        if importance is None:
            importance = 1.0
            source = "default"
        rows.append(
            {
                "scope": scope,
                "category": category,
                "attribute": attribute,
                "importance": float(importance),
                "raw_importance": None if raw is None else float(raw),
                "importance_source": source,
                "occurrences": statistics.occurrences,
                "scored_occurrences": statistics.scored_occurrences,
                "mean_token_count": (
                    0.0
                    if not statistics.scored_occurrences
                    else statistics.token_count_sum
                    / statistics.scored_occurrences
                ),
                "truncated_fraction": (
                    0.0
                    if not statistics.present_sections
                    else 1.0
                    - statistics.included_sections / statistics.present_sections
                ),
            }
        )

    for attribute, statistics in global_statistics.items():
        append_row("global", None, attribute, statistics)
    for (category, attribute), statistics in category_statistics.items():
        append_row("category", category, attribute, statistics)
    if not rows:
        raise ValueError("attribute analysis found no raw attributes in the sample")
    rows.sort(
        key=lambda row: (
            row["scope"],
            "" if row["category"] is None else row["category"],
            -row["importance"],
            row["attribute"],
        )
    )
    priorities: dict[tuple[str, str | None], int] = defaultdict(int)
    for row in rows:
        group = (row["scope"], row["category"])
        priorities[group] += 1
        row["priority"] = priorities[group]
    return pl.DataFrame(rows)


def _validate_model(
    model: Any,
    model_path: Path,
) -> tuple[PoolingSequenceClassifier, int, bool, int | None]:
    if not isinstance(model, PoolingSequenceClassifier):
        raise TypeError(
            "attribute_importance requires a trained pooling Transformer; "
            f"artifact is not compatible: {model_path}"
        )
    if bool(getattr(model.config, "match_initialized_only", False)):
        raise ValueError("attribute_importance requires a trained Transformer")
    if "attention" not in model.head.poolings:
        raise ValueError(
            "attribute_importance requires Transformer head.poolings to contain "
            "'attention'"
        )
    max_length = getattr(model.config, "match_max_length", None)
    if not isinstance(max_length, int) or max_length < 1:
        raise ValueError("trained Transformer does not define match_max_length")
    use_field_tokens = bool(getattr(model.config, "match_use_field_tokens", True))
    value_limit = getattr(
        model.config,
        "match_max_attribute_value_tokens",
        None,
    )
    return model, max_length, use_field_tokens, value_limit


def analyze_transformer_attribute_importance(
    config: AppConfig,
    pairs: Sequence[PreparedPair],
) -> AttributeImportanceResult:
    """Run one inference pass and persist category/global attribute scores."""
    settings: AttributeImportanceAnalysisSettings = (
        config.analysis_models.attribute_importance
    )
    model_path = config.model_description.transformer.artifact_dir
    tokenizer, loaded_model = load_trained_classifier(
        model_path,
        device=config.runtime.device,
    )
    model, max_length, use_field_tokens, value_limit = _validate_model(
        loaded_model,
        model_path,
    )
    observations = _collect_observations(
        model,
        tokenizer,
        pairs,
        batch_size=config.inference.transformer.batch_size,
        max_length=max_length,
        use_field_tokens=use_field_tokens,
        max_attribute_value_tokens=value_limit,
    )
    frame = _aggregate(
        observations,
        group_by_category=settings.group_by_category,
        min_occurrences=settings.min_occurrences,
    )
    output_dir = settings.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / settings.output_file
    metadata_path = output_dir / settings.metadata_file
    frame.write_parquet(output_path)
    metadata = {
        "analysis_model": config.analysis.analysis_model,
        "data_model": config.analysis.data_model,
        "augmentation_model": config.analysis.augmentation_model,
        "data_postprocessing_model": (
            config.analysis.data_postprocessing_model
        ),
        "model": settings.model,
        "model_path": str(model_path),
        "sample_size": len(pairs),
        "score_type": settings.score_type,
        "group_by_category": settings.group_by_category,
        "min_occurrences": settings.min_occurrences,
        "seed": config.runtime.seed,
        "max_length": max_length,
        "use_field_tokens": use_field_tokens,
        "max_attribute_value_tokens": value_limit,
        "attention_num_heads": int(
            model.config.head_config.get("attention_num_heads", 1)
        ),
    }
    if config.analysis.augmentation_model == "attribute_shuffle":
        augmentation = config.augmentation_models.attribute_shuffle
        metadata["augmentation"] = {
            "shuffled_copies": augmentation.shuffled_copies,
            "keep_original": augmentation.keep_original,
            "seed": augmentation.seed,
            "shuffle_cards_independently": (
                augmentation.shuffle_cards_independently
            ),
            "skip_oversized": augmentation.skip_oversized,
        }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "Saved attribute importance: rows={}, sample_rows={}, output={!s}",
        frame.height,
        len(pairs),
        output_path,
    )
    return AttributeImportanceResult(
        output_path=output_path,
        metadata_path=metadata_path,
        sample_rows=len(pairs),
        attribute_rows=frame.height,
    )


__all__ = [
    "AttributeImportanceResult",
    "analyze_transformer_attribute_importance",
]
