"""Transformer profile rules and the persisted inference contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from ...pair_serialization import DEFAULT_MAX_ATTRIBUTE_VALUE_TOKENS


SEQUENCE_CLASSIFIER_PROFILE = "sequence_classifier"
PROMPTED_BINARY_RERANKER_PROFILE = "prompted_binary_reranker"
QWEN3_RERANKER_PROFILE = "qwen3_reranker"
TRANSFORMER_PROFILES = frozenset(
    {
        SEQUENCE_CLASSIFIER_PROFILE,
        PROMPTED_BINARY_RERANKER_PROFILE,
        QWEN3_RERANKER_PROFILE,
    }
)

PROFILE_HEAD_TYPES = {
    SEQUENCE_CLASSIFIER_PROFILE: frozenset({"default", "pooling"}),
    PROMPTED_BINARY_RERANKER_PROFILE: frozenset({"native", "attention_pooling"}),
    QWEN3_RERANKER_PROFILE: frozenset({"native"}),
}


def normalize_profile(value: str) -> str:
    profile = value.strip().lower()
    if profile not in TRANSFORMER_PROFILES:
        raise ValueError(f"unsupported Transformer profile: {value!r}")
    return profile


def validate_profile_head(profile: str, head_type: str) -> tuple[str, str]:
    normalized_profile = normalize_profile(profile)
    normalized_head = head_type.strip().lower()
    if normalized_head not in PROFILE_HEAD_TYPES[normalized_profile]:
        supported = ", ".join(sorted(PROFILE_HEAD_TYPES[normalized_profile]))
        raise ValueError(
            f"head_type {head_type!r} is invalid for profile "
            f"{normalized_profile!r}; expected one of: {supported}"
        )
    return normalized_profile, normalized_head


def is_prompted_profile(profile: str) -> bool:
    return normalize_profile(profile) in {
        PROMPTED_BINARY_RERANKER_PROFILE,
        QWEN3_RERANKER_PROFILE,
    }


def is_nemotron_profile(profile: str) -> bool:
    return normalize_profile(profile) == PROMPTED_BINARY_RERANKER_PROFILE


def is_qwen3_reranker_profile(profile: str) -> bool:
    return normalize_profile(profile) == QWEN3_RERANKER_PROFILE


def requires_trust_remote_code(profile: str) -> bool:
    """Return whether the pretrained tokenizer/model uses bundled Python code."""
    return is_nemotron_profile(profile)


def _read(config: Mapping[str, Any] | Any, name: str, default: Any) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


@dataclass(frozen=True, slots=True)
class TransformerArtifactContract:
    """Output and encoding semantics persisted alongside model weights."""

    profile: str
    head_type: str
    num_logits: int
    probability_transform: str

    def __post_init__(self) -> None:
        normalize_profile(self.profile)
        if self.num_logits not in {1, 2}:
            raise ValueError("Transformer artifacts must expose one or two logits")
        expected = "sigmoid" if self.num_logits == 1 else "softmax"
        if self.probability_transform != expected:
            raise ValueError(
                f"{self.num_logits}-logit artifacts require {expected!r}, got "
                f"{self.probability_transform!r}"
            )
        if is_prompted_profile(self.profile) and self.num_logits != 1:
            raise ValueError("prompted binary rerankers must expose exactly one logit")

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | Any,
    ) -> TransformerArtifactContract:
        num_labels = int(_read(config, "num_labels", 2))
        num_logits = int(_read(config, "match_num_logits", num_labels))
        return cls(
            profile=str(
                _read(config, "match_profile", SEQUENCE_CLASSIFIER_PROFILE)
            ),
            head_type=str(_read(config, "match_head_type", "default")),
            num_logits=num_logits,
            probability_transform=str(
                _read(
                    config,
                    "match_probability_transform",
                    "sigmoid" if num_logits == 1 else "softmax",
                )
            ),
        )

    @classmethod
    def for_training(
        cls,
        *,
        profile: str,
        head_type: str,
        num_logits: int,
    ) -> TransformerArtifactContract:
        normalized_profile, normalized_head = validate_profile_head(
            profile, head_type
        )
        return cls(
            profile=normalized_profile,
            head_type=normalized_head,
            num_logits=num_logits,
            probability_transform="sigmoid" if num_logits == 1 else "softmax",
        )

    @property
    def uses_prompted_pairs(self) -> bool:
        return is_prompted_profile(self.profile)

    @property
    def requires_trust_remote_code(self) -> bool:
        return requires_trust_remote_code(self.profile)

    @property
    def encoder_pooling(self) -> str:
        return "mean" if self.uses_prompted_pairs else "cls"

    def apply_to(self, config: Any) -> None:
        config.match_profile = self.profile
        config.match_head_type = self.head_type
        config.match_num_logits = self.num_logits
        config.match_probability_transform = self.probability_transform

    def validate_logits(self, logits: np.ndarray) -> np.ndarray:
        values = np.asarray(logits)
        if values.ndim != 2 or values.shape[1] != self.num_logits:
            raise ValueError(
                f"expected logits with shape (n_samples, {self.num_logits}), "
                f"got {values.shape}"
            )
        return values

    def positive_probabilities(self, logits: np.ndarray) -> np.ndarray:
        values = self.validate_logits(logits).astype(np.float64, copy=False)
        if self.num_logits == 1:
            scores = values[:, 0]
            result = np.empty_like(scores)
            positive = scores >= 0
            result[positive] = 1.0 / (1.0 + np.exp(-scores[positive]))
            exponentials = np.exp(scores[~positive])
            result[~positive] = exponentials / (1.0 + exponentials)
            return result
        shifted = values - values.max(axis=1, keepdims=True)
        exponentiated = np.exp(shifted)
        return exponentiated[:, 1] / exponentiated.sum(axis=1)

    def logit_margin(self, logits: np.ndarray) -> np.ndarray:
        values = self.validate_logits(logits)
        if self.num_logits == 1:
            return values[:, 0]
        return values[:, 1] - values[:, 0]


@dataclass(frozen=True, slots=True)
class TransformerRuntimeContract:
    """Complete persisted contract consumed by every inference backend."""

    output: TransformerArtifactContract
    hidden_size: int
    max_length: int | None
    use_field_tokens: bool
    max_attribute_value_chars: int | None
    max_attribute_value_tokens: int | None

    def __post_init__(self) -> None:
        if self.hidden_size < 1:
            raise ValueError("Transformer config does not expose hidden_size")
        if self.max_length is not None and self.max_length < 1:
            raise ValueError("artifact max_length must be positive or None")
        if (
            self.max_attribute_value_chars is not None
            and self.max_attribute_value_chars < 1
        ):
            raise ValueError(
                "artifact max_attribute_value_chars must be positive or None"
            )
        if (
            self.max_attribute_value_tokens is not None
            and self.max_attribute_value_tokens < 1
        ):
            raise ValueError(
                "artifact max_attribute_value_tokens must be positive or None"
            )

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any] | Any,
    ) -> TransformerRuntimeContract:
        hidden_size = _read(config, "hidden_size", 0)
        if not hidden_size:
            backbone = _read(config, "backbone_config", {})
            hidden_size = _read(backbone, "hidden_size", 0)
        max_length = _read(config, "match_max_length", None)
        max_chars = _read(config, "match_max_attribute_value_chars", None)
        max_tokens = _read(
            config,
            "match_max_attribute_value_tokens",
            DEFAULT_MAX_ATTRIBUTE_VALUE_TOKENS,
        )
        return cls(
            output=TransformerArtifactContract.from_config(config),
            hidden_size=int(hidden_size or 0),
            max_length=None if max_length is None else int(max_length),
            use_field_tokens=bool(
                _read(config, "match_use_field_tokens", True)
            ),
            max_attribute_value_chars=(
                None if max_chars is None else int(max_chars)
            ),
            max_attribute_value_tokens=(
                None if max_tokens is None else int(max_tokens)
            ),
        )

    def apply_encoding_to(self, config: Any) -> None:
        config.match_max_length = self.max_length
        config.match_use_field_tokens = self.use_field_tokens
        config.match_max_attribute_value_chars = self.max_attribute_value_chars
        config.match_max_attribute_value_tokens = self.max_attribute_value_tokens

    @property
    def profile(self) -> str:
        return self.output.profile

    @property
    def num_logits(self) -> int:
        return self.output.num_logits


def positive_probabilities(logits: np.ndarray) -> np.ndarray:
    """Backward-compatible conversion inferred from a logits matrix."""
    values = np.asarray(logits)
    if values.ndim != 2 or values.shape[1] not in {1, 2}:
        raise ValueError("expected logits with shape (n_samples, 1 or 2)")
    contract = TransformerArtifactContract(
        profile=(
            PROMPTED_BINARY_RERANKER_PROFILE
            if values.shape[1] == 1
            else SEQUENCE_CLASSIFIER_PROFILE
        ),
        head_type="native" if values.shape[1] == 1 else "default",
        num_logits=int(values.shape[1]),
        probability_transform="sigmoid" if values.shape[1] == 1 else "softmax",
    )
    return contract.positive_probabilities(values)


__all__ = [
    "PROFILE_HEAD_TYPES",
    "PROMPTED_BINARY_RERANKER_PROFILE",
    "QWEN3_RERANKER_PROFILE",
    "SEQUENCE_CLASSIFIER_PROFILE",
    "TRANSFORMER_PROFILES",
    "TransformerArtifactContract",
    "TransformerRuntimeContract",
    "is_nemotron_profile",
    "is_prompted_profile",
    "is_qwen3_reranker_profile",
    "normalize_profile",
    "positive_probabilities",
    "requires_trust_remote_code",
    "validate_profile_head",
]
