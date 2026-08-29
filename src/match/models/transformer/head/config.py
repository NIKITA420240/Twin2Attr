"""Configuration for a composable Transformer pooling head."""

from __future__ import annotations

from dataclasses import dataclass


SUPPORTED_POOLINGS = frozenset({"cls", "mean", "max", "attention"})


@dataclass(frozen=True, slots=True)
class PoolingHeadConfig:
    """Describe pooling branches and the MLP applied to their concatenation."""

    poolings: tuple[str, ...] = ("cls",)
    mlp_hidden_dims: tuple[int, ...] = ()
    dropout: float = 0.1
    attention_hidden_dim: int | None = None
    attention_num_heads: int = 1
    native_logit_weight: float = 1.0
    attention_logit_weight: float = 0.0

    def __post_init__(self) -> None:
        normalized = tuple(pooling.strip().lower() for pooling in self.poolings)
        if not normalized:
            raise ValueError("Transformer head poolings must not be empty")
        unknown = set(normalized) - SUPPORTED_POOLINGS
        if unknown:
            raise ValueError(
                "unsupported Transformer head poolings: "
                + ", ".join(sorted(unknown))
            )
        if len(set(normalized)) != len(normalized):
            raise ValueError("Transformer head poolings must be unique")
        if any(dimension < 1 for dimension in self.mlp_hidden_dims):
            raise ValueError("Transformer head MLP dimensions must be positive")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("Transformer head dropout must be in [0, 1)")
        if self.attention_hidden_dim is not None and self.attention_hidden_dim < 1:
            raise ValueError("attention_hidden_dim must be positive or None")
        if self.attention_num_heads < 1:
            raise ValueError("attention_num_heads must be positive")
        if self.native_logit_weight < 0.0 or self.attention_logit_weight < 0.0:
            raise ValueError("hybrid logit weights must be non-negative")
        if self.native_logit_weight == 0.0 and self.attention_logit_weight == 0.0:
            raise ValueError("at least one hybrid logit weight must be positive")
        object.__setattr__(self, "poolings", normalized)

    def to_dict(self) -> dict[str, object]:
        return {
            "poolings": list(self.poolings),
            "mlp_hidden_dims": list(self.mlp_hidden_dims),
            "dropout": self.dropout,
            "attention_hidden_dim": self.attention_hidden_dim,
            "attention_num_heads": self.attention_num_heads,
            "native_logit_weight": self.native_logit_weight,
            "attention_logit_weight": self.attention_logit_weight,
        }

    @classmethod
    def from_dict(cls, values: dict[str, object]) -> PoolingHeadConfig:
        return cls(
            poolings=tuple(str(value) for value in values.get("poolings", ("cls",))),
            mlp_hidden_dims=tuple(
                int(value) for value in values.get("mlp_hidden_dims", ())
            ),
            dropout=float(values.get("dropout", 0.1)),
            attention_hidden_dim=(
                None
                if values.get("attention_hidden_dim") is None
                else int(values["attention_hidden_dim"])
            ),
            attention_num_heads=int(values.get("attention_num_heads", 1)),
            native_logit_weight=float(values.get("native_logit_weight", 1.0)),
            attention_logit_weight=float(values.get("attention_logit_weight", 0.0)),
        )


__all__ = ["PoolingHeadConfig", "SUPPORTED_POOLINGS"]
