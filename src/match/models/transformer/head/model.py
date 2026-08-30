"""Composition of token pooling branches and a classification MLP."""

from __future__ import annotations

import torch
from torch import nn

from .config import PoolingHeadConfig
from .pooling import AttentionPooling, CLSPooling, MaxPooling, MeanPooling


def pooling_output_width(hidden_size: int, config: PoolingHeadConfig) -> int:
    return hidden_size * sum(
        config.attention_num_heads if name == "attention" else 1
        for name in config.poolings
    )


def _classification_mlp(
    input_width: int,
    hidden_dims: tuple[int, ...],
    num_labels: int,
    dropout: float,
) -> nn.Sequential:
    dimensions = [input_width, *hidden_dims]
    layers: list[nn.Module] = [nn.Dropout(dropout)]
    for input_dim, output_dim in zip(dimensions, dimensions[1:]):
        layers.extend(
            [nn.Linear(input_dim, output_dim), nn.GELU(), nn.Dropout(dropout)]
        )
    layers.append(nn.Linear(dimensions[-1], num_labels))
    return nn.Sequential(*layers)


class TransformerPoolingHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        config: PoolingHeadConfig,
    ) -> None:
        super().__init__()
        factories = {
            "cls": CLSPooling,
            "mean": MeanPooling,
            "max": MaxPooling,
        }
        branches: dict[str, nn.Module] = {}
        for name in config.poolings:
            if name == "attention":
                branches[name] = AttentionPooling(
                    hidden_size,
                    config.attention_hidden_dim,
                    config.attention_num_heads,
                )
            else:
                branches[name] = factories[name]()
        self.poolings = nn.ModuleDict(branches)

        self.output_width = pooling_output_width(hidden_size, config)
        self.classifier = _classification_mlp(
            self.output_width,
            config.mlp_hidden_dims,
            num_labels,
            config.dropout,
        )

    def pool(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            [
                pooling(hidden_states, attention_mask)
                for pooling in self.poolings.values()
            ],
            dim=-1,
        )

    def attention_weights(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        pooling = self.poolings["attention"] if "attention" in self.poolings else None
        if not isinstance(pooling, AttentionPooling):
            raise TypeError("Transformer pooling head has no attention branch")
        return pooling.attention_weights(hidden_states, attention_mask)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.classifier(self.pool(hidden_states, attention_mask))


class TypedAttributeFusionHead(nn.Module):
    """Fuse pooled Transformer context with deterministic pair features."""

    def __init__(
        self,
        hidden_size: int,
        typed_feature_count: int,
        num_labels: int,
        config: PoolingHeadConfig,
    ) -> None:
        super().__init__()
        if typed_feature_count < 1:
            raise ValueError("typed fusion requires a positive feature count")
        self.typed_feature_count = int(typed_feature_count)
        self.text_head = TransformerPoolingHead(
            hidden_size,
            num_labels,
            config,
        )
        # Only the pooling branches are used; avoid unused classifier parameters.
        self.text_head.classifier = nn.Identity()

        typed_layers: list[nn.Module] = []
        if config.typed_layer_norm:
            typed_layers.append(nn.LayerNorm(self.typed_feature_count))
        typed_dimensions = [self.typed_feature_count, *config.typed_hidden_dims]
        for input_dim, output_dim in zip(
            typed_dimensions,
            typed_dimensions[1:],
        ):
            typed_layers.extend(
                [
                    nn.Linear(input_dim, output_dim),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                ]
            )
        self.typed_encoder = nn.Sequential(*typed_layers)
        typed_width = typed_dimensions[-1]
        self.classifier = _classification_mlp(
            self.text_head.output_width + typed_width,
            config.mlp_hidden_dims,
            num_labels,
            config.dropout,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        typed_features: torch.Tensor,
    ) -> torch.Tensor:
        if typed_features.ndim != 2:
            raise ValueError("typed_features must have shape [batch, features]")
        if typed_features.shape[0] != hidden_states.shape[0]:
            raise ValueError("typed_features batch dimension does not match text")
        if typed_features.shape[1] != self.typed_feature_count:
            raise ValueError(
                f"expected {self.typed_feature_count} typed features, got "
                f"{typed_features.shape[1]}"
            )
        if not torch.isfinite(typed_features).all():
            raise ValueError("typed_features must contain only finite values")
        text = self.text_head.pool(hidden_states, attention_mask)
        typed = self.typed_encoder(typed_features.to(dtype=text.dtype))
        return self.classifier(torch.cat((text, typed), dim=-1))


__all__ = [
    "TransformerPoolingHead",
    "TypedAttributeFusionHead",
    "pooling_output_width",
]
