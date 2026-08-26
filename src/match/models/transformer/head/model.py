"""Composition of token pooling branches and a classification MLP."""

from __future__ import annotations

import torch
from torch import nn

from .config import PoolingHeadConfig
from .pooling import AttentionPooling, CLSPooling, MaxPooling, MeanPooling


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

        pooled_width = hidden_size * sum(
            config.attention_num_heads if name == "attention" else 1
            for name in config.poolings
        )
        dimensions = [pooled_width, *config.mlp_hidden_dims]
        layers: list[nn.Module] = [nn.Dropout(config.dropout)]
        for input_dim, output_dim in zip(dimensions, dimensions[1:]):
            layers.extend(
                [nn.Linear(input_dim, output_dim), nn.GELU(), nn.Dropout(config.dropout)]
            )
        layers.append(nn.Linear(dimensions[-1], num_labels))
        self.classifier = nn.Sequential(*layers)

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


__all__ = ["TransformerPoolingHead"]
