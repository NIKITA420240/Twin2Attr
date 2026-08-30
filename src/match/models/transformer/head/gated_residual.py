"""Zero-initialized text and typed residuals gated by pair context."""

from __future__ import annotations

import torch
from torch import nn

from .config import PoolingHeadConfig
from .model import TransformerPoolingHead


def _residual_mlp(
    input_width: int,
    hidden_dims: tuple[int, ...],
    dropout: float,
) -> nn.Sequential:
    dimensions = [input_width, *hidden_dims]
    layers: list[nn.Module] = [nn.Dropout(dropout)]
    for input_dim, output_dim in zip(dimensions, dimensions[1:]):
        layers.extend(
            [nn.Linear(input_dim, output_dim), nn.GELU(), nn.Dropout(dropout)]
        )
    layers.append(nn.Linear(dimensions[-1], 1))
    return nn.Sequential(*layers)


def _last_linear(module: nn.Sequential) -> nn.Linear:
    layer = module[-1]
    if not isinstance(layer, nn.Linear):
        raise RuntimeError("residual branch must end with a linear layer")
    return layer


class GatedResidualFusionHead(nn.Module):
    """Add zero-initialized text and context-gated typed score corrections."""

    def __init__(
        self,
        hidden_size: int,
        typed_feature_count: int,
        config: PoolingHeadConfig,
    ) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError("gated residual fusion requires a hidden size")
        if typed_feature_count < 1:
            raise ValueError("gated residual fusion requires typed features")
        self.typed_feature_count = int(typed_feature_count)
        self.text_pool = TransformerPoolingHead(hidden_size, 1, config)
        self.text_pool.classifier = nn.Identity()
        text_width = self.text_pool.output_width
        self.text_residual = _residual_mlp(
            text_width,
            config.mlp_hidden_dims,
            config.dropout,
        )

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
        self.typed_residual = nn.Linear(typed_width, 1)
        self.typed_gate = nn.Linear(text_width + typed_width, 1)

    def reset_residual_outputs(self) -> None:
        """Make the complete head an exact no-op at initialization."""
        text_output = _last_linear(self.text_residual)
        nn.init.zeros_(text_output.weight)
        nn.init.zeros_(text_output.bias)
        nn.init.zeros_(self.typed_residual.weight)
        nn.init.zeros_(self.typed_residual.bias)

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

        text = self.text_pool.pool(hidden_states, attention_mask)
        typed = self.typed_encoder(
            typed_features.to(device=text.device, dtype=text.dtype)
        )
        text_delta = self.text_residual(text)
        typed_delta = self.typed_residual(typed)
        gate = torch.sigmoid(self.typed_gate(torch.cat((text, typed), dim=-1)))
        return text_delta + gate * typed_delta


__all__ = ["GatedResidualFusionHead"]
