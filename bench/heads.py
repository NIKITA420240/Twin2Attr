"""The final trainable head over two frozen cross-encoder orders."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as functional


@dataclass(frozen=True)
class HeadOutput:
    """Averaged match logit and the two directional logits."""

    logit: torch.Tensor
    forward_logit: torch.Tensor
    reverse_logit: torch.Tensor


def _validate_features(features: torch.Tensor, hidden_size: int) -> None:
    if features.ndim != 3 or features.shape[1:] != (5, hidden_size):
        raise ValueError(
            "features must have shape "
            f"[batch, 5, {hidden_size}], got {tuple(features.shape)}"
        )


class _CardProjector(nn.Module):
    """One shared projector for the mean/max pools of either card."""

    def __init__(self, hidden_size: int, projection_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(2 * hidden_size),
            nn.Linear(2 * hidden_size, projection_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(projection_dim),
        )

    def forward(self, mean: torch.Tensor, maximum: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat((mean, maximum), dim=-1))


class _CategoryConditionedDirectionalHead(nn.Module):
    """Symmetric card interactions followed by category-conditioned FiLM."""

    def __init__(
        self,
        hidden_size: int,
        projection_dim: int,
        mlp_dim: int,
        dropout: float,
        num_categories: int,
        category_embedding_dim: int,
    ) -> None:
        super().__init__()
        if num_categories < 1:
            raise ValueError("num_categories must be positive")
        self.hidden_size = hidden_size
        self.card_projector = _CardProjector(hidden_size, projection_dim, dropout)
        pair_dim = hidden_size + 3 * projection_dim
        self.representation = nn.Sequential(
            nn.LayerNorm(pair_dim),
            nn.Linear(pair_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.category_embedding = nn.Embedding(num_categories, category_embedding_dim)
        self.film = nn.Linear(category_embedding_dim, 2 * mlp_dim)
        self.classifier = nn.Linear(mlp_dim, 1)

    def forward(
        self,
        features: torch.Tensor,
        category_ids: torch.Tensor,
    ) -> torch.Tensor:
        _validate_features(features, self.hidden_size)
        cls, left_mean, left_max, right_mean, right_max = features.unbind(dim=1)
        left = self.card_projector(left_mean, left_max)
        right = self.card_projector(right_mean, right_max)
        pair = torch.cat(
            (cls, left + right, torch.abs(left - right), left * right),
            dim=-1,
        )
        representation = self.representation(pair)
        scale, shift = self.film(self.category_embedding(category_ids)).chunk(2, dim=-1)
        conditioned = representation * (1.0 + 0.1 * torch.tanh(scale)) + shift
        return self.classifier(conditioned).squeeze(-1)


class BidirectionalCategoryConditionedHead(nn.Module):
    """Average one shared category-conditioned head over both input orders."""

    def __init__(
        self,
        hidden_size: int,
        projection_dim: int,
        mlp_dim: int,
        dropout: float,
        num_categories: int,
        category_embedding_dim: int,
    ) -> None:
        super().__init__()
        self.directional_head = _CategoryConditionedDirectionalHead(
            hidden_size,
            projection_dim,
            mlp_dim,
            dropout,
            num_categories,
            category_embedding_dim,
        )

    def forward(
        self,
        features: torch.Tensor,
        *,
        reverse_features: torch.Tensor,
        category_ids: torch.Tensor,
    ) -> HeadOutput:
        forward = self.directional_head(features, category_ids)
        reverse = self.directional_head(reverse_features, category_ids)
        return HeadOutput(
            logit=0.5 * (forward + reverse),
            forward_logit=forward,
            reverse_logit=reverse,
        )


def head_loss(
    output: HeadOutput,
    target: torch.Tensor,
    *,
    positive_weight: torch.Tensor,
    consistency_weight: float,
) -> torch.Tensor:
    """Weighted match loss plus agreement between both encoder orders."""

    match_loss = functional.binary_cross_entropy_with_logits(
        output.logit,
        target,
        pos_weight=positive_weight,
    )
    consistency_loss = functional.smooth_l1_loss(
        output.forward_logit,
        output.reverse_logit,
    )
    return match_loss + consistency_weight * consistency_loss


__all__ = [
    "BidirectionalCategoryConditionedHead",
    "HeadOutput",
    "head_loss",
]
