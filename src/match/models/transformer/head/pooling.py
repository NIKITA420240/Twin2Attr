"""Mask-aware pooling components for contextual token embeddings."""

from __future__ import annotations

import torch
from torch import nn


class CLSPooling(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        valid = attention_mask.any(dim=1, keepdim=True)
        cls_embedding = hidden_states[:, 0]
        return torch.where(valid, cls_embedding, torch.zeros_like(cls_embedding))


class MeanPooling(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
        return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


class MaxPooling(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(dtype=torch.bool)
        pooled = hidden_states.masked_fill(
            ~mask,
            torch.finfo(hidden_states.dtype).min,
        ).amax(dim=1)
        return torch.where(
            attention_mask.any(dim=1, keepdim=True),
            pooled,
            torch.zeros_like(pooled),
        )


class AttentionPooling(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        attention_hidden_dim: int | None,
        num_heads: int = 1,
    ) -> None:
        super().__init__()
        if num_heads < 1:
            raise ValueError("num_heads must be positive")
        inner_size = attention_hidden_dim or max(1, hidden_size // 2)
        self.num_heads = num_heads
        self.score = nn.Sequential(
            nn.Linear(hidden_size, inner_size),
            nn.Tanh(),
            nn.Linear(inner_size, num_heads),
        )

    def attention_weights(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        scores = self.score(hidden_states)
        mask = attention_mask.to(dtype=torch.bool).unsqueeze(-1)
        scores = scores.masked_fill(
            ~mask,
            torch.finfo(scores.dtype).min,
        )
        weights = torch.softmax(scores.float(), dim=1)
        weights = weights * mask.to(dtype=weights.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(
            torch.finfo(weights.dtype).eps
        )
        return weights.to(dtype=hidden_states.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        weights = self.attention_weights(hidden_states, attention_mask)
        pooled = torch.einsum("blk,blh->bkh", weights, hidden_states).flatten(1)
        return torch.where(
            attention_mask.any(dim=1, keepdim=True),
            pooled,
            torch.zeros_like(pooled),
        )


__all__ = ["AttentionPooling", "CLSPooling", "MaxPooling", "MeanPooling"]
