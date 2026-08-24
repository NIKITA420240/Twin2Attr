"""Differential learning-rate groups for Transformer fine-tuning."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
from torch import nn
from transformers import PreTrainedModel


NamedParameters = Iterable[tuple[str, nn.Parameter]]


@dataclass(frozen=True, slots=True)
class LearningRateMultipliers:
    """LR ratios kept stable when Trainer or HPO changes backbone LR."""

    embeddings: float = 1.0
    head: float = 1.0
    layerwise_decay: float = 1.0

    def __post_init__(self) -> None:
        if self.embeddings <= 0.0 or self.head <= 0.0:
            raise ValueError("learning-rate multipliers must be positive")
        if not 0.0 < self.layerwise_decay <= 1.0:
            raise ValueError("layerwise_decay must be in (0, 1]")

    @classmethod
    def from_learning_rates(
        cls,
        *,
        embeddings_lr: float,
        backbone_lr: float,
        head_lr: float,
        layerwise_decay: float,
    ) -> LearningRateMultipliers:
        if min(embeddings_lr, backbone_lr, head_lr) <= 0.0:
            raise ValueError("learning rates must be positive")
        return cls(
            embeddings=embeddings_lr / backbone_lr,
            head=head_lr / backbone_lr,
            layerwise_decay=layerwise_decay,
        )


def _encoder_layers(backbone: nn.Module) -> Sequence[nn.Module]:
    """Find encoder layers on BERT/RoBERTa or DistilBERT backbones."""

    encoder = getattr(backbone, "encoder", None)
    if encoder is not None and hasattr(encoder, "layer"):
        return encoder.layer
    transformer = getattr(backbone, "transformer", None)
    if transformer is not None and hasattr(transformer, "layer"):
        return transformer.layer
    raise AttributeError(
        "unsupported Transformer backbone: expected encoder.layer or transformer.layer"
    )


def _uses_weight_decay(parameter_name: str) -> bool:
    lowered = parameter_name.lower()
    return not (
        lowered.endswith(".bias")
        or lowered.endswith("layernorm.weight")
        or lowered.endswith("layer_norm.weight")
    )


def _append_groups(
    groups: list[dict[str, object]],
    named_parameters: NamedParameters,
    *,
    learning_rate: float,
    weight_decay: float,
    group_name: str,
) -> None:
    parameters = list(named_parameters)
    decay = [
        parameter
        for name, parameter in parameters
        if _uses_weight_decay(name)
    ]
    no_decay = [
        parameter
        for name, parameter in parameters
        if not _uses_weight_decay(name)
    ]
    if decay:
        groups.append(
            {
                "params": decay,
                "lr": learning_rate,
                "weight_decay": weight_decay,
                "group_name": f"{group_name}.decay",
            }
        )
    if no_decay:
        groups.append(
            {
                "params": no_decay,
                "lr": learning_rate,
                "weight_decay": 0.0,
                "group_name": f"{group_name}.no_decay",
            }
        )


def build_transformer_optimizer(
    model: PreTrainedModel,
    *,
    backbone_lr: float,
    multipliers: LearningRateMultipliers,
    weight_decay: float,
) -> torch.optim.AdamW:
    """Create non-overlapping AdamW groups for embeddings/backbone/head."""

    if backbone_lr <= 0.0:
        raise ValueError("backbone_lr must be positive")
    if weight_decay < 0.0:
        raise ValueError("weight_decay must be non-negative")

    backbone = model.base_model
    embeddings = getattr(backbone, "embeddings", None)
    if embeddings is None:
        raise AttributeError("Transformer backbone has no embeddings module")
    encoder_layers = _encoder_layers(backbone)
    if not encoder_layers:
        raise ValueError("Transformer backbone has no encoder layers")

    groups: list[dict[str, object]] = []
    assigned_ids: set[int] = set()

    def claim(named_parameters: NamedParameters) -> list[tuple[str, nn.Parameter]]:
        claimed: list[tuple[str, nn.Parameter]] = []
        for name, parameter in named_parameters:
            if not parameter.requires_grad:
                continue
            parameter_id = id(parameter)
            if parameter_id in assigned_ids:
                raise RuntimeError(f"parameter {name!r} was assigned more than once")
            assigned_ids.add(parameter_id)
            claimed.append((name, parameter))
        return claimed

    _append_groups(
        groups,
        claim(embeddings.named_parameters(prefix="backbone.embeddings")),
        learning_rate=backbone_lr * multipliers.embeddings,
        weight_decay=weight_decay,
        group_name="embeddings",
    )

    layer_count = len(encoder_layers)
    for layer_index, layer in enumerate(encoder_layers):
        distance_from_top = layer_count - layer_index - 1
        layer_lr = backbone_lr * multipliers.layerwise_decay**distance_from_top
        _append_groups(
            groups,
            claim(
                layer.named_parameters(
                    prefix=f"backbone.encoder.layer.{layer_index}"
                )
            ),
            learning_rate=layer_lr,
            weight_decay=weight_decay,
            group_name=f"backbone.layer.{layer_index}",
        )

    _append_groups(
        groups,
        claim(
            (name, parameter)
            for name, parameter in backbone.named_parameters(prefix="backbone")
            if parameter.requires_grad and id(parameter) not in assigned_ids
        ),
        learning_rate=backbone_lr,
        weight_decay=weight_decay,
        group_name="backbone.other",
    )

    backbone_parameter_ids = {id(parameter) for parameter in backbone.parameters()}
    _append_groups(
        groups,
        claim(
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and id(parameter) not in backbone_parameter_ids
        ),
        learning_rate=backbone_lr * multipliers.head,
        weight_decay=weight_decay,
        group_name="head",
    )

    trainable_ids = {
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    }
    if assigned_ids != trainable_ids:
        raise RuntimeError(
            "optimizer parameter coverage mismatch: "
            f"{len(trainable_ids - assigned_ids)} missing, "
            f"{len(assigned_ids - trainable_ids)} unexpected"
        )

    return torch.optim.AdamW(groups)


def optimizer_group_summary(
    optimizer: torch.optim.Optimizer,
) -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "name": str(group.get("group_name", "unnamed")),
            "learning_rate": float(group["lr"]),
            "weight_decay": float(group["weight_decay"]),
            "parameter_count": sum(
                parameter.numel() for parameter in group["params"]
            ),
        }
        for group in optimizer.param_groups
    )


__all__ = [
    "LearningRateMultipliers",
    "build_transformer_optimizer",
    "optimizer_group_summary",
]
