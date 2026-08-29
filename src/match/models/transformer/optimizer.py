"""Differential learning-rate groups for Transformer fine-tuning."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import torch
from torch import nn
from transformers import PreTrainedModel


NamedParameters = Iterable[tuple[str, nn.Parameter]]
_TRAINABLE_TOKEN_IDS_ATTRIBUTE = "_match_trainable_token_ids"
_GRADIENT_MASK_HANDLE_ATTRIBUTE = "_match_gradient_mask_handle"


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
    layers = getattr(backbone, "layers", None)
    if layers is not None:
        return layers
    raise AttributeError(
        "unsupported Transformer backbone: expected encoder.layer, "
        "transformer.layer or layers"
    )


def freeze_backbone_except_last_layers(
    model: PreTrainedModel,
    last_n_layers: int,
    *,
    train_input_word_embeddings: bool = False,
) -> tuple[int, ...]:
    """Freeze the backbone except its last encoder layers and optional word rows."""
    if last_n_layers < 1:
        raise ValueError("last_n_layers must be positive")
    backbone = model.base_model
    encoder_layers = _encoder_layers(backbone)
    layer_count = len(encoder_layers)
    if last_n_layers > layer_count:
        raise ValueError(
            f"cannot train last {last_n_layers} layers of a {layer_count}-layer backbone"
        )

    for parameter in backbone.parameters():
        parameter.requires_grad_(False)

    first_trainable_layer = layer_count - last_n_layers
    for layer in encoder_layers[first_trainable_layer:]:
        for parameter in layer.parameters():
            parameter.requires_grad_(True)

    if train_input_word_embeddings:
        embeddings = model.get_input_embeddings()
        if embeddings is None or not hasattr(embeddings, "weight"):
            raise ValueError("model does not expose trainable input embeddings")
        embeddings.weight.requires_grad_(True)

    return tuple(range(first_trainable_layer, layer_count))


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


def restrict_word_embedding_updates(
    model: PreTrainedModel,
    token_ids: Sequence[int],
) -> None:
    """Restrict word-embedding optimizer updates to selected vocabulary rows.

    The embedding matrix remains a standard Hugging Face parameter, so saved
    checkpoints need no custom loading code. ``build_transformer_optimizer``
    reads the marker installed here and disables weight decay for this matrix.
    A post-accumulation hook masks non-selected rows before gradient clipping.
    """
    embeddings = model.get_input_embeddings()
    if embeddings is None or not hasattr(embeddings, "weight"):
        raise ValueError("model does not expose trainable input embeddings")
    weight = embeddings.weight
    normalized_ids = tuple(sorted({int(token_id) for token_id in token_ids}))
    if not normalized_ids:
        raise ValueError("at least one trainable token id is required")
    vocabulary_size = int(weight.shape[0])
    if normalized_ids[0] < 0 or normalized_ids[-1] >= vocabulary_size:
        raise ValueError("trainable token id is outside the model vocabulary")
    setattr(weight, _TRAINABLE_TOKEN_IDS_ATTRIBUTE, normalized_ids)

    previous_handle = getattr(weight, _GRADIENT_MASK_HANDLE_ATTRIBUTE, None)
    if previous_handle is not None:
        previous_handle.remove()

    def mask_after_accumulation(parameter: nn.Parameter) -> None:
        _mask_unselected_embedding_gradients(parameter, normalized_ids)

    handle = weight.register_post_accumulate_grad_hook(mask_after_accumulation)
    setattr(weight, _GRADIENT_MASK_HANDLE_ATTRIBUTE, handle)


def _restricted_token_ids(parameter: nn.Parameter) -> tuple[int, ...] | None:
    value = getattr(parameter, _TRAINABLE_TOKEN_IDS_ATTRIBUTE, None)
    return None if value is None else tuple(int(token_id) for token_id in value)


def _remove_embedding_restriction(model: PreTrainedModel) -> None:
    embeddings = model.get_input_embeddings()
    if embeddings is None or not hasattr(embeddings, "weight"):
        return
    weight = embeddings.weight
    handle = getattr(weight, _GRADIENT_MASK_HANDLE_ATTRIBUTE, None)
    if handle is not None:
        handle.remove()
        delattr(weight, _GRADIENT_MASK_HANDLE_ATTRIBUTE)
    if hasattr(weight, _TRAINABLE_TOKEN_IDS_ATTRIBUTE):
        delattr(weight, _TRAINABLE_TOKEN_IDS_ATTRIBUTE)


def _mask_unselected_embedding_gradients(
    parameter: nn.Parameter,
    token_ids: tuple[int, ...],
) -> None:
    gradient = parameter.grad
    if gradient is None:
        return
    if gradient.is_sparse:
        raise RuntimeError("restricted embedding updates require dense gradients")
    indices = torch.tensor(token_ids, device=gradient.device, dtype=torch.long)
    selected_gradients = gradient.index_select(0, indices).clone()
    gradient.zero_()
    gradient.index_copy_(0, indices, selected_gradients)


def _install_final_embedding_gradient_mask(
    optimizer: torch.optim.Optimizer,
    parameter: nn.Parameter,
    token_ids: tuple[int, ...],
) -> None:
    """Reapply the row mask after any distributed gradient synchronization."""

    def mask_before_step(
        _optimizer: torch.optim.Optimizer,
        _args: tuple[object, ...],
        _kwargs: dict[str, object],
    ) -> None:
        _mask_unselected_embedding_gradients(parameter, token_ids)

    optimizer.register_step_pre_hook(mask_before_step)


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
        embeddings = model.get_input_embeddings()
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

    embedding_parameters = list(
        embeddings.named_parameters(prefix="backbone.embeddings")
    )
    input_embeddings = model.get_input_embeddings()
    if input_embeddings is None or not hasattr(input_embeddings, "weight"):
        raise AttributeError("Transformer backbone has no word-embedding weight")
    word_embedding_weight = input_embeddings.weight
    trainable_token_ids = _restricted_token_ids(word_embedding_weight)
    if trainable_token_ids is None:
        _append_groups(
            groups,
            claim(embedding_parameters),
            learning_rate=backbone_lr * multipliers.embeddings,
            weight_decay=weight_decay,
            group_name="embeddings",
        )
    else:
        other_embedding_parameters = [
            (name, parameter)
            for name, parameter in embedding_parameters
            if id(parameter) != id(word_embedding_weight)
        ]
        _append_groups(
            groups,
            claim(other_embedding_parameters),
            learning_rate=backbone_lr * multipliers.embeddings,
            weight_decay=weight_decay,
            group_name="embeddings",
        )
        claimed_word_embeddings = claim(
            [("backbone.embeddings.word_embeddings.weight", word_embedding_weight)]
        )
        if len(claimed_word_embeddings) != 1:
            raise RuntimeError("word-embedding weight is not trainable")
        groups.append(
            {
                "params": [word_embedding_weight],
                "lr": backbone_lr * multipliers.embeddings,
                # Decoupled AdamW decay would alter frozen vocabulary rows.
                "weight_decay": 0.0,
                "group_name": "new_token_embeddings.no_decay",
            }
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

    optimizer = torch.optim.AdamW(groups)
    if trainable_token_ids is not None:
        _install_final_embedding_gradient_mask(
            optimizer,
            word_embedding_weight,
            trainable_token_ids,
        )
    return optimizer


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


def finish_special_token_adaptation(
    model: PreTrainedModel,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: object | None,
    *,
    last_n_layers: int,
    main_backbone_lr: float,
    main_multipliers: LearningRateMultipliers,
    reset_learning_rates: bool,
) -> tuple[int, ...]:
    """Switch from the short adaptation phase to the normal top-N phase."""
    _remove_embedding_restriction(model)
    trainable_layers = freeze_backbone_except_last_layers(model, last_n_layers)
    if not reset_learning_rates:
        return trainable_layers

    layer_count = len(_encoder_layers(model.base_model))
    scheduler_base_lrs = getattr(lr_scheduler, "base_lrs", None)
    for index, group in enumerate(optimizer.param_groups):
        name = str(group.get("group_name", ""))
        if name.startswith("head."):
            target_lr = main_backbone_lr * main_multipliers.head
        elif name.startswith("backbone.layer."):
            layer_index = int(name.split(".")[2])
            distance_from_top = layer_count - layer_index - 1
            target_lr = (
                main_backbone_lr
                * main_multipliers.layerwise_decay**distance_from_top
            )
        else:
            target_lr = main_backbone_lr
        old_base_lr = (
            float(scheduler_base_lrs[index])
            if scheduler_base_lrs is not None
            else float(group["lr"])
        )
        schedule_factor = (
            float(group["lr"]) / old_base_lr if old_base_lr > 0.0 else 1.0
        )
        group["initial_lr"] = target_lr
        group["lr"] = target_lr * schedule_factor
        if scheduler_base_lrs is not None:
            scheduler_base_lrs[index] = target_lr
    return trainable_layers


__all__ = [
    "LearningRateMultipliers",
    "build_transformer_optimizer",
    "freeze_backbone_except_last_layers",
    "finish_special_token_adaptation",
    "optimizer_group_summary",
    "restrict_word_embedding_updates",
]
