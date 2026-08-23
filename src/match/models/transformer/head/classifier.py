"""Serializable Hugging Face classifier with a composable pooling head."""

from __future__ import annotations

import inspect
from typing import Any

import torch
from torch import nn
from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput

from .config import PoolingHeadConfig
from .model import TransformerPoolingHead


def _restore_backbone_config(values: dict[str, Any]) -> PretrainedConfig:
    parameters = dict(values)
    try:
        model_type = str(parameters.pop("model_type"))
    except KeyError as error:
        raise ValueError("backbone config must contain model_type") from error
    return AutoConfig.for_model(model_type, **parameters)


class PoolingSequenceClassifierConfig(PretrainedConfig):
    model_type = "match_pooling_sequence_classifier"

    def __init__(
        self,
        *,
        backbone_config: dict[str, Any] | None = None,
        head_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.backbone_config = dict(backbone_config or {})
        self.head_config = dict(head_config or PoolingHeadConfig().to_dict())
        self.hidden_size = 0
        if self.backbone_config:
            restored_config = _restore_backbone_config(self.backbone_config)
            self.hidden_size = int(getattr(restored_config, "hidden_size", 0))
            if self.hidden_size < 1:
                raise ValueError("backbone config must expose a hidden size")
        self.match_head_type = "pooling"


class PoolingSequenceClassifier(PreTrainedModel):
    config_class = PoolingSequenceClassifierConfig
    base_model_prefix = "backbone"

    def __init__(self, config: PoolingSequenceClassifierConfig) -> None:
        super().__init__(config)
        backbone_config = _restore_backbone_config(config.backbone_config)
        self.backbone = AutoModel.from_config(backbone_config)
        self.head = TransformerPoolingHead(
            hidden_size=config.hidden_size,
            num_labels=config.num_labels,
            config=PoolingHeadConfig.from_dict(config.head_config),
        )
        self.post_init()

    @classmethod
    def from_backbone_pretrained(
        cls,
        model_path: str,
        *,
        head_config: PoolingHeadConfig,
        num_labels: int,
        id2label: dict[int, str],
        label2id: dict[str, int],
    ) -> PoolingSequenceClassifier:
        backbone_config = AutoConfig.from_pretrained(model_path)
        config = PoolingSequenceClassifierConfig(
            backbone_config=backbone_config.to_dict(),
            head_config=head_config.to_dict(),
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )
        model = cls(config)
        model.backbone = AutoModel.from_pretrained(
            model_path,
            config=backbone_config,
        )
        return model

    def get_input_embeddings(self) -> nn.Module:
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.backbone.set_input_embeddings(value)

    def resize_token_embeddings(
        self,
        new_num_tokens: int | None = None,
        pad_to_multiple_of: int | None = None,
        mean_resizing: bool = True,
    ) -> nn.Embedding:
        resize_parameters = inspect.signature(
            self.backbone.resize_token_embeddings
        ).parameters
        resize_kwargs: dict[str, Any] = {
            "pad_to_multiple_of": pad_to_multiple_of,
        }
        if "mean_resizing" in resize_parameters:
            resize_kwargs["mean_resizing"] = mean_resizing
        embeddings = self.backbone.resize_token_embeddings(
            new_num_tokens,
            **resize_kwargs,
        )
        vocabulary_size = int(embeddings.num_embeddings)
        self.config.vocab_size = vocabulary_size
        self.config.backbone_config["vocab_size"] = vocabulary_size
        return embeddings

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> SequenceClassifierOutput | tuple[torch.Tensor, ...]:
        if attention_mask is None:
            if input_ids is None:
                raise ValueError("attention_mask is required when input_ids is absent")
            attention_mask = torch.ones_like(input_ids)
        backbone_kwargs = dict(kwargs)
        if token_type_ids is not None:
            backbone_kwargs["token_type_ids"] = token_type_ids
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **backbone_kwargs,
        )
        logits = self.head(outputs.last_hidden_state, attention_mask)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
        use_return_dict = (
            self.config.use_return_dict if return_dict is None else return_dict
        )
        if not use_return_dict:
            optional_outputs = tuple(
                value
                for value in (outputs.hidden_states, outputs.attentions)
                if value is not None
            )
            result = (logits, *optional_outputs)
            return (loss, *result) if loss is not None else result
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = ["PoolingSequenceClassifier", "PoolingSequenceClassifierConfig"]
