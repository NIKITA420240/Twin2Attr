"""Serializable Hugging Face classifiers with composable pooling heads."""

from __future__ import annotations

import inspect
from typing import Any

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForSequenceClassification,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from .config import PoolingHeadConfig
from .gated_residual import GatedResidualFusionHead
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
        attention_implementation: str = "auto",
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
            **(
                {}
                if attention_implementation == "auto"
                else {"attn_implementation": attention_implementation}
            ),
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
        # The pooling head only consumes the final contextual representation.
        # Disable diagnostic tensors even when the pretrained config enables them.
        backbone_kwargs["output_hidden_states"] = False
        backbone_kwargs["output_attentions"] = False
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
            result = (logits,)
            return (loss, *result) if loss is not None else result
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
        )


class HybridSequenceClassifierConfig(PoolingSequenceClassifierConfig):
    model_type = "match_hybrid_sequence_classifier"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.match_head_type = "hybrid"


class HybridSequenceClassifier(PreTrainedModel):
    """BGE's pretrained scalar head plus a trainable pooling branch.

    The two scalar logits are mixed before converting to the repository's
    two-class ``[different, match]`` representation.  Keeping the native head
    as a real module preserves its checkpoint weights and makes the mix
    backwards-compatible with the existing Trainer and predictor contracts.
    """

    config_class = HybridSequenceClassifierConfig
    base_model_prefix = "backbone"

    def __init__(self, config: HybridSequenceClassifierConfig) -> None:
        super().__init__(config)
        backbone_config = _restore_backbone_config(config.backbone_config)
        self.backbone = AutoModel.from_config(backbone_config)
        native_model = AutoModelForSequenceClassification.from_config(backbone_config)
        native_head = getattr(native_model, "classifier", None)
        if native_head is None:
            native_head = getattr(native_model, "score", None)
        if native_head is None:
            raise ValueError("pretrained classifier exposes neither classifier nor score")
        self.native_head = native_head
        self.attention_head = TransformerPoolingHead(
            hidden_size=config.hidden_size,
            num_labels=1,
            config=PoolingHeadConfig.from_dict(config.head_config),
        )
        head_config = PoolingHeadConfig.from_dict(config.head_config)
        logit_weights = torch.tensor(
            [head_config.native_logit_weight, head_config.attention_logit_weight],
            dtype=torch.float32,
        )
        if head_config.train_logit_weights:
            self.logit_weights = nn.Parameter(logit_weights)
        else:
            self.register_buffer("logit_weights", logit_weights)
        self.native_only = (
            not head_config.train_logit_weights
            and head_config.attention_logit_weight == 0.0
        )
        if self.native_only:
            self.attention_head.requires_grad_(False)
        self.post_init()

    @classmethod
    def from_backbone_pretrained(
        cls,
        model_path: str,
        *,
        head_config: PoolingHeadConfig,
        id2label: dict[int, str],
        label2id: dict[str, int],
        attention_implementation: str = "auto",
    ) -> HybridSequenceClassifier:
        pretrained = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            **(
                {}
                if attention_implementation == "auto"
                else {"attn_implementation": attention_implementation}
            ),
        )
        backbone_config = AutoConfig.from_pretrained(model_path)
        config = HybridSequenceClassifierConfig(
            backbone_config=backbone_config.to_dict(),
            head_config=head_config.to_dict(),
            num_labels=2,
            id2label=id2label,
            label2id=label2id,
        )
        model = cls(config)
        model.backbone = pretrained.base_model
        native_head = getattr(pretrained, "classifier", None)
        if native_head is None:
            native_head = getattr(pretrained, "score", None)
        if native_head is None:
            raise ValueError("pretrained classifier exposes neither classifier nor score")
        model.native_head = native_head
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
        resize_parameters = inspect.signature(self.backbone.resize_token_embeddings).parameters
        resize_kwargs: dict[str, Any] = {"pad_to_multiple_of": pad_to_multiple_of}
        if "mean_resizing" in resize_parameters:
            resize_kwargs["mean_resizing"] = mean_resizing
        embeddings = self.backbone.resize_token_embeddings(new_num_tokens, **resize_kwargs)
        self.config.vocab_size = int(embeddings.num_embeddings)
        self.config.backbone_config["vocab_size"] = int(embeddings.num_embeddings)
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
            output_hidden_states=False,
            output_attentions=False,
            **backbone_kwargs,
        )
        native_logits = self.native_head(outputs.last_hidden_state)
        if native_logits.ndim != 2 or native_logits.shape[-1] != 1:
            raise ValueError(
                "hybrid head requires a pretrained scalar reranker head with "
                "logits shaped (batch_size, 1); use head.type='default' or "
                "head.type='pooling' for a multi-class classifier"
            )
        weights = self.logit_weights.to(dtype=native_logits.dtype)
        scalar = weights[0] * native_logits
        if not self.native_only:
            attention_logits = self.attention_head(
                outputs.last_hidden_state,
                attention_mask,
            )
            scalar = scalar + weights[1] * attention_logits
        # ``[0, score]`` is exactly sigmoid(score) under softmax, preserving
        # the calibration of BGE's original one-logit reranker head.
        logits = torch.cat((torch.zeros_like(scalar), scalar), dim=-1)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
        use_return_dict = self.config.use_return_dict if return_dict is None else return_dict
        if not use_return_dict:
            result = (logits,)
            return (loss, *result) if loss is not None else result
        return SequenceClassifierOutput(loss=loss, logits=logits)


class GatedResidualFusionSequenceClassifierConfig(PoolingSequenceClassifierConfig):
    model_type = "match_gated_residual_fusion_classifier"

    def __init__(self, *, typed_feature_count: int = 0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.match_head_type = "gated_residual_fusion"
        self.typed_feature_count = int(typed_feature_count)


class GatedResidualFusionSequenceClassifier(PreTrainedModel):
    """Scalar sequence reranker plus shared text/typed residual fusion."""

    config_class = GatedResidualFusionSequenceClassifierConfig
    base_model_prefix = "backbone"

    def __init__(self, config: GatedResidualFusionSequenceClassifierConfig) -> None:
        super().__init__(config)
        if config.typed_feature_count < 1:
            raise ValueError("gated residual fusion model requires typed features")
        backbone_config = _restore_backbone_config(config.backbone_config)
        self.backbone = AutoModel.from_config(backbone_config)
        native_model = AutoModelForSequenceClassification.from_config(backbone_config)
        native_head = getattr(native_model, "classifier", None)
        if native_head is None:
            native_head = getattr(native_model, "score", None)
        if native_head is None:
            raise ValueError("pretrained classifier exposes neither classifier nor score")
        self.native_head = native_head
        self.fusion_head = GatedResidualFusionHead(
            config.hidden_size,
            config.typed_feature_count,
            PoolingHeadConfig.from_dict(config.head_config),
        )
        self.post_init()
        self.fusion_head.reset_residual_outputs()

    @classmethod
    def from_backbone_pretrained(
        cls,
        model_path: str,
        *,
        head_config: PoolingHeadConfig,
        typed_feature_count: int,
        id2label: dict[int, str],
        label2id: dict[str, int],
        attention_implementation: str = "auto",
    ) -> GatedResidualFusionSequenceClassifier:
        attention_kwargs = (
            {}
            if attention_implementation == "auto"
            else {"attn_implementation": attention_implementation}
        )
        pretrained = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            **attention_kwargs,
        )
        backbone_config = AutoConfig.from_pretrained(model_path)
        config = GatedResidualFusionSequenceClassifierConfig(
            backbone_config=backbone_config.to_dict(),
            head_config=head_config.to_dict(),
            typed_feature_count=typed_feature_count,
            num_labels=2,
            id2label=id2label,
            label2id=label2id,
        )
        model = cls(config)
        model.backbone = pretrained.base_model
        native_head = getattr(pretrained, "classifier", None)
        if native_head is None:
            native_head = getattr(pretrained, "score", None)
        if native_head is None:
            raise ValueError("pretrained classifier exposes neither classifier nor score")
        model.native_head = native_head
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
        parameters = inspect.signature(
            self.backbone.resize_token_embeddings
        ).parameters
        resize_kwargs: dict[str, Any] = {"pad_to_multiple_of": pad_to_multiple_of}
        if "mean_resizing" in parameters:
            resize_kwargs["mean_resizing"] = mean_resizing
        embeddings = self.backbone.resize_token_embeddings(
            new_num_tokens,
            **resize_kwargs,
        )
        self.config.vocab_size = int(embeddings.num_embeddings)
        self.config.backbone_config["vocab_size"] = int(embeddings.num_embeddings)
        return embeddings

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        typed_features: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> SequenceClassifierOutput | tuple[torch.Tensor, ...]:
        if typed_features is None:
            raise ValueError("gated residual fusion requires typed_features")
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
            output_hidden_states=False,
            output_attentions=False,
            **backbone_kwargs,
        )
        native_logits = self.native_head(outputs.last_hidden_state)
        if native_logits.ndim != 2 or native_logits.shape[-1] != 1:
            raise ValueError(
                "gated residual fusion requires a pretrained scalar reranker head"
            )
        correction = self.fusion_head(
            outputs.last_hidden_state,
            attention_mask,
            typed_features,
        ).to(dtype=native_logits.dtype)
        score = native_logits + correction
        logits = torch.cat((torch.zeros_like(score), score), dim=-1)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
        use_return_dict = (
            self.config.use_return_dict if return_dict is None else return_dict
        )
        if not use_return_dict:
            result = (logits,)
            return (loss, *result) if loss is not None else result
        return SequenceClassifierOutput(loss=loss, logits=logits)


__all__ = [
    "GatedResidualFusionSequenceClassifier",
    "GatedResidualFusionSequenceClassifierConfig",
    "HybridSequenceClassifier",
    "HybridSequenceClassifierConfig",
    "PoolingSequenceClassifier",
    "PoolingSequenceClassifierConfig",
]
