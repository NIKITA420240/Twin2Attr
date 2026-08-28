"""Nemotron-specific one-logit classifier with a trainable pooling head."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from .head import PoolingHeadConfig, TransformerPoolingHead
from .profile import (
    PROMPTED_BINARY_RERANKER_PROFILE,
    TransformerArtifactContract,
)


NEMOTRON_BACKBONE_CONFIG_DIRECTORY = "nemotron_backbone_config"


class NemotronAttentionConfig(PretrainedConfig):
    model_type = "match_nemotron_attention_classifier"

    def __init__(
        self,
        *,
        head_config: dict[str, Any] | None = None,
        hidden_size: int = 0,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("num_labels", 1)
        super().__init__(**kwargs)
        self.head_config = dict(head_config or {})
        self.hidden_size = int(hidden_size)
        TransformerArtifactContract.for_training(
            profile=PROMPTED_BINARY_RERANKER_PROFILE,
            head_type="attention_pooling",
            num_logits=1,
        ).apply_to(self)


class NemotronAttentionSequenceClassifier(PreTrainedModel):
    """Apply the project pooling head to NVIDIA's bidirectional Llama backbone."""

    config_class = NemotronAttentionConfig
    base_model_prefix = "backbone"

    def __init__(self, config: NemotronAttentionConfig, backbone: nn.Module) -> None:
        super().__init__(config)
        self.backbone = backbone
        self.head = TransformerPoolingHead(
            config.hidden_size,
            1,
            PoolingHeadConfig.from_dict(config.head_config),
        )
        # Only the new head should receive random initialization. The supplied
        # bidirectional backbone already contains pretrained weights.
        self.head.apply(self._init_weights)

    @classmethod
    def from_backbone_pretrained(
        cls,
        model_path: str,
        *,
        head_config: PoolingHeadConfig,
    ) -> NemotronAttentionSequenceClassifier:
        native = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
        if int(native.config.num_labels) != 1 or not hasattr(native, "model"):
            raise ValueError(
                "Nemotron attention pooling requires a one-logit model with a "
                "bidirectional .model backbone"
            )
        native.config.use_cache = False
        config = NemotronAttentionConfig(
            head_config=head_config.to_dict(),
            hidden_size=int(native.config.hidden_size),
            num_labels=1,
            id2label={0: "match"},
            label2id={"match": 0},
        )
        return cls(config, native.model)

    @classmethod
    def from_artifact(
        cls,
        model_directory: str | Path,
    ) -> NemotronAttentionSequenceClassifier:
        directory = Path(model_directory)
        config = NemotronAttentionConfig.from_pretrained(directory)
        backbone_config = AutoConfig.from_pretrained(
            directory / NEMOTRON_BACKBONE_CONFIG_DIRECTORY,
            trust_remote_code=True,
        )
        native = AutoModelForSequenceClassification.from_config(
            backbone_config,
            trust_remote_code=True,
        )
        model = cls(config, native.model)
        safetensors_path = directory / "model.safetensors"
        if safetensors_path.is_file():
            from safetensors.torch import load_file

            state_dict = load_file(str(safetensors_path))
        else:
            state_dict = torch.load(
                directory / "pytorch_model.bin",
                map_location="cpu",
                weights_only=True,
            )
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                "invalid Nemotron attention artifact state: "
                f"missing={missing}, unexpected={unexpected}"
            )
        return model

    def save_pretrained(self, save_directory: str | Path, *args: Any, **kwargs: Any):
        result = super().save_pretrained(save_directory, *args, **kwargs)
        backbone_config_directory = (
            Path(save_directory) / NEMOTRON_BACKBONE_CONFIG_DIRECTORY
        )
        self.backbone.config.save_pretrained(backbone_config_directory)
        return result

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
        return embeddings

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> SequenceClassifierOutput | tuple[torch.Tensor, ...]:
        del labels
        if attention_mask is None:
            if input_ids is None:
                raise ValueError("attention_mask is required when input_ids is absent")
            attention_mask = torch.ones_like(input_ids)
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **kwargs,
        )
        logits = self.head(outputs.last_hidden_state, attention_mask)
        use_return_dict = (
            self.config.use_return_dict if return_dict is None else return_dict
        )
        if not use_return_dict:
            return (logits,)
        return SequenceClassifierOutput(logits=logits)


__all__ = [
    "NEMOTRON_BACKBONE_CONFIG_DIRECTORY",
    "NemotronAttentionConfig",
    "NemotronAttentionSequenceClassifier",
]
