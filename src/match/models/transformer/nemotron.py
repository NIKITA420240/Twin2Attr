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

from .head import (
    GatedResidualFusionHead,
    PoolingHeadConfig,
    TransformerPoolingHead,
)
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
        attention_implementation: str = "auto",
    ) -> NemotronAttentionSequenceClassifier:
        attention_kwargs = (
            {}
            if attention_implementation == "auto"
            else {"attn_implementation": attention_implementation}
        )
        native = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            trust_remote_code=True,
            **attention_kwargs,
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


class NemotronTypedFusionConfig(PretrainedConfig):
    model_type = "match_nemotron_typed_fusion_classifier"

    def __init__(
        self,
        *,
        head_config: dict[str, Any] | None = None,
        hidden_size: int = 0,
        typed_feature_count: int = 0,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("num_labels", 1)
        super().__init__(**kwargs)
        self.head_config = dict(head_config or {})
        self.hidden_size = int(hidden_size)
        self.typed_feature_count = int(typed_feature_count)
        TransformerArtifactContract.for_training(
            profile=PROMPTED_BINARY_RERANKER_PROFILE,
            head_type="typed_attribute_fusion",
            num_logits=1,
        ).apply_to(self)


class _TypedResidualHead(nn.Module):
    """Learn a typed correction whose zero initialization preserves native logits."""

    def __init__(self, feature_count: int, config: PoolingHeadConfig) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        if config.typed_layer_norm:
            layers.append(nn.LayerNorm(feature_count))
        dimensions = [feature_count, *config.typed_hidden_dims]
        for input_width, output_width in zip(dimensions, dimensions[1:]):
            layers.extend(
                [
                    nn.Linear(input_width, output_width),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                ]
            )
        self.encoder = nn.Sequential(*layers)
        self.output = nn.Linear(dimensions[-1], 1)
        self.feature_count = feature_count

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.feature_count:
            raise ValueError(
                "typed_features must have shape "
                f"[batch, {self.feature_count}]"
            )
        if not torch.isfinite(features).all():
            raise ValueError("typed_features must contain only finite values")
        encoded = self.encoder(features.to(dtype=self.output.weight.dtype))
        return self.output(encoded)


class NemotronTypedFusionSequenceClassifier(PreTrainedModel):
    """Preserve Nemotron's native score and learn a typed residual correction."""

    config_class = NemotronTypedFusionConfig
    base_model_prefix = "native_model"

    def __init__(
        self,
        config: NemotronTypedFusionConfig,
        native_model: PreTrainedModel,
    ) -> None:
        super().__init__(config)
        if config.typed_feature_count < 1:
            raise ValueError("Nemotron typed fusion requires typed features")
        self.native_model = native_model
        self.typed_head = _TypedResidualHead(
            config.typed_feature_count,
            PoolingHeadConfig.from_dict(config.head_config),
        )
        self.typed_head.apply(self._init_weights)
        # Exact native behavior at initialization; training can only add a
        # learned correction after seeing evidence in the typed branch.
        nn.init.zeros_(self.typed_head.output.weight)
        nn.init.zeros_(self.typed_head.output.bias)

    @property
    def base_model(self) -> nn.Module:
        return self.native_model.base_model

    @classmethod
    def from_backbone_pretrained(
        cls,
        model_path: str,
        *,
        head_config: PoolingHeadConfig,
        typed_feature_count: int,
        attention_implementation: str = "auto",
    ) -> NemotronTypedFusionSequenceClassifier:
        attention_kwargs = (
            {}
            if attention_implementation == "auto"
            else {"attn_implementation": attention_implementation}
        )
        native = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            trust_remote_code=True,
            **attention_kwargs,
        )
        if int(native.config.num_labels) != 1:
            raise ValueError("Nemotron typed fusion requires a one-logit model")
        native.config.use_cache = False
        config = NemotronTypedFusionConfig(
            head_config=head_config.to_dict(),
            hidden_size=int(native.config.hidden_size),
            typed_feature_count=typed_feature_count,
            num_labels=1,
            id2label={0: "match"},
            label2id={"match": 0},
        )
        return cls(config, native)

    @classmethod
    def from_artifact(
        cls,
        model_directory: str | Path,
    ) -> NemotronTypedFusionSequenceClassifier:
        directory = Path(model_directory)
        config = NemotronTypedFusionConfig.from_pretrained(directory)
        native_config = AutoConfig.from_pretrained(
            directory / NEMOTRON_BACKBONE_CONFIG_DIRECTORY,
            trust_remote_code=True,
        )
        native = AutoModelForSequenceClassification.from_config(
            native_config,
            trust_remote_code=True,
        )
        model = cls(config, native)
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
                "invalid Nemotron typed fusion artifact state: "
                f"missing={missing}, unexpected={unexpected}"
            )
        return model

    def save_pretrained(self, save_directory: str | Path, *args: Any, **kwargs: Any):
        result = super().save_pretrained(save_directory, *args, **kwargs)
        native_config_directory = (
            Path(save_directory) / NEMOTRON_BACKBONE_CONFIG_DIRECTORY
        )
        self.native_model.config.save_pretrained(native_config_directory)
        return result

    def get_input_embeddings(self) -> nn.Module:
        return self.native_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.native_model.set_input_embeddings(value)

    def resize_token_embeddings(
        self,
        new_num_tokens: int | None = None,
        pad_to_multiple_of: int | None = None,
        mean_resizing: bool = True,
    ) -> nn.Embedding:
        parameters = inspect.signature(
            self.native_model.resize_token_embeddings
        ).parameters
        resize_kwargs: dict[str, Any] = {"pad_to_multiple_of": pad_to_multiple_of}
        if "mean_resizing" in parameters:
            resize_kwargs["mean_resizing"] = mean_resizing
        embeddings = self.native_model.resize_token_embeddings(
            new_num_tokens,
            **resize_kwargs,
        )
        self.config.vocab_size = int(embeddings.num_embeddings)
        return embeddings

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        typed_features: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> SequenceClassifierOutput | tuple[torch.Tensor, ...]:
        del labels
        if typed_features is None:
            raise ValueError("Nemotron typed fusion requires typed_features")
        native_output = self.native_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            **kwargs,
        )
        native_logits = native_output.logits
        if native_logits.ndim != 2 or native_logits.shape[1] != 1:
            raise ValueError("Nemotron native model must return [batch, 1] logits")
        correction = self.typed_head(
            typed_features.to(device=native_logits.device)
        ).to(dtype=native_logits.dtype)
        logits = native_logits + correction
        use_return_dict = (
            self.config.use_return_dict if return_dict is None else return_dict
        )
        if not use_return_dict:
            return (logits,)
        return SequenceClassifierOutput(logits=logits)


class NemotronGatedResidualFusionConfig(PretrainedConfig):
    model_type = "match_nemotron_gated_residual_fusion_classifier"

    def __init__(
        self,
        *,
        head_config: dict[str, Any] | None = None,
        hidden_size: int = 0,
        typed_feature_count: int = 0,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("num_labels", 1)
        super().__init__(**kwargs)
        self.head_config = dict(head_config or {})
        self.hidden_size = int(hidden_size)
        self.typed_feature_count = int(typed_feature_count)
        TransformerArtifactContract.for_training(
            profile=PROMPTED_BINARY_RERANKER_PROFILE,
            head_type="gated_residual_fusion",
            num_logits=1,
        ).apply_to(self)


class NemotronGatedResidualFusionSequenceClassifier(PreTrainedModel):
    """Native Nemotron score plus zero-initialized gated residuals."""

    config_class = NemotronGatedResidualFusionConfig
    base_model_prefix = "native_model"

    def __init__(
        self,
        config: NemotronGatedResidualFusionConfig,
        native_model: PreTrainedModel,
    ) -> None:
        super().__init__(config)
        if config.typed_feature_count < 1:
            raise ValueError("Nemotron gated residual fusion requires typed features")
        self.native_model = native_model
        self.fusion_head = GatedResidualFusionHead(
            config.hidden_size,
            config.typed_feature_count,
            PoolingHeadConfig.from_dict(config.head_config),
        )
        self.fusion_head.apply(self._init_weights)
        self.fusion_head.reset_residual_outputs()

    @property
    def base_model(self) -> nn.Module:
        return self.native_model.base_model

    @classmethod
    def from_backbone_pretrained(
        cls,
        model_path: str,
        *,
        head_config: PoolingHeadConfig,
        typed_feature_count: int,
        attention_implementation: str = "auto",
    ) -> NemotronGatedResidualFusionSequenceClassifier:
        attention_kwargs = (
            {}
            if attention_implementation == "auto"
            else {"attn_implementation": attention_implementation}
        )
        native = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            trust_remote_code=True,
            **attention_kwargs,
        )
        if int(native.config.num_labels) != 1:
            raise ValueError("Nemotron gated residual fusion requires one logit")
        native.config.use_cache = False
        config = NemotronGatedResidualFusionConfig(
            head_config=head_config.to_dict(),
            hidden_size=int(native.config.hidden_size),
            typed_feature_count=typed_feature_count,
            num_labels=1,
            id2label={0: "match"},
            label2id={"match": 0},
        )
        return cls(config, native)

    @classmethod
    def from_artifact(
        cls,
        model_directory: str | Path,
    ) -> NemotronGatedResidualFusionSequenceClassifier:
        directory = Path(model_directory)
        config = NemotronGatedResidualFusionConfig.from_pretrained(directory)
        native_config = AutoConfig.from_pretrained(
            directory / NEMOTRON_BACKBONE_CONFIG_DIRECTORY,
            trust_remote_code=True,
        )
        native = AutoModelForSequenceClassification.from_config(
            native_config,
            trust_remote_code=True,
        )
        model = cls(config, native)
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
                "invalid Nemotron gated residual fusion artifact state: "
                f"missing={missing}, unexpected={unexpected}"
            )
        return model

    def save_pretrained(self, save_directory: str | Path, *args: Any, **kwargs: Any):
        result = super().save_pretrained(save_directory, *args, **kwargs)
        native_config_directory = (
            Path(save_directory) / NEMOTRON_BACKBONE_CONFIG_DIRECTORY
        )
        self.native_model.config.save_pretrained(native_config_directory)
        return result

    def get_input_embeddings(self) -> nn.Module:
        return self.native_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.native_model.set_input_embeddings(value)

    def resize_token_embeddings(
        self,
        new_num_tokens: int | None = None,
        pad_to_multiple_of: int | None = None,
        mean_resizing: bool = True,
    ) -> nn.Embedding:
        parameters = inspect.signature(
            self.native_model.resize_token_embeddings
        ).parameters
        resize_kwargs: dict[str, Any] = {"pad_to_multiple_of": pad_to_multiple_of}
        if "mean_resizing" in parameters:
            resize_kwargs["mean_resizing"] = mean_resizing
        embeddings = self.native_model.resize_token_embeddings(
            new_num_tokens,
            **resize_kwargs,
        )
        self.config.vocab_size = int(embeddings.num_embeddings)
        return embeddings

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        typed_features: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_dict: bool | None = None,
        **kwargs: Any,
    ) -> SequenceClassifierOutput | tuple[torch.Tensor, ...]:
        del labels
        if typed_features is None:
            raise ValueError("Nemotron gated residual fusion requires typed_features")
        if attention_mask is None:
            if input_ids is None:
                raise ValueError("attention_mask is required when input_ids is absent")
            attention_mask = torch.ones_like(input_ids)
        native_kwargs = dict(kwargs)
        native_kwargs.pop("output_hidden_states", None)
        native_kwargs.pop("output_attentions", None)
        native_output = self.native_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
            output_attentions=False,
            **native_kwargs,
        )
        native_logits = native_output.logits
        if native_logits.ndim != 2 or native_logits.shape[1] != 1:
            raise ValueError("Nemotron native model must return [batch, 1] logits")
        hidden_states = getattr(native_output, "hidden_states", None)
        if hidden_states:
            final_hidden_state = hidden_states[-1]
        else:
            backbone_output = self.native_model.base_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                **native_kwargs,
            )
            final_hidden_state = backbone_output.last_hidden_state
        correction = self.fusion_head(
            final_hidden_state,
            attention_mask,
            typed_features,
        ).to(dtype=native_logits.dtype)
        logits = native_logits + correction
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
    "NemotronGatedResidualFusionConfig",
    "NemotronGatedResidualFusionSequenceClassifier",
    "NemotronTypedFusionConfig",
    "NemotronTypedFusionSequenceClassifier",
]
