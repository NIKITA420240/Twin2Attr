"""Loading boundary for persisted Transformer classifier artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from .head import (
    HybridSequenceClassifier,
    HybridSequenceClassifierConfig,
    PoolingSequenceClassifier,
    PoolingSequenceClassifierConfig,
)
from .nemotron import (
    NemotronAttentionConfig,
    NemotronAttentionSequenceClassifier,
    NemotronTypedFusionConfig,
    NemotronTypedFusionSequenceClassifier,
)
from .profile import TransformerArtifactContract
from .precision import torch_inference_dtype


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_trained_classifier(
    model_dir: str | Path,
    *,
    device: str | torch.device | None = None,
    dtype: str = "float32",
    attention_implementation: str = "auto",
) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    """Restore native, project-pooling, or Nemotron-attention artifacts."""
    target_device = resolve_device(device)
    target_dtype = torch_inference_dtype(dtype)
    if (
        target_device.type == "cuda"
        and target_dtype == torch.bfloat16
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("the selected CUDA device does not support bfloat16")
    if target_device.type == "mps" and target_dtype == torch.bfloat16:
        raise RuntimeError("bfloat16 inference is not supported on MPS")

    directory = Path(model_dir)
    config_values = json.loads(
        (directory / "config.json").read_text(encoding="utf-8")
    )
    contract = TransformerArtifactContract.from_config(config_values)
    tokenizer = AutoTokenizer.from_pretrained(
        directory,
        use_fast=True,
        trust_remote_code=contract.uses_prompted_pairs,
    )
    model_type = config_values.get("model_type")
    attention_kwargs = (
        {}
        if attention_implementation == "auto"
        else {"attn_implementation": attention_implementation}
    )
    if model_type == NemotronAttentionConfig.model_type:
        model = NemotronAttentionSequenceClassifier.from_artifact(directory)
    elif model_type == NemotronTypedFusionConfig.model_type:
        model = NemotronTypedFusionSequenceClassifier.from_artifact(directory)
    elif model_type == HybridSequenceClassifierConfig.model_type:
        model = HybridSequenceClassifier.from_pretrained(directory)
    elif model_type == PoolingSequenceClassifierConfig.model_type:
        model = PoolingSequenceClassifier.from_pretrained(directory)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            directory,
            trust_remote_code=contract.uses_prompted_pairs,
            **attention_kwargs,
        )
    if attention_implementation != "auto" and model_type in {
        NemotronAttentionConfig.model_type,
        HybridSequenceClassifierConfig.model_type,
        PoolingSequenceClassifierConfig.model_type,
    }:
        model.set_attn_implementation(attention_implementation)
    restored_contract = TransformerArtifactContract.from_config(model.config)
    if restored_contract != contract:
        raise RuntimeError(
            "restored Transformer model does not match its persisted artifact contract"
        )
    model.to(device=target_device, dtype=target_dtype).eval()
    return tokenizer, model


__all__ = ["load_trained_classifier", "resolve_device"]
