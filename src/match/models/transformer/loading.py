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

from .head import PoolingSequenceClassifier, PoolingSequenceClassifierConfig
from .int8_artifact import (
    is_qwen3_pytorch_int8_config,
    load_qwen3_pytorch_int8_model,
)
from .nemotron import NemotronAttentionConfig, NemotronAttentionSequenceClassifier
from .profile import TransformerArtifactContract, is_qwen3_reranker_profile
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
    tokenizer_kwargs = {
        "use_fast": True,
        "trust_remote_code": contract.requires_trust_remote_code,
    }
    if is_qwen3_reranker_profile(contract.profile):
        tokenizer_kwargs["fix_mistral_regex"] = True
    tokenizer = AutoTokenizer.from_pretrained(directory, **tokenizer_kwargs)
    model_type = config_values.get("model_type")
    if is_qwen3_pytorch_int8_config(config_values):
        model = load_qwen3_pytorch_int8_model(
            directory,
            device=target_device,
            dtype=target_dtype,
        )
    elif model_type == NemotronAttentionConfig.model_type:
        model = NemotronAttentionSequenceClassifier.from_artifact(directory)
    elif model_type == PoolingSequenceClassifierConfig.model_type:
        model = PoolingSequenceClassifier.from_pretrained(directory)
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            directory,
            trust_remote_code=contract.requires_trust_remote_code,
        )
    restored_contract = TransformerArtifactContract.from_config(model.config)
    if restored_contract != contract:
        raise RuntimeError(
            "restored Transformer model does not match its persisted artifact contract"
        )
    if not is_qwen3_pytorch_int8_config(config_values):
        model.to(device=target_device, dtype=target_dtype).eval()
    return tokenizer, model


__all__ = ["load_trained_classifier", "resolve_device"]
