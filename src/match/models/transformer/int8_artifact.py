"""Standalone PyTorch CUDA artifact for weight-only INT8 Qwen3 inference."""

from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .int8 import (
    Int8QuantizationResult,
    quantize_module_weights_to_int8,
    replace_module_weights_with_int8_shells,
)
from .profile import TransformerRuntimeContract, is_qwen3_reranker_profile
from .qwen3 import Qwen3YesNoReranker, qwen3_score_token_ids


PYTORCH_INT8_FORMAT = "qwen3_weight_only_int8_per_row_v1"
PYTORCH_INT8_WEIGHTS_NAME = "model.safetensors"
PYTORCH_INT8_METADATA_NAME = "pytorch_int8_metadata.json"


@dataclass(frozen=True, slots=True)
class PytorchInt8ArtifactResult:
    directory: Path
    weights_path: Path
    quantization: Int8QuantizationResult


def _source_directory(config: Any) -> Path:
    value = getattr(config, "match_source_model_path", None)
    if not value:
        raise ValueError("Qwen3 artifact does not record match_source_model_path")
    source = Path(str(value)).expanduser()
    if not source.is_absolute():
        source = source.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Qwen3 source model directory is missing: {source}")
    return source


def export_qwen3_pytorch_int8_artifact(
    model_directory: str | Path,
) -> PytorchInt8ArtifactResult:
    """Quantize Qwen3 weights and persist a source-independent checkpoint."""
    directory = Path(model_directory)
    artifact_config = AutoConfig.from_pretrained(directory, local_files_only=True)
    contract = TransformerRuntimeContract.from_config(artifact_config)
    if not is_qwen3_reranker_profile(contract.profile):
        raise ValueError("PyTorch INT8 export currently supports Qwen3 reranker only")
    tokenizer = AutoTokenizer.from_pretrained(
        directory,
        local_files_only=True,
        fix_mistral_regex=True,
    )
    tokenizer.padding_side = "left"
    no_token_id, yes_token_id = qwen3_score_token_ids(tokenizer)
    causal_lm = AutoModelForCausalLM.from_pretrained(
        _source_directory(artifact_config),
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="eager",
    )
    model = Qwen3YesNoReranker(
        causal_lm,
        no_token_id=no_token_id,
        yes_token_id=yes_token_id,
    )
    model.config = artifact_config
    del causal_lm
    quantization = quantize_module_weights_to_int8(model.backbone)
    gc.collect()

    weights_path = directory / PYTORCH_INT8_WEIGHTS_NAME
    weights_path.unlink(missing_ok=True)
    state = {
        name: tensor.detach().cpu().contiguous()
        for name, tensor in model.state_dict().items()
    }
    save_file(
        state,
        str(weights_path),
        metadata={"format": PYTORCH_INT8_FORMAT},
    )
    del state, model
    gc.collect()

    config_path = directory / "config.json"
    config_values = json.loads(config_path.read_text(encoding="utf-8"))
    config_values["match_initialized_only"] = False
    config_values["match_pytorch_quantization"] = {
        "format": PYTORCH_INT8_FORMAT,
        "weights_file": PYTORCH_INT8_WEIGHTS_NAME,
        **asdict(quantization),
    }
    config_path.write_text(
        json.dumps(config_values, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "format": PYTORCH_INT8_FORMAT,
        "weights_file": PYTORCH_INT8_WEIGHTS_NAME,
        "compute_dtype": "configured at runtime",
        "quantization": asdict(quantization),
    }
    (directory / PYTORCH_INT8_METADATA_NAME).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return PytorchInt8ArtifactResult(directory, weights_path, quantization)


def is_qwen3_pytorch_int8_config(config_values: dict[str, Any]) -> bool:
    value = config_values.get("match_pytorch_quantization")
    return (
        isinstance(value, dict)
        and value.get("format") == PYTORCH_INT8_FORMAT
        and value.get("weights_file") == PYTORCH_INT8_WEIGHTS_NAME
    )


def load_qwen3_pytorch_int8_model(
    directory: str | Path,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Qwen3YesNoReranker:
    """Restore a quantized Qwen3 model without allocating FP16 source weights."""
    # Keep this PyTorch-only dependency out of module initialization. Native
    # TensorRT imports the artifact helpers through the shared loading module,
    # but never needs an empty-weights context. Newer Transformers releases no
    # longer expose this helper from transformers.modeling_utils.
    from accelerate import init_empty_weights

    model_dir = Path(directory)
    config_values = json.loads(
        (model_dir / "config.json").read_text(encoding="utf-8")
    )
    if not is_qwen3_pytorch_int8_config(config_values):
        raise ValueError("artifact is not a supported PyTorch INT8 Qwen3 checkpoint")
    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    no_token_id = config_values.get("match_no_token_id")
    yes_token_id = config_values.get("match_yes_token_id")
    if (
        isinstance(no_token_id, bool)
        or not isinstance(no_token_id, int)
        or isinstance(yes_token_id, bool)
        or not isinstance(yes_token_id, int)
        or no_token_id == yes_token_id
    ):
        raise ValueError("PyTorch INT8 Qwen3 artifact has invalid score token IDs")

    with init_empty_weights(include_buffers=False):
        causal_lm = AutoModelForCausalLM.from_config(
            config,
            attn_implementation="eager",
            dtype=dtype,
        )
        model = Qwen3YesNoReranker(
            causal_lm,
            no_token_id=no_token_id,
            yes_token_id=yes_token_id,
        )
    model.config = config
    del causal_lm
    expected = replace_module_weights_with_int8_shells(model.backbone)
    recorded = config_values["match_pytorch_quantization"]
    for field in (
        "linear_layers",
        "embedding_layers",
        "int8_weight_bytes",
        "scale_bytes",
    ):
        if recorded.get(field) != getattr(expected, field):
            raise ValueError(
                f"PyTorch INT8 artifact {field} does not match its architecture"
            )

    weights_path = model_dir / PYTORCH_INT8_WEIGHTS_NAME
    if not weights_path.is_file():
        raise FileNotFoundError(f"PyTorch INT8 weights are missing: {weights_path}")
    state = load_file(str(weights_path), device=str(device))
    model.load_state_dict(state, strict=True, assign=True)
    del state
    gc.collect()
    model.to(device=device, dtype=dtype).eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_directory", type=Path)
    args = parser.parse_args()
    result = export_qwen3_pytorch_int8_artifact(args.model_directory)
    print(
        f"PyTorch INT8 artifact saved to {result.directory}; "
        f"weights={result.weights_path.stat().st_size} bytes; "
        f"linear_layers={result.quantization.linear_layers}; "
        f"embedding_layers={result.quantization.embedding_layers}"
    )


if __name__ == "__main__":
    main()


__all__ = [
    "PYTORCH_INT8_FORMAT",
    "PYTORCH_INT8_METADATA_NAME",
    "PYTORCH_INT8_WEIGHTS_NAME",
    "PytorchInt8ArtifactResult",
    "export_qwen3_pytorch_int8_artifact",
    "is_qwen3_pytorch_int8_config",
    "load_qwen3_pytorch_int8_model",
]
