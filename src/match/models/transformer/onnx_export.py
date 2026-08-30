"""Export trained Transformer inference graphs to ONNX."""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from .fp8 import Fp8QuantizationResult, quantize_module_weights_to_fp8
from .int8 import Int8QuantizationResult, quantize_module_weights_to_int8
from .loading import load_trained_classifier
from .profile import (
    TransformerRuntimeContract,
    is_mxbai_reranker_profile,
    is_qwen3_reranker_profile,
)
from .qwen3 import Qwen3YesNoReranker, qwen3_score_token_ids

ONNX_DIRECTORY_NAME = "onnx"
CLASSIFIER_ONNX_NAME = "classifier.onnx"
ENCODER_ONNX_NAME = "encoder.onnx"
ONNX_METADATA_NAME = "metadata.json"
ONNX_EXTERNAL_DATA_SUFFIX = ".data"


@dataclass(frozen=True, slots=True)
class OnnxExportResult:
    directory: Path
    classifier_path: Path | None
    encoder_path: Path | None
    precision: str
    opset: int


class _ClassifierGraph(nn.Module):
    def __init__(self, model: nn.Module, *, token_type_ids: bool) -> None:
        super().__init__()
        self.model = model
        self.token_type_ids = token_type_ids

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_dict": True,
        }
        if self.token_type_ids:
            inputs["token_type_ids"] = token_type_ids
        return self.model(**inputs).logits


class _EncoderGraph(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        *,
        token_type_ids: bool,
        mean_pooling: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = model.base_model
        self.token_type_ids = token_type_ids
        self.mean_pooling = mean_pooling

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "return_dict": True,
        }
        if self.token_type_ids:
            inputs["token_type_ids"] = token_type_ids
        hidden_state = self.backbone(**inputs).last_hidden_state
        if self.mean_pooling:
            mask = attention_mask.unsqueeze(-1).to(dtype=hidden_state.dtype)
            return (hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return hidden_state[:, 0, :]


def _require_onnx() -> Any:
    try:
        import onnx
    except ImportError as error:
        raise RuntimeError(
            "ONNX export requires the optional dependencies; install the "
            "project with the 'onnx-cpu' or 'onnx-gpu' extra"
        ) from error
    return onnx


def _convert_to_float16(source: Path, destination: Path) -> None:
    onnx = _require_onnx()
    try:
        from onnxconverter_common import float16
    except ImportError as error:
        raise RuntimeError(
            "float16 ONNX export requires onnxconverter-common"
        ) from error
    # In-memory shape inference calls ModelProto.SerializeToString(), which
    # fails for FP32 graphs above protobuf's 2 GiB limit. Do shape inference
    # through paths, then disable the converter's second inference pass.
    # onnxconverter-common's path helper keeps a NamedTemporaryFile open and
    # fails on Windows, so explicitly close our adjacent temporary file first.
    temporary = tempfile.NamedTemporaryFile(
        dir=source.parent,
        prefix=f".{source.stem}.shape-inferred-",
        suffix=source.suffix,
        delete=False,
    )
    inferred_path = Path(temporary.name)
    temporary.close()
    inferred_path.unlink(missing_ok=True)
    try:
        onnx.shape_inference.infer_shapes_path(
            str(source),
            str(inferred_path),
        )
        graph = onnx.load(str(inferred_path))
        converted = float16.convert_float_to_float16(
            graph,
            keep_io_types=True,
            disable_shape_infer=True,
        )
    finally:
        inferred_path.unlink(missing_ok=True)
    # onnxconverter-common does not currently update Cast(to=FLOAT) nodes
    # whose inferred output annotation it changed to FLOAT16. Such a graph
    # passes onnx.checker but ONNX Runtime rejects it as internally
    # inconsistent. Keep the Cast attribute aligned with the converted type.
    value_types = {
        value.name: value.type.tensor_type.elem_type
        for value in (
            *converted.graph.value_info,
            *converted.graph.output,
        )
        if value.type.HasField("tensor_type")
    }
    for node in converted.graph.node:
        if node.op_type != "Cast" or not node.output:
            continue
        if value_types.get(node.output[0]) != onnx.TensorProto.FLOAT16:
            continue
        for attribute in node.attribute:
            if attribute.name == "to" and attribute.i == onnx.TensorProto.FLOAT:
                attribute.i = onnx.TensorProto.FLOAT16
    # A 1B-parameter FP16 model is still larger than protobuf's 2 GiB message
    # limit. Keep the graph in ``destination`` and write tensor payloads to one
    # adjacent file, which ONNX Runtime resolves automatically. Remove a stale
    # payload first because ONNX appends when the target file already exists.
    external_data_path = destination.with_name(
        f"{destination.name}{ONNX_EXTERNAL_DATA_SUFFIX}"
    )
    destination.unlink(missing_ok=True)
    external_data_path.unlink(missing_ok=True)
    try:
        onnx.save_model(
            converted,
            str(destination),
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location=external_data_path.name,
            size_threshold=1024,
        )
    except Exception:
        destination.unlink(missing_ok=True)
        external_data_path.unlink(missing_ok=True)
        raise


def _copy_onnx_bundle(source: Path, destination: Path) -> None:
    """Copy a graph and every relative external-data payload it references."""
    onnx = _require_onnx()
    graph = onnx.load(str(source), load_external_data=False)
    locations = {
        item.value
        for initializer in graph.graph.initializer
        for item in initializer.external_data
        if item.key == "location"
    }
    destination.unlink(missing_ok=True)
    for location in locations:
        relative = Path(location)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"unsafe ONNX external-data location: {location!r}")
        source_payload = source.parent / relative
        destination_payload = destination.parent / relative
        if not source_payload.is_file():
            raise FileNotFoundError(
                f"ONNX external-data payload is missing: {source_payload}"
            )
        destination_payload.parent.mkdir(parents=True, exist_ok=True)
        destination_payload.unlink(missing_ok=True)
        shutil.copy2(source_payload, destination_payload)
    shutil.copy2(source, destination)


def _export_graph(
    graph: nn.Module,
    destination: Path,
    *,
    max_length: int,
    use_token_type_ids: bool,
    opset: int,
    precision: str,
    dynamic_batch: bool,
    dynamic_sequence_length: bool,
    output_name: str,
    convert_float16: bool = True,
    sample_batch_size: int = 2,
) -> None:
    input_ids = torch.ones((sample_batch_size, max_length), dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    args: tuple[torch.Tensor, ...]
    input_names = ["input_ids", "attention_mask"]
    if use_token_type_ids:
        args = (input_ids, attention_mask, torch.zeros_like(input_ids))
        input_names.append("token_type_ids")
    else:
        args = (input_ids, attention_mask)

    dynamic_axes: dict[str, dict[int, str]] = {}
    for name in (*input_names, output_name):
        axes: dict[int, str] = {}
        if dynamic_batch:
            axes[0] = "batch_size"
        if dynamic_sequence_length and name in input_names:
            axes[1] = "sequence_length"
        if axes:
            dynamic_axes[name] = axes

    with tempfile.TemporaryDirectory(dir=destination.parent) as directory:
        fp32_path = Path(directory) / destination.name
        export_kwargs = {
            "input_names": input_names,
            "output_names": [output_name],
            "dynamic_axes": dynamic_axes or None,
            "opset_version": opset,
            "do_constant_folding": True,
            "external_data": True,
        }
        # PyTorch 2.9 changed torch.onnx.export to default to the dynamo
        # exporter, which adds an onnxscript dependency. The mature legacy
        # exporter is sufficient for these fixed Hugging Face forward graphs
        # and also keeps compatibility with the project's torch>=2.2 floor.
        if "dynamo" in inspect.signature(torch.onnx.export).parameters:
            export_kwargs["dynamo"] = False
        torch.onnx.export(
            graph.eval(),
            args,
            str(fp32_path),
            **export_kwargs,
        )
        if precision == "float16" and convert_float16:
            _convert_to_float16(fp32_path, destination)
        else:
            _copy_onnx_bundle(fp32_path, destination)


def _load_causal_reranker_export_model(
    model_dir: Path,
    *,
    precision: str,
) -> tuple[
    Any,
    nn.Module,
    Fp8QuantizationResult | Int8QuantizationResult | None,
]:
    artifact_config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    source_value = getattr(artifact_config, "match_source_model_path", None)
    if not source_value:
        raise ValueError("causal reranker artifact does not record match_source_model_path")
    source = Path(str(source_value)).expanduser()
    if not source.is_absolute():
        source = source.resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"causal reranker source model directory is missing: {source}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        local_files_only=True,
        fix_mistral_regex=True,
    )
    tokenizer.padding_side = "left"
    if is_mxbai_reranker_profile(
        TransformerRuntimeContract.from_config(artifact_config).profile
    ):
        no_token_id = int(getattr(artifact_config, "match_no_token_id"))
        yes_token_id = int(getattr(artifact_config, "match_yes_token_id"))
    else:
        no_token_id, yes_token_id = qwen3_score_token_ids(tokenizer)
    model_dtype = torch.float32 if precision == "float32" else torch.float16
    causal_lm = AutoModelForCausalLM.from_pretrained(
        source,
        dtype=model_dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="eager",
    )
    model = Qwen3YesNoReranker(
        causal_lm,
        no_token_id=no_token_id,
        yes_token_id=yes_token_id,
        prepare_4d_attention_mask=True,
    )
    del causal_lm
    gc.collect()
    model.config = artifact_config
    quantization = None
    if precision == "float8":
        quantization = quantize_module_weights_to_fp8(model.backbone)
        gc.collect()
    elif precision == "int8":
        quantization = quantize_module_weights_to_int8(model.backbone)
        gc.collect()
    return tokenizer, model, quantization


def export_transformer_to_onnx(
    model_directory: str | Path,
    *,
    opset: int = 18,
    precision: str = "float16",
    dynamic_batch: bool = True,
    dynamic_sequence_length: bool = True,
    export_classifier: bool = True,
    export_encoder: bool = True,
) -> OnnxExportResult:
    """Export classifier logits and/or pooled embeddings next to a trained model."""
    if precision not in {"float32", "float16", "float8", "int8"}:
        raise ValueError("ONNX precision must be float32, float16, float8, or int8")
    if opset < 14:
        raise ValueError("ONNX opset must be at least 14")
    if precision == "float8" and opset < 19:
        raise ValueError("float8 ONNX export requires opset 19 or newer")
    if not (export_classifier or export_encoder):
        raise ValueError("at least one ONNX graph must be selected")
    _require_onnx()

    model_dir = Path(model_directory)
    config_values = json.loads(
        (model_dir / "config.json").read_text(encoding="utf-8")
    )
    configured_contract = TransformerRuntimeContract.from_config(config_values)
    causal_reranker = is_qwen3_reranker_profile(
        configured_contract.profile
    ) or is_mxbai_reranker_profile(configured_contract.profile)
    quantization: Fp8QuantizationResult | Int8QuantizationResult | None = None
    if causal_reranker:
        tokenizer, model, quantization = _load_causal_reranker_export_model(
            model_dir,
            precision=precision,
        )
    else:
        if precision in {"float8", "int8"}:
            raise ValueError(
                f"{precision} export is currently supported only for causal rerankers"
            )
        tokenizer, model = load_trained_classifier(
            model_dir,
            device="cpu",
            dtype="float32",
        )
    model.eval()
    output_dir = model_dir / ONNX_DIRECTORY_NAME
    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_contract = TransformerRuntimeContract.from_config(model.config)
    max_length = runtime_contract.max_length or 256
    contract = runtime_contract.output
    prompted = contract.uses_prompted_pairs
    # Prompted rerankers always receive one sequence and do not expose BERT
    # segment ids even when a test/surrogate tokenizer happens to advertise them.
    use_token_type_ids = (
        not prompted and "token_type_ids" in tokenizer.model_input_names
    )

    common = {
        "max_length": max_length,
        "use_token_type_ids": use_token_type_ids,
        "opset": opset,
        "precision": precision,
        "dynamic_batch": dynamic_batch,
        "dynamic_sequence_length": dynamic_sequence_length,
        "convert_float16": not causal_reranker,
        "sample_batch_size": 1 if causal_reranker else 2,
    }
    classifier_path = output_dir / CLASSIFIER_ONNX_NAME if export_classifier else None
    encoder_path = output_dir / ENCODER_ONNX_NAME if export_encoder else None
    if classifier_path is not None:
        _export_graph(
            _ClassifierGraph(model, token_type_ids=use_token_type_ids),
            classifier_path,
            output_name="logits",
            **common,
        )
    if encoder_path is not None:
        _export_graph(
            _EncoderGraph(
                model,
                token_type_ids=use_token_type_ids,
                mean_pooling=prompted,
            ),
            encoder_path,
            output_name="cls_embedding",
            **common,
        )

    result = OnnxExportResult(
        directory=output_dir,
        classifier_path=classifier_path,
        encoder_path=encoder_path,
        precision=precision,
        opset=opset,
    )
    metadata = {
        **asdict(result),
        "directory": ".",
        "classifier_path": (None if classifier_path is None else classifier_path.name),
        "encoder_path": None if encoder_path is None else encoder_path.name,
        "max_length": max_length,
        "dynamic_batch": dynamic_batch,
        "dynamic_sequence_length": dynamic_sequence_length,
        "input_names": [
            "input_ids",
            "attention_mask",
            *(["token_type_ids"] if use_token_type_ids else []),
        ],
        "profile": contract.profile,
        "head_type": contract.head_type,
        "num_logits": contract.num_logits,
        "probability_transform": contract.probability_transform,
        "encoder_pooling": contract.encoder_pooling,
        "fp8_quantization": (
            asdict(quantization)
            if isinstance(quantization, Fp8QuantizationResult)
            else None
        ),
        "int8_quantization": (
            asdict(quantization)
            if isinstance(quantization, Int8QuantizationResult)
            else None
        ),
    }
    (output_dir / ONNX_METADATA_NAME).write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    logger.info("Exported ONNX Transformer graphs to {!s}", output_dir)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_directory", type=Path)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument(
        "--precision",
        choices=("float32", "float16", "float8", "int8"),
        default="float16",
    )
    parser.add_argument("--classifier-only", action="store_true")
    args = parser.parse_args()
    export_transformer_to_onnx(
        args.model_directory,
        opset=args.opset,
        precision=args.precision,
        export_encoder=not args.classifier_only,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "ONNX_EXTERNAL_DATA_SUFFIX",
    "OnnxExportResult",
    "export_transformer_to_onnx",
]
