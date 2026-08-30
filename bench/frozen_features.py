"""Compact feature extraction from an immutable pair Transformer."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer


def sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_frozen_backbone(
    config_dir: Path,
    checkpoint_file: Path,
    *,
    device: torch.device,
) -> tuple[torch.nn.Module, Any, int]:
    """Load only ``backbone.*`` tensors and disable every gradient."""

    outer = json.loads((config_dir / "config.json").read_text(encoding="utf-8"))
    values = dict(outer.get("backbone_config") or outer)
    model_type = str(values.pop("model_type"))
    config = AutoConfig.for_model(model_type, **values)
    backbone = AutoModel.from_config(config)
    state: dict[str, torch.Tensor] = {}
    with safe_open(checkpoint_file, framework="pt", device="cpu") as tensors:
        for key in tensors.keys():
            if key.startswith("backbone."):
                state[key.removeprefix("backbone.")] = tensors.get_tensor(key)
    missing, unexpected = backbone.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"backbone state mismatch: missing={missing}, unexpected={unexpected}"
        )
    backbone.requires_grad_(False)
    backbone.eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(
        config_dir,
        local_files_only=True,
        use_fast=True,
    )
    hidden_size = int(outer.get("hidden_size") or getattr(config, "hidden_size"))
    return backbone, tokenizer, hidden_size


def pair_segment_masks(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    sep_token_id: int,
    cls_token_id: int | None,
    pad_token_id: int | None,
    token_type_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return content masks for BERT and RoBERTa/XLM-R pair layouts."""

    valid = attention_mask.bool()
    special = input_ids.eq(sep_token_id)
    if cls_token_id is not None:
        special |= input_ids.eq(cls_token_id)
    if pad_token_id is not None:
        special |= input_ids.eq(pad_token_id)
    content = valid & ~special
    if token_type_ids is not None and bool(token_type_ids.eq(1).any()):
        return content & token_type_ids.eq(0), content & token_type_ids.eq(1)

    left = torch.zeros_like(content)
    right = torch.zeros_like(content)
    for row in range(input_ids.shape[0]):
        separators = torch.nonzero(
            input_ids[row].eq(sep_token_id) & valid[row],
            as_tuple=False,
        ).flatten()
        if separators.numel() == 2:
            left[row, 1 : int(separators[0])] = True
            right[row, int(separators[0]) + 1 : int(separators[1])] = True
        elif separators.numel() == 3:
            left[row, 1 : int(separators[0])] = True
            right[row, int(separators[1]) + 1 : int(separators[2])] = True
        else:
            raise ValueError(
                "unsupported pair layout: expected two or three separator tokens"
            )
    left &= content
    right &= content
    if not bool(left.any(dim=1).all()) or not bool(right.any(dim=1).all()):
        raise ValueError("every encoded pair must contain both cards")
    return left, right


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(values.dtype)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def _masked_max(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return values.masked_fill(
        ~mask.unsqueeze(-1),
        torch.finfo(values.dtype).min,
    ).amax(dim=1)


def extract_frozen_features(
    pairs: Sequence[Any],
    *,
    backbone: torch.nn.Module,
    tokenizer: Any,
    collator: Any,
    hidden_size: int,
    output_path: Path,
    batch_size: int,
    device: torch.device,
    description: str,
) -> Path:
    """Write ``[CLS, L-mean, L-max, R-mean, R-max]`` as float16 NPY."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(pairs), 5, hidden_size),
    )
    loader = DataLoader(
        pairs,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
        pin_memory=device.type == "cuda",
    )
    offset = 0
    with torch.inference_mode():
        for batch in tqdm(loader, desc=description, unit="batch"):
            batch.pop("labels", None)
            batch.pop("sample_weights", None)
            model_inputs = {
                key: value.to(device, non_blocking=True)
                for key, value in batch.items()
            }
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                hidden = backbone(**model_inputs, return_dict=True).last_hidden_state
                left_mask, right_mask = pair_segment_masks(
                    model_inputs["input_ids"],
                    model_inputs["attention_mask"],
                    sep_token_id=int(tokenizer.sep_token_id),
                    cls_token_id=tokenizer.cls_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                    token_type_ids=model_inputs.get("token_type_ids"),
                )
                pooled = torch.stack(
                    (
                        hidden[:, 0],
                        _masked_mean(hidden, left_mask),
                        _masked_max(hidden, left_mask),
                        _masked_mean(hidden, right_mask),
                        _masked_max(hidden, right_mask),
                    ),
                    dim=1,
                )
            current = pooled.detach().float().cpu().numpy().astype(np.float16)
            features[offset : offset + len(current)] = current
            offset += len(current)
    features.flush()
    if offset != len(pairs):
        raise RuntimeError(f"feature row mismatch: expected {len(pairs)}, got {offset}")
    return output_path


def parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


__all__ = [
    "extract_frozen_features",
    "load_frozen_backbone",
    "pair_segment_masks",
    "parameter_count",
    "sha256_file",
]
