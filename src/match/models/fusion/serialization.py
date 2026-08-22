"""Fusion checkpoint persistence."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Mapping

import torch
from loguru import logger

from .model import FusionClassifier, FusionConfig, resolve_device


def save_fusion_classifier(
    model: FusionClassifier,
    config: FusionConfig,
    state_dict: Mapping[str, torch.Tensor],
    *,
    best_validation_macro_pr_auc: float,
    model_path: str | Path,
) -> Path:
    target_path = Path(model_path).expanduser().resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 1,
            "cls_dim": model.cls_dim,
            "maxpooling_dim": model.maxpooling_dim,
            "config": asdict(config),
            "state_dict": {key: value.cpu() for key, value in state_dict.items()},
            "best_validation_macro_pr_auc": best_validation_macro_pr_auc,
        },
        target_path,
    )
    logger.info(
        "Saved fusion head: path={!s}, best_val_macro_pr_auc={:.6f}",
        target_path,
        best_validation_macro_pr_auc,
    )
    return target_path


def load_fusion_classifier(
    model_path: str | Path,
    *,
    device: str | torch.device | None = None,
) -> FusionClassifier:
    target_device = resolve_device(device)
    checkpoint = torch.load(
        Path(model_path).expanduser().resolve(),
        map_location=target_device,
        weights_only=True,
    )
    if checkpoint.get("format_version") != 1:
        raise ValueError("unsupported fusion checkpoint format")
    saved_config = FusionConfig(**checkpoint["config"])
    model = FusionClassifier(
        int(checkpoint["cls_dim"]),
        int(checkpoint["maxpooling_dim"]),
        hidden_dim=saved_config.hidden_dim,
        dropout=saved_config.dropout,
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(target_device).eval()


__all__ = ["load_fusion_classifier", "save_fusion_classifier"]
