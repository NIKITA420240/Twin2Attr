"""Load packaged word-level NER artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from transformers import AutoTokenizer

from .config import WordNerModelConfig
from .model import WordNERModel
from .postprocessing import NerPostprocessor

if TYPE_CHECKING:
    from .predictor import WordNerPredictor


def _resolve_device(device: str | torch.device | None) -> torch.device:
    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_word_ner_predictor(
    model_dir: str | Path,
    *,
    cluster_centers_path: str | Path | None,
    batch_size: int,
    max_length: int,
    use_amp: bool,
    semantic_cleanup: bool,
    device: str | torch.device | None = None,
) -> WordNerPredictor:
    from .predictor import WordNerPredictor

    directory = Path(model_dir).expanduser().resolve()
    metadata_path = directory / "word_ner_config.json"
    weights_path = directory / "model.pt"
    for path in (directory, metadata_path, weights_path):
        if not path.exists():
            raise FileNotFoundError(f"NER artifact does not exist: {path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise ValueError("word_ner_config.json must contain a JSON object")
    model_config = WordNerModelConfig.from_mapping(metadata)
    architecture_source = directory if (directory / "config.json").is_file() else None
    model = WordNERModel(
        model_config,
        architecture_source=architecture_source,
    )
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    target_device = _resolve_device(device)
    tokenizer = AutoTokenizer.from_pretrained(
        directory,
        use_fast=True,
        local_files_only=True,
    )
    if not tokenizer.is_fast:
        raise TypeError("Word NER requires a fast tokenizer with offset mappings")
    centers: dict[str, torch.Tensor] | None = None
    if semantic_cleanup:
        if cluster_centers_path is None:
            raise ValueError("semantic_cleanup requires cluster_centers_path")
        centers_path = Path(cluster_centers_path).expanduser().resolve()
        if not centers_path.is_file():
            raise FileNotFoundError(
                f"NER cluster centers do not exist: {centers_path}"
            )
        loaded_centers = torch.load(
            centers_path,
            map_location="cpu",
            weights_only=True,
        )
        if not isinstance(loaded_centers, dict):
            raise TypeError("cluster_centers.pt must contain a mapping")
        centers = loaded_centers
    postprocessor = NerPostprocessor(
        model_config.class_names,
        cluster_centers=centers,
        device=target_device,
    )
    return WordNerPredictor(
        model=model,
        tokenizer=tokenizer,
        postprocessor=postprocessor,
        device=target_device,
        batch_size=batch_size,
        max_length=max_length,
        use_amp=use_amp,
    )


__all__ = ["load_word_ner_predictor"]
