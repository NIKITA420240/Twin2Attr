"""Mixedbread causal-reranker scoring contract."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


MXBAI_LOGIT_SCORE_CONFIG = Path("1_LogitScore") / "config.json"


def mxbai_score_token_ids(
    model_directory: str | Path,
    tokenizer: Any,
) -> tuple[int, int]:
    """Return the checkpoint's false/true token IDs and validate its tokenizer."""
    path = Path(model_directory) / MXBAI_LOGIT_SCORE_CONFIG
    if not path.is_file():
        raise FileNotFoundError(f"Mixedbread LogitScore config is missing: {path}")
    values = json.loads(path.read_text(encoding="utf-8"))
    false_token_id = int(values["false_token_id"])
    true_token_id = int(values["true_token_id"])
    if false_token_id == true_token_id:
        raise ValueError("Mixedbread true and false score token IDs must differ")
    if list(tokenizer.encode("0", add_special_tokens=False)) != [false_token_id]:
        raise ValueError("Mixedbread false_token_id does not encode label '0'")
    if list(tokenizer.encode("1", add_special_tokens=False)) != [true_token_id]:
        raise ValueError("Mixedbread true_token_id does not encode label '1'")
    return false_token_id, true_token_id


__all__ = ["MXBAI_LOGIT_SCORE_CONFIG", "mxbai_score_token_ids"]
