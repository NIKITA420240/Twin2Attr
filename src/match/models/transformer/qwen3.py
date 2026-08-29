"""Zero-shot yes/no scoring wrapper for Qwen3 reranker checkpoints."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput


QWEN3_NO_TOKEN = "no"
QWEN3_YES_TOKEN = "yes"


def qwen3_score_token_ids(tokenizer: Any) -> tuple[int, int]:
    """Resolve the exact one-token labels used by the official reranker."""
    identifiers: list[int] = []
    for token in (QWEN3_NO_TOKEN, QWEN3_YES_TOKEN):
        values = list(tokenizer.encode(token, add_special_tokens=False))
        if len(values) != 1:
            raise ValueError(
                f"Qwen3 score label {token!r} must encode to one token, got {values}"
            )
        identifiers.append(int(values[0]))
    if identifiers[0] == identifiers[1]:
        raise ValueError("Qwen3 yes/no score labels resolve to the same token")
    return identifiers[0], identifiers[1]


class Qwen3YesNoReranker(nn.Module):
    """Return one stable logit: ``yes_logit - no_logit``.

    The full language-model head is deliberately not retained. Only its two
    required rows are copied, avoiding a sequence-by-vocabulary output and
    allowing ONNX dead-code elimination to omit the unused output projection.
    """

    def __init__(
        self,
        causal_lm: nn.Module,
        *,
        no_token_id: int,
        yes_token_id: int,
        prepare_4d_attention_mask: bool = False,
    ) -> None:
        super().__init__()
        backbone = getattr(causal_lm, "model", None)
        output_embeddings = causal_lm.get_output_embeddings()
        if backbone is None or output_embeddings is None:
            raise ValueError("Qwen3 causal LM must expose model and output embeddings")
        weight = output_embeddings.weight.detach()
        self.backbone = backbone
        self.register_buffer(
            "score_weight",
            weight[[no_token_id, yes_token_id]].clone(),
            persistent=True,
        )
        self.prepare_4d_attention_mask = prepare_4d_attention_mask
        self.config = causal_lm.config

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        **_: Any,
    ) -> SequenceClassifierOutput:
        model_attention_mask = attention_mask
        if self.prepare_4d_attention_mask:
            # Transformers >=4.57 builds causal masks through torch.vmap,
            # which the legacy ONNX tracer cannot represent. Qwen accepts an
            # already prepared additive 4D mask, so construct the equivalent
            # causal + key-padding mask with traceable tensor operations.
            expanded = attention_mask[:, None, None, :].expand(
                -1,
                1,
                input_ids.shape[1],
                -1,
            )
            allowed = torch.tril(expanded).to(dtype=torch.bool)
            zero = torch.zeros((), dtype=self.score_weight.dtype, device=allowed.device)
            blocked = torch.full(
                (),
                torch.finfo(self.score_weight.dtype).min,
                dtype=self.score_weight.dtype,
                device=allowed.device,
            )
            model_attention_mask = torch.where(allowed, zero, blocked)
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=model_attention_mask,
            use_cache=False,
            return_dict=True,
        )
        # Qwen prompts use left padding and always end in the assistant suffix,
        # so the final hidden state predicts the next yes/no token.
        final_hidden = outputs.last_hidden_state[:, -1, :]
        pair_logits = torch.nn.functional.linear(final_hidden, self.score_weight)
        margin = (pair_logits[:, 1] - pair_logits[:, 0]).unsqueeze(1)
        return SequenceClassifierOutput(logits=margin)


__all__ = [
    "QWEN3_NO_TOKEN",
    "QWEN3_YES_TOKEN",
    "Qwen3YesNoReranker",
    "qwen3_score_token_ids",
]
