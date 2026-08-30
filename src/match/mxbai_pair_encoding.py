"""Official chat-template encoding for Mixedbread causal rerankers."""

from __future__ import annotations

from collections.abc import Sequence

from transformers import PreTrainedTokenizerBase

from .pair_serialization import serialize_card
from .prepare_data import PreparedPair
from .qwen3_pair_encoding import _batch_truncate_pairs


MXBAI_RERANKER_PREFIX = (
    "<|im_start|>system\n"
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    "<|im_end|>\n<|im_start|>user\nquery: "
)
MXBAI_RERANKER_DOCUMENT_SEPARATOR = "\ndocument: "
MXBAI_RERANKER_SUFFIX = (
    "\nYou are a search relevance expert who evaluates how well documents match "
    "search queries. For each query-document pair, carefully analyze the semantic "
    "relationship between them, then provide your binary relevance judgment (0 for "
    "not relevant, 1 for relevant).\nRelevance:<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def serialize_mxbai_pair(
    pair: PreparedPair,
    *,
    use_field_tokens: bool = False,
) -> str:
    """Render one pair exactly like the checkpoint's chat template."""
    left = serialize_card(pair.left, use_field_tokens=use_field_tokens)
    right = serialize_card(pair.right, use_field_tokens=use_field_tokens)
    return (
        MXBAI_RERANKER_PREFIX
        + left
        + MXBAI_RERANKER_DOCUMENT_SEPARATOR
        + right
        + MXBAI_RERANKER_SUFFIX
    )


def encode_mxbai_pairs(
    tokenizer: PreTrainedTokenizerBase,
    pairs: Sequence[PreparedPair],
    *,
    max_length: int,
    use_field_tokens: bool,
    max_attribute_value_chars: int | None,
    max_attribute_value_tokens: int | None,
    batch_fields: bool = False,
    field_chunk_size: int = 16_384,
) -> list[dict[str, list[int]]]:
    """Batch-tokenize prompts and preserve the assistant scoring suffix."""
    prepared = _batch_truncate_pairs(
        tokenizer,
        pairs,
        max_attribute_value_chars=max_attribute_value_chars,
        max_attribute_value_tokens=max_attribute_value_tokens,
        chunk_size=field_chunk_size,
    )
    prompts = [
        serialize_mxbai_pair(pair, use_field_tokens=use_field_tokens)
        for pair in prepared
    ]
    suffix_ids = list(
        tokenizer.encode(MXBAI_RERANKER_SUFFIX, add_special_tokens=False)
    )
    suffix_token_count = len(suffix_ids)
    if max_length <= suffix_token_count:
        raise ValueError("max_length is too small for the Mixedbread scoring suffix")

    prompt_ids: list[list[int]] = []
    for start in range(0, len(prompts), field_chunk_size):
        batch = tokenizer(
            prompts[start : start + field_chunk_size],
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )
        prompt_ids.extend(list(map(int, values)) for values in batch["input_ids"])

    encoded: list[dict[str, list[int]]] = []
    for values in prompt_ids:
        if len(values) > max_length:
            # Qwen BPE can merge trailing document whitespace with the suffix's
            # first newline. Preserve the contextualized tail from the complete
            # prompt instead of replacing it with the suffix encoded in isolation.
            contextual_suffix_ids = values[-suffix_token_count:]
            values = (
                values[: max_length - suffix_token_count]
                + contextual_suffix_ids
            )
        encoded.append(
            {
                "input_ids": values,
                "attention_mask": [1] * len(values),
            }
        )
    return encoded


__all__ = [
    "MXBAI_RERANKER_DOCUMENT_SEPARATOR",
    "MXBAI_RERANKER_PREFIX",
    "MXBAI_RERANKER_SUFFIX",
    "encode_mxbai_pairs",
    "serialize_mxbai_pair",
]
