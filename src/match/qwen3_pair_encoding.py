"""Official instruction prompt encoding for Qwen3 reranker models."""

from __future__ import annotations

from collections.abc import Sequence

from transformers import PreTrainedTokenizerBase

from .pair_serialization import serialize_card
from .prepare_data import PreparedCard, PreparedPair


QWEN3_RERANKER_INSTRUCTION = (
    "Same exact SKU? Any conflicting variant attribute means no."
)
QWEN3_RERANKER_PREFIX = (
    "<|im_start|>system\n"
    "Judge whether the Document meets the requirements based on the Query and the "
    'Instruct provided. Note that the answer can only be "yes" or "no".'
    "<|im_end|>\n<|im_start|>user\n"
)
QWEN3_RERANKER_SUFFIX = (
    "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
)


def _truncate_card(
    tokenizer: PreTrainedTokenizerBase,
    card: PreparedCard,
    *,
    max_attribute_value_chars: int | None,
    max_attribute_value_tokens: int | None,
) -> PreparedCard:
    attributes: list[tuple[str, str]] = []
    for key, value in card.attributes:
        if max_attribute_value_chars is not None:
            value = value[:max_attribute_value_chars]
        if max_attribute_value_tokens is not None:
            token_ids = tokenizer.encode(value, add_special_tokens=False)
            value = tokenizer.decode(
                token_ids[:max_attribute_value_tokens],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
        attributes.append((key, value))
    return PreparedCard(card.item_id, card.name, card.category, tuple(attributes))


def _batch_encode_texts(
    tokenizer: PreTrainedTokenizerBase,
    texts: Sequence[str],
    *,
    chunk_size: int,
) -> list[list[int]]:
    """Tokenize strings in bounded chunks while preserving input order."""
    encoded: list[list[int]] = []
    for start in range(0, len(texts), chunk_size):
        chunk = list(texts[start : start + chunk_size])
        batch = tokenizer(
            chunk,
            add_special_tokens=False,
            padding=False,
            truncation=False,
        )
        encoded.extend(list(map(int, token_ids)) for token_ids in batch["input_ids"])
    return encoded


def _batch_truncate_pairs(
    tokenizer: PreTrainedTokenizerBase,
    pairs: Sequence[PreparedPair],
    *,
    max_attribute_value_chars: int | None,
    max_attribute_value_tokens: int | None,
    chunk_size: int,
) -> list[PreparedPair]:
    """Apply per-value limits with batched tokenizer calls."""
    values = [
        (
            value
            if max_attribute_value_chars is None
            else value[:max_attribute_value_chars]
        )
        for pair in pairs
        for card in (pair.left, pair.right)
        for _, value in card.attributes
    ]
    if max_attribute_value_tokens is None:
        truncated_values = values
    else:
        value_ids = _batch_encode_texts(
            tokenizer,
            values,
            chunk_size=chunk_size,
        )
        truncated_values = []
        for start in range(0, len(value_ids), chunk_size):
            truncated_values.extend(
                tokenizer.batch_decode(
                    [
                        token_ids[:max_attribute_value_tokens]
                        for token_ids in value_ids[start : start + chunk_size]
                    ],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            )

    value_index = 0

    def truncate_card(card: PreparedCard) -> PreparedCard:
        nonlocal value_index
        attributes: list[tuple[str, str]] = []
        for key, _ in card.attributes:
            attributes.append((key, truncated_values[value_index]))
            value_index += 1
        return PreparedCard(card.item_id, card.name, card.category, tuple(attributes))

    prepared: list[PreparedPair] = []
    for pair in pairs:
        prepared.append(
            PreparedPair(
                truncate_card(pair.left),
                truncate_card(pair.right),
                pair.label,
                pair.category,
                pair.sample_weight,
                pair.preserve_attribute_order,
                pair.skip_oversized_attributes,
            )
        )
    return prepared


def _truncate_card_characters(
    card: PreparedCard,
    max_attribute_value_chars: int | None,
) -> PreparedCard:
    return PreparedCard(
        card.item_id,
        card.name,
        card.category,
        tuple(
            (
                key,
                value
                if max_attribute_value_chars is None
                else value[:max_attribute_value_chars],
            )
            for key, value in card.attributes
        ),
    )


def _body(left: str, right: str) -> str:
    return (
        f"<Instruct>: {QWEN3_RERANKER_INSTRUCTION}\n"
        f"<Query>: {left}\n"
        f"<Document>: {right}"
    )


def serialize_qwen3_pair(
    pair: PreparedPair,
    *,
    use_field_tokens: bool = False,
    max_attribute_value_chars: int | None = None,
) -> str:
    """Serialize a product pair with Qwen3's official reranking chat template."""
    left = serialize_card(
        _truncate_card_characters(pair.left, max_attribute_value_chars),
        use_field_tokens=use_field_tokens,
    )
    right = serialize_card(
        _truncate_card_characters(pair.right, max_attribute_value_chars),
        use_field_tokens=use_field_tokens,
    )
    return QWEN3_RERANKER_PREFIX + _body(left, right) + QWEN3_RERANKER_SUFFIX


def serialize_qwen3_pair_for_tokenizer(
    tokenizer: PreTrainedTokenizerBase,
    pair: PreparedPair,
    *,
    use_field_tokens: bool,
    max_attribute_value_chars: int | None,
    max_attribute_value_tokens: int | None,
) -> str:
    """Apply per-value limits before assembling the Qwen3 prompt."""
    prepared = PreparedPair(
        _truncate_card(
            tokenizer,
            pair.left,
            max_attribute_value_chars=max_attribute_value_chars,
            max_attribute_value_tokens=max_attribute_value_tokens,
        ),
        _truncate_card(
            tokenizer,
            pair.right,
            max_attribute_value_chars=max_attribute_value_chars,
            max_attribute_value_tokens=max_attribute_value_tokens,
        ),
        pair.label,
        pair.category,
        pair.sample_weight,
        pair.preserve_attribute_order,
        pair.skip_oversized_attributes,
    )
    return serialize_qwen3_pair(prepared, use_field_tokens=use_field_tokens)


def encode_qwen3_pairs(
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
    """Tokenize Qwen3 prompts while always preserving its assistant suffix."""
    prefix_ids = list(
        tokenizer.encode(QWEN3_RERANKER_PREFIX, add_special_tokens=False)
    )
    suffix_ids = list(
        tokenizer.encode(QWEN3_RERANKER_SUFFIX, add_special_tokens=False)
    )
    body_limit = max_length - len(prefix_ids) - len(suffix_ids)
    if body_limit < 1:
        raise ValueError(
            "max_length is too small for the Qwen3 reranker prefix and suffix"
        )

    prepared_pairs = (
        _batch_truncate_pairs(
            tokenizer,
            pairs,
            max_attribute_value_chars=max_attribute_value_chars,
            max_attribute_value_tokens=max_attribute_value_tokens,
            chunk_size=field_chunk_size,
        )
        if batch_fields
        else None
    )
    prompts = [
        (
            serialize_qwen3_pair(pair, use_field_tokens=use_field_tokens)
            if prepared_pairs is not None
            else serialize_qwen3_pair_for_tokenizer(
                tokenizer,
                pair,
                use_field_tokens=use_field_tokens,
                max_attribute_value_chars=max_attribute_value_chars,
                max_attribute_value_tokens=max_attribute_value_tokens,
            )
        )
        for pair in (prepared_pairs if prepared_pairs is not None else pairs)
    ]
    bodies = [
        prompt[len(QWEN3_RERANKER_PREFIX) : -len(QWEN3_RERANKER_SUFFIX)]
        for prompt in prompts
    ]
    batched_body_ids = (
        _batch_encode_texts(
            tokenizer,
            bodies,
            chunk_size=field_chunk_size,
        )
        if batch_fields
        else None
    )

    encoded: list[dict[str, list[int]]] = []
    for index, body in enumerate(bodies):
        body_ids = (
            batched_body_ids[index]
            if batched_body_ids is not None
            else list(tokenizer.encode(body, add_special_tokens=False))
        )[:body_limit]
        input_ids = [*prefix_ids, *body_ids, *suffix_ids]
        encoded.append(
            {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
            }
        )
    return encoded


__all__ = [
    "QWEN3_RERANKER_INSTRUCTION",
    "QWEN3_RERANKER_PREFIX",
    "QWEN3_RERANKER_SUFFIX",
    "encode_qwen3_pairs",
    "serialize_qwen3_pair",
    "serialize_qwen3_pair_for_tokenizer",
]
