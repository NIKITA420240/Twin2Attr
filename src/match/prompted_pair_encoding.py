"""Single-sequence prompt encoding for binary reranker models."""

from __future__ import annotations

from collections.abc import Sequence

from transformers import PreTrainedTokenizerBase

from .pair_serialization import serialize_card
from .prepare_data import PreparedCard, PreparedPair


def _truncate_card_values(
    card: PreparedCard,
    *,
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


def serialize_prompted_pair(
    pair: PreparedPair,
    *,
    use_field_tokens: bool = True,
    max_attribute_value_chars: int | None = None,
) -> str:
    """Serialize ``question:<A>\n\npassage:<B>`` without token truncation."""
    left = serialize_card(
        _truncate_card_values(
            pair.left,
            max_attribute_value_chars=max_attribute_value_chars,
        ),
        use_field_tokens=use_field_tokens,
    )
    right = serialize_card(
        _truncate_card_values(
            pair.right,
            max_attribute_value_chars=max_attribute_value_chars,
        ),
        use_field_tokens=use_field_tokens,
    )
    return f"question:{left}\n\npassage:{right}"


def serialize_prompted_pair_for_tokenizer(
    tokenizer: PreTrainedTokenizerBase,
    pair: PreparedPair,
    *,
    use_field_tokens: bool,
    max_attribute_value_chars: int | None,
    max_attribute_value_tokens: int | None,
) -> str:
    """Apply per-value limits before assembling the reranker prompt."""
    if max_attribute_value_tokens is None:
        return serialize_prompted_pair(
            pair,
            use_field_tokens=use_field_tokens,
            max_attribute_value_chars=max_attribute_value_chars,
        )

    def truncated(card: PreparedCard) -> PreparedCard:
        attributes: list[tuple[str, str]] = []
        for key, value in card.attributes:
            if max_attribute_value_chars is not None:
                value = value[:max_attribute_value_chars]
            token_ids = tokenizer.encode(value, add_special_tokens=False)
            value = tokenizer.decode(
                token_ids[:max_attribute_value_tokens],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            attributes.append((key, value))
        return PreparedCard(card.item_id, card.name, card.category, tuple(attributes))

    return serialize_prompted_pair(
        PreparedPair(
            truncated(pair.left),
            truncated(pair.right),
            pair.label,
            pair.category,
            pair.sample_weight,
            pair.preserve_attribute_order,
            pair.skip_oversized_attributes,
            pair.training_target,
        ),
        use_field_tokens=use_field_tokens,
    )


def encode_prompted_pairs(
    tokenizer: PreTrainedTokenizerBase,
    pairs: Sequence[PreparedPair],
    *,
    max_length: int,
    use_field_tokens: bool,
    max_attribute_value_chars: int | None,
    max_attribute_value_tokens: int | None,
) -> list[dict[str, list[int]]]:
    prompts = [
        serialize_prompted_pair_for_tokenizer(
            tokenizer,
            pair,
            use_field_tokens=use_field_tokens,
            max_attribute_value_chars=max_attribute_value_chars,
            max_attribute_value_tokens=max_attribute_value_tokens,
        )
        for pair in pairs
    ]
    encoded = tokenizer(
        prompts,
        add_special_tokens=True,
        padding=False,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
        return_token_type_ids=False,
    )
    return [
        {"input_ids": list(input_ids), "attention_mask": list(attention_mask)}
        for input_ids, attention_mask in zip(
            encoded["input_ids"], encoded["attention_mask"]
        )
    ]


__all__ = [
    "encode_prompted_pairs",
    "serialize_prompted_pair",
    "serialize_prompted_pair_for_tokenizer",
]
