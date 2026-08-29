"""Pure text serialization for prepared cards and pairs."""

from __future__ import annotations

from .prepare_data import PreparedCard, PreparedPair


KEY_TOKEN = "[KEY]"
VAL_TOKEN = "[VAL]"
PAIR_SPECIAL_TOKENS = (KEY_TOKEN, VAL_TOKEN)
DEFAULT_MAX_ATTRIBUTE_VALUE_TOKENS = 32


def serialize_card(card: PreparedCard, *, use_field_tokens: bool = True) -> str:
    """Serialize fields in source order without tokenization or truncation."""
    fields = (("name", card.name), ("category", card.category), *card.attributes)
    if use_field_tokens:
        return " ".join(f"{KEY_TOKEN} {key} {VAL_TOKEN} {value}" for key, value in fields)
    return " ".join(f"{key}: {value}" for key, value in fields)


def serialize_pair(
    pair: PreparedPair,
    *,
    use_field_tokens: bool = True,
) -> tuple[str, str]:
    """Return complete left and right card texts."""
    return (
        serialize_card(pair.left, use_field_tokens=use_field_tokens),
        serialize_card(pair.right, use_field_tokens=use_field_tokens),
    )


__all__ = [
    "KEY_TOKEN",
    "DEFAULT_MAX_ATTRIBUTE_VALUE_TOKENS",
    "PAIR_SPECIAL_TOKENS",
    "VAL_TOKEN",
    "serialize_card",
    "serialize_pair",
]
