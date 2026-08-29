"""Model-aware serialization and tokenization of prepared product pairs."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from .models.transformer.profile import (
    PROMPTED_BINARY_RERANKER_PROFILE,
    QWEN3_RERANKER_PROFILE,
    SEQUENCE_CLASSIFIER_PROFILE,
    normalize_profile,
)
from .qwen3_pair_encoding import (
    QWEN3_RERANKER_PREFIX,
    QWEN3_RERANKER_SUFFIX,
    encode_qwen3_pairs as _encode_qwen3_pairs,
    serialize_qwen3_pair,
    serialize_qwen3_pair_for_tokenizer as _serialize_qwen3_pair_for_tokenizer,
)
from .pair_serialization import (
    DEFAULT_MAX_ATTRIBUTE_VALUE_TOKENS,
    KEY_TOKEN,
    PAIR_SPECIAL_TOKENS,
    VAL_TOKEN,
    serialize_card,
    serialize_pair,
)
from .prepare_data import PreparedCard, PreparedPair
from .prompted_pair_encoding import (
    encode_prompted_pairs as _encode_prompted_pairs,
    serialize_prompted_pair,
    serialize_prompted_pair_for_tokenizer as _serialize_prompted_pair_for_tokenizer,
)

__all__ = [
    "DEFAULT_MAX_ATTRIBUTE_VALUE_TOKENS",
    "KEY_TOKEN",
    "VAL_TOKEN",
    "PairEncodingCollator",
    "PairEncodingWithAttributes",
    "PreparedPairDataset",
    "add_pair_special_tokens",
    "encode_prepared_pair",
    "encode_prepared_pair_with_attributes",
    "infer_pair_max_length",
    "pair_special_token_ids",
    "serialize_card",
    "serialize_pair",
    "serialize_prompted_pair",
    "serialize_qwen3_pair",
]


class PreparedPairDataset(Dataset):
    """Expose prepared pairs to a PyTorch ``DataLoader`` without copying them."""

    def __init__(
        self,
        pairs: Sequence[PreparedPair],
        *,
        transform: Callable[[PreparedPair], PreparedPair] | None = None,
    ) -> None:
        self._pairs = pairs
        self._transform = transform

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> PreparedPair:
        pair = self._pairs[index]
        return pair if self._transform is None else self._transform(pair)


def add_pair_special_tokens(
    tokenizer: PreTrainedTokenizerBase,
    model: PreTrainedModel | None = None,
) -> int:
    """Register ``[KEY]`` and ``[VAL]`` and optionally resize model embeddings.

    Call this once after loading a tokenizer and every freshly initialized
    model. Existing special tokens are not changed.

    Returns
    -------
    int
        Number of tokens newly added to the tokenizer vocabulary.
    """
    added_count = tokenizer.add_tokens(list(PAIR_SPECIAL_TOKENS), special_tokens=True)

    if model is not None:
        embeddings = model.get_input_embeddings()
        if embeddings is None:
            raise ValueError("model does not expose input embeddings")
        if embeddings.num_embeddings != len(tokenizer):
            model.resize_token_embeddings(len(tokenizer))
    return added_count


def pair_special_token_ids(
    tokenizer: PreTrainedTokenizerBase,
) -> tuple[int, ...]:
    """Return the single vocabulary id assigned to each pair field token."""
    token_ids: list[int] = []
    for token in PAIR_SPECIAL_TOKENS:
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if len(encoded) != 1 or tokenizer.convert_ids_to_tokens(encoded[0]) != token:
            raise ValueError(
                f"tokenizer does not contain {token!r}; call "
                "add_pair_special_tokens(tokenizer, model) first"
            )
        token_ids.append(int(encoded[0]))
    return tuple(token_ids)


def _require_pair_special_tokens(tokenizer: PreTrainedTokenizerBase) -> None:
    pair_special_token_ids(tokenizer)


def _pair_special_token_count(tokenizer: PreTrainedTokenizerBase) -> int:
    """Return the pair-token overhead across Transformers tokenizer APIs.

    Transformers v5 tokenizer backends may omit the legacy pair-builder
    methods and may not retain pair post-processor metadata. In that case the
    special-token names still distinguish BERT's
    ``[CLS] A [SEP] B [SEP]`` layout from RoBERTa/XLM-R's
    ``<s> A </s></s> B </s>`` layout.
    """
    build_inputs = getattr(tokenizer, "build_inputs_with_special_tokens", None)
    if callable(build_inputs):
        return int(tokenizer.num_special_tokens_to_add(pair=True))

    reported_count = int(tokenizer.num_special_tokens_to_add(pair=True))
    if reported_count == 3:
        return reported_count

    cls_token_id = getattr(tokenizer, "cls_token_id", None)
    sep_token_id = getattr(tokenizer, "sep_token_id", None)
    cls_token = getattr(tokenizer, "cls_token", None)
    sep_token = getattr(tokenizer, "sep_token", None)
    uses_bert_tokens = cls_token in (None, "[CLS]") and sep_token in (
        None,
        "[SEP]",
    )
    if cls_token_id is not None and sep_token_id is not None and uses_bert_tokens:
        return 3
    uses_roberta_tokens = cls_token == "<s>" and sep_token == "</s>"
    if cls_token_id is not None and sep_token_id is not None and uses_roberta_tokens:
        return 4
    raise ValueError(
        "tokenizer without build_inputs_with_special_tokens must use a supported "
        "BERT or RoBERTa/XLM-R pair layout"
    )


def _encode_card_sections(
    tokenizer: PreTrainedTokenizerBase,
    card: PreparedCard,
    *,
    use_field_tokens: bool,
    max_attribute_value_chars: int | None,
    max_attribute_value_tokens: int | None,
    preserve_attribute_order: bool = False,
) -> tuple[list[int], list[tuple[str, list[int]]], int]:
    required_texts, attribute_texts = _card_section_texts(
        card,
        use_field_tokens=use_field_tokens,
        max_attribute_value_chars=max_attribute_value_chars,
    )
    required: list[int] = []
    for text in required_texts:
        required.extend(tokenizer.encode(text, add_special_tokens=False))

    attributes: list[tuple[str, list[int]]] = []
    for key, prefix, value in attribute_texts:
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        value_ids = tokenizer.encode(value, add_special_tokens=False)
        if max_attribute_value_tokens is not None:
            value_ids = value_ids[:max_attribute_value_tokens]
        attributes.append((str(key), prefix_ids + value_ids))
    if not preserve_attribute_order:
        attributes.sort(key=lambda attribute: len(attribute[1]))
    demand = len(required) + sum(len(token_ids) for _, token_ids in attributes)
    return required, attributes, demand


def _card_section_texts(
    card: PreparedCard,
    *,
    use_field_tokens: bool,
    max_attribute_value_chars: int | None,
) -> tuple[tuple[str, str], list[tuple[str, str, str]]]:
    required = tuple(
        f"{KEY_TOKEN} {key} {VAL_TOKEN} {value}"
        if use_field_tokens
        else f"{key}: {value}"
        for key, value in (("name", card.name), ("category", card.category))
    )
    attributes: list[tuple[str, str, str]] = []
    for key, value in card.attributes:
        if max_attribute_value_chars is not None:
            value = value[:max_attribute_value_chars]
        prefix = (
            f"{KEY_TOKEN} {key} {VAL_TOKEN}"
            if use_field_tokens
            else f"{key}:"
        )
        attributes.append((str(key), prefix, f" {value}"))
    return required, attributes


class _PreencodedTokenizer:
    """Replay batched field encodings through the scalar assembly path."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        texts: Sequence[str],
        input_ids: Sequence[Sequence[int]],
    ) -> None:
        self._tokenizer = tokenizer
        self._texts = texts
        self._input_ids = input_ids
        self._index = 0

    def __getattr__(self, name: str):
        return getattr(self._tokenizer, name)

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        if add_special_tokens:
            raise ValueError("batched field encoding does not add special tokens")
        if self._index >= len(self._texts) or self._texts[self._index] != text:
            raise RuntimeError("batched field tokenization order is inconsistent")
        result = list(self._input_ids[self._index])
        self._index += 1
        return result

    def require_consumed(self) -> None:
        if self._index != len(self._texts):
            raise RuntimeError("batched field tokenization left unused encodings")


def _batch_encode_prepared_pairs(
    tokenizer: PreTrainedTokenizerBase,
    pairs: Sequence[PreparedPair],
    *,
    max_length: int,
    special_token_count: int,
    use_field_tokens: bool,
    max_attribute_value_chars: int | None,
    max_attribute_value_tokens: int | None,
    field_chunk_size: int,
) -> list[dict[str, list[int]]]:
    texts: list[str] = []
    for pair in pairs:
        for card in (pair.left, pair.right):
            required, attributes = _card_section_texts(
                card,
                use_field_tokens=use_field_tokens,
                max_attribute_value_chars=max_attribute_value_chars,
            )
            texts.extend(required)
            for _, prefix, value in attributes:
                texts.extend((prefix, value))

    input_ids: list[list[int]] = []
    for offset in range(0, len(texts), field_chunk_size):
        encoded = tokenizer(
            texts[offset : offset + field_chunk_size],
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        input_ids.extend(encoded["input_ids"])

    queued_tokenizer = _PreencodedTokenizer(tokenizer, texts, input_ids)
    encoded_pairs = [
        _encode_prepared_pair(
            queued_tokenizer,
            pair,
            max_length=max_length,
            special_token_count=special_token_count,
            use_field_tokens=use_field_tokens,
            max_attribute_value_chars=max_attribute_value_chars,
            max_attribute_value_tokens=max_attribute_value_tokens,
        )
        for pair in pairs
    ]
    queued_tokenizer.require_consumed()
    return encoded_pairs


@dataclass(frozen=True, slots=True)
class _FittedCard:
    input_ids: list[int]
    token_attributes: list[str | None]
    included_attributes: tuple[str, ...]


def _fit_pair_to_budget(
    left_sections: tuple[list[int], list[tuple[str, list[int]]], int],
    right_sections: tuple[list[int], list[tuple[str, list[int]]], int],
    content_budget: int,
    *,
    skip_oversized_attributes: bool = False,
) -> tuple[_FittedCard, _FittedCard]:
    def fit_card(
        sections: tuple[list[int], list[tuple[str, list[int]]], int],
        budget: int,
    ) -> _FittedCard:
        required, attributes, _ = sections
        if len(required) >= budget:
            fitted = required[:budget]
            return _FittedCard(fitted, [None] * len(fitted), ())

        fitted = list(required)
        token_attributes: list[str | None] = [None] * len(required)
        included: list[str] = []
        for key, token_ids in attributes:
            if len(fitted) + len(token_ids) > budget:
                if skip_oversized_attributes:
                    continue
                break
            fitted.extend(token_ids)
            token_attributes.extend([key] * len(token_ids))
            included.append(key)
        return _FittedCard(fitted, token_attributes, tuple(included))

    left_demand = left_sections[2]
    right_demand = right_sections[2]
    left_budget = min(left_demand, (content_budget + 1) // 2)
    right_budget = min(right_demand, content_budget // 2)
    remaining = content_budget - left_budget - right_budget

    left_extra = min(left_demand - left_budget, remaining)
    left_budget += left_extra
    remaining -= left_extra
    right_budget += min(right_demand - right_budget, remaining)

    left = fit_card(left_sections, left_budget)
    right = fit_card(right_sections, right_budget)

    while (
        remaining := content_budget - len(left.input_ids) - len(right.input_ids)
    ) > 0:
        left_candidate = fit_card(
            left_sections,
            len(left.input_ids) + remaining,
        )
        right_candidate = fit_card(
            right_sections,
            len(right.input_ids) + remaining,
        )
        left_gain = len(left_candidate.input_ids) - len(left.input_ids)
        right_gain = len(right_candidate.input_ids) - len(right.input_ids)
        if left_gain <= 0 and right_gain <= 0:
            break
        if left_gain >= right_gain:
            left = left_candidate
        else:
            right = right_candidate
    return left, right


def _encode_prepared_pair_with_fitted(
    tokenizer: PreTrainedTokenizerBase,
    pair: PreparedPair,
    *,
    max_length: int,
    special_token_count: int,
    use_field_tokens: bool,
    max_attribute_value_chars: int | None,
    max_attribute_value_tokens: int | None,
) -> tuple[dict[str, list[int]], _FittedCard, _FittedCard]:
    content_budget = max_length - special_token_count
    if content_budget < 1:
        raise ValueError("max_length must leave room for pair content after special tokens")

    left_sections = _encode_card_sections(
        tokenizer,
        pair.left,
        use_field_tokens=use_field_tokens,
        max_attribute_value_chars=max_attribute_value_chars,
        max_attribute_value_tokens=max_attribute_value_tokens,
        preserve_attribute_order=pair.preserve_attribute_order,
    )
    right_sections = _encode_card_sections(
        tokenizer,
        pair.right,
        use_field_tokens=use_field_tokens,
        max_attribute_value_chars=max_attribute_value_chars,
        max_attribute_value_tokens=max_attribute_value_tokens,
        preserve_attribute_order=pair.preserve_attribute_order,
    )
    left, right = _fit_pair_to_budget(
        left_sections,
        right_sections,
        content_budget,
        skip_oversized_attributes=pair.skip_oversized_attributes,
    )
    left_ids = left.input_ids
    right_ids = right.input_ids

    build_inputs = getattr(tokenizer, "build_inputs_with_special_tokens", None)
    if callable(build_inputs):
        input_ids = build_inputs(left_ids, right_ids)
    else:
        cls_token_id = tokenizer.cls_token_id
        sep_token_id = tokenizer.sep_token_id
        if cls_token_id is None or sep_token_id is None:
            raise ValueError("pair encoding requires CLS and SEP token ids")
        cls_token = getattr(tokenizer, "cls_token", None)
        sep_token = getattr(tokenizer, "sep_token", None)
        if special_token_count == 3 and cls_token in (None, "[CLS]") and sep_token in (
            None,
            "[SEP]",
        ):
            input_ids = [cls_token_id, *left_ids, sep_token_id, *right_ids, sep_token_id]
        elif special_token_count == 4 and cls_token == "<s>" and sep_token == "</s>":
            input_ids = [
                cls_token_id,
                *left_ids,
                sep_token_id,
                sep_token_id,
                *right_ids,
                sep_token_id,
            ]
        else:
            raise ValueError(
                "tokenizer without build_inputs_with_special_tokens must use a "
                "supported BERT or RoBERTa/XLM-R pair layout"
            )
    encoded: dict[str, list[int]] = {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
    }
    if "token_type_ids" in tokenizer.model_input_names:
        create_token_types = getattr(
            tokenizer,
            "create_token_type_ids_from_sequences",
            None,
        )
        if callable(create_token_types):
            token_type_ids = create_token_types(left_ids, right_ids)
        else:
            token_type_ids = [0] * (len(left_ids) + 2) + [1] * (
                len(right_ids) + 1
            )
        encoded["token_type_ids"] = token_type_ids
    return encoded, left, right


def _encode_prepared_pair(
    tokenizer: PreTrainedTokenizerBase,
    pair: PreparedPair,
    *,
    max_length: int,
    special_token_count: int,
    use_field_tokens: bool,
    max_attribute_value_chars: int | None,
    max_attribute_value_tokens: int | None,
) -> dict[str, list[int]]:
    encoded, _, _ = _encode_prepared_pair_with_fitted(
        tokenizer,
        pair,
        max_length=max_length,
        special_token_count=special_token_count,
        use_field_tokens=use_field_tokens,
        max_attribute_value_chars=max_attribute_value_chars,
        max_attribute_value_tokens=max_attribute_value_tokens,
    )
    return encoded


def encode_prepared_pair(
    tokenizer: PreTrainedTokenizerBase,
    pair: PreparedPair,
    *,
    max_length: int,
    use_field_tokens: bool = True,
    max_attribute_value_chars: int | None = None,
    max_attribute_value_tokens: int | None = DEFAULT_MAX_ATTRIBUTE_VALUE_TOKENS,
) -> dict[str, list[int]]:
    """Tokenize a pair while dropping attributes only at field boundaries.

    The content budget is initially split equally. If one card is shorter than
    its share, its unused tokens are assigned to the other card. ``name`` and
    ``category`` have priority. Attributes are stably sorted from shortest to
    longest tokenized field, and only complete field sections are appended.

    ``use_field_tokens=False`` disables ``[KEY]`` and ``[VAL]`` without
    affecting the model-specific pair tokens such as ``[CLS]`` and ``[SEP]``.
    ``max_attribute_value_chars`` is applied before tokenization and
    ``max_attribute_value_tokens`` afterwards. ``None`` disables the
    corresponding per-value limit.
    """
    if max_attribute_value_chars is not None and max_attribute_value_chars < 1:
        raise ValueError("max_attribute_value_chars must be positive or None")
    if max_attribute_value_tokens is not None and max_attribute_value_tokens < 1:
        raise ValueError("max_attribute_value_tokens must be positive or None")
    if use_field_tokens:
        _require_pair_special_tokens(tokenizer)
    return _encode_prepared_pair(
        tokenizer,
        pair,
        max_length=max_length,
        special_token_count=_pair_special_token_count(tokenizer),
        use_field_tokens=use_field_tokens,
        max_attribute_value_chars=max_attribute_value_chars,
        max_attribute_value_tokens=max_attribute_value_tokens,
    )


@dataclass(frozen=True, slots=True)
class PairEncodingWithAttributes:
    """Encoded pair plus the raw-attribute owner of every token position."""

    inputs: dict[str, list[int]]
    token_attributes: tuple[str | None, ...]
    present_attribute_sections: dict[str, int]
    included_attribute_sections: dict[str, int]


def _pair_token_attributes(
    tokenizer: PreTrainedTokenizerBase,
    left: _FittedCard,
    right: _FittedCard,
    special_token_count: int,
) -> tuple[str | None, ...]:
    cls_token = getattr(tokenizer, "cls_token", None)
    sep_token = getattr(tokenizer, "sep_token", None)
    if special_token_count == 3 and cls_token in (None, "[CLS]") and sep_token in (
        None,
        "[SEP]",
    ):
        values = [
            None,
            *left.token_attributes,
            None,
            *right.token_attributes,
            None,
        ]
    elif special_token_count == 4 and cls_token == "<s>" and sep_token == "</s>":
        values = [
            None,
            *left.token_attributes,
            None,
            None,
            *right.token_attributes,
            None,
        ]
    else:
        raise ValueError(
            "attribute analysis supports BERT and RoBERTa/XLM-R pair layouts"
        )
    return tuple(values)


def encode_prepared_pair_with_attributes(
    tokenizer: PreTrainedTokenizerBase,
    pair: PreparedPair,
    *,
    max_length: int,
    use_field_tokens: bool = True,
    max_attribute_value_chars: int | None = None,
    max_attribute_value_tokens: int | None = DEFAULT_MAX_ATTRIBUTE_VALUE_TOKENS,
) -> PairEncodingWithAttributes:
    """Encode one pair and preserve token-to-raw-attribute alignment."""
    if max_attribute_value_chars is not None and max_attribute_value_chars < 1:
        raise ValueError("max_attribute_value_chars must be positive or None")
    if max_attribute_value_tokens is not None and max_attribute_value_tokens < 1:
        raise ValueError("max_attribute_value_tokens must be positive or None")
    if use_field_tokens:
        _require_pair_special_tokens(tokenizer)
    special_token_count = _pair_special_token_count(tokenizer)
    inputs, left, right = _encode_prepared_pair_with_fitted(
        tokenizer,
        pair,
        max_length=max_length,
        special_token_count=special_token_count,
        use_field_tokens=use_field_tokens,
        max_attribute_value_chars=max_attribute_value_chars,
        max_attribute_value_tokens=max_attribute_value_tokens,
    )
    token_attributes = _pair_token_attributes(
        tokenizer,
        left,
        right,
        special_token_count,
    )
    if len(token_attributes) != len(inputs["input_ids"]):
        raise RuntimeError("attribute token alignment does not match encoded input")
    present = Counter(
        str(key)
        for card in (pair.left, pair.right)
        for key, _ in card.attributes
    )
    included = Counter((*left.included_attributes, *right.included_attributes))
    return PairEncodingWithAttributes(
        inputs=inputs,
        token_attributes=token_attributes,
        present_attribute_sections=dict(present),
        included_attribute_sections=dict(included),
    )


@dataclass(frozen=True, slots=True)
class _PromptedPairBatchEncoder:
    tokenizer: PreTrainedTokenizerBase
    max_length: int
    use_field_tokens: bool
    max_attribute_value_chars: int | None
    max_attribute_value_tokens: int | None
    special_token_count: int

    def encode(
        self,
        pairs: Sequence[PreparedPair],
    ) -> list[dict[str, list[int]]]:
        return _encode_prompted_pairs(
            self.tokenizer,
            pairs,
            max_length=self.max_length,
            use_field_tokens=self.use_field_tokens,
            max_attribute_value_chars=self.max_attribute_value_chars,
            max_attribute_value_tokens=self.max_attribute_value_tokens,
        )


@dataclass(frozen=True, slots=True)
class _Qwen3PairBatchEncoder:
    tokenizer: PreTrainedTokenizerBase
    max_length: int
    use_field_tokens: bool
    max_attribute_value_chars: int | None
    max_attribute_value_tokens: int | None
    special_token_count: int

    def encode(
        self,
        pairs: Sequence[PreparedPair],
    ) -> list[dict[str, list[int]]]:
        return _encode_qwen3_pairs(
            self.tokenizer,
            pairs,
            max_length=self.max_length,
            use_field_tokens=self.use_field_tokens,
            max_attribute_value_chars=self.max_attribute_value_chars,
            max_attribute_value_tokens=self.max_attribute_value_tokens,
        )


@dataclass(frozen=True, slots=True)
class _LegacyPairBatchEncoder:
    tokenizer: PreTrainedTokenizerBase
    max_length: int
    use_field_tokens: bool
    max_attribute_value_chars: int | None
    max_attribute_value_tokens: int | None
    batch_fields: bool
    field_chunk_size: int
    special_token_count: int

    def encode(
        self,
        pairs: Sequence[PreparedPair],
    ) -> list[dict[str, list[int]]]:
        if self.batch_fields:
            return _batch_encode_prepared_pairs(
                self.tokenizer,
                pairs,
                max_length=self.max_length,
                special_token_count=self.special_token_count,
                use_field_tokens=self.use_field_tokens,
                max_attribute_value_chars=self.max_attribute_value_chars,
                max_attribute_value_tokens=self.max_attribute_value_tokens,
                field_chunk_size=self.field_chunk_size,
            )
        return [
            _encode_prepared_pair(
                self.tokenizer,
                pair,
                max_length=self.max_length,
                special_token_count=self.special_token_count,
                use_field_tokens=self.use_field_tokens,
                max_attribute_value_chars=self.max_attribute_value_chars,
                max_attribute_value_tokens=self.max_attribute_value_tokens,
            )
            for pair in pairs
        ]


def _pair_batch_encoder(
    tokenizer: PreTrainedTokenizerBase,
    *,
    profile: str,
    max_length: int,
    use_field_tokens: bool,
    max_attribute_value_chars: int | None,
    max_attribute_value_tokens: int | None,
    batch_fields: bool,
    field_chunk_size: int,
) -> _PromptedPairBatchEncoder | _Qwen3PairBatchEncoder | _LegacyPairBatchEncoder:
    if profile == PROMPTED_BINARY_RERANKER_PROFILE:
        return _PromptedPairBatchEncoder(
            tokenizer=tokenizer,
            max_length=max_length,
            use_field_tokens=use_field_tokens,
            max_attribute_value_chars=max_attribute_value_chars,
            max_attribute_value_tokens=max_attribute_value_tokens,
            special_token_count=int(tokenizer.num_special_tokens_to_add(pair=False)),
        )
    if profile == QWEN3_RERANKER_PROFILE:
        # The official Qwen reranker reads the next-token score after the
        # assistant suffix, so padding must stay to the left of that suffix.
        tokenizer.padding_side = "left"
        prompt_overhead = len(
            tokenizer.encode(QWEN3_RERANKER_PREFIX, add_special_tokens=False)
        ) + len(tokenizer.encode(QWEN3_RERANKER_SUFFIX, add_special_tokens=False))
        return _Qwen3PairBatchEncoder(
            tokenizer=tokenizer,
            max_length=max_length,
            use_field_tokens=use_field_tokens,
            max_attribute_value_chars=max_attribute_value_chars,
            max_attribute_value_tokens=max_attribute_value_tokens,
            special_token_count=prompt_overhead,
        )
    return _LegacyPairBatchEncoder(
        tokenizer=tokenizer,
        max_length=max_length,
        use_field_tokens=use_field_tokens,
        max_attribute_value_chars=max_attribute_value_chars,
        max_attribute_value_tokens=max_attribute_value_tokens,
        batch_fields=batch_fields,
        field_chunk_size=field_chunk_size,
        special_token_count=_pair_special_token_count(tokenizer),
    )


class PairEncodingCollator:
    """Encode structured pairs and dynamically pad them to the longest batch row."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        max_length: int,
        *,
        use_field_tokens: bool = True,
        max_attribute_value_chars: int | None = None,
        max_attribute_value_tokens: int | None = DEFAULT_MAX_ATTRIBUTE_VALUE_TOKENS,
        include_labels: bool = True,
        padding_length_buckets: tuple[int, ...] | None = None,
        batch_fields: bool = False,
        field_chunk_size: int = 16_384,
        profile: str = SEQUENCE_CLASSIFIER_PROFILE,
    ) -> None:
        if max_length < 1:
            raise ValueError("max_length must be positive")
        if max_attribute_value_chars is not None and max_attribute_value_chars < 1:
            raise ValueError("max_attribute_value_chars must be positive or None")
        if max_attribute_value_tokens is not None and max_attribute_value_tokens < 1:
            raise ValueError("max_attribute_value_tokens must be positive or None")
        if field_chunk_size < 1:
            raise ValueError("field_chunk_size must be positive")
        profile = normalize_profile(profile)
        if padding_length_buckets is not None:
            if not padding_length_buckets or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 1
                for value in padding_length_buckets
            ):
                raise ValueError(
                    "padding_length_buckets must contain positive integers"
                )
            if tuple(sorted(set(padding_length_buckets))) != padding_length_buckets:
                raise ValueError(
                    "padding_length_buckets must be strictly increasing"
                )
            if padding_length_buckets[-1] != max_length:
                raise ValueError(
                    "the final padding length bucket must equal max_length "
                    f"({max_length})"
                )
        if use_field_tokens:
            _require_pair_special_tokens(tokenizer)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_field_tokens = use_field_tokens
        self.max_attribute_value_chars = max_attribute_value_chars
        self.max_attribute_value_tokens = max_attribute_value_tokens
        self.include_labels = include_labels
        self.padding_length_buckets = padding_length_buckets
        self.batch_fields = batch_fields
        self.field_chunk_size = field_chunk_size
        self.profile = profile
        self._encoder = _pair_batch_encoder(
            tokenizer,
            profile=profile,
            max_length=max_length,
            use_field_tokens=use_field_tokens,
            max_attribute_value_chars=max_attribute_value_chars,
            max_attribute_value_tokens=max_attribute_value_tokens,
            batch_fields=batch_fields,
            field_chunk_size=field_chunk_size,
        )
        self.special_token_count = self._encoder.special_token_count
        if max_length <= self.special_token_count:
            raise ValueError("max_length must leave room for pair content after special tokens")

    def __call__(self, pairs: list[PreparedPair]) -> dict[str, torch.Tensor]:
        if not pairs:
            raise ValueError("cannot collate an empty batch")
        encoded_pairs = self._encoder.encode(pairs)
        padding: bool | str = True
        padding_max_length: int | None = None
        if self.padding_length_buckets is not None:
            longest = max(len(encoded["input_ids"]) for encoded in encoded_pairs)
            padding_max_length = next(
                bucket
                for bucket in self.padding_length_buckets
                if bucket >= longest
            )
            padding = "max_length"
        batch = self.tokenizer.pad(
            encoded_pairs,
            padding=padding,
            max_length=padding_max_length,
            return_tensors="pt",
        )

        if self.include_labels:
            labels = [pair.label for pair in pairs]
            if any(label is None for label in labels):
                if not all(label is None for label in labels):
                    raise ValueError("a batch cannot mix labeled and unlabeled pairs")
            else:
                batch["labels"] = torch.tensor(labels, dtype=torch.long)
                batch["sample_weights"] = torch.tensor(
                    [pair.sample_weight for pair in pairs],
                    dtype=torch.float32,
                )
        return batch


def infer_pair_max_length(
    tokenizer: PreTrainedTokenizerBase,
    pairs: Sequence[PreparedPair],
    *,
    quantile: float = 0.95,
    sample_size: int = 10_000,
    hard_cap: int = 512,
    use_field_tokens: bool = True,
    max_attribute_value_chars: int | None = None,
    max_attribute_value_tokens: int | None = DEFAULT_MAX_ATTRIBUTE_VALUE_TOKENS,
    profile: str = SEQUENCE_CLASSIFIER_PROFILE,
) -> int:
    """Infer a rounded token limit from complete structured pair lengths."""
    if not pairs:
        raise ValueError("pairs must not be empty")
    if not 0.0 < quantile <= 1.0:
        raise ValueError("quantile must be in (0, 1]")
    if sample_size < 1 or hard_cap < 8:
        raise ValueError("sample_size must be positive and hard_cap at least 8")
    if max_attribute_value_chars is not None and max_attribute_value_chars < 1:
        raise ValueError("max_attribute_value_chars must be positive or None")
    if max_attribute_value_tokens is not None and max_attribute_value_tokens < 1:
        raise ValueError("max_attribute_value_tokens must be positive or None")
    if use_field_tokens:
        _require_pair_special_tokens(tokenizer)

    if len(pairs) > sample_size:
        indices = np.linspace(0, len(pairs) - 1, sample_size, dtype=int)
        sample = [pairs[int(index)] for index in indices]
    else:
        sample = pairs

    profile = normalize_profile(profile)
    if profile in {PROMPTED_BINARY_RERANKER_PROFILE, QWEN3_RERANKER_PROFILE}:
        serialize = (
            _serialize_qwen3_pair_for_tokenizer
            if profile == QWEN3_RERANKER_PROFILE
            else _serialize_prompted_pair_for_tokenizer
        )
        lengths = np.fromiter(
            (
                len(
                    tokenizer.encode(
                        serialize(
                            tokenizer,
                            pair,
                            use_field_tokens=use_field_tokens,
                            max_attribute_value_chars=max_attribute_value_chars,
                            max_attribute_value_tokens=max_attribute_value_tokens,
                        ),
                        add_special_tokens=True,
                    )
                )
                for pair in sample
            ),
            dtype=np.int32,
        )
        selected = int(np.quantile(lengths, quantile, method="higher"))
        selected = max(8, int(math.ceil(selected / 8) * 8))
        model_limit = getattr(tokenizer, "model_max_length", hard_cap)
        if not isinstance(model_limit, int) or model_limit <= 0 or model_limit > 100_000:
            model_limit = hard_cap
        return min(selected, model_limit, hard_cap)

    special_token_count = _pair_special_token_count(tokenizer)
    lengths = np.fromiter(
        (
            _encode_card_sections(
                tokenizer,
                pair.left,
                use_field_tokens=use_field_tokens,
                max_attribute_value_chars=max_attribute_value_chars,
                max_attribute_value_tokens=max_attribute_value_tokens,
            )[2]
            + _encode_card_sections(
                tokenizer,
                pair.right,
                use_field_tokens=use_field_tokens,
                max_attribute_value_chars=max_attribute_value_chars,
                max_attribute_value_tokens=max_attribute_value_tokens,
            )[2]
            + special_token_count
            for pair in sample
        ),
        dtype=np.int32,
    )
    selected = int(np.quantile(lengths, quantile, method="higher"))
    selected = max(8, int(math.ceil(selected / 8) * 8))

    model_limit = getattr(tokenizer, "model_max_length", hard_cap)
    if not isinstance(model_limit, int) or model_limit <= 0 or model_limit > 100_000:
        model_limit = hard_cap
    return min(selected, model_limit, hard_cap)
