import unittest

from match.mxbai_pair_encoding import (
    MXBAI_RERANKER_SUFFIX,
    encode_mxbai_pairs,
    serialize_mxbai_pair,
)
from match.prepare_data import PreparedCard, PreparedPair


def _pair(trailing_whitespace: str) -> PreparedPair:
    return PreparedPair(
        PreparedCard(1, "Red chair", "Furniture", (("color", "red"),)),
        PreparedCard(
            2,
            "Crimson chair",
            "Furniture",
            (("color", "crimson" + trailing_whitespace),),
        ),
        1,
        "Furniture",
    )


class _ContextualSuffixTokenizer:
    """Minimal tokenizer that models Qwen's whitespace/newline BPE merge."""

    suffix_ids = [900, 901, 902]
    contextual_first_suffix_id = 800

    def encode(self, text, *, add_special_tokens=False):
        del add_special_tokens
        if text == MXBAI_RERANKER_SUFFIX:
            return list(self.suffix_ids)
        return self._encode_prompt(text)

    def __call__(
        self,
        texts,
        *,
        add_special_tokens=False,
        padding=False,
        truncation=False,
    ):
        del add_special_tokens, padding, truncation
        return {"input_ids": [self._encode_prompt(text) for text in texts]}

    def _encode_prompt(self, text: str) -> list[int]:
        if not text.endswith(MXBAI_RERANKER_SUFFIX):
            return [100 + index for index, _ in enumerate(text.split(), start=1)]
        body = text[: -len(MXBAI_RERANKER_SUFFIX)]
        body_ids = [100 + index for index, _ in enumerate(body.split(), start=1)]
        suffix_ids = list(self.suffix_ids)
        if body and body[-1].isspace():
            suffix_ids[0] = self.contextual_first_suffix_id
        return body_ids + suffix_ids


class MixedbreadProfileTests(unittest.TestCase):
    def test_contextual_suffix_is_preserved_for_trailing_whitespace(self) -> None:
        tokenizer = _ContextualSuffixTokenizer()

        for trailing_whitespace in (" ", "\t", "\n"):
            with self.subTest(trailing_whitespace=repr(trailing_whitespace)):
                pair = _pair(trailing_whitespace)
                full_ids = tokenizer.encode(
                    serialize_mxbai_pair(pair),
                    add_special_tokens=False,
                )
                encoded = encode_mxbai_pairs(
                    tokenizer,
                    [pair],
                    max_length=8,
                    use_field_tokens=False,
                    max_attribute_value_chars=None,
                    max_attribute_value_tokens=None,
                )[0]["input_ids"]

                self.assertEqual(len(encoded), 8)
                self.assertEqual(encoded[-3:], full_ids[-3:])
                self.assertEqual(
                    encoded[-3],
                    tokenizer.contextual_first_suffix_id,
                )
                self.assertEqual(encoded[:5], full_ids[:5])

    def test_untruncated_contextual_prompt_is_unchanged(self) -> None:
        tokenizer = _ContextualSuffixTokenizer()
        pair = _pair(" ")
        full_ids = tokenizer.encode(
            serialize_mxbai_pair(pair),
            add_special_tokens=False,
        )

        encoded = encode_mxbai_pairs(
            tokenizer,
            [pair],
            max_length=len(full_ids) + 1,
            use_field_tokens=False,
            max_attribute_value_chars=None,
            max_attribute_value_tokens=None,
        )[0]["input_ids"]

        self.assertEqual(encoded, full_ids)


if __name__ == "__main__":
    unittest.main()
