import unittest

import torch
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPast

from match.config import load_app_config_file
from match.models.transformer.construction import model_factory
from match.models.transformer.fp8 import quantize_module_weights_to_fp8
from match.models.transformer.profile import (
    QWEN3_RERANKER_PROFILE,
    TransformerArtifactContract,
)
from match.models.transformer.qwen3 import (
    Qwen3YesNoReranker,
    qwen3_score_token_ids,
)
from match.pair_encoding import PairEncodingCollator, serialize_qwen3_pair
from match.paths import PROJECT_ROOT
from match.prepare_data import PreparedCard, PreparedPair
from match.qwen3_pair_encoding import (
    QWEN3_RERANKER_INSTRUCTION,
    QWEN3_RERANKER_PREFIX,
    QWEN3_RERANKER_SUFFIX,
    encode_qwen3_pairs,
)


def _pair() -> PreparedPair:
    return PreparedPair(
        PreparedCard(1, "Red chair", "Furniture", (("color", "red"),)),
        PreparedCard(2, "Crimson chair", "Furniture", (("color", "crimson"),)),
        1,
        "Furniture",
    )


class _WhitespaceTokenizer:
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(self) -> None:
        self._ids: dict[str, int] = {}
        self._tokens: dict[int, str] = {}
        self.batch_encode_calls = 0

    def encode(self, text, *, add_special_tokens=False):
        del add_special_tokens
        values: list[int] = []
        for token in text.split():
            if token not in self._ids:
                identifier = len(self._ids) + 1
                self._ids[token] = identifier
                self._tokens[identifier] = token
            values.append(self._ids[token])
        return values

    def decode(self, values, **kwargs):
        del kwargs
        return " ".join(self._tokens[value] for value in values)

    def __call__(
        self,
        texts,
        *,
        add_special_tokens=False,
        padding=False,
        truncation=False,
    ):
        del padding, truncation
        self.batch_encode_calls += 1
        return {
            "input_ids": [
                self.encode(text, add_special_tokens=add_special_tokens)
                for text in texts
            ]
        }

    def batch_decode(self, values, **kwargs):
        return [self.decode(value, **kwargs) for value in values]

    def num_special_tokens_to_add(self, *, pair=False):
        del pair
        return 0

    def pad(self, rows, *, padding, max_length=None, return_tensors):
        del padding, return_tensors
        width = max_length or max(len(row["input_ids"]) for row in rows)
        return {
            name: torch.tensor(
                [row[name] + [0] * (width - len(row[name])) for row in rows]
            )
            for name in rows[0]
        }


class _Backbone(nn.Module):
    def __init__(self, embeddings: nn.Embedding) -> None:
        super().__init__()
        self.embeddings = embeddings

    def forward(self, input_ids, attention_mask, **kwargs):
        del attention_mask, kwargs
        return BaseModelOutputWithPast(last_hidden_state=self.embeddings(input_ids))


class _CausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        embeddings = nn.Embedding(16, 4)
        with torch.no_grad():
            embeddings.weight.copy_(torch.arange(64).reshape(16, 4))
        self.model = _Backbone(embeddings)
        self.lm_head = nn.Linear(4, 16, bias=False)
        self.lm_head.weight = embeddings.weight
        self.config = type("Config", (), {})()

    def get_output_embeddings(self):
        return self.lm_head


class Qwen3ProfileTests(unittest.TestCase):
    def test_fp8_weight_storage_preserves_small_model_outputs(self) -> None:
        model = nn.Sequential(nn.Embedding(16, 4), nn.Linear(4, 3, bias=False))
        inputs = torch.tensor([[1, 2], [3, 4]])
        expected = model(inputs).detach()

        result = quantize_module_weights_to_fp8(model)
        actual = model(inputs).detach()

        self.assertEqual(result.embedding_layers, 1)
        self.assertEqual(result.linear_layers, 1)
        self.assertGreater(result.fp8_weight_bytes, 0)
        torch.testing.assert_close(
            actual.float(),
            expected.float(),
            atol=0.08,
            rtol=0.08,
        )

    def test_yes_no_wrapper_returns_logit_margin(self) -> None:
        model = Qwen3YesNoReranker(
            _CausalLM(),
            no_token_id=2,
            yes_token_id=3,
        )
        result = model(
            input_ids=torch.tensor([[1, 2], [3, 4]]),
            attention_mask=torch.ones((2, 2), dtype=torch.long),
        )

        final_hidden = model.backbone.embeddings(
            torch.tensor([[1, 2], [3, 4]])
        )[:, -1, :]
        expected = (
            torch.nn.functional.linear(final_hidden, model.score_weight)[:, 1]
            - torch.nn.functional.linear(final_hidden, model.score_weight)[:, 0]
        )
        torch.testing.assert_close(result.logits[:, 0], expected)
        self.assertEqual(tuple(result.logits.shape), (2, 1))

    def test_score_labels_must_be_distinct_single_tokens(self) -> None:
        tokenizer = _WhitespaceTokenizer()

        no_token_id, yes_token_id = qwen3_score_token_ids(tokenizer)

        self.assertNotEqual(no_token_id, yes_token_id)

    def test_pipeline_selects_qwen3_zero_shot_artifact(self) -> None:
        config = load_app_config_file(PROJECT_ROOT / "configs" / "pipeline.yaml")
        transformer = config.model_description.transformer

        self.assertEqual(transformer.profile, QWEN3_RERANKER_PROFILE)
        self.assertEqual(
            transformer.pretrained_model_path,
            "models/qwen3-reranker-4b",
        )
        self.assertEqual(transformer.head.type, "native")
        self.assertFalse(transformer.pair_encoding.use_field_tokens)

    def test_serializes_official_qwen3_yes_no_prompt(self) -> None:
        prompt = serialize_qwen3_pair(_pair())

        self.assertEqual(
            prompt,
            QWEN3_RERANKER_PREFIX
            + f"<Instruct>: {QWEN3_RERANKER_INSTRUCTION}\n"
            + "<Query>: name: Red chair category: Furniture color: red\n"
            + "<Document>: name: Crimson chair category: Furniture color: crimson"
            + QWEN3_RERANKER_SUFFIX,
        )

    def test_collator_preserves_assistant_suffix_when_truncating(self) -> None:
        tokenizer = _WhitespaceTokenizer()
        suffix_ids = tokenizer.encode(
            QWEN3_RERANKER_SUFFIX,
            add_special_tokens=False,
        )
        batch = PairEncodingCollator(
            tokenizer,
            40,
            use_field_tokens=False,
            max_attribute_value_tokens=None,
            profile=QWEN3_RERANKER_PROFILE,
        )([_pair()])

        input_ids = batch["input_ids"][0].tolist()
        self.assertEqual(tokenizer.padding_side, "left")
        self.assertLessEqual(len(input_ids), 40)
        self.assertEqual(input_ids[-len(suffix_ids) :], suffix_ids)

    def test_batched_qwen_tokenization_matches_scalar_ids(self) -> None:
        tokenizer = _WhitespaceTokenizer()
        pairs = [
            _pair(),
            PreparedPair(
                PreparedCard(
                    3,
                    "Long product",
                    "Furniture",
                    (("description", "one two three four five six"),),
                ),
                PreparedCard(
                    4,
                    "Short product",
                    "Furniture",
                    (("description", "one two"),),
                ),
                0,
                "Furniture",
            ),
        ]
        scalar = encode_qwen3_pairs(
            tokenizer,
            pairs,
            max_length=40,
            use_field_tokens=False,
            max_attribute_value_chars=256,
            max_attribute_value_tokens=3,
        )

        batched = encode_qwen3_pairs(
            tokenizer,
            pairs,
            max_length=40,
            use_field_tokens=False,
            max_attribute_value_chars=256,
            max_attribute_value_tokens=3,
            batch_fields=True,
            field_chunk_size=2,
        )

        self.assertEqual(batched, scalar)
        # Two value chunks plus one body chunk for two pairs.
        self.assertEqual(tokenizer.batch_encode_calls, 3)

    def test_qwen_collator_uses_batched_tokenization_setting(self) -> None:
        tokenizer = _WhitespaceTokenizer()
        collator = PairEncodingCollator(
            tokenizer,
            40,
            use_field_tokens=False,
            max_attribute_value_tokens=3,
            batch_fields=True,
            field_chunk_size=2,
            profile=QWEN3_RERANKER_PROFILE,
        )

        collator([_pair(), _pair()])

        self.assertGreater(tokenizer.batch_encode_calls, 0)

    def test_contract_requires_one_logit(self) -> None:
        contract = TransformerArtifactContract.for_training(
            profile=QWEN3_RERANKER_PROFILE,
            head_type="native",
            num_logits=1,
        )

        self.assertTrue(contract.uses_prompted_pairs)
        self.assertFalse(contract.requires_trust_remote_code)

    def test_model_construction_waits_for_yes_no_wrapper(self) -> None:
        factory = model_factory(
            "qwen",
            _WhitespaceTokenizer(),
            use_field_tokens=False,
            profile=QWEN3_RERANKER_PROFILE,
            head_type="native",
        )

        with self.assertRaisesRegex(ValueError, "yes/no scoring wrapper"):
            factory()


if __name__ == "__main__":
    unittest.main()
