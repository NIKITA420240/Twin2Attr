import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import torch
from transformers import AutoTokenizer

from match.models.transformer.config import SequenceClassifierConfig
from match.models.transformer.token_cache import (
    load_or_build_sharded_token_cache,
    load_or_build_token_cache,
)
from match.models.transformer.training import _prepare_pair_datasets
from match.pair_encoding import PairEncodingCollator, PreparedPairDataset
from match.paths import PROJECT_ROOT
from match.prepare_data import PreparedCard, PreparedPair


def _pairs() -> list[PreparedPair]:
    return [
        PreparedPair(
            PreparedCard(1, "left product", "category", (("color", "black"),)),
            PreparedCard(2, "right product", "category", (("color", "white"),)),
            0,
            "category",
            sample_weight=2.0,
            training_target=0.25,
        ),
        PreparedPair(
            PreparedCard(3, "another left", "category", ()),
            PreparedCard(4, "another right", "category", ()),
            1,
            "category",
        ),
    ]


class TransformerTokenCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tokenizer = AutoTokenizer.from_pretrained(
            PROJECT_ROOT / "weights" / "rubert-tiny2"
        )

    def _collator(self) -> PairEncodingCollator:
        return PairEncodingCollator(
            self.tokenizer,
            32,
            use_field_tokens=False,
            padding_length_buckets=(16, 32),
        )

    def test_cached_batch_matches_direct_tokenization_and_reuses_files(self) -> None:
        pairs = _pairs()
        collator = self._collator()
        with tempfile.TemporaryDirectory() as directory:
            cached = load_or_build_token_cache(
                pairs,
                self.tokenizer,
                collator,
                directory,
                split_name="train",
                chunk_size=1,
            )
            direct_batch = collator(pairs)
            cached_batch = collator([cached[0], cached[1]])
            self.assertEqual(set(direct_batch), set(cached_batch))
            for name in direct_batch:
                torch.testing.assert_close(direct_batch[name], cached_batch[name])

            with patch.object(
                collator,
                "encode_pairs",
                side_effect=AssertionError("cache should be reused"),
            ):
                reused = load_or_build_token_cache(
                    pairs,
                    self.tokenizer,
                    collator,
                    directory,
                    split_name="train",
                    chunk_size=1,
                )
            self.assertEqual(reused[0].inputs, cached[0].inputs)

    def test_dynamic_augmentation_bypasses_only_train_cache(self) -> None:
        pairs = _pairs()
        config = SequenceClassifierConfig(
            model_path="model",
            max_length=32,
            token_cache_enabled=True,
            token_cache_directory=Path("cache"),
        )
        transform = Mock(side_effect=lambda pair: pair)
        validation_cache = object()
        with patch(
            "match.models.transformer.training.load_or_build_sharded_token_cache",
            return_value=validation_cache,
        ) as load_cache:
            train_dataset, validation_dataset = _prepare_pair_datasets(
                pairs,
                pairs,
                self.tokenizer,
                self._collator(),
                config,
                train_pair_transform=transform,
            )

        self.assertIsInstance(train_dataset, PreparedPairDataset)
        self.assertIs(validation_dataset, validation_cache)
        self.assertEqual(load_cache.call_count, 1)
        self.assertEqual(load_cache.call_args.kwargs["split_name"], "validation")

    def test_reuses_unchanged_source_shard_when_another_source_changes(self) -> None:
        pairs = _pairs()
        collator = self._collator()
        with tempfile.TemporaryDirectory() as directory:
            load_or_build_sharded_token_cache(
                pairs,
                ("human", "llm"),
                self.tokenizer,
                collator,
                directory,
                split_name="train",
                chunk_size=8,
            )
            changed_pairs = [
                pairs[0],
                replace(
                    pairs[1],
                    left=replace(pairs[1].left, name="changed LLM product"),
                ),
            ]
            with patch.object(
                collator,
                "encode_pairs",
                wraps=collator.encode_pairs,
            ) as encode:
                cached = load_or_build_sharded_token_cache(
                    changed_pairs,
                    ("human", "llm"),
                    self.tokenizer,
                    collator,
                    directory,
                    split_name="train",
                    chunk_size=8,
                )

            encode.assert_called_once()
            self.assertEqual(encode.call_args.args[0], [changed_pairs[1]])
            self.assertEqual(len(cached), 2)


if __name__ == "__main__":
    unittest.main()
