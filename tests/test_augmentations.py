import unittest
from dataclasses import replace

from match.augmentations import apply_attribute_shuffle, apply_pair_augmentation
from match.config import AttributeShuffleSettings, load_app_config_file
from match.paths import PROJECT_ROOT
from match.prepare_data import PreparedCard, PreparedPair


def _pair() -> PreparedPair:
    left = PreparedCard(
        1,
        "left",
        "category",
        tuple((f"left_{index}", str(index)) for index in range(8)),
    )
    right = PreparedCard(
        2,
        "right",
        "category",
        tuple((f"right_{index}", str(index)) for index in range(8)),
    )
    return PreparedPair(left, right, 1, "category")


class AttributeShuffleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.settings = AttributeShuffleSettings(
            type="attribute_shuffle",
            shuffled_copies=2,
            keep_original=False,
            seed=42,
            shuffle_cards_independently=True,
            skip_oversized=True,
        )

    def test_shuffle_is_deterministic_and_retains_source_alignment(self) -> None:
        pair = _pair()

        first = apply_attribute_shuffle([pair], self.settings)
        second = apply_attribute_shuffle([pair], self.settings)

        self.assertEqual(first, second)
        self.assertEqual(first.source_indices, (0, 0))
        self.assertEqual(len(first.pairs), 2)
        self.assertNotEqual(first.pairs[0].left.attributes, pair.left.attributes)
        self.assertNotEqual(
            first.pairs[0].left.attributes,
            first.pairs[1].left.attributes,
        )
        self.assertTrue(first.pairs[0].preserve_attribute_order)
        self.assertTrue(first.pairs[0].skip_oversized_attributes)

    def test_null_model_is_an_exact_noop(self) -> None:
        pair = _pair()
        config = load_app_config_file(PROJECT_ROOT / "configs" / "pipeline.yaml")

        result = apply_pair_augmentation([pair], config, model_name=None)

        self.assertEqual(result.pairs, (pair,))
        self.assertEqual(result.source_indices, (0,))

    def test_keep_original_adds_unmodified_pair(self) -> None:
        pair = _pair()

        result = apply_attribute_shuffle(
            [pair],
            replace(self.settings, shuffled_copies=1, keep_original=True),
        )

        self.assertEqual(result.pairs[0], pair)
        self.assertFalse(result.pairs[0].preserve_attribute_order)
        self.assertTrue(result.pairs[1].preserve_attribute_order)


if __name__ == "__main__":
    unittest.main()
