import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from match.models.transformer.model import WeightedSequenceTrainer
from match.prepare_data import PreparedCard, PreparedPair
from match.text_augmentation import (
    ProductTextAugmenter,
    TextAugmentationConfig,
)


def _pair() -> PreparedPair:
    return PreparedPair(
        left=PreparedCard(
            item_id=1,
            name="красный игровой телефон",
            category="Электроника",
            attributes=(
                ("цвет", "ярко красный"),
                ("материал", "алюминий металл"),
            ),
        ),
        right=PreparedCard(
            item_id=2,
            name="black gaming phone",
            category="Электроника",
            attributes=(("color", "deep black"),),
        ),
        label=1,
        category="Электроника",
        sample_weight=2.0,
    )


def _config(**overrides: object) -> TextAugmentationConfig:
    values = {
        "enabled": True,
        "alpha": 0.7,
        "attribute_dropout_probability": 0.0,
        "word_shuffle_probability": 0.0,
        "keyboard_typo_probability": 0.0,
        "word_dropout_probability": 0.0,
    }
    values.update(overrides)
    return TextAugmentationConfig(**values)


class ProductTextAugmenterTests(unittest.TestCase):
    def test_attribute_dropout_removes_complete_attributes_only(self) -> None:
        original = _pair()
        augmented = ProductTextAugmenter(
            _config(attribute_dropout_probability=1.0),
            seed=7,
        ).augment_pair(original)

        self.assertEqual(augmented.left.attributes, ())
        self.assertEqual(augmented.right.attributes, ())
        self.assertEqual(augmented.left.name, original.left.name)
        self.assertEqual(augmented.left.category, original.left.category)
        self.assertEqual(augmented.label, original.label)
        self.assertEqual(augmented.sample_weight, original.sample_weight)

    def test_word_order_shuffle_preserves_words_and_schema(self) -> None:
        original = _pair()
        augmented = ProductTextAugmenter(
            _config(word_shuffle_probability=1.0),
            seed=11,
        ).augment_pair(original)

        self.assertNotEqual(augmented.left.name, original.left.name)
        self.assertCountEqual(
            augmented.left.name.split(),
            original.left.name.split(),
        )
        self.assertEqual(
            [key for key, _ in augmented.left.attributes],
            [key for key, _ in original.left.attributes],
        )
        self.assertCountEqual(
            augmented.left.attributes[0][1].split(),
            original.left.attributes[0][1].split(),
        )

    def test_keyboard_typos_support_russian_and_latin_layouts(self) -> None:
        original = _pair()
        augmented = ProductTextAugmenter(
            _config(keyboard_typo_probability=1.0),
            seed=19,
        ).augment_pair(original)

        self.assertNotEqual(augmented.left.name, original.left.name)
        self.assertNotEqual(augmented.right.name, original.right.name)
        self.assertEqual(len(augmented.left.name), len(original.left.name))
        self.assertEqual(len(augmented.right.name), len(original.right.name))
        self.assertEqual(augmented.left.category, original.left.category)
        self.assertEqual(augmented.left.attributes[0][0], "цвет")

    def test_seed_makes_augmentation_reproducible(self) -> None:
        config = _config(
            attribute_dropout_probability=0.25,
            word_shuffle_probability=0.5,
            keyboard_typo_probability=0.2,
            word_dropout_probability=0.2,
        )

        first = ProductTextAugmenter(config, seed=42).augment_pair(_pair())
        second = ProductTextAugmenter(config, seed=42).augment_pair(_pair())

        self.assertEqual(first, second)


class _TwoPassModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.original_logits = nn.Parameter(torch.tensor([[2.0, -1.0], [0.5, 1.5]]))
        self.augmented_logits = nn.Parameter(torch.tensor([[0.2, 0.8], [1.0, -0.5]]))
        self.call_count = 0

    def forward(self, input_ids: torch.Tensor) -> SimpleNamespace:
        self.call_count += 1
        logits = (
            self.original_logits if int(input_ids[0, 0]) == 1 else self.augmented_logits
        )
        return SimpleNamespace(logits=logits)


class AugmentedLossTests(unittest.TestCase):
    def test_combines_original_and_augmented_weighted_losses(self) -> None:
        alpha = 0.25
        labels = torch.tensor([0, 1])
        sample_weights = torch.tensor([1.0, 3.0])
        class_weights = torch.tensor([0.75, 1.25])
        model = _TwoPassModel()
        trainer = object.__new__(WeightedSequenceTrainer)
        trainer.class_weights = class_weights
        trainer.augmentation_alpha = alpha
        inputs = {
            "input_ids": torch.tensor([[1], [1]]),
            "augmented_input_ids": torch.tensor([[2], [2]]),
            "labels": labels,
            "sample_weights": sample_weights,
        }
        expected_original = (
            torch.sum(
                F.cross_entropy(
                    model.original_logits,
                    labels,
                    weight=class_weights,
                    reduction="none",
                )
                * sample_weights
            )
            / sample_weights.sum()
        )
        expected_augmented = (
            torch.sum(
                F.cross_entropy(
                    model.augmented_logits,
                    labels,
                    weight=class_weights,
                    reduction="none",
                )
                * sample_weights
            )
            / sample_weights.sum()
        )

        loss = trainer.compute_loss(model, inputs)

        torch.testing.assert_close(
            loss,
            alpha * expected_original + (1.0 - alpha) * expected_augmented,
        )
        self.assertEqual(model.call_count, 2)
        loss.backward()
        self.assertIsNotNone(model.original_logits.grad)
        self.assertIsNotNone(model.augmented_logits.grad)


if __name__ == "__main__":
    unittest.main()
