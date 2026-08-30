import unittest

import numpy as np

from match.pair_features import (
    TypedAttributeComparator,
    TypedAttributeOptions,
    aggregate_typed_attribute_features,
)
from match.prepare_data import PreparedCard, PreparedPair
from match.models.transformer.typed_fusion import build_typed_feature_matrix


def _pair(
    left_attributes: tuple[tuple[str, str], ...],
    right_attributes: tuple[tuple[str, str], ...],
) -> PreparedPair:
    return PreparedPair(
        left=PreparedCard(1, "left", "category", left_attributes),
        right=PreparedCard(2, "right", "category", right_attributes),
        label=None,
        category="category",
    )


class TypedAttributeFeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.options = TypedAttributeOptions(enabled=True)
        self.comparator = TypedAttributeComparator(self.options)

    def test_detects_types_and_uses_type_specific_metrics(self) -> None:
        comparisons = self.comparator.compare_pair(
            _pair(
                (
                    ("Код модели", "AB-123"),
                    ("Вес, г", "1000"),
                    ("Цвет", "красный; черный"),
                    ("Описание", "защитный чехол"),
                ),
                (
                    ("код модели", "ab123"),
                    ("вес, г", "1000"),
                    ("цвет", "красный"),
                    ("описание", "чехол защитный"),
                ),
            )
        )
        by_key = {item.key: item for item in comparisons}

        self.assertEqual(by_key["код модели"].semantic_type, "CODE")
        self.assertEqual(by_key["код модели"].exact, 1.0)
        self.assertEqual(by_key["код модели"].metrics["digit_exact"], 1.0)
        self.assertEqual(by_key["вес, г"].semantic_type, "PHYSICAL")
        self.assertEqual(by_key["вес, г"].metrics["relative_difference"], 0.0)
        self.assertEqual(by_key["цвет"].semantic_type, "SET")
        self.assertAlmostEqual(by_key["цвет"].metrics["jaccard"], 0.5)
        self.assertEqual(by_key["цвет"].metrics["containment"], 1.0)
        self.assertEqual(by_key["описание"].semantic_type, "TEXT")
        self.assertGreater(by_key["описание"].similarity, 0.5)

    def test_missing_value_preserves_semantic_type(self) -> None:
        comparison = self.comparator.compare_pair(
            _pair((("Цвет", "красный"),), ())
        )[0]

        self.assertEqual(comparison.semantic_type, "SET")
        self.assertFalse(comparison.both_present)
        self.assertTrue(comparison.one_missing)
        self.assertEqual(comparison.conflict, 0.0)

    def test_comparisons_and_aggregates_are_pair_order_invariant(self) -> None:
        pair = _pair(
            (("Артикул", "AB-123"), ("Цвет", "красный; черный")),
            (("Артикул", "AB124"), ("Цвет", "красный")),
        )
        reversed_pair = PreparedPair(
            left=pair.right,
            right=pair.left,
            label=None,
            category=pair.category,
        )

        direct = aggregate_typed_attribute_features(
            self.comparator.compare_pair(pair),
            enabled_types=self.options.enabled_types,
        )
        reverse = aggregate_typed_attribute_features(
            self.comparator.compare_pair(reversed_pair),
            enabled_types=self.options.enabled_types,
        )

        self.assertEqual(direct, reverse)
        self.assertEqual(direct["typed_model_code_conflict"], 1.0)
        self.assertTrue(np.isfinite(np.asarray(list(direct.values()))).all())

    def test_disabled_types_are_absent_from_comparisons_and_schema(self) -> None:
        options = TypedAttributeOptions(
            enabled=True,
            code=True,
            physical=False,
            numeric=False,
            set=False,
            text=False,
        )
        comparator = TypedAttributeComparator(options)
        comparisons = comparator.compare_pair(
            _pair((("Артикул", "A-1"), ("Цвет", "красный")), (("Артикул", "A1"),))
        )
        features = aggregate_typed_attribute_features(
            comparisons,
            enabled_types=options.enabled_types,
        )

        self.assertEqual({item.semantic_type for item in comparisons}, {"CODE"})
        self.assertIn("typed_code_comparisons", features)
        self.assertNotIn("typed_set_comparisons", features)

    def test_options_manifest_round_trip(self) -> None:
        restored = TypedAttributeOptions.from_dict(self.options.to_dict())
        self.assertEqual(restored, self.options)

    def test_transformer_matrix_has_stable_finite_schema(self) -> None:
        pair = _pair(
            (("Артикул", "AB-123"), ("Вес, г", "1000")),
            (("Артикул", "AB123"), ("Вес, кг", "1")),
        )

        names, matrix = build_typed_feature_matrix([pair, pair], self.options)

        self.assertEqual(len(names), 52)
        self.assertEqual(matrix.shape, (2, 52))
        self.assertEqual(matrix.dtype, np.float32)
        self.assertTrue(np.isfinite(matrix).all())
        np.testing.assert_array_equal(matrix[0], matrix[1])


if __name__ == "__main__":
    unittest.main()
