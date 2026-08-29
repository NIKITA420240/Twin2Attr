import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl

from match.models.boosting.features import BoostingFeatureBuilder
from match.models.boosting.serialization import save_boosting_model
from match.models.cascade.predictor import CascadePredictor
from match.models.contracts import MatchPredictor, PredictionBatch
from match.models.stacking.features import (
    TRANSFORMER_LOGIT_MARGIN,
    build_stacking_features,
)
from match.models.stacking.predictor import StackingPredictor
from match.pair_features import TypedAttributeOptions


def _batch() -> PredictionBatch:
    items = pl.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "name": ["Phone X100", "Phone X100", "Case X100", "Phone Z900"],
            "category": ["phones", "phones", "accessories", "phones"],
            "attributes": [
                '{"brand":"Acme","article":"X100"}',
                '{"brand":"Acme","article":"X100"}',
                '{"brand":"Acme","article":"X100"}',
                '{"brand":"Other","article":"Z900"}',
            ],
        }
    )
    matches = pl.DataFrame({"id1": [1, 1, 1], "id2": [2, 3, 4]})
    return PredictionBatch(items, matches, "attributes")


class _FixedPredictor:
    def __init__(self, probabilities: list[float]) -> None:
        self.probabilities = np.asarray(probabilities, dtype=np.float32)
        self.received_rows: list[int] = []

    def predict_proba(self, batch: PredictionBatch) -> np.ndarray:
        self.received_rows.append(batch.matches.height)
        return self.probabilities.copy()


class _FixedMarginPredictor:
    def predict_logit_margin(self, batch: PredictionBatch) -> np.ndarray:
        return np.arange(batch.matches.height, dtype=np.float32) - 1.0


class _FixedCatBoost:
    def predict_proba(self, features, *, thread_count: int):
        del thread_count
        margins = features[TRANSFORMER_LOGIT_MARGIN].to_numpy()
        probabilities = 1.0 / (1.0 + np.exp(-margins))
        return np.column_stack((1.0 - probabilities, probabilities))


class _SavingCatBoost:
    def save_model(self, path: str) -> None:
        Path(path).write_text("model", encoding="utf-8")


class BoostingAndCascadeTests(unittest.TestCase):
    def test_boosting_features_are_symmetric_and_schema_is_stable(self) -> None:
        batch = _batch().take_indices([1])
        reverse_matches = batch.matches.select(
            pl.col("id2").alias("id1"),
            pl.col("id1").alias("id2"),
        )
        reverse = PredictionBatch(batch.items, reverse_matches, "attributes")

        features = BoostingFeatureBuilder().transform(batch)
        reverse_features = BoostingFeatureBuilder().transform(reverse)

        self.assertEqual(features.shape, (1, 85))
        self.assertEqual(list(features.columns), list(reverse_features.columns))
        self.assertTrue(features.equals(reverse_features))
        self.assertEqual(features.columns[0], "category")

    def test_typed_boosting_features_extend_schema_and_remain_symmetric(self) -> None:
        batch = _batch().take_indices([1])
        reverse_matches = batch.matches.select(
            pl.col("id2").alias("id1"),
            pl.col("id1").alias("id2"),
        )
        reverse = PredictionBatch(batch.items, reverse_matches, "attributes")
        builder = BoostingFeatureBuilder(TypedAttributeOptions(enabled=True))

        features = builder.transform(batch)
        reverse_features = builder.transform(reverse)

        self.assertEqual(features.shape, (1, 137))
        self.assertEqual(list(features.columns), list(reverse_features.columns))
        self.assertTrue(features.equals(reverse_features))
        self.assertIn("typed_code_exact_matches", features.columns)
        self.assertIn("typed_attribute_one_missing", features.columns)

        restored = BoostingFeatureBuilder.from_feature_options(
            builder.feature_options
        )
        self.assertEqual(restored.feature_options, builder.feature_options)
        self.assertTrue(restored.transform(batch).equals(features))

    def test_boosting_manifest_persists_typed_feature_options(self) -> None:
        builder = BoostingFeatureBuilder(TypedAttributeOptions(enabled=True))
        with tempfile.TemporaryDirectory() as directory:
            output = save_boosting_model(
                _SavingCatBoost(),
                directory,
                ["category", "typed_attribute_comparisons"],
                feature_options=builder.feature_options,
            )
            manifest = json.loads(
                (output / "manifest.json").read_text(encoding="utf-8")
            )

        self.assertEqual(manifest["feature_schema_version"], 2)
        self.assertEqual(manifest["feature_options"], builder.feature_options)

    def test_cascade_routes_only_uncertain_rows_and_restores_order(self) -> None:
        fast = _FixedPredictor([0.001, 0.5, 0.999])
        main = _FixedPredictor([0.8])
        predictor = CascadePredictor(
            fast,
            main,
            negative_threshold=0.01,
            positive_threshold=0.99,
        )

        result = predictor.predict_proba(_batch())

        np.testing.assert_allclose(result, [0.001, 0.8, 0.999])
        self.assertEqual(fast.received_rows, [3])
        self.assertEqual(main.received_rows, [1])
        self.assertIsInstance(predictor, MatchPredictor)

    def test_cascade_skips_main_model_when_every_score_is_confident(self) -> None:
        fast = _FixedPredictor([0.001, 0.999, 0.0])
        main = _FixedPredictor([])
        predictor = CascadePredictor(
            fast,
            main,
            negative_threshold=0.01,
            positive_threshold=0.99,
        )

        result = predictor.predict_proba(_batch())

        np.testing.assert_allclose(result, fast.probabilities)
        self.assertEqual(main.received_rows, [])

    def test_stacking_appends_transformer_margin_to_base_features(self) -> None:
        batch = _batch()
        margins = np.array([-2.0, 0.0, 3.0], dtype=np.float32)

        features = build_stacking_features(batch, margins)

        self.assertEqual(features.shape, (3, 86))
        self.assertEqual(features.columns[-1], TRANSFORMER_LOGIT_MARGIN)
        np.testing.assert_array_equal(
            features[TRANSFORMER_LOGIT_MARGIN].to_numpy(),
            margins,
        )

    def test_stacking_predictor_uses_margin_and_returns_probabilities(self) -> None:
        batch = _batch()
        feature_names = list(
            build_stacking_features(
                batch,
                np.zeros(batch.matches.height, dtype=np.float32),
            ).columns
        )
        predictor = StackingPredictor(
            _FixedCatBoost(),
            feature_names,
            transformer=_FixedMarginPredictor(),
        )

        probabilities = predictor.predict_proba(batch)

        expected = 1.0 / (1.0 + np.exp(-np.array([-1.0, 0.0, 1.0])))
        np.testing.assert_allclose(probabilities, expected, rtol=1e-6)
        self.assertIsInstance(predictor, MatchPredictor)


if __name__ == "__main__":
    unittest.main()
