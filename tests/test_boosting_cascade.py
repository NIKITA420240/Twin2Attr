import unittest

import numpy as np
import polars as pl

from match.models.boosting.features import BoostingFeatureBuilder
from match.models.cascade.predictor import CascadePredictor
from match.models.contracts import MatchPredictor, PredictionBatch


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


if __name__ == "__main__":
    unittest.main()
