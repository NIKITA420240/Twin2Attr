import math
import unittest
from pathlib import Path

import polars as pl

from match.config import (
    DatasetSourceSettings,
    DatasetSplitterSettings,
    SampleWeightModelSettings,
)
from match.data_models.preparation import prepare_source_matches
from match.sample_weight_models import TransitivitySampleWeightModel


def _triangle(confidences=(1.0, 1.0, 1.0)) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id1": [1, 2, 1],
            "id2": [2, 3, 3],
            "target": [1, 1, 0],
            "_annotation_confidence": confidences,
        }
    )


class TransitivitySampleWeightTests(unittest.TestCase):
    def test_penalizes_every_edge_in_a_contradictory_triangle(self) -> None:
        model = TransitivitySampleWeightModel(
            penalty_strength=1.0,
            min_weight_multiplier=0.1,
            min_comparable_neighbors=1,
            confidence_weighted_violations=False,
        )

        result = model.apply(_triangle(), source_name="llm")

        self.assertEqual(
            result.get_column("transitivity_violations").to_list(),
            [1, 1, 1],
        )
        for value in result.get_column("weight_multiplier"):
            self.assertAlmostEqual(value, math.exp(-1.0), places=6)

    def test_requires_configured_number_of_comparable_neighbors(self) -> None:
        model = TransitivitySampleWeightModel(min_comparable_neighbors=2)

        result = model.apply(_triangle(), source_name="llm")

        self.assertEqual(set(result.get_column("weight_multiplier")), {1.0})
        self.assertEqual(
            result.get_column("transitivity_violations").to_list(),
            [1, 1, 1],
        )

    def test_confidence_penalizes_the_weaker_edge_more(self) -> None:
        model = TransitivitySampleWeightModel(
            min_comparable_neighbors=1,
            confidence_weighted_violations=True,
        )

        result = model.apply(
            _triangle((5 / 9, 1.0, 1.0)),
            source_name="llm",
        )
        multipliers = result.get_column("weight_multiplier").to_list()

        self.assertLess(multipliers[0], multipliers[1])
        self.assertLess(multipliers[0], multipliers[2])

    def test_source_weight_is_multiplied_after_vote_labeling(self) -> None:
        items = pl.DataFrame(
            {"id": [1, 2, 3], "category": ["a", "a", "a"]}
        )
        matches = pl.DataFrame(
            {
                "id1": [1, 2, 1],
                "id2": [2, 3, 3],
                "target": [8 / 9, 8 / 9, 1 / 9],
            }
        )
        source = DatasetSourceSettings(
            name="llm",
            matches=Path("matches.parquet"),
            weight=2.0,
            max_rows=None,
            sampling_strategy="random",
            splitter=DatasetSplitterSettings(
                splitter_type="binary",
                score_type="votes",
                total_votes=9,
                negative_threshold=2,
                positive_threshold=7,
                uncertain_action="drop",
            ),
            weight_model=SampleWeightModelSettings(
                type="transitivity",
                enabled=True,
                penalty_strength=1.0,
                min_weight_multiplier=0.1,
                min_comparable_neighbors=1,
                confidence_weighted_violations=False,
            ),
        )

        result = prepare_source_matches(items, matches, source, seed=42)

        for value in result.get_column("sample_weight"):
            self.assertAlmostEqual(value, 2.0 * math.exp(-1.0), places=6)
        self.assertIn("weight_multiplier", result.columns)
        self.assertIn("transitivity_violations", result.columns)


if __name__ == "__main__":
    unittest.main()
