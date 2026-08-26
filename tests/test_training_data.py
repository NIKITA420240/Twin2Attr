import json
import unittest

import polars as pl

from match.data import prepare_training_data


class TrainingDataTests(unittest.TestCase):
    def test_prepares_aligned_train_and_validation_pairs(self) -> None:
        items = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["one", "two", "three"],
                "category": ["a", "a", "b"],
                "normalized_attributes": [
                    json.dumps({"color": "black"}),
                    json.dumps({"color": "white"}),
                    json.dumps({"material": "steel"}),
                ],
            }
        )
        train_matches = pl.DataFrame(
            {
                "id1": [1, 2],
                "id2": [2, 1],
                "target": [1, 1],
                "sample_weight": [3.0, 1.0],
            }
        )
        validation_matches = pl.DataFrame(
            {"id1": [1], "id2": [3], "target": [0]}
        )

        data = prepare_training_data(
            items,
            train_matches,
            validation_matches,
            attributes_column="normalized_attributes",
        )

        self.assertIs(data.items, items)
        self.assertIs(data.train_matches, train_matches)
        self.assertIs(data.validation_matches, validation_matches)
        self.assertEqual(data.attributes_column, "normalized_attributes")
        self.assertEqual([pair.label for pair in data.train_pairs], [1, 1])
        self.assertEqual(
            [pair.sample_weight for pair in data.train_pairs],
            [3.0, 1.0],
        )
        self.assertEqual([pair.label for pair in data.validation_pairs], [0])
        self.assertEqual(data.validation_pairs[0].sample_weight, 1.0)
        self.assertEqual(data.train_pairs[0].left.item_id, 1)
        self.assertEqual(data.validation_pairs[0].right.item_id, 3)

    def test_keeps_stacking_pairs_separate_and_aligned(self) -> None:
        items = pl.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "name": ["one", "two", "three", "four"],
                "category": ["a", "a", "b", "b"],
                "attributes": ["{}", "{}", "{}", "{}"],
            }
        )
        train = pl.DataFrame({"id1": [1], "id2": [2], "target": [1]})
        stacking = pl.DataFrame({"id1": [2], "id2": [3], "target": [0]})
        validation = pl.DataFrame({"id1": [3], "id2": [4], "target": [1]})

        data = prepare_training_data(
            items,
            train,
            validation,
            attributes_column="attributes",
            stacking_matches=stacking,
        )

        self.assertIs(data.stacking_matches, stacking)
        self.assertEqual([pair.left.item_id for pair in data.train_pairs], [1])
        self.assertEqual([pair.left.item_id for pair in data.stacking_pairs], [2])
        self.assertEqual([pair.left.item_id for pair in data.validation_pairs], [3])


if __name__ == "__main__":
    unittest.main()
