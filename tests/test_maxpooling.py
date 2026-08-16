import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import polars as pl

from match import encode_attribute_pairs, train_maxpooling_model


class MaxPoolingPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        directory = Path(self.temp_directory.name)
        self.data_path = directory / "items.parquet"
        self.train_matches_path = directory / "train_matches.parquet"
        self.inference_matches_path = directory / "inference_matches.parquet"

        pl.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "name": ["one", "two", "three", "four"],
                "attributes": [
                    json.dumps({"цвет": "черный", "вес": "10 кг"}, ensure_ascii=False),
                    json.dumps({"цвет": "черный", "вес": "11 кг"}, ensure_ascii=False),
                    json.dumps({"материал": "сталь", "длина": "20 см"}, ensure_ascii=False),
                    json.dumps({"материал": "пластик", "длина": "30 см"}, ensure_ascii=False),
                ],
                "category": ["a", "a", "b", "b"],
            }
        ).write_parquet(self.data_path)

        pl.DataFrame(
            {
                "id1": [1, 2, 1, 2, 3, 4, 3, 4],
                "id2": [2, 1, 1, 2, 4, 3, 3, 4],
                "target": [1, 1, 1, 1, 0, 0, 0, 0],
            }
        ).write_parquet(self.train_matches_path)

        pl.DataFrame({"id1": [1, 2], "id2": [2, 1]}).write_parquet(
            self.inference_matches_path
        )

    def tearDown(self) -> None:
        self.temp_directory.cleanup()

    def test_trains_pipeline_and_encodes_symmetric_pairs(self) -> None:
        model = train_maxpooling_model(
            self.data_path,
            self.train_matches_path,
            vector_size=4,
            fasttext_epochs=1,
            classifier_epochs=1,
            batch_size=4,
            workers=1,
            validation_fraction=0.25,
            random_state=7,
        )

        embeddings = encode_attribute_pairs(
            self.data_path,
            self.inference_matches_path,
            model,
        )

        self.assertEqual(model.vector_size, 4)
        self.assertTrue(np.isfinite(model.best_validation_pr_auc))
        self.assertEqual(embeddings.shape, (2, 8))
        self.assertEqual(embeddings.dtype, np.float32)
        self.assertTrue(np.isfinite(embeddings).all())
        np.testing.assert_allclose(embeddings[0], embeddings[1], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
