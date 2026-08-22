import json
import unittest

import numpy as np
import polars as pl

from match import (
    encode_attribute_pairs,
    predict_maxpooling_probabilities,
    train_maxpooling_model,
)


class MaxPoolingPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.items = pl.DataFrame(
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
        )

        self.train_matches = pl.DataFrame(
            {
                "id1": [1, 2, 1, 2, 3, 4, 3, 4],
                "id2": [2, 1, 1, 2, 4, 3, 3, 4],
                "target": [1, 1, 1, 1, 0, 0, 0, 0],
            }
        )

        self.inference_matches = pl.DataFrame(
            {"id1": [1, 2], "id2": [2, 1]}
        )

    def test_trains_pipeline_and_encodes_symmetric_pairs(self) -> None:
        items = self.items.rename({"attributes": "normalized_attributes"})
        model = train_maxpooling_model(
            items,
            self.train_matches,
            attributes_column="normalized_attributes",
            vector_size=4,
            fasttext_epochs=1,
            classifier_epochs=1,
            batch_size=4,
            workers=1,
            validation_fraction=0.25,
            random_state=7,
        )

        embeddings = encode_attribute_pairs(
            items,
            self.inference_matches,
            model,
            attributes_column="normalized_attributes",
        )
        probabilities = predict_maxpooling_probabilities(
            items,
            self.inference_matches,
            model,
            attributes_column="normalized_attributes",
            batch_size=2,
        )

        self.assertEqual(model.vector_size, 4)
        self.assertTrue(np.isfinite(model.best_validation_pr_auc))
        self.assertEqual(embeddings.shape, (2, 8))
        self.assertEqual(embeddings.dtype, np.float32)
        self.assertTrue(np.isfinite(embeddings).all())
        np.testing.assert_allclose(embeddings[0], embeddings[1], atol=1e-6)
        self.assertEqual(probabilities.shape, (2,))
        self.assertTrue(np.isfinite(probabilities).all())


if __name__ == "__main__":
    unittest.main()
