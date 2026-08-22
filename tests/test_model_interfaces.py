import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import polars as pl

from match.models import (
    FusionPredictor,
    MatchPredictor,
    MaxPoolingPredictor,
    PairEncoder,
    PredictionBatch,
    TransformerPredictor,
)


def _batch() -> PredictionBatch:
    return PredictionBatch(
        items=pl.DataFrame({"id": [1], "attributes": ["{}"]}),
        matches=pl.DataFrame({"id1": [1], "id2": [1]}),
        pairs=[object()],
        attributes_column="attributes",
    )


class ModelInterfaceTests(unittest.TestCase):
    def test_transformer_supports_encoding_and_prediction(self) -> None:
        model = SimpleNamespace(config=SimpleNamespace(hidden_size=16))
        predictor = TransformerPredictor(object(), model, batch_size=8)
        encoded = np.ones((1, 16), dtype=np.float32)
        probabilities = np.array([0.8], dtype=np.float32)

        with (
            patch("match.transformer.encode_pair_cls", return_value=encoded),
            patch(
                "match.transformer.predict_match_probabilities",
                return_value=probabilities,
            ),
        ):
            self.assertIs(predictor.encode(_batch()), encoded)
            self.assertIs(predictor.predict_proba(_batch()), probabilities)

        self.assertIsInstance(predictor, PairEncoder)
        self.assertIsInstance(predictor, MatchPredictor)
        self.assertEqual(predictor.output_dim, 16)

    def test_maxpooling_supports_encoding_and_prediction(self) -> None:
        predictor = MaxPoolingPredictor(
            SimpleNamespace(vector_size=4),
            batch_size=8,
        )
        encoded = np.ones((1, 8), dtype=np.float32)
        probabilities = np.array([0.7], dtype=np.float32)

        with (
            patch("match.maxpooling.encode_attribute_pairs", return_value=encoded),
            patch(
                "match.maxpooling.predict_maxpooling_probabilities",
                return_value=probabilities,
            ),
        ):
            self.assertIs(predictor.encode(_batch()), encoded)
            self.assertIs(predictor.predict_proba(_batch()), probabilities)

        self.assertIsInstance(predictor, PairEncoder)
        self.assertIsInstance(predictor, MatchPredictor)
        self.assertEqual(predictor.output_dim, 8)

    def test_fusion_composes_two_encoders(self) -> None:
        transformer = Mock(spec=PairEncoder)
        maxpooling = Mock(spec=PairEncoder)
        transformer.encode.return_value = np.ones((1, 16), dtype=np.float32)
        maxpooling.encode.return_value = np.ones((1, 8), dtype=np.float32)
        predictor = FusionPredictor(
            model=object(),
            transformer=transformer,
            maxpooling=maxpooling,
        )
        probabilities = np.array([0.9], dtype=np.float32)

        with patch(
            "match.fusion.predict_fusion_probabilities",
            return_value=probabilities,
        ) as predict:
            result = predictor.predict_proba(_batch())

        self.assertIs(result, probabilities)
        transformer.encode.assert_called_once()
        maxpooling.encode.assert_called_once()
        predict.assert_called_once()
        self.assertIsInstance(predictor, MatchPredictor)


if __name__ == "__main__":
    unittest.main()
