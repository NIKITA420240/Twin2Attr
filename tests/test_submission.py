import unittest
from pathlib import Path
from unittest.mock import patch

from match.submission import _load_predictor


class PredictorLoadingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("/tmp/solution")

    def test_loads_only_transformer_for_transformer_prediction(self) -> None:
        transformer = object()
        with (
            patch(
                "match.submission.TransformerPredictor.load",
                return_value=transformer,
            ) as load_transformer,
            patch("match.submission.MaxPoolingPredictor.load") as load_maxpooling,
        ):
            result = _load_predictor(
                {
                    "predictor": "transformer",
                    "model_directory": "models/transformer",
                    "batch_size": 16,
                },
                self.root,
            )

        self.assertIs(result, transformer)
        load_transformer.assert_called_once_with(
            self.root / "models" / "transformer",
            batch_size=16,
            device=None,
        )
        load_maxpooling.assert_not_called()

    def test_loads_only_maxpooling_for_maxpooling_prediction(self) -> None:
        maxpooling = object()
        with (
            patch("match.submission.TransformerPredictor.load") as load_transformer,
            patch(
                "match.submission.MaxPoolingPredictor.load",
                return_value=maxpooling,
            ) as load_maxpooling,
        ):
            result = _load_predictor(
                {
                    "predictor": "maxpooling",
                    "maxpooling_path": "models/maxpooling.joblib",
                },
                self.root,
            )

        self.assertIs(result, maxpooling)
        load_transformer.assert_not_called()
        load_maxpooling.assert_called_once_with(
            self.root / "models" / "maxpooling.joblib",
            batch_size=512,
            device=None,
        )

    def test_composes_fusion_from_both_encoders(self) -> None:
        transformer = object()
        maxpooling = object()
        fusion = object()
        with (
            patch(
                "match.submission.TransformerPredictor.load",
                return_value=transformer,
            ),
            patch(
                "match.submission.MaxPoolingPredictor.load",
                return_value=maxpooling,
            ),
            patch(
                "match.submission.FusionPredictor.load",
                return_value=fusion,
            ) as load_fusion,
        ):
            result = _load_predictor(
                {
                    "predictor": "fusion",
                    "model_directory": "models/transformer",
                    "maxpooling_path": "models/maxpooling.joblib",
                    "fusion_path": "models/fusion.pt",
                },
                self.root,
            )

        self.assertIs(result, fusion)
        load_fusion.assert_called_once_with(
            self.root / "models" / "fusion.pt",
            transformer=transformer,
            maxpooling=maxpooling,
            batch_size=512,
            device=None,
        )

    def test_rejects_unknown_predictor(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported predictor"):
            _load_predictor({"predictor": "unknown"}, self.root)


if __name__ == "__main__":
    unittest.main()
