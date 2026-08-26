import unittest
from pathlib import Path
from unittest.mock import patch

from match.models.factory import build_predictor


class PredictorLoadingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("/tmp/solution")

    def test_loads_only_transformer_for_transformer_prediction(self) -> None:
        transformer = object()
        with (
            patch(
                "match.models.transformer.predictor.TransformerPredictor.load",
                return_value=transformer,
            ) as load_transformer,
            patch(
                "match.models.maxpooling.predictor.MaxPoolingPredictor.load"
            ) as load_maxpooling,
        ):
            result = build_predictor(
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
            patch(
                "match.models.transformer.predictor.TransformerPredictor.load"
            ) as load_transformer,
            patch(
                "match.models.maxpooling.predictor.MaxPoolingPredictor.load",
                return_value=maxpooling,
            ) as load_maxpooling,
        ):
            result = build_predictor(
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
                "match.models.transformer.predictor.TransformerPredictor.load",
                return_value=transformer,
            ),
            patch(
                "match.models.maxpooling.predictor.MaxPoolingPredictor.load",
                return_value=maxpooling,
            ),
            patch(
                "match.models.fusion.predictor.FusionPredictor.load",
                return_value=fusion,
            ) as load_fusion,
        ):
            result = build_predictor(
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

    def test_composes_cascade_from_boosting_and_transformer(self) -> None:
        boosting = object()
        transformer = object()
        cascade = object()
        with (
            patch(
                "match.models.boosting.predictor.BoostingPredictor.load",
                return_value=boosting,
            ) as load_boosting,
            patch(
                "match.models.transformer.predictor.TransformerPredictor.load",
                return_value=transformer,
            ) as load_transformer,
            patch(
                "match.models.cascade.predictor.CascadePredictor",
                return_value=cascade,
            ) as create_cascade,
        ):
            result = build_predictor(
                {
                    "predictor": "cascade",
                    "fast_model": "boosting",
                    "main_model": "transformer",
                    "boosting_directory": "models/boosting",
                    "model_directory": "models/transformer",
                    "negative_threshold": 0.02,
                    "positive_threshold": 0.98,
                },
                self.root,
            )

        self.assertIs(result, cascade)
        load_boosting.assert_called_once_with(
            self.root / "models" / "boosting",
            thread_count=-1,
        )
        load_transformer.assert_called_once_with(
            self.root / "models" / "transformer",
            batch_size=64,
            device=None,
        )
        create_cascade.assert_called_once_with(
            boosting,
            transformer,
            negative_threshold=0.02,
            positive_threshold=0.98,
        )

    def test_composes_stacking_from_transformer_and_catboost(self) -> None:
        transformer = object()
        stacking = object()
        with (
            patch(
                "match.models.transformer.predictor.TransformerPredictor.load",
                return_value=transformer,
            ) as load_transformer,
            patch(
                "match.models.stacking.predictor.StackingPredictor.load",
                return_value=stacking,
            ) as load_stacking,
        ):
            result = build_predictor(
                {
                    "predictor": "stacking",
                    "base_model": "transformer",
                    "stacking_model": "boosting",
                    "model_directory": "models/transformer",
                    "stacking_directory": "models/stacking",
                    "batch_size": 32,
                    "stacking_thread_count": 4,
                },
                self.root,
            )

        self.assertIs(result, stacking)
        load_transformer.assert_called_once_with(
            self.root / "models" / "transformer",
            batch_size=32,
            device=None,
        )
        load_stacking.assert_called_once_with(
            self.root / "models" / "stacking",
            transformer=transformer,
            thread_count=4,
        )

    def test_rejects_unknown_predictor(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported predictor"):
            build_predictor({"predictor": "unknown"}, self.root)


if __name__ == "__main__":
    unittest.main()
