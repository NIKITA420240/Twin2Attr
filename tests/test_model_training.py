import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from match.config import load_app_config_file
from match.models.artifacts import TrainingArtifacts, save_solution_manifest
from match.models.training import (
    FusionTrainer,
    MaxPoolingTrainer,
    TransformerTrainer,
    build_trainer,
)
from match.paths import PROJECT_ROOT


class ModelTrainingStrategyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_app_config_file(
            PROJECT_ROOT / "configs" / "pipeline.yaml"
        )

    def test_factory_selects_one_training_strategy(self) -> None:
        transformer = build_trainer(self.config)
        maxpooling = build_trainer(
            replace(
                self.config,
                training=replace(self.config.training, model="maxpooling"),
            )
        )
        fusion = build_trainer(
            replace(
                self.config,
                training=replace(self.config.training, model="fusion"),
            )
        )

        self.assertIsInstance(transformer, TransformerTrainer)
        self.assertIsInstance(maxpooling, MaxPoolingTrainer)
        self.assertIsInstance(fusion, FusionTrainer)

    def test_solution_manifest_matches_training_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = replace(
                self.config,
                artifacts=replace(
                    self.config.artifacts,
                    transformer_dir=root / "models" / "transformer",
                    maxpooling_path=root / "models" / "maxpooling.joblib",
                    fusion_path=root / "models" / "fusion.pt",
                    solution_path=root / "solution.json",
                ),
            )
            artifacts = TrainingArtifacts(
                predictor="fusion",
                transformer_dir=config.artifacts.transformer_dir,
                maxpooling_path=config.artifacts.maxpooling_path,
                fusion_path=config.artifacts.fusion_path,
            )
            output_path = save_solution_manifest(config, artifacts)
            solution = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(solution["predictor"], "fusion")
        self.assertEqual(solution["model_directory"], "models/transformer")
        self.assertEqual(
            solution["maxpooling_path"],
            "models/maxpooling.joblib",
        )
        self.assertEqual(solution["fusion_path"], "models/fusion.pt")


if __name__ == "__main__":
    unittest.main()
