import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from match.config import load_app_config_file
from match.models.artifacts import TrainingArtifacts, save_solution_manifest
from match.models.factory import build_trainer
from match.models.fusion.training import FusionTrainer
from match.models.maxpooling.training import MaxPoolingTrainer
from match.models.stacking.training import StackingTrainer
from match.models.transformer.training import TransformerTrainer
from match.paths import PROJECT_ROOT


class ModelTrainingStrategyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_app_config_file(
            PROJECT_ROOT / "configs" / "pipeline.yaml"
        )

    def test_factory_selects_one_training_strategy(self) -> None:
        stacking = build_trainer(self.config)
        transformer = build_trainer(
            replace(
                self.config,
                training=replace(self.config.training, model="transformer"),
            )
        )
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

        self.assertIsInstance(stacking, StackingTrainer)
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

    def test_solution_manifest_persists_ner_preprocessing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = replace(
                self.config,
                features=replace(
                    self.config.features,
                    ner=replace(
                        self.config.features.ner,
                        enabled=True,
                        model_dir=root / "models" / "ner",
                        cluster_centers_path=root / "models" / "centers.pt",
                    ),
                ),
                artifacts=replace(
                    self.config.artifacts,
                    transformer_dir=root / "models" / "transformer",
                    solution_path=root / "solution.json",
                ),
            )
            artifacts = TrainingArtifacts(
                predictor="transformer",
                transformer_dir=config.artifacts.transformer_dir,
            )
            output_path = save_solution_manifest(config, artifacts)
            solution = json.loads(output_path.read_text(encoding="utf-8"))

        ner = solution["features"]["ner"]
        self.assertTrue(ner["enabled"])
        self.assertEqual(ner["provider"], "word_ner")
        self.assertEqual(ner["model_dir"], "models/ner")
        self.assertEqual(ner["cluster_centers_path"], "models/centers.pt")
        self.assertEqual(ner["enriched_column"], "enriched_attributes")

    def test_solution_manifest_persists_physical_provider(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = replace(
                self.config,
                features=replace(
                    self.config.features,
                    physical=replace(
                        self.config.features.physical,
                        enabled=True,
                    ),
                ),
                artifacts=replace(
                    self.config.artifacts,
                    transformer_dir=root / "models" / "transformer",
                    solution_path=root / "solution.json",
                ),
            )
            artifacts = TrainingArtifacts(
                predictor="transformer",
                transformer_dir=config.artifacts.transformer_dir,
            )
            output_path = save_solution_manifest(config, artifacts)
            solution = json.loads(output_path.read_text(encoding="utf-8"))

        physical = solution["features"]["physical"]
        self.assertTrue(physical["enabled"])
        self.assertTrue(physical["normalize_units"])
        self.assertEqual(physical["enriched_column"], "feature_attributes")

    def test_solution_manifest_uses_relative_paths_for_shared_resources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = root / "experiments" / "baseline"
            config = replace(
                self.config,
                features=replace(
                    self.config.features,
                    normalization=replace(
                        self.config.features.normalization,
                        synonyms_path=root / "data" / "synonyms.parquet",
                        unique_attributes_path=(
                            root / "data" / "unique_attributes.parquet"
                        ),
                    ),
                ),
                artifacts=replace(
                    self.config.artifacts,
                    transformer_dir=experiment / "models" / "transformer",
                    solution_path=experiment / "solution.json",
                ),
            )
            artifacts = TrainingArtifacts(
                predictor="transformer",
                transformer_dir=config.artifacts.transformer_dir,
            )

            output_path = save_solution_manifest(config, artifacts)
            solution = json.loads(output_path.read_text(encoding="utf-8"))

        normalization = solution["features"]["normalization"]
        self.assertEqual(
            normalization["synonyms_path"],
            "../../data/synonyms.parquet",
        )
        self.assertFalse(Path(normalization["synonyms_path"]).is_absolute())


if __name__ == "__main__":
    unittest.main()
