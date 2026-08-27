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
                model_description=replace(
                    self.config.model_description,
                    transformer=replace(
                        self.config.model_description.transformer,
                        artifact_dir=root / "models" / "transformer",
                    ),
                    maxpooling=replace(
                        self.config.model_description.maxpooling,
                        artifact_path=root / "models" / "maxpooling.joblib",
                    ),
                    fusion=replace(
                        self.config.model_description.fusion,
                        artifact_path=root / "models" / "fusion.pt",
                    ),
                ),
                training=replace(
                    self.config.training,
                    solution_path=root / "solution.json",
                ),
            )
            artifacts = TrainingArtifacts(
                predictor="fusion",
                transformer_dir=config.model_description.transformer.artifact_dir,
                maxpooling_path=config.model_description.maxpooling.artifact_path,
                fusion_path=config.model_description.fusion.artifact_path,
            )
            output_path = save_solution_manifest(config, artifacts)
            solution = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(solution["predictor"], "fusion")
        self.assertEqual(solution["model_directory"], "models/transformer")
        self.assertEqual(solution["backend"], "pytorch")
        self.assertEqual(solution["batch_size"], 512)
        self.assertEqual(solution["dtype"], "bfloat16")
        self.assertEqual(solution["num_workers"], 8)
        self.assertEqual(solution["prefetch_factor"], 2)
        self.assertTrue(solution["pin_memory"])
        self.assertTrue(solution["non_blocking_transfer"])
        self.assertEqual(
            solution["tokenizer"],
            {
                "batch_fields": {
                    "enabled": True,
                    "chunk_size": 16384,
                }
            },
        )
        self.assertEqual(
            solution["length_bucketing"],
            {
                "enabled": False,
                "padding_length_buckets": [64, 96, 128, 160, 192, 224, 256],
            },
        )
        self.assertEqual(
            solution["torch_compile"],
            {
                "enabled": False,
                "mode": "reduce-overhead",
                "dynamic": True,
            },
        )
        self.assertEqual(
            solution["onnxruntime"],
            {
                "provider": "cuda",
                "device_id": 0,
                "io_binding": True,
                "graph_optimization": "all",
                "fallback_to_pytorch": True,
                "tensorrt": {
                    "engine_cache": {
                        "enabled": True,
                        "path": "onnx/trt_cache",
                    },
                    "timing_cache": {
                        "enabled": True,
                        "path": None,
                    },
                    "profiles": {
                        "min_batch_size": 1,
                        "opt_batch_size": 2048,
                        "max_batch_size": 2048,
                        "sequence_lengths": [64, 96, 128],
                    },
                },
            },
        )
        self.assertEqual(
            solution["onnx_artifacts"],
            {
                "enabled": False,
                "precision": "float16",
                "opset": 18,
                "classifier_path": "onnx/classifier.onnx",
                "encoder_path": "onnx/encoder.onnx",
                "dynamic_batch": True,
                "dynamic_sequence_length": True,
            },
        )
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
                model_description=replace(
                    self.config.model_description,
                    transformer=replace(
                        self.config.model_description.transformer,
                        artifact_dir=root / "models" / "transformer",
                    ),
                ),
                training=replace(
                    self.config.training,
                    solution_path=root / "solution.json",
                ),
            )
            artifacts = TrainingArtifacts(
                predictor="transformer",
                transformer_dir=config.model_description.transformer.artifact_dir,
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
                model_description=replace(
                    self.config.model_description,
                    transformer=replace(
                        self.config.model_description.transformer,
                        artifact_dir=root / "models" / "transformer",
                    ),
                ),
                training=replace(
                    self.config.training,
                    solution_path=root / "solution.json",
                ),
            )
            artifacts = TrainingArtifacts(
                predictor="transformer",
                transformer_dir=config.model_description.transformer.artifact_dir,
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
                model_description=replace(
                    self.config.model_description,
                    transformer=replace(
                        self.config.model_description.transformer,
                        artifact_dir=experiment / "models" / "transformer",
                    ),
                ),
                training=replace(
                    self.config.training,
                    solution_path=experiment / "solution.json",
                ),
            )
            artifacts = TrainingArtifacts(
                predictor="transformer",
                transformer_dir=config.model_description.transformer.artifact_dir,
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
