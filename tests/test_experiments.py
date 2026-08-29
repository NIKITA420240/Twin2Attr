import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import polars as pl

from match.config import load_app_config_file
from match.experiments import (
    EXPERIMENT_REGISTRY_COLUMNS,
    configure_experiment,
    save_experiment_record,
    validate_experiment_name,
    validation_pairs_hash,
)
from match.models.artifacts import TrainingArtifacts
from match.paths import PROJECT_ROOT


class ExperimentTrackingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_app_config_file(
            PROJECT_ROOT / "configs" / "pipeline.yaml"
        )

    def test_configures_all_generated_paths_under_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, experiment_dir, registry_path = configure_experiment(
                self.config,
                "mmarco-human-baseline",
                experiments_root=root,
            )

            self.assertEqual(
                experiment_dir,
                root.resolve() / "mmarco-human-baseline",
            )
            self.assertEqual(registry_path, root.resolve() / "experiments.csv")
            self.assertEqual(
                config.model_description.transformer.artifact_dir,
                experiment_dir / "models" / "twin2attr" / "transformer",
            )
            self.assertEqual(
                config.training.solution_path,
                experiment_dir / "solution.json",
            )
            self.assertEqual(
                config.inference.solution_path,
                config.training.solution_path,
            )
            self.assertEqual(
                config.training.resolved_config_path,
                experiment_dir
                / "models"
                / "twin2attr"
                / "pipeline_config.yaml",
            )
            self.assertEqual(
                config.logging.file,
                experiment_dir / "logs" / "pipeline.log",
            )

    def test_rejects_path_like_experiment_name(self) -> None:
        with self.assertRaisesRegex(ValueError, "EXPERIMENT_NAME"):
            validate_experiment_name("../overwrite")

    def test_validation_hash_does_not_depend_on_row_order(self) -> None:
        matches = pl.DataFrame(
            {
                "id1": [2, 1],
                "id2": [4, 3],
                "target": [0, 1],
            }
        )

        self.assertEqual(
            validation_pairs_hash(matches),
            validation_pairs_hash(matches.reverse()),
        )
        reversed_pairs = matches.select(
            pl.col("id2").alias("id1"),
            pl.col("id1").alias("id2"),
            "target",
        )
        self.assertEqual(
            validation_pairs_hash(matches),
            validation_pairs_hash(reversed_pairs),
        )

    def test_saves_json_and_appends_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, experiment_dir, registry_path = configure_experiment(
                self.config,
                "mmarco-human-baseline",
                experiments_root=root,
            )
            splits = SimpleNamespace(
                items=pl.DataFrame({"id": [1, 2, 3, 4]}),
                train_matches=pl.DataFrame(
                    {"id1": [1], "id2": [2], "target": [1]}
                ),
                validation_matches=pl.DataFrame(
                    {"id1": [3], "id2": [4], "target": [0]}
                ),
            )
            artifacts = TrainingArtifacts(
                predictor="transformer",
                transformer_dir=(
                    config.model_description.transformer.artifact_dir
                ),
                metrics=(("transformer.validation_macro_pr_auc", 0.74),),
            )
            metadata_path = artifacts.transformer_dir / "training_metadata.json"
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            metadata_path.write_text(
                json.dumps(
                    {
                        "completed_epochs": 2,
                        "resolved_config": {
                            "train_batch_size": 256,
                            "gradient_accumulation_steps": 2,
                            "max_length": 256,
                        },
                        "performance_history": [
                            {
                                "train_seconds": 100.0,
                                "examples_per_second": 900.0,
                                "real_tokens_per_second": 200_000.0,
                                "padding_efficiency": 0.8,
                                "peak_cuda_memory_gib": 12.0,
                            },
                            {
                                "train_seconds": 120.0,
                                "examples_per_second": 1_000.0,
                                "real_tokens_per_second": 220_000.0,
                                "padding_efficiency": 0.9,
                                "peak_cuda_memory_gib": 14.0,
                            },
                        ],
                        "runtime": {
                            "gpu_name": "NVIDIA H100 80GB HBM3",
                            "gpu_count": 1,
                            "world_size": 1,
                            "precision": "bf16",
                            "trainable_parameters": 123_456,
                        },
                    }
                ),
                encoding="utf-8",
            )

            record_path, saved_registry = save_experiment_record(
                config,
                artifacts,
                splits,
                experiment_name="mmarco-human-baseline",
                registry_path=registry_path,
            )

            record = json.loads(record_path.read_text(encoding="utf-8"))
            self.assertEqual(record["experiment_name"], "mmarco-human-baseline")
            self.assertEqual(record["macro_pr_auc_human"], 0.74)
            self.assertEqual(record["train_data"], "Human + LLM")
            self.assertEqual(record["split"]["train_rows"], 1)
            self.assertTrue(record["split"]["validation_pairs_hash"])
            with saved_registry.open(encoding="utf-8", newline="") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["experiment_name"], "mmarco-human-baseline")
            self.assertEqual(
                rows[0]["s3_path"],
                "experiments/mmarco-human-baseline",
            )
            self.assertEqual(rows[0]["registry_schema_version"], "2")
            self.assertEqual(rows[0]["completed_epochs"], "2")
            self.assertEqual(rows[0]["performance_epochs_observed"], "2")
            self.assertEqual(rows[0]["train_batch_size"], "256")
            self.assertEqual(rows[0]["effective_batch_size"], "512")
            self.assertEqual(rows[0]["avg_train_epoch_seconds"], "110.0")
            self.assertEqual(rows[0]["total_train_seconds"], "220.0")
            self.assertEqual(rows[0]["avg_real_tokens_per_second"], "210000.0")
            self.assertAlmostEqual(
                float(rows[0]["avg_padding_efficiency"]),
                0.85,
            )
            self.assertEqual(rows[0]["max_train_peak_cuda_memory_gib"], "14.0")
            self.assertEqual(rows[0]["precision"], "bf16")
            self.assertTrue(experiment_dir.is_dir())

    def test_migrates_legacy_registry_and_preserves_existing_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _, registry_path = configure_experiment(
                self.config,
                "new-experiment",
                experiments_root=root,
            )
            legacy_columns = EXPERIMENT_REGISTRY_COLUMNS[:18]
            with registry_path.open("w", encoding="utf-8", newline="") as output:
                writer = csv.DictWriter(output, fieldnames=legacy_columns)
                writer.writeheader()
                writer.writerow(
                    {
                        "experiment_name": "old-experiment",
                        "created_at": "2026-01-01T00:00:00Z",
                    }
                )
            splits = SimpleNamespace(
                train_matches=pl.DataFrame(
                    {"id1": [1], "id2": [2], "target": [1]}
                ),
                validation_matches=pl.DataFrame(
                    {"id1": [3], "id2": [4], "target": [0]}
                ),
            )
            artifacts = TrainingArtifacts(
                predictor="transformer",
                transformer_dir=(
                    config.model_description.transformer.artifact_dir
                ),
            )

            save_experiment_record(
                config,
                artifacts,
                splits,
                experiment_name="new-experiment",
                registry_path=registry_path,
            )

            with registry_path.open(encoding="utf-8", newline="") as source:
                reader = csv.DictReader(source)
                rows = list(reader)
            self.assertEqual(reader.fieldnames, list(EXPERIMENT_REGISTRY_COLUMNS))
            self.assertEqual(rows[0]["experiment_name"], "old-experiment")
            self.assertEqual(rows[0]["created_at"], "2026-01-01T00:00:00Z")
            self.assertEqual(rows[0]["registry_schema_version"], "1")
            self.assertEqual(rows[0]["avg_train_epoch_seconds"], "")
            self.assertEqual(rows[1]["experiment_name"], "new-experiment")
            self.assertEqual(rows[1]["registry_schema_version"], "2")

    def test_rejects_registry_with_another_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, _, registry_path = configure_experiment(
                self.config,
                "baseline",
                experiments_root=root,
            )
            registry_path.write_text("model,score\nold,0.7\n", encoding="utf-8")
            splits = SimpleNamespace(
                train_matches=pl.DataFrame(
                    {"id1": [1], "id2": [2], "target": [1]}
                ),
                validation_matches=pl.DataFrame(
                    {"id1": [3], "id2": [4], "target": [0]}
                ),
            )
            artifacts = TrainingArtifacts(
                predictor="transformer",
                transformer_dir=(
                    config.model_description.transformer.artifact_dir
                ),
            )

            with self.assertRaisesRegex(ValueError, "unexpected schema"):
                save_experiment_record(
                    config,
                    artifacts,
                    splits,
                    experiment_name="baseline",
                    registry_path=registry_path,
                )

if __name__ == "__main__":
    unittest.main()
