import csv
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import polars as pl

from match.config import load_app_config_file
from match.experiments import (
    _sample_weighting_summary,
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

    def test_sample_weighting_summary_records_targets_and_votes(self) -> None:
        matches = pl.DataFrame(
            {
                "target": [0, 0, 1],
                "sample_weight": [1.0, 1.0, 1.0],
                "weight_multiplier": [1.0, 1.0, 1.0],
                "data_source": ["llm", "llm", "llm"],
                "annotation_votes": [0, 2, 9],
            }
        )

        summary = _sample_weighting_summary(matches)["llm"]

        self.assertEqual(summary["target_counts"], {"0": 2, "1": 1})
        self.assertEqual(summary["vote_counts"], {"0": 1, "2": 1, "9": 1})

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
                    {
                        "id1": [1],
                        "id2": [2],
                        "target": [1],
                        "sample_weight": [1.0],
                    }
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
            self.assertEqual(record["train_data"], "Human")
            self.assertEqual(record["split"]["train_rows"], 1)
            self.assertTrue(record["split"]["validation_pairs_hash"])
            self.assertEqual(record["sample_weighting"]["all"]["rows"], 1)
            with saved_registry.open(encoding="utf-8", newline="") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["experiment_name"], "mmarco-human-baseline")
            self.assertEqual(
                rows[0]["s3_path"],
                "experiments/mmarco-human-baseline",
            )
            self.assertTrue(experiment_dir.is_dir())

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
