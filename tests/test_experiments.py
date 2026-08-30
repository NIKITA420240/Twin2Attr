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
    _sample_weighting_summary,
    configure_experiment,
    rebuild_experiment_registry,
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
            config, experiment_dir = configure_experiment(
                self.config,
                "mmarco-human-baseline",
                experiments_root=root,
            )

            self.assertEqual(
                experiment_dir,
                root.resolve() / "mmarco-human-baseline",
            )
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

    def test_saves_self_contained_json_without_shared_csv(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, experiment_dir = configure_experiment(
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

            record_path = save_experiment_record(
                config,
                artifacts,
                splits,
                experiment_name="mmarco-human-baseline",
            )

            record = json.loads(record_path.read_text(encoding="utf-8"))
            self.assertEqual(record["experiment_name"], "mmarco-human-baseline")
            self.assertEqual(record["macro_pr_auc_human"], 0.74)
            self.assertEqual(
                record["train_data"],
                "Human + Codex Reviewed + Neural Review + LLM",
            )
            self.assertEqual(record["split"]["train_rows"], 1)
            self.assertTrue(record["split"]["validation_pairs_hash"])
            self.assertEqual(record["sample_weighting"]["all"]["rows"], 1)
            self.assertFalse((root / "experiments.csv").exists())
            self.assertTrue(experiment_dir.is_dir())

    def test_rebuilds_registry_from_all_experiment_records(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = [
                {
                    "experiment_name": "second",
                    "created_at": "2026-08-30T02:00:00Z",
                },
                {
                    "experiment_name": "first",
                    "created_at": "2026-08-30T01:00:00Z",
                },
            ]
            for record in records:
                name = record["experiment_name"]
                experiment_dir = root / name
                experiment_dir.mkdir()
                complete = {
                    column: record.get(column, f"{column}-{name}")
                    for column in EXPERIMENT_REGISTRY_COLUMNS[:18]
                }
                complete["schema_version"] = 1
                (experiment_dir / "experiment.json").write_text(
                    json.dumps(complete),
                    encoding="utf-8",
                )

            registry_path, row_count = rebuild_experiment_registry(root)

            self.assertEqual(row_count, 2)
            with registry_path.open(encoding="utf-8", newline="") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(
                [row["experiment_name"] for row in rows],
                ["first", "second"],
            )
            self.assertEqual(rows[0]["registry_schema_version"], "1")
            self.assertEqual(rows[0]["completed_epochs"], "")

    def test_rebuild_rejects_incomplete_record(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment_dir = root / "incomplete"
            experiment_dir.mkdir()
            (experiment_dir / "experiment.json").write_text(
                json.dumps({"experiment_name": "incomplete"}),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "missing registry fields"):
                rebuild_experiment_registry(root)

if __name__ == "__main__":
    unittest.main()
