import unittest
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

from match.config import load_app_config_file
from match.models.artifacts import TrainingArtifacts
from match.paths import PROJECT_ROOT
from match.workflows.train import _data_split_config, _training_match_paths, train


class TrainWorkflowConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        config = load_app_config_file(
            PROJECT_ROOT / "configs" / "pipeline.yaml"
        )
        self.config = replace(
            config,
            paths=replace(
                config.paths,
                train_matches=PROJECT_ROOT / "data" / "train.parquet",
                validation_matches=PROJECT_ROOT / "data" / "validation.parquet",
            ),
            split=replace(
                config.split,
                validation_fraction=0.25,
                leakage_scope="item",
                seed=17,
                candidate_splits=32,
                train_output_path=PROJECT_ROOT / "artifacts" / "train.parquet",
                validation_output_path=(
                    PROJECT_ROOT / "artifacts" / "validation.parquet"
                ),
            ),
        )

    def test_builds_named_training_match_paths(self) -> None:
        paths = _training_match_paths(self.config)

        self.assertEqual(paths.source.name, "train.parquet")
        self.assertEqual(paths.validation.name, "validation.parquet")
        self.assertEqual(paths.generated_train.name, "train.parquet")
        self.assertEqual(paths.generated_validation.name, "validation.parquet")

    def test_builds_data_split_config(self) -> None:
        split = _data_split_config(self.config)

        self.assertEqual(split.validation_fraction, 0.25)
        self.assertEqual(split.leakage_scope, "item")
        self.assertEqual(split.seed, 17)
        self.assertEqual(split.candidate_splits, 32)

    def test_delegates_training_to_selected_strategy(self) -> None:
        items = object()
        train_matches = object()
        validation_matches = object()
        data = object()
        artifacts = TrainingArtifacts(
            predictor="transformer",
            transformer_dir=self.config.artifacts.transformer_dir,
        )
        solution_path = self.config.artifacts.solution_path
        trainer = Mock()
        trainer.train.return_value = artifacts

        with (
            patch(
                "match.workflows.train.workflow_logging",
                return_value=nullcontext(),
            ),
            patch("match.workflows.train.read_parquet", return_value=items),
            patch(
                "match.workflows.train.prepare_configured_items",
                return_value=SimpleNamespace(
                    frame=items,
                    attributes_column="attributes",
                ),
            ) as prepare_items,
            patch(
                "match.workflows.train.load_training_matches",
                return_value=(train_matches, validation_matches),
            ),
            patch(
                "match.workflows.train.prepare_training_data",
                return_value=data,
            ),
            patch(
                "match.workflows.train.build_trainer",
                return_value=trainer,
            ) as build_trainer,
            patch("match.workflows.train.save_app_config") as save_config,
            patch(
                "match.workflows.train.save_solution_manifest",
                return_value=solution_path,
            ) as save_manifest,
        ):
            result = train(self.config)

        self.assertEqual(result.predictor, "transformer")
        self.assertEqual(
            result.resolved_config_path,
            self.config.artifacts.resolved_config_path,
        )
        self.assertEqual(result.solution_path, solution_path)
        build_trainer.assert_called_once_with(self.config)
        prepare_items.assert_called_once_with(items, self.config)
        trainer.train.assert_called_once_with(data)
        save_config.assert_called_once_with(
            self.config,
            self.config.artifacts.resolved_config_path,
        )
        save_manifest.assert_called_once_with(self.config, artifacts)


if __name__ == "__main__":
    unittest.main()
