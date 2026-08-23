import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock, patch

from match.config import load_app_config_file
from match.models.artifacts import TrainingArtifacts
from match.paths import PROJECT_ROOT
from match.workflows.train import train


class TrainWorkflowConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        config = load_app_config_file(
            PROJECT_ROOT / "configs" / "pipeline.yaml"
        )
        self.config = config

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
        data_model = Mock()
        data_model.load_training_splits.return_value = SimpleNamespace(
            items=items,
            train_matches=train_matches,
            validation_matches=validation_matches,
        )

        with (
            patch(
                "match.workflows.train.workflow_logging",
                return_value=nullcontext(),
            ),
            patch(
                "match.workflows.train.build_data_model",
                return_value=data_model,
            ) as build_data_model,
            patch(
                "match.workflows.train.prepare_configured_items",
                return_value=SimpleNamespace(
                    frame=items,
                    attributes_column="attributes",
                ),
            ) as prepare_items,
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
        build_data_model.assert_called_once_with(self.config)
        data_model.load_training_splits.assert_called_once_with()
        prepare_items.assert_called_once_with(items, self.config)
        trainer.train.assert_called_once_with(data)
        save_config.assert_called_once_with(
            self.config,
            self.config.artifacts.resolved_config_path,
        )
        save_manifest.assert_called_once_with(self.config, artifacts)


if __name__ == "__main__":
    unittest.main()
