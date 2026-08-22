import unittest
from dataclasses import replace

from match.config import load_app_config_file
from match.paths import PROJECT_ROOT
from match.workflows.train import _data_split_config, _training_match_paths


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


if __name__ == "__main__":
    unittest.main()
