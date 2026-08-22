import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from match.config import AppConfig, load_app_config_file, save_app_config
from match.paths import PROJECT_ROOT


class AppConfigTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_app_config_file(
            PROJECT_ROOT / "configs" / "pipeline.yaml"
        )

    def test_loads_resolved_typed_config(self) -> None:
        self.assertIsInstance(self.config, AppConfig)
        self.assertIsInstance(self.config.paths.items, Path)
        self.assertTrue(self.config.paths.items.is_absolute())
        self.assertEqual(self.config.split.seed, self.config.runtime.seed)
        self.assertEqual(self.config.training.model, "transformer")
        self.assertIsInstance(
            self.config.pair_encoding.max_attribute_value_tokens,
            int,
        )
        self.assertIsNone(self.config.pair_encoding.max_length)

    def test_applies_overrides_before_creating_typed_config(self) -> None:
        config = load_app_config_file(
            PROJECT_ROOT / "configs" / "pipeline.yaml",
            ["runtime.seed=99", "training.model=fusion"],
        )

        self.assertEqual(config.runtime.seed, 99)
        self.assertEqual(config.split.seed, 99)
        self.assertEqual(config.training.model, "fusion")

    def test_applies_transformer_optimizer_overrides(self) -> None:
        config = load_app_config_file(
            PROJECT_ROOT / "configs" / "pipeline.yaml",
            [
                "models_parameters.transformer.hpo_trials=1",
                "models_parameters.transformer.learning_rate=0.00001",
                "models_parameters.transformer.train_batch_size=16",
                "models_parameters.transformer.gradient_accumulation_steps=4",
            ],
        )

        parameters = config.models_parameters.transformer
        self.assertEqual(parameters.hpo_trials, 1)
        self.assertEqual(parameters.learning_rate, 1e-5)
        self.assertEqual(parameters.train_batch_size, 16)
        self.assertEqual(parameters.gradient_accumulation_steps, 4)

    def test_rejects_invalid_transformer_learning_rate(self) -> None:
        with self.assertRaisesRegex(ValueError, "optimizer parameters"):
            load_app_config_file(
                PROJECT_ROOT / "configs" / "pipeline.yaml",
                ["models_parameters.transformer.learning_rate=0"],
            )

    def test_config_is_immutable(self) -> None:
        with self.assertRaises(FrozenInstanceError):
            self.config.runtime.seed = 100

    def test_rejects_unknown_training_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "training.model"):
            load_app_config_file(
                PROJECT_ROOT / "configs" / "pipeline.yaml",
                ["training.model=unknown"],
            )

    def test_saved_config_can_be_loaded_again(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "pipeline_config.yaml"
            save_app_config(self.config, output_path)
            restored = load_app_config_file(output_path)

        self.assertEqual(restored, self.config)


if __name__ == "__main__":
    unittest.main()
