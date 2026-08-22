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
        self.assertEqual(self.config.split.seed, self.config.model.seed)
        self.assertIsInstance(
            self.config.pair_encoding.max_attribute_value_tokens,
            int,
        )
        self.assertIsNone(self.config.pair_encoding.max_length)

    def test_applies_overrides_before_creating_typed_config(self) -> None:
        config = load_app_config_file(
            PROJECT_ROOT / "configs" / "pipeline.yaml",
            ["model.seed=99", "features.maxpooling.enabled=true"],
        )

        self.assertEqual(config.model.seed, 99)
        self.assertEqual(config.split.seed, 99)
        self.assertTrue(config.features.maxpooling.enabled)

    def test_config_is_immutable(self) -> None:
        with self.assertRaises(FrozenInstanceError):
            self.config.model.seed = 100

    def test_saved_config_can_be_loaded_again(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "pipeline_config.yaml"
            save_app_config(self.config, output_path)
            restored = load_app_config_file(output_path)

        self.assertEqual(restored, self.config)


if __name__ == "__main__":
    unittest.main()
