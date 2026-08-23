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
        base = self.config.data_model_description.base_dataset
        self.assertIsInstance(base.items, Path)
        self.assertTrue(base.items.is_absolute())
        self.assertEqual(base.seed, self.config.runtime.seed)
        self.assertEqual(self.config.training.model, "transformer")
        self.assertEqual(self.config.training.data_model, "base_dataset")
        self.assertEqual(self.config.inference.model, "transformer")
        self.assertEqual(
            self.config.inference.transformer_dir,
            PROJECT_ROOT / "weights" / "rubert-tiny2",
        )
        self.assertIsInstance(
            self.config.pair_encoding.max_attribute_value_tokens,
            int,
        )
        self.assertIsNone(self.config.pair_encoding.max_length)

    def test_applies_overrides_before_creating_typed_config(self) -> None:
        config = load_app_config_file(
            PROJECT_ROOT / "configs" / "pipeline.yaml",
            [
                "runtime.seed=99",
                "training.model=fusion",
                "training.data_model=mix_dataset",
            ],
        )

        self.assertEqual(config.runtime.seed, 99)
        self.assertEqual(config.data_model_description.base_dataset.seed, 99)
        self.assertEqual(config.training.model, "fusion")
        self.assertEqual(config.training.data_model, "mix_dataset")

    def test_loads_mixed_dataset_sources(self) -> None:
        mixed = self.config.data_model_description.mix_dataset

        self.assertEqual(mixed.validation_source, "human")
        self.assertEqual(mixed.validation_fraction, 0.2)
        self.assertEqual([source.name for source in mixed.sources], ["human", "llm"])
        llm = mixed.sources[1]
        self.assertEqual(llm.weight, 1.0)
        self.assertEqual(llm.max_rows, 750_000)
        self.assertEqual(llm.splitter.total_votes, 9)
        self.assertEqual(llm.splitter.negative_threshold, 2)
        self.assertEqual(llm.splitter.positive_threshold, 7)

    def test_rejects_fractional_vote_threshold(self) -> None:
        with self.assertRaisesRegex(ValueError, "whole vote counts"):
            load_app_config_file(
                PROJECT_ROOT / "configs" / "pipeline.yaml",
                [
                    "data_model_description.mix_dataset.sources.llm.splitter.negative_threshold=2.5"
                ],
            )

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

    def test_loads_composable_transformer_head(self) -> None:
        head = self.config.models_parameters.transformer.head

        self.assertEqual(head.type, "pooling")
        self.assertEqual(head.poolings, ("cls", "mean", "attention"))
        self.assertEqual(head.mlp_hidden_dims, (312,))
        self.assertEqual(head.attention_hidden_dim, 156)

    def test_can_select_original_transformer_head(self) -> None:
        config = load_app_config_file(
            PROJECT_ROOT / "configs" / "pipeline.yaml",
            ["models_parameters.transformer.head.type=default"],
        )

        self.assertEqual(config.models_parameters.transformer.head.type, "default")

    def test_rejects_invalid_transformer_learning_rate(self) -> None:
        with self.assertRaisesRegex(ValueError, "optimizer parameters"):
            load_app_config_file(
                PROJECT_ROOT / "configs" / "pipeline.yaml",
                ["models_parameters.transformer.learning_rate=0"],
            )

    def test_loads_disabled_ner_settings_without_artifacts(self) -> None:
        config = load_app_config_file(
            PROJECT_ROOT / "configs" / "pipeline.yaml",
            ["features.ner.enabled=false"],
        )
        ner = config.features.ner

        self.assertFalse(ner.enabled)
        self.assertEqual(ner.provider, "word_ner")
        self.assertEqual(ner.source_column, "name")
        self.assertEqual(ner.enriched_column, "enriched_attributes")

    def test_enabled_ner_requires_its_artifact_paths(self) -> None:
        with self.assertRaisesRegex(ValueError, "model_dir"):
            load_app_config_file(
                PROJECT_ROOT / "configs" / "pipeline.yaml",
                [
                    "features.ner.enabled=true",
                    "features.ner.model_dir=null",
                ],
            )

    def test_loads_physical_feature_provider_settings(self) -> None:
        physical = self.config.features.physical

        self.assertFalse(physical.enabled)
        self.assertEqual(physical.source_column, "name")
        self.assertTrue(physical.normalize_units)
        self.assertEqual(physical.enriched_column, "feature_attributes")

    def test_config_is_immutable(self) -> None:
        with self.assertRaises(FrozenInstanceError):
            self.config.runtime.seed = 100

    def test_rejects_unknown_training_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "training.model"):
            load_app_config_file(
                PROJECT_ROOT / "configs" / "pipeline.yaml",
                ["training.model=unknown"],
            )

    def test_rejects_unknown_inference_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "inference.model"):
            load_app_config_file(
                PROJECT_ROOT / "configs" / "pipeline.yaml",
                ["inference.model=unknown"],
            )

    def test_saved_config_can_be_loaded_again(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "pipeline_config.yaml"
            save_app_config(self.config, output_path)
            restored = load_app_config_file(output_path)

        self.assertEqual(restored, self.config)


if __name__ == "__main__":
    unittest.main()
