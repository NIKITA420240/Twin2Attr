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
        self.assertEqual(self.config.training.model, "stacking")
        self.assertEqual(self.config.training.data_model, "base_dataset")
        self.assertEqual(
            self.config.training.augmentation_model,
            "attribute_word_dropout",
        )
        self.assertIsNone(self.config.training.data_postprocessing_model)
        self.assertEqual(self.config.inference.model, "stacking")
        self.assertIsNone(self.config.inference.augmentation_model)
        self.assertIsNone(self.config.inference.data_postprocessing_model)
        self.assertEqual(
            self.config.training.resolved_config_path,
            PROJECT_ROOT / "models" / "twin2attr" / "pipeline_config.yaml",
        )
        self.assertEqual(
            self.config.training.solution_path,
            PROJECT_ROOT / "solution.json",
        )
        self.assertEqual(
            self.config.inference.solution_path,
            self.config.training.solution_path,
        )
        self.assertEqual(self.config.inference.transformer.batch_size, 512)
        self.assertEqual(self.config.inference.transformer.dtype, "bfloat16")
        self.assertEqual(self.config.inference.transformer.num_workers, 8)
        self.assertEqual(self.config.inference.transformer.prefetch_factor, 2)
        self.assertTrue(self.config.inference.transformer.pin_memory)
        self.assertTrue(
            self.config.inference.transformer.non_blocking_transfer
        )
        self.assertTrue(self.config.inference.transformer.length_bucketing)
        self.assertFalse(
            self.config.inference.transformer.torch_compile.enabled
        )
        self.assertEqual(
            self.config.inference.transformer.torch_compile.mode,
            "reduce-overhead",
        )
        self.assertTrue(
            self.config.inference.transformer.torch_compile.dynamic
        )
        self.assertFalse(hasattr(self.config, "artifacts"))
        self.assertEqual(
            self.config.analysis.analysis_model,
            "attribute_importance",
        )
        self.assertEqual(self.config.analysis.data_model, "base_dataset")
        self.assertEqual(
            self.config.analysis.augmentation_model,
            "attribute_shuffle",
        )
        self.assertIsNone(self.config.analysis.data_postprocessing_model)
        augmentation = self.config.augmentation_models.attribute_shuffle
        self.assertEqual(augmentation.shuffled_copies, 1)
        self.assertFalse(augmentation.keep_original)
        self.assertTrue(augmentation.shuffle_cards_independently)
        self.assertTrue(augmentation.skip_oversized)
        text_augmentation = (
            self.config.augmentation_models.attribute_word_dropout
        )
        self.assertEqual(text_augmentation.pair_probability, 0.5)
        self.assertEqual(
            text_augmentation.attribute_dropout_probability,
            0.15,
        )
        self.assertEqual(text_augmentation.word_dropout_probability, 0.03)
        self.assertEqual(text_augmentation.keyboard_typo_probability, 0.01)
        self.assertEqual(text_augmentation.word_shuffle_probability, 0.0)
        self.assertEqual(
            self.config.data_postprocessing_models.attribute_sort.priorities_path,
            PROJECT_ROOT
            / "analysis"
            / "attribute_importance"
            / "attribute_importance.parquet",
        )
        analysis = self.config.analysis_models.attribute_importance
        self.assertEqual(analysis.model, "transformer")
        self.assertIsNone(analysis.sample_size)
        self.assertTrue(analysis.group_by_category)
        self.assertEqual(analysis.min_occurrences, 30)
        self.assertEqual(analysis.score_type, "normalized_mean_attention")
        self.assertTrue(self.config.features.normalization.enabled)
        self.assertEqual(
            self.config.features.execution_order,
            ("normalization", "ner", "physical"),
        )
        self.assertEqual(
            self.config.features.normalization.output_column,
            "normalized_attributes",
        )
        self.assertEqual(
            self.config.model_description.transformer.artifact_dir,
            PROJECT_ROOT / "models" / "twin2attr" / "stacking" / "transformer",
        )
        self.assertEqual(
            self.config.model_description.stacking.artifact_dir,
            PROJECT_ROOT / "models" / "twin2attr" / "stacking" / "boosting",
        )
        self.assertEqual(
            self.config.model_description.transformer.pretrained_model_path,
            "models/rubert-base-cased",
        )
        encoding = self.config.model_description.transformer.pair_encoding
        self.assertTrue(encoding.use_field_tokens)
        self.assertFalse(
            self.config.model_description.transformer.train_new_token_embeddings_only
        )
        self.assertIsNone(
            self.config.model_description.transformer.train_last_n_layers
        )
        self.assertEqual(
            self.config.model_description.transformer.lr_scheduler_type,
            "cosine",
        )
        self.assertIsInstance(
            encoding.max_attribute_value_tokens,
            int,
        )
        self.assertEqual(encoding.max_length, 256)
        self.assertEqual(base.stacking_train_fraction, 0.15)
        self.assertEqual(
            self.config.model_description.stacking.base_model,
            "transformer",
        )
        self.assertEqual(
            self.config.model_description.stacking.stacking_model,
            "boosting",
        )

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
                "model_description.transformer.hpo_trials=1",
                "model_description.transformer.learning_rate=0.00001",
                "model_description.transformer.embeddings_learning_rate=0.000005",
                "model_description.transformer.train_new_token_embeddings_only=true",
                "model_description.transformer.train_last_n_layers=3",
                "model_description.transformer.lr_scheduler_type=cosine",
                "model_description.transformer.head_learning_rate=0.00003",
                "model_description.transformer.layerwise_lr_decay=0.8",
                "model_description.transformer.train_batch_size=16",
                "model_description.transformer.gradient_accumulation_steps=4",
            ],
        )

        parameters = config.model_description.transformer
        self.assertEqual(parameters.hpo_trials, 1)
        self.assertEqual(parameters.learning_rate, 1e-5)
        self.assertEqual(parameters.embeddings_learning_rate, 5e-6)
        self.assertTrue(parameters.train_new_token_embeddings_only)
        self.assertEqual(parameters.train_last_n_layers, 3)
        self.assertEqual(parameters.lr_scheduler_type, "cosine")
        self.assertEqual(parameters.head_learning_rate, 3e-5)
        self.assertEqual(parameters.layerwise_lr_decay, 0.8)
        self.assertEqual(parameters.train_batch_size, 16)
        self.assertEqual(parameters.gradient_accumulation_steps, 4)

    def test_loads_composable_transformer_head(self) -> None:
        head = self.config.model_description.transformer.head

        self.assertEqual(head.type, "pooling")
        self.assertEqual(head.poolings, ("cls", "attention"))
        self.assertEqual(head.mlp_hidden_dims, (384,))
        self.assertEqual(head.attention_hidden_dim, 192)
        self.assertEqual(head.attention_num_heads, 1)

    def test_can_select_original_transformer_head(self) -> None:
        config = load_app_config_file(
            PROJECT_ROOT / "configs" / "pipeline.yaml",
            ["model_description.transformer.head.type=default"],
        )

        self.assertEqual(config.model_description.transformer.head.type, "default")

    def test_rejects_invalid_transformer_learning_rate(self) -> None:
        with self.assertRaisesRegex(ValueError, "optimizer parameters"):
            load_app_config_file(
                PROJECT_ROOT / "configs" / "pipeline.yaml",
                ["model_description.transformer.learning_rate=0"],
            )

    def test_rejects_non_positive_trainable_layer_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "optimizer parameters"):
            load_app_config_file(
                PROJECT_ROOT / "configs" / "pipeline.yaml",
                ["model_description.transformer.train_last_n_layers=0"],
            )

    def test_rejects_unknown_lr_scheduler(self) -> None:
        with self.assertRaisesRegex(ValueError, "lr_scheduler_type"):
            load_app_config_file(
                PROJECT_ROOT / "configs" / "pipeline.yaml",
                ["model_description.transformer.lr_scheduler_type=cyclic"],
            )

    def test_rejects_invalid_transformer_inference_dtype(self) -> None:
        with self.assertRaisesRegex(ValueError, "inference.transformer.dtype"):
            load_app_config_file(
                PROJECT_ROOT / "configs" / "pipeline.yaml",
                ["inference.transformer.dtype=int8"],
            )

    def test_rejects_negative_transformer_inference_workers(self) -> None:
        with self.assertRaisesRegex(ValueError, "num_workers"):
            load_app_config_file(
                PROJECT_ROOT / "configs" / "pipeline.yaml",
                ["inference.transformer.num_workers=-1"],
            )

    def test_rejects_unknown_torch_compile_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "torch_compile.mode"):
            load_app_config_file(
                PROJECT_ROOT / "configs" / "pipeline.yaml",
                ["inference.transformer.torch_compile.mode=fastest"],
            )

    def test_rejects_incomplete_feature_execution_order(self) -> None:
        with self.assertRaisesRegex(ValueError, "every feature exactly once"):
            load_app_config_file(
                PROJECT_ROOT / "configs" / "pipeline.yaml",
                ["features.execution_order=[normalization,ner]"],
            )

    def test_rejects_duplicate_feature_execution_order(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not contain duplicates"):
            load_app_config_file(
                PROJECT_ROOT / "configs" / "pipeline.yaml",
                [
                    "features.execution_order="
                    "[normalization,ner,physical,normalization]"
                ],
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

    def test_rejects_overlapping_stacking_and_validation_fractions(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be less than one"):
            load_app_config_file(
                PROJECT_ROOT / "configs" / "pipeline.yaml",
                [
                    "data_model_description.base_dataset.validation_fraction=0.6",
                    "data_model_description.base_dataset.stacking_train_fraction=0.4",
                ],
            )

    def test_saved_config_can_be_loaded_again(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "pipeline_config.yaml"
            save_app_config(self.config, output_path)
            restored = load_app_config_file(output_path)

        self.assertEqual(restored, self.config)


if __name__ == "__main__":
    unittest.main()
