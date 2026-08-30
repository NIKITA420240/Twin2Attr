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
        overlap = (
            self.config.data_model_description.mix_dataset.overlap_resolution
        )
        self.assertTrue(overlap.enabled)
        self.assertEqual(overlap.source_priority, ("human", "llm"))
        sources = {
            source.name: source
            for source in self.config.data_model_description.mix_dataset.sources
        }
        self.assertEqual(sources["human"].weight_model.type, "constant")
        self.assertFalse(sources["human"].weight_model.enabled)
        hard_sources = {
            source.name: source
            for source in self.config.data_model_description
            .mix_dataset_hard_negative.sources
        }
        hard_negative = hard_sources["hard_negative"]
        self.assertEqual(hard_negative.weight, 0.5)
        self.assertEqual(hard_negative.max_rows, 130_000)
        self.assertEqual(
            hard_negative.items,
            PROJECT_ROOT / "data" / "hard_negative_items.parquet",
        )
        self.assertEqual(
            hard_negative.matches,
            PROJECT_ROOT / "data" / "hard_negative_matches.parquet",
        )
        self.assertFalse(hard_negative.weight_model.enabled)
        llm_weight_model = sources["llm"].weight_model
        self.assertEqual(llm_weight_model.type, "transitivity")
        self.assertTrue(llm_weight_model.enabled)
        self.assertEqual(llm_weight_model.penalty_strength, 1.0)
        self.assertEqual(llm_weight_model.min_weight_multiplier, 0.25)
        self.assertEqual(llm_weight_model.min_comparable_neighbors, 2)
        self.assertTrue(llm_weight_model.confidence_weighted_violations)
        self.assertEqual(sources["llm"].splitter.target_mode, "soft")
        confidence_weighting = sources["llm"].confidence_weighting
        self.assertFalse(confidence_weighting.enabled)
        self.assertEqual(
            confidence_weighting.method,
            "distance_from_midpoint",
        )
        self.assertEqual(confidence_weighting.min_weight_multiplier, 0.2)
        self.assertEqual(confidence_weighting.power, 1.0)
        self.assertEqual(self.config.training.model, "transformer")
        self.assertEqual(
            self.config.training.data_model,
            "mix_dataset_hard_negative",
        )
        self.assertIsNone(self.config.training.augmentation_model)
        self.assertIsNone(self.config.training.data_postprocessing_model)
        self.assertEqual(self.config.inference.model, "transformer")
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
        self.assertEqual(self.config.inference.transformer.batch_size, 1)
        self.assertEqual(self.config.inference.transformer.dtype, "bfloat16")
        self.assertEqual(self.config.inference.transformer.backend, "pytorch")
        self.assertEqual(self.config.inference.transformer.num_workers, 8)
        self.assertEqual(self.config.inference.transformer.prefetch_factor, 2)
        self.assertTrue(self.config.inference.transformer.pin_memory)
        self.assertTrue(
            self.config.inference.transformer.non_blocking_transfer
        )
        self.assertEqual(
            self.config.inference.transformer.attention.implementation,
            "sdpa",
        )
        length_bucketing = self.config.inference.transformer.length_bucketing
        self.assertFalse(length_bucketing.enabled)
        self.assertEqual(
            length_bucketing.padding_length_buckets,
            (64, 96, 128, 160, 192, 224, 256),
        )
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
        training_runtime = self.config.model_description.transformer.training_runtime
        self.assertTrue(training_runtime.optimizer.fused)
        self.assertTrue(training_runtime.length_bucketing.enabled)
        self.assertEqual(
            training_runtime.length_bucketing.mega_batch_multiplier,
            50,
        )
        self.assertEqual(
            training_runtime.length_bucketing.padding_length_buckets,
            (32, 64, 96, 128),
        )
        self.assertEqual(training_runtime.dataloader.num_workers, 4)
        self.assertEqual(training_runtime.dataloader.prefetch_factor, 2)
        self.assertTrue(training_runtime.dataloader.persistent_workers)
        self.assertTrue(training_runtime.dataloader.pin_memory)
        self.assertTrue(training_runtime.dataloader.non_blocking_transfer)
        self.assertTrue(training_runtime.performance_logging.enabled)
        self.assertTrue(training_runtime.token_cache.enabled)
        self.assertEqual(training_runtime.attention.implementation, "sdpa")
        self.assertEqual(
            training_runtime.token_cache.directory,
            PROJECT_ROOT / ".cache" / "tokenized_pairs",
        )
        self.assertEqual(training_runtime.token_cache.build_chunk_size, 4_096)
        self.assertTrue(training_runtime.torch_compile.enabled)
        fast_dev = self.config.model_description.transformer.validation.fast_dev
        self.assertTrue(fast_dev.enabled)
        self.assertEqual(fast_dev.max_rows, 20_000)
        self.assertEqual(fast_dev.every_n_optimizer_steps, 1_000)
        onnxruntime = self.config.inference.transformer.onnxruntime
        self.assertEqual(onnxruntime.provider, "cuda")
        self.assertEqual(onnxruntime.device_id, 0)
        self.assertTrue(onnxruntime.io_binding)
        self.assertEqual(onnxruntime.graph_optimization, "all")
        self.assertTrue(onnxruntime.fallback_to_pytorch)
        tensorrt = onnxruntime.tensorrt
        self.assertTrue(tensorrt.engine_cache.enabled)
        self.assertEqual(tensorrt.engine_cache.path, "onnx/trt_cache")
        self.assertTrue(tensorrt.timing_cache.enabled)
        self.assertIsNone(tensorrt.timing_cache.path)
        self.assertEqual(tensorrt.profiles.min_batch_size, 1)
        self.assertEqual(tensorrt.profiles.opt_batch_size, 2048)
        self.assertEqual(tensorrt.profiles.max_batch_size, 2048)
        self.assertEqual(tensorrt.profiles.sequence_lengths, (64, 96, 128))
        native = self.config.inference.transformer.tensorrt
        self.assertEqual(native.device_id, 0)
        self.assertEqual(native.workspace_size_gb, 8.0)
        self.assertEqual(native.builder_optimization_level, 3)
        self.assertTrue(native.fallback_to_onnxruntime)
        self.assertEqual(native.profiles.sequence_lengths, (1, 256, 472))
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
        labeling = self.config.labeling
        self.assertEqual(labeling.lower_p, 0.54)
        self.assertEqual(labeling.upper_p, 0.56)
        self.assertEqual(labeling.sample_size, 100000)
        self.assertEqual(labeling.seed, self.config.runtime.seed)
        self.assertEqual(labeling.llm.max_concurrency, 64)
        self.assertEqual(labeling.llm.min_concurrency, 32)
        self.assertEqual(labeling.llm.max_rounds, 1000)
        self.assertEqual(labeling.llm.token_env, "LLM_PROXY_TOKEN")
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
        typed = self.config.pair_features.typed_attributes
        self.assertTrue(typed.enabled)
        self.assertTrue(typed.symmetric)
        self.assertTrue(typed.preserve_semantic_type_when_missing)
        self.assertEqual(
            typed.enabled_types,
            ("CODE", "PHYSICAL", "NUMERIC", "SET", "TEXT"),
        )
        self.assertEqual(
            self.config.features.normalization.output_column,
            "normalized_attributes",
        )
        self.assertEqual(
            self.config.model_description.transformer.artifact_dir,
            PROJECT_ROOT / "models" / "twin2attr" / "nemotron-native",
        )
        self.assertEqual(
            self.config.model_description.stacking.artifact_dir,
            PROJECT_ROOT / "models" / "twin2attr" / "stacking" / "boosting",
        )
        self.assertEqual(
            self.config.model_description.transformer.pretrained_model_path,
            "models/llama-nemotron-rerank-1b-v2",
        )
        self.assertEqual(
            self.config.model_description.transformer.profile,
            "prompted_binary_reranker",
        )
        encoding = self.config.model_description.transformer.pair_encoding
        batch_fields = (
            self.config.model_description.transformer.tokenizer.batch_fields
        )
        self.assertTrue(batch_fields.enabled)
        self.assertEqual(batch_fields.chunk_size, 16384)
        onnx_export = self.config.model_description.transformer.export.onnx
        self.assertFalse(onnx_export.enabled)
        self.assertEqual(onnx_export.opset, 18)
        self.assertEqual(onnx_export.precision, "float16")
        self.assertTrue(onnx_export.export_classifier)
        self.assertFalse(onnx_export.export_encoder)
        self.assertFalse(encoding.use_field_tokens)
        self.assertFalse(
            self.config.model_description.transformer.special_token_initialization.enabled
        )
        self.assertFalse(
            self.config.model_description.transformer.train_new_token_embeddings_only
        )
        self.assertEqual(
            self.config.model_description.transformer.train_last_n_layers,
            3,
        )
        self.assertEqual(self.config.model_description.transformer.max_epochs, 2)
        self.assertEqual(
            self.config.model_description.transformer.train_batch_size,
            256,
        )
        self.assertEqual(
            self.config.model_description.transformer.gradient_accumulation_steps,
            1,
        )
        self.assertEqual(
            self.config.model_description.transformer.lr_scheduler_type,
            "cosine",
        )
        self.assertIsInstance(
            encoding.max_attribute_value_tokens,
            int,
        )
        self.assertEqual(encoding.max_attribute_value_chars, 256)
        self.assertEqual(encoding.max_length, 128)
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
        self.assertEqual(
            [source.name for source in mixed.sources],
            ["human", "llm"],
        )
        llm = next(source for source in mixed.sources if source.name == "llm")
        self.assertEqual(llm.weight, 1.0)
        self.assertEqual(llm.max_rows, 750_000)
        self.assertEqual(llm.splitter.total_votes, 9)
        self.assertEqual(llm.splitter.negative_threshold, 2)
        self.assertEqual(llm.splitter.positive_threshold, 7)
        self.assertEqual(llm.confidence_power, 2.0)

        hard_mixed = (
            self.config.data_model_description.mix_dataset_hard_negative
        )
        self.assertEqual(
            [source.name for source in hard_mixed.sources],
            ["human", "hard_negative", "llm"],
        )
        self.assertEqual(
            hard_mixed.overlap_resolution.source_priority,
            ("human", "hard_negative", "llm"),
        )

    def test_loads_codex_mixed_dataset(self) -> None:
        mixed = self.config.data_model_description.mix_dataset_codex
        sources = {source.name: source for source in mixed.sources}

        self.assertEqual(
            list(sources),
            ["human", "codex_reviewed", "llm"],
        )
        self.assertEqual(
            mixed.overlap_resolution.source_priority,
            ("human", "codex_reviewed", "llm"),
        )
        codex = sources["codex_reviewed"]
        self.assertEqual(
            codex.matches,
            PROJECT_ROOT / "data" / "matches_llm_ambiguous_reviewed.parquet",
        )
        self.assertEqual(codex.weight, 1.0)
        self.assertIsNone(codex.max_rows)
        self.assertEqual(codex.splitter.score_type, "label")
        llm = sources["llm"]
        self.assertEqual(llm.splitter.target_mode, "soft")
        self.assertFalse(llm.confidence_weighting.enabled)

    def test_loads_neural_review_mixed_dataset(self) -> None:
        mixed = self.config.data_model_description.mix_dataset_neural_review
        sources = {source.name: source for source in mixed.sources}

        self.assertEqual(
            list(sources),
            ["human", "neural_review", "llm"],
        )
        self.assertEqual(
            mixed.overlap_resolution.source_priority,
            ("human", "neural_review", "llm"),
        )
        review = sources["neural_review"]
        self.assertEqual(review.target_column, "source_score")
        self.assertEqual(review.splitter.score_type, "probability")
        self.assertEqual(review.splitter.negative_threshold, 0.5)
        self.assertEqual(review.splitter.positive_threshold, 0.5)
        self.assertEqual(review.splitter.target_mode, "hard")
        self.assertFalse(review.confidence_weighting.enabled)
        self.assertFalse(review.weight_model.enabled)
        self.assertEqual(sources["llm"].max_rows, 649_663)

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
                "model_description.transformer.profile=sequence_classifier",
                "model_description.transformer.head.type=default",
                "model_description.transformer.pair_encoding.use_field_tokens=true",
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

        self.assertEqual(head.type, "native")
        self.assertEqual(head.poolings, ("cls", "attention"))
        self.assertEqual(head.mlp_hidden_dims, (384,))
        self.assertEqual(head.attention_hidden_dim, 192)
        self.assertEqual(head.attention_num_heads, 1)
        self.assertEqual(head.native_logit_weight, 1.0)
        self.assertEqual(head.attention_logit_weight, 0.0)
        self.assertFalse(head.train_logit_weights)

    def test_can_select_original_transformer_head(self) -> None:
        config = load_app_config_file(
            PROJECT_ROOT / "configs" / "pipeline.yaml",
            [
                "model_description.transformer.profile=sequence_classifier",
                "model_description.transformer.head.type=default",
            ],
        )

        self.assertEqual(config.model_description.transformer.head.type, "default")

    def test_rejects_invalid_transformer_learning_rate(self) -> None:
        with self.assertRaisesRegex(ValueError, "optimizer parameters"):
            load_app_config_file(
                PROJECT_ROOT / "configs" / "pipeline.yaml",
                ["model_description.transformer.learning_rate=0"],
            )

    def test_rejects_non_positive_attribute_value_character_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "max_attribute_value_chars"):
            load_app_config_file(
                PROJECT_ROOT / "configs" / "pipeline.yaml",
                [
                    "model_description.transformer.pair_encoding."
                    "max_attribute_value_chars=0"
                ],
            )

    def test_rejects_non_positive_tokenizer_field_chunk_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "chunk_size"):
            load_app_config_file(
                PROJECT_ROOT / "configs" / "pipeline.yaml",
                [
                    "model_description.transformer.tokenizer.batch_fields."
                    "chunk_size=0"
                ],
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

    def test_rejects_unknown_transformer_backend(self) -> None:
        with self.assertRaisesRegex(ValueError, "backend"):
            load_app_config_file(
                PROJECT_ROOT / "configs" / "pipeline.yaml",
                ["inference.transformer.backend=unknown"],
            )

    def test_rejects_unknown_onnxruntime_provider(self) -> None:
        with self.assertRaisesRegex(ValueError, "provider"):
            load_app_config_file(
                PROJECT_ROOT / "configs" / "pipeline.yaml",
                ["inference.transformer.onnxruntime.provider=unknown"],
            )

    def test_rejects_negative_onnxruntime_device_id(self) -> None:
        with self.assertRaisesRegex(ValueError, "device_id"):
            load_app_config_file(
                PROJECT_ROOT / "configs" / "pipeline.yaml",
                ["inference.transformer.onnxruntime.device_id=-1"],
            )

    def test_rejects_invalid_tensorrt_profile(self) -> None:
        with self.assertRaisesRegex(ValueError, "min <= opt <= max"):
            load_app_config_file(
                PROJECT_ROOT / "configs" / "pipeline.yaml",
                [
                    "inference.transformer.onnxruntime.tensorrt."
                    "profiles.min_batch_size=4096"
                ],
            )

    def test_rejects_unknown_sample_weight_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "weight_model.type"):
            load_app_config_file(
                PROJECT_ROOT / "configs" / "pipeline.yaml",
                [
                    "data_model_description.mix_dataset.sources.llm."
                    "weight_model.type=unknown"
                ],
            )

    def test_rejects_invalid_sample_weight_floor(self) -> None:
        with self.assertRaisesRegex(ValueError, "min_weight_multiplier"):
            load_app_config_file(
                PROJECT_ROOT / "configs" / "pipeline.yaml",
                [
                    "data_model_description.mix_dataset.sources.llm."
                    "weight_model.min_weight_multiplier=0"
                ],
            )

    def test_rejects_invalid_onnx_export_precision(self) -> None:
        with self.assertRaisesRegex(ValueError, "precision"):
            load_app_config_file(
                PROJECT_ROOT / "configs" / "pipeline.yaml",
                ["model_description.transformer.export.onnx.precision=int8"],
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

    def test_rejects_unknown_attention_implementation(self) -> None:
        with self.assertRaisesRegex(ValueError, "attention.implementation"):
            load_app_config_file(
                PROJECT_ROOT / "configs" / "pipeline.yaml",
                ["inference.transformer.attention.implementation=magic"],
            )

    def test_rejects_invalid_training_token_cache_chunk_size(self) -> None:
        with self.assertRaisesRegex(ValueError, "token_cache.build_chunk_size"):
            load_app_config_file(
                PROJECT_ROOT / "configs" / "pipeline.yaml",
                [
                    "model_description.transformer.training_runtime."
                    "token_cache.build_chunk_size=0"
                ],
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

    def test_enabled_overlap_resolution_requires_every_source_once(self) -> None:
        with self.assertRaisesRegex(ValueError, "every configured source"):
            load_app_config_file(
                PROJECT_ROOT / "configs" / "pipeline.yaml",
                [
                    "data_model_description.mix_dataset."
                    "overlap_resolution.source_priority=[human]",
                ],
            )

    def test_saved_config_can_be_loaded_again(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output_path = Path(directory) / "pipeline_config.yaml"
            save_app_config(self.config, output_path)
            serialized = output_path.read_text(encoding="utf-8")
            restored = load_app_config_file(output_path)

        self.assertEqual(restored, self.config)
        self.assertIn("sequence_lengths:", serialized)
        self.assertNotIn("min_sequence_length:", serialized)


if __name__ == "__main__":
    unittest.main()
