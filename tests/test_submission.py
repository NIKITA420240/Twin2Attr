import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import polars as pl

from match.models.factory import build_predictor
from match.models.transformer.tensorrt_common import TensorRTInitializationError
from match.submission import _predict


class PredictorLoadingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = Path("/tmp/solution")

    def test_loads_only_transformer_for_transformer_prediction(self) -> None:
        transformer = object()
        with (
            patch(
                "match.models.transformer.predictor.TransformerPredictor.load",
                return_value=transformer,
            ) as load_transformer,
            patch(
                "match.models.maxpooling.predictor.MaxPoolingPredictor.load"
            ) as load_maxpooling,
        ):
            result = build_predictor(
                {
                    "predictor": "transformer",
                    "model_directory": "models/transformer",
                    "batch_size": 16,
                    "dtype": "bfloat16",
                    "length_bucketing": {
                        "enabled": True,
                        "padding_length_buckets": [64, 128],
                    },
                    "tokenizer": {
                        "batch_fields": {
                            "enabled": True,
                            "chunk_size": 8192,
                        }
                    },
                    "torch_compile": {
                        "enabled": True,
                        "mode": "reduce-overhead",
                        "dynamic": True,
                    },
                },
                self.root,
            )

        self.assertIs(result, transformer)
        load_transformer.assert_called_once_with(
            self.root / "models" / "transformer",
            batch_size=16,
            dtype="bfloat16",
            num_workers=0,
            prefetch_factor=2,
            pin_memory=True,
            non_blocking_transfer=True,
            length_bucketing=True,
            padding_length_buckets=(64, 128),
            batch_fields=True,
            field_chunk_size=8192,
            compile_enabled=True,
            compile_mode="reduce-overhead",
            compile_dynamic=True,
            device=None,
        )
        load_maxpooling.assert_not_called()

    def test_loads_onnxruntime_transformer_backend(self) -> None:
        transformer = Mock()
        with (
            patch(
                "match.models.transformer.factory.build_transformer_predictor",
                return_value=transformer,
            ) as load_transformer,
            patch(
                "match.models.transformer.predictor.TransformerPredictor.load"
            ) as load_pytorch,
        ):
            result = build_predictor(
                {
                    "predictor": "transformer",
                    "backend": "onnxruntime",
                    "model_directory": "models/transformer",
                    "batch_size": 2048,
                    "onnxruntime": {
                        "provider": "cuda",
                        "io_binding": True,
                        "graph_optimization": "all",
                        "fallback_to_pytorch": False,
                    },
                    "onnx_artifacts": {
                        "classifier_path": "onnx/classifier.onnx",
                        "encoder_path": "onnx/encoder.onnx",
                    },
                },
                self.root,
            )

        self.assertIs(result, transformer)
        load_transformer.assert_called_once()
        call = load_transformer.call_args
        self.assertEqual(call.args[1], self.root)
        self.assertEqual(call.kwargs, {"usage": "classifier"})
        load_pytorch.assert_not_called()

    def test_onnx_fallback_handles_only_expected_initialization_errors(self) -> None:
        transformer = object()
        solution = {
            "predictor": "transformer",
            "backend": "onnxruntime",
            "model_directory": "models/transformer",
            "onnxruntime": {"fallback_to_pytorch": True},
        }
        executor = Mock()
        executor.prepare.side_effect = FileNotFoundError("graph missing")
        with (
            patch(
                "match.models.transformer.factory.AutoTokenizer.from_pretrained",
                return_value=object(),
            ),
            patch(
                "pathlib.Path.read_text",
                return_value='{"hidden_size": 16}',
            ),
            patch(
                "match.models.transformer.factory.OnnxRuntimeTransformerExecutor",
                return_value=executor,
            ),
            patch(
                "match.models.transformer.predictor.TransformerPredictor.load",
                return_value=transformer,
            ) as load_pytorch,
        ):
            result = build_predictor(solution, self.root)

        self.assertIs(result, transformer)
        load_pytorch.assert_called_once()

    def test_onnx_fallback_does_not_hide_configuration_errors(self) -> None:
        solution = {
            "predictor": "transformer",
            "backend": "onnxruntime",
            "model_directory": "models/transformer",
            "onnxruntime": {"fallback_to_pytorch": True},
        }
        with (
            patch(
                "match.models.transformer.factory.AutoTokenizer.from_pretrained",
                side_effect=ValueError("invalid tokenizer configuration"),
            ),
            patch(
                "pathlib.Path.read_text",
                return_value='{"hidden_size": 16}',
            ),
            patch(
                "match.models.transformer.predictor.TransformerPredictor.load"
            ) as load_pytorch,
            self.assertRaisesRegex(ValueError, "invalid tokenizer"),
        ):
            build_predictor(solution, self.root)

        load_pytorch.assert_not_called()

    def test_onnx_factory_passes_runtime_options_to_executor(self) -> None:
        solution = {
            "predictor": "transformer",
            "backend": "onnxruntime",
            "model_directory": "models/transformer",
            "onnxruntime": {
                "provider": "cuda",
                "device_id": 2,
                "fallback_to_pytorch": False,
                "tensorrt": {
                    "engine_cache": {
                        "enabled": True,
                        "path": "cache/engines",
                    },
                    "timing_cache": {
                        "enabled": True,
                        "path": "cache/timing",
                    },
                    "profiles": {
                        "min_batch_size": 1,
                        "opt_batch_size": 512,
                        "max_batch_size": 1024,
                        "sequence_lengths": [64, 96, 128],
                    },
                },
            },
            "onnx_artifacts": {"precision": "float16"},
        }
        executor = Mock()
        predictor = object()
        with (
            patch(
                "match.models.transformer.factory.AutoTokenizer.from_pretrained",
                return_value=SimpleNamespace(
                    model_input_names=[
                        "input_ids",
                        "attention_mask",
                        "token_type_ids",
                    ]
                ),
            ),
            patch(
                "pathlib.Path.read_text",
                return_value='{"hidden_size": 16}',
            ),
            patch(
                "match.models.transformer.factory.OnnxRuntimeTransformerExecutor",
                return_value=executor,
            ) as executor_type,
            patch(
                "match.models.transformer.factory.TransformerPredictor",
                return_value=predictor,
            ),
        ):
            result = build_predictor(solution, self.root)

        self.assertIs(result, predictor)
        executor_kwargs = executor_type.call_args.kwargs
        self.assertEqual(executor_kwargs["device_id"], 2)
        tensorrt = executor_kwargs["tensorrt"]
        self.assertEqual(
            tensorrt.engine_cache_path,
            self.root / "models/transformer/cache/engines",
        )
        self.assertEqual(tensorrt.profile.opt_batch_size, 512)
        self.assertEqual(tensorrt.profile.max_batch_size, 1024)
        self.assertEqual(tensorrt.profile.sequence_lengths, (64, 96, 128))
        self.assertIn("token_type_ids", tensorrt.profile.input_names)
        self.assertTrue(tensorrt.fp16_enabled)
        executor.prepare.assert_called_once_with(classifier=True, encoder=False)
        executor.warmup.assert_called_once_with(classifier=True, encoder=False)

    def test_native_tensorrt_factory_passes_build_options(self) -> None:
        solution = {
            "predictor": "transformer",
            "backend": "tensorrt",
            "model_directory": "models/transformer",
            "batch_size": 256,
            "tensorrt": {
                "device_id": 1,
                "workspace_size_gb": 4,
                "builder_optimization_level": 2,
                "fallback_to_onnxruntime": False,
                "engine_cache": {"path": "onnx/native-cache"},
                "timing_cache": {"path": "onnx/native-cache/timing.cache"},
                "profiles": {
                    "min_batch_size": 1,
                    "opt_batch_size": 256,
                    "max_batch_size": 512,
                    "sequence_lengths": [1, 256, 472],
                },
            },
            "onnx_artifacts": {
                "precision": "float16",
                "classifier_path": "onnx/classifier.onnx",
            },
        }
        executor = Mock()
        predictor = object()
        with (
            patch(
                "match.models.transformer.factory.AutoTokenizer.from_pretrained",
                return_value=object(),
            ),
            patch(
                "pathlib.Path.read_text",
                return_value='{"hidden_size": 768, "match_max_length": 472}',
            ),
            patch(
                "match.models.transformer.factory.TensorRTTransformerExecutor",
                return_value=executor,
            ) as executor_type,
            patch(
                "match.models.transformer.factory.TransformerPredictor",
                return_value=predictor,
            ),
        ):
            result = build_predictor(solution, self.root)

        self.assertIs(result, predictor)
        options = executor_type.call_args.kwargs["options"]
        self.assertEqual(options.device_id, 1)
        self.assertEqual(options.workspace_size_bytes, 4 * 1024**3)
        self.assertEqual(options.builder_optimization_level, 2)
        self.assertEqual(options.profile.max_batch_size, 512)
        self.assertEqual(options.profile.max_sequence_length, 472)
        self.assertTrue(options.fp16_enabled)
        executor.prepare.assert_called_once_with(classifier=True, encoder=False)
        executor.warmup.assert_called_once_with(classifier=True, encoder=False)

    def test_native_tensorrt_fallback_builds_onnx_directly(self) -> None:
        solution = {
            "predictor": "transformer",
            "backend": "tensorrt",
            "model_directory": "models/transformer",
            "batch_size": 256,
            "tensorrt": {"fallback_to_onnxruntime": True},
            "onnxruntime": {"device_id": 2},
            "onnx_artifacts": {"precision": "float16"},
        }
        onnx_executor = Mock()
        predictor = object()
        with (
            patch(
                "match.models.transformer.factory.AutoTokenizer.from_pretrained",
                return_value=SimpleNamespace(
                    model_input_names=["input_ids", "attention_mask"]
                ),
            ),
            patch(
                "pathlib.Path.read_text",
                return_value='{"hidden_size": 768, "match_max_length": 472}',
            ),
            patch(
                "match.models.transformer.factory.TensorRTTransformerExecutor",
                side_effect=TensorRTInitializationError("engine failed"),
            ),
            patch(
                "match.models.transformer.factory.OnnxRuntimeTransformerExecutor",
                return_value=onnx_executor,
            ) as onnx_type,
            patch(
                "match.models.transformer.factory.TransformerPredictor",
                return_value=predictor,
            ),
        ):
            result = build_predictor(solution, self.root)

        self.assertIs(result, predictor)
        self.assertEqual(onnx_type.call_args.kwargs["provider"], "cuda")
        self.assertEqual(onnx_type.call_args.kwargs["device_id"], 2)
        onnx_executor.prepare.assert_called_once_with(classifier=True, encoder=False)
        onnx_executor.warmup.assert_called_once_with(classifier=True, encoder=False)

    def test_loads_only_maxpooling_for_maxpooling_prediction(self) -> None:
        maxpooling = object()
        with (
            patch(
                "match.models.transformer.predictor.TransformerPredictor.load"
            ) as load_transformer,
            patch(
                "match.models.maxpooling.predictor.MaxPoolingPredictor.load",
                return_value=maxpooling,
            ) as load_maxpooling,
        ):
            result = build_predictor(
                {
                    "predictor": "maxpooling",
                    "maxpooling_path": "models/maxpooling.joblib",
                },
                self.root,
            )

        self.assertIs(result, maxpooling)
        load_transformer.assert_not_called()
        load_maxpooling.assert_called_once_with(
            self.root / "models" / "maxpooling.joblib",
            batch_size=512,
            device=None,
        )

    def test_composes_fusion_from_both_encoders(self) -> None:
        transformer = object()
        maxpooling = object()
        fusion = object()
        with (
            patch(
                "match.models.transformer.predictor.TransformerPredictor.load",
                return_value=transformer,
            ),
            patch(
                "match.models.maxpooling.predictor.MaxPoolingPredictor.load",
                return_value=maxpooling,
            ),
            patch(
                "match.models.fusion.predictor.FusionPredictor.load",
                return_value=fusion,
            ) as load_fusion,
        ):
            result = build_predictor(
                {
                    "predictor": "fusion",
                    "model_directory": "models/transformer",
                    "maxpooling_path": "models/maxpooling.joblib",
                    "fusion_path": "models/fusion.pt",
                },
                self.root,
            )

        self.assertIs(result, fusion)
        load_fusion.assert_called_once_with(
            self.root / "models" / "fusion.pt",
            transformer=transformer,
            maxpooling=maxpooling,
            batch_size=512,
            device=None,
        )

    def test_composes_cascade_from_boosting_and_transformer(self) -> None:
        boosting = object()
        transformer = object()
        cascade = object()
        with (
            patch(
                "match.models.boosting.predictor.BoostingPredictor.load",
                return_value=boosting,
            ) as load_boosting,
            patch(
                "match.models.transformer.predictor.TransformerPredictor.load",
                return_value=transformer,
            ) as load_transformer,
            patch(
                "match.models.cascade.predictor.CascadePredictor",
                return_value=cascade,
            ) as create_cascade,
        ):
            result = build_predictor(
                {
                    "predictor": "cascade",
                    "fast_model": "boosting",
                    "main_model": "transformer",
                    "boosting_directory": "models/boosting",
                    "model_directory": "models/transformer",
                    "negative_threshold": 0.02,
                    "positive_threshold": 0.98,
                },
                self.root,
            )

        self.assertIs(result, cascade)
        load_boosting.assert_called_once_with(
            self.root / "models" / "boosting",
            thread_count=-1,
        )
        load_transformer.assert_called_once_with(
            self.root / "models" / "transformer",
            batch_size=64,
            dtype="float32",
            num_workers=0,
            prefetch_factor=2,
            pin_memory=True,
            non_blocking_transfer=True,
            length_bucketing=False,
            padding_length_buckets=None,
            batch_fields=False,
            field_chunk_size=16384,
            compile_enabled=False,
            compile_mode="reduce-overhead",
            compile_dynamic=True,
            device=None,
        )
        create_cascade.assert_called_once_with(
            boosting,
            transformer,
            negative_threshold=0.02,
            positive_threshold=0.98,
        )

    def test_composes_stacking_from_transformer_and_catboost(self) -> None:
        transformer = object()
        stacking = object()
        with (
            patch(
                "match.models.transformer.predictor.TransformerPredictor.load",
                return_value=transformer,
            ) as load_transformer,
            patch(
                "match.models.stacking.predictor.StackingPredictor.load",
                return_value=stacking,
            ) as load_stacking,
        ):
            result = build_predictor(
                {
                    "predictor": "stacking",
                    "base_model": "transformer",
                    "stacking_model": "boosting",
                    "model_directory": "models/transformer",
                    "stacking_directory": "models/stacking",
                    "batch_size": 32,
                    "stacking_thread_count": 4,
                },
                self.root,
            )

        self.assertIs(result, stacking)
        load_transformer.assert_called_once_with(
            self.root / "models" / "transformer",
            batch_size=32,
            dtype="float32",
            num_workers=0,
            prefetch_factor=2,
            pin_memory=True,
            non_blocking_transfer=True,
            length_bucketing=False,
            padding_length_buckets=None,
            batch_fields=False,
            field_chunk_size=16384,
            compile_enabled=False,
            compile_mode="reduce-overhead",
            compile_dynamic=True,
            device=None,
        )
        load_stacking.assert_called_once_with(
            self.root / "models" / "stacking",
            transformer=transformer,
            thread_count=4,
        )

    def test_rejects_unknown_predictor(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported predictor"):
            build_predictor({"predictor": "unknown"}, self.root)

    def test_inference_averages_shuffled_copies_per_source_pair(self) -> None:
        items = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["one", "two", "three"],
                "category": ["category"] * 3,
                "attributes": [
                    '{"a":"1","b":"2","c":"3"}',
                    '{"a":"1","b":"2","c":"3"}',
                    '{"a":"1","b":"2","c":"3"}',
                ],
            }
        )
        matches = pl.DataFrame({"id1": [1, 2], "id2": [2, 3]})
        predictor = Mock()
        predictor.predict_proba.return_value = np.asarray(
            [0.2, 0.4, 0.6, 0.8],
            dtype=np.float32,
        )
        solution = {
            "augmentation_model": "attribute_shuffle",
            "augmentation_models": {
                "attribute_shuffle": {
                    "shuffled_copies": 2,
                    "keep_original": False,
                    "seed": 42,
                    "shuffle_cards_independently": True,
                    "skip_oversized": True,
                }
            },
        }
        with (
            patch(
                "match.submission.prepare_manifest_items",
                return_value=SimpleNamespace(
                    frame=items,
                    attributes_column="attributes",
                ),
            ),
            patch(
                "match.submission.build_predictor",
                return_value=predictor,
            ),
        ):
            result = _predict(items, matches, solution, self.root)

        np.testing.assert_allclose(result, [0.3, 0.7])
        augmented_batch = predictor.predict_proba.call_args.args[0]
        self.assertEqual(augmented_batch.matches.height, 4)
        self.assertTrue(
            all(pair.preserve_attribute_order for pair in augmented_batch.pairs)
        )


if __name__ == "__main__":
    unittest.main()
