import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import polars as pl
import torch

from match.models import (
    FusionPredictor,
    MatchPredictor,
    MaxPoolingPredictor,
    PairEncoder,
    PredictionBatch,
    TransformerPredictor,
)
from match.models.transformer.predictor import _CompiledForward
from match.models.transformer.executor import TransformerExecutorOutOfMemoryError
from match.models.transformer.batching import TransformerBatchingSettings
from match.models.transformer.pytorch_executor import PyTorchTransformerExecutor


class _FakeExecutor:
    device = torch.device("cpu")
    output_dim = 16
    max_length = 32
    use_field_tokens = False
    max_attribute_value_chars = None
    max_attribute_value_tokens = 16

    def predict_logits(self, batch, *, non_blocking):
        del batch, non_blocking
        return np.array([[0.0, 1.0]], dtype=np.float32)

    def encode_cls(self, batch, *, non_blocking):
        del batch, non_blocking
        return np.ones((1, 16), dtype=np.float32)

    def prepare(self, *, classifier, encoder):
        del classifier, encoder

    def warmup(self, *, classifier, encoder):
        del classifier, encoder

    def clear_cache(self):
        pass


def _batch() -> PredictionBatch:
    return PredictionBatch(
        items=pl.DataFrame({"id": [1], "attributes": ["{}"]}),
        matches=pl.DataFrame({"id1": [1], "id2": [1]}),
        pairs=[object()],
        attributes_column="attributes",
    )


class ModelInterfaceTests(unittest.TestCase):
    def test_transformer_executes_every_token_budget_sub_batch(self) -> None:
        executor = _FakeExecutor()
        predictor = TransformerPredictor(
            object(),
            executor,
            batch_size=512,
            max_tokens_per_batch=32_768,
        )
        collated = [
            {"input_ids": torch.zeros((3, 32), dtype=torch.int64)},
            {"input_ids": torch.zeros((2, 96), dtype=torch.int64)},
        ]

        def infer(batch, *, non_blocking):
            del non_blocking
            return np.ones((batch["input_ids"].shape[0], 1), dtype=np.float32)

        with (
            patch.object(TransformerBatchingSettings, "collator", return_value=object()),
            patch(
                "match.models.transformer.predictor.inference_dataset",
                return_value=([object()] * 5, None),
            ),
            patch(
                "match.models.transformer.predictor.inference_loader",
                return_value=([collated], False),
            ),
        ):
            result = predictor._execute_pairs(
                [object()] * 5,
                operation=infer,
                output_width=1,
            )

        self.assertEqual(result.shape, (5, 1))

    def test_transformer_can_fail_fast_instead_of_reducing_oom_batch(self) -> None:
        executor = _FakeExecutor()
        predictor = TransformerPredictor(
            object(),
            executor,
            batch_size=256,
            retry_on_oom=False,
        )

        def fail(batch, *, non_blocking):
            del batch, non_blocking
            raise TransformerExecutorOutOfMemoryError

        with (
            patch.object(TransformerBatchingSettings, "collator", return_value=object()),
            patch(
                "match.models.transformer.predictor.inference_dataset",
                return_value=([object()], None),
            ),
            patch(
                "match.models.transformer.predictor.inference_loader",
                return_value=([{}], False),
            ),
            self.assertRaisesRegex(
                TransformerExecutorOutOfMemoryError,
                "batch_size=256.*fallback is disabled",
            ),
        ):
            predictor._execute_pairs(
                [object()],
                operation=fail,
                output_width=1,
            )

    def test_transformer_retries_with_half_batch_after_oom(self) -> None:
        executor = _FakeExecutor()
        predictor = TransformerPredictor(
            object(),
            executor,
            batch_size=256,
            retry_on_oom=True,
        )
        attempted_batch_sizes = []
        operation_calls = 0

        def loader(dataset, collator, *, batch_size, **kwargs):
            del dataset, collator, kwargs
            attempted_batch_sizes.append(batch_size)
            return ([{}], False)

        def infer(batch, *, non_blocking):
            nonlocal operation_calls
            del batch, non_blocking
            operation_calls += 1
            if operation_calls == 1:
                raise TransformerExecutorOutOfMemoryError
            return np.array([[1.0]], dtype=np.float32)

        with (
            patch.object(TransformerBatchingSettings, "collator", return_value=object()),
            patch(
                "match.models.transformer.predictor.inference_dataset",
                return_value=([object()], None),
            ),
            patch(
                "match.models.transformer.predictor.inference_loader",
                side_effect=loader,
            ),
        ):
            result = predictor._execute_pairs(
                [object()],
                operation=infer,
                output_width=1,
            )

        self.assertEqual(attempted_batch_sizes, [256, 128])
        np.testing.assert_array_equal(result, np.array([[1.0]], dtype=np.float32))

    def test_transformer_compiles_classifier_and_backbone(self) -> None:
        model = SimpleNamespace(
            config=SimpleNamespace(hidden_size=16),
            base_model=object(),
            eval=Mock(),
        )
        with patch(
            "match.models.transformer.pytorch_executor.torch.compile",
            side_effect=lambda module, **kwargs: Mock(),
        ) as compile_model:
            executor = PyTorchTransformerExecutor(
                model,
                compile_enabled=True,
                compile_mode="reduce-overhead",
                compile_dynamic=True,
            )

        self.assertIsNotNone(executor._compiled_model)
        self.assertIsNotNone(executor._compiled_backbone)
        self.assertEqual(compile_model.call_count, 2)
        compile_model.assert_any_call(
            model,
            mode="reduce-overhead",
            dynamic=True,
        )
        compile_model.assert_any_call(
            model.base_model,
            mode="reduce-overhead",
            dynamic=True,
        )

    def test_compiled_forward_falls_back_to_eager_execution(self) -> None:
        eager = Mock(return_value="eager result")
        compiled = Mock(side_effect=RuntimeError("unsupported graph"))
        with patch(
            "match.models.transformer.pytorch_executor.torch.compile",
            return_value=compiled,
        ):
            forward = _CompiledForward(
                eager,
                name="test model",
                mode="reduce-overhead",
                dynamic=True,
            )

        result = forward(input_ids=Mock())

        self.assertEqual(result, "eager result")
        compiled.assert_called_once()
        eager.assert_called_once()

    def test_transformer_supports_encoding_and_prediction(self) -> None:
        predictor = TransformerPredictor(object(), _FakeExecutor(), batch_size=8)
        encoded = np.ones((1, 16), dtype=np.float32)
        probabilities = np.array([0.8], dtype=np.float32)

        with (
            patch.object(TransformerPredictor, "encode", return_value=encoded),
            patch.object(
                TransformerPredictor,
                "predict_proba",
                return_value=probabilities,
            ),
        ):
            self.assertIs(predictor.encode(_batch()), encoded)
            self.assertIs(predictor.predict_proba(_batch()), probabilities)

        self.assertIsInstance(predictor, PairEncoder)
        self.assertIsInstance(predictor, MatchPredictor)
        self.assertEqual(predictor.output_dim, 16)

    def test_maxpooling_supports_encoding_and_prediction(self) -> None:
        predictor = MaxPoolingPredictor(
            SimpleNamespace(vector_size=4),
            batch_size=8,
        )
        encoded = np.ones((1, 8), dtype=np.float32)
        probabilities = np.array([0.7], dtype=np.float32)

        with (
            patch(
                "match.models.maxpooling.predictor.encode_attribute_pairs",
                return_value=encoded,
            ),
            patch(
                "match.models.maxpooling.predictor.predict_maxpooling_probabilities",
                return_value=probabilities,
            ),
        ):
            self.assertIs(predictor.encode(_batch()), encoded)
            self.assertIs(predictor.predict_proba(_batch()), probabilities)

        self.assertIsInstance(predictor, PairEncoder)
        self.assertIsInstance(predictor, MatchPredictor)
        self.assertEqual(predictor.output_dim, 8)

    def test_fusion_composes_two_encoders(self) -> None:
        transformer = Mock(spec=PairEncoder)
        maxpooling = Mock(spec=PairEncoder)
        transformer.encode.return_value = np.ones((1, 16), dtype=np.float32)
        maxpooling.encode.return_value = np.ones((1, 8), dtype=np.float32)
        predictor = FusionPredictor(
            model=object(),
            transformer=transformer,
            maxpooling=maxpooling,
        )
        probabilities = np.array([0.9], dtype=np.float32)

        with patch(
            "match.models.fusion.predictor.predict_fusion_probabilities",
            return_value=probabilities,
        ) as predict:
            result = predictor.predict_proba(_batch())

        self.assertIs(result, probabilities)
        transformer.encode.assert_called_once()
        maxpooling.encode.assert_called_once()
        predict.assert_called_once()
        self.assertIsInstance(predictor, MatchPredictor)


if __name__ == "__main__":
    unittest.main()
