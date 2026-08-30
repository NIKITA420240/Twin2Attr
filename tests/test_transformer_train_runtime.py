import unittest
from unittest.mock import Mock

import numpy as np
import torch

from match.models.transformer.train_runtime import (
    EpochPerformanceTracker,
    LengthAwareSampler,
    stratified_sample_indices,
)
from match.models.transformer.model import WeightedSequenceTrainer
from match.models.transformer.config import SequenceClassifierConfig
from match.models.transformer.training import (
    _select_fast_dev_typed_features,
    _training_arguments,
)


class LengthAwareSamplerTests(unittest.TestCase):
    def test_visits_every_example_once_and_groups_similar_lengths(self) -> None:
        sampler = LengthAwareSampler(
            list(range(1, 33)),
            batch_size=4,
            mega_batch_multiplier=8,
            seed=13,
        )

        indices = list(sampler)

        self.assertEqual(sorted(indices), list(range(32)))
        for start in range(0, len(indices), 4):
            batch_lengths = [index + 1 for index in indices[start : start + 4]]
            self.assertLessEqual(max(batch_lengths) - min(batch_lengths), 3)

    def test_epoch_changes_the_randomized_order(self) -> None:
        sampler = LengthAwareSampler(
            [index % 7 + 1 for index in range(64)],
            batch_size=4,
            mega_batch_multiplier=4,
            seed=13,
        )

        first_epoch = list(sampler)
        sampler.set_epoch(1)
        second_epoch = list(sampler)

        self.assertNotEqual(first_epoch, second_epoch)
        self.assertEqual(sorted(second_epoch), list(range(64)))


class EpochPerformanceTrackerTests(unittest.TestCase):
    def test_aggregates_padding_metrics_at_epoch_end(self) -> None:
        tracker = EpochPerformanceTracker()
        tracker.start_epoch()
        tracker.observe(torch.tensor([[1, 1, 0], [1, 0, 0]]))
        tracker.observe(torch.tensor([[1, 1, 1]]))

        record = tracker.finish_epoch(1)

        self.assertIsNotNone(record)
        assert record is not None
        self.assertEqual(record["epoch"], 1)
        self.assertEqual(record["examples"], 3)
        self.assertEqual(record["real_tokens"], 6)
        self.assertEqual(record["padded_tokens"], 9)
        self.assertAlmostEqual(record["padding_efficiency"], 6 / 9)
        self.assertEqual(len(tracker.history), 1)


class FastDevSamplingTests(unittest.TestCase):
    def test_tuple_indices_select_rows_from_typed_feature_matrix(self) -> None:
        features = np.arange(20, dtype=np.float32).reshape(5, 4)

        selected = _select_fast_dev_typed_features(features, (3, 1))

        np.testing.assert_array_equal(selected, features[[3, 1]])

    def test_is_deterministic_class_stratified_and_bounded(self) -> None:
        labels = [0] * 80 + [1] * 20

        first = stratified_sample_indices(labels, max_rows=20, seed=7)
        second = stratified_sample_indices(labels, max_rows=20, seed=7)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 20)
        sampled_labels = [labels[index] for index in first]
        self.assertEqual(sampled_labels.count(0), 16)
        self.assertEqual(sampled_labels.count(1), 4)

    def test_fast_dev_uses_a_separate_eval_dataloader_cache_key(self) -> None:
        trainer = object.__new__(WeightedSequenceTrainer)
        validation_dataset = object()
        fast_dev_dataset = object()
        trainer.eval_dataset = validation_dataset
        trainer.fast_dev_dataset = fast_dev_dataset
        selected_dataset = {}

        def get_eval_dataloader(name):
            selected_dataset["value"] = trainer.eval_dataset[name]
            return "fast-dev-loader"

        trainer.get_eval_dataloader = Mock(side_effect=get_eval_dataloader)

        validation_metric = Mock()
        trainer.compute_metrics = validation_metric
        trainer.fast_dev_compute_metrics = Mock()
        trainer.evaluation_loop = Mock(
            return_value=type(
                "EvaluationOutput",
                (),
                {"metrics": {}, "num_samples": 1},
            )()
        )
        trainer.log = Mock()
        trainer.state = type("TrainerState", (), {"global_step": 1})()

        trainer._run_fast_dev_evaluation()

        trainer.get_eval_dataloader.assert_called_once_with("fast_dev")
        self.assertIs(selected_dataset["value"], fast_dev_dataset)
        self.assertIs(trainer.eval_dataset, validation_dataset)
        self.assertIs(trainer.compute_metrics, validation_metric)


class TrainingArgumentsTests(unittest.TestCase):
    def test_passes_dataloader_and_compile_options_to_transformers(self) -> None:
        config = SequenceClassifierConfig(
            model_path="model",
            dataloader_num_workers=2,
            dataloader_prefetch_factor=3,
            dataloader_persistent_workers=True,
            dataloader_pin_memory=False,
            torch_compile=True,
            torch_compile_mode="reduce-overhead",
        )

        arguments = _training_arguments(
            "artifacts",
            config,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.assertEqual(arguments.dataloader_num_workers, 2)
        self.assertEqual(arguments.dataloader_prefetch_factor, 3)
        self.assertTrue(arguments.dataloader_persistent_workers)
        self.assertFalse(arguments.dataloader_pin_memory)
        self.assertTrue(arguments.torch_compile)
        self.assertEqual(arguments.torch_compile_mode, "reduce-overhead")
