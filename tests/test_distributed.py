import os
import unittest
from pathlib import Path
from unittest.mock import patch

from match.config import TransformerDistributedSettings
from match.distributed import (
    current_process,
    validate_distributed_launch,
)
from match.models.transformer.config import SequenceClassifierConfig
from match.models.transformer.training import _training_arguments


class DistributedTrainingTests(unittest.TestCase):
    def test_reads_torchrun_topology(self) -> None:
        with patch.dict(
            os.environ,
            {"WORLD_SIZE": "2", "RANK": "1", "LOCAL_RANK": "1"},
            clear=True,
        ):
            process = current_process()

        self.assertEqual(process.world_size, 2)
        self.assertEqual(process.rank, 1)
        self.assertEqual(process.local_rank, 1)
        self.assertFalse(process.is_main_process)

    def test_rejects_wrong_torchrun_world_size(self) -> None:
        settings = TransformerDistributedSettings(
            enabled=True,
            expected_world_size=2,
        )
        with (
            patch.dict(
                os.environ,
                {"WORLD_SIZE": "1", "RANK": "0", "LOCAL_RANK": "0"},
                clear=True,
            ),
            self.assertRaisesRegex(RuntimeError, "world-size mismatch"),
        ):
            validate_distributed_launch(settings)

    def test_training_arguments_enable_nccl_ddp(self) -> None:
        config = SequenceClassifierConfig(
            "checkpoint",
            hpo_trials=1,
            auto_find_batch_size=False,
            distributed_enabled=True,
            distributed_expected_world_size=2,
            ddp_backend="nccl",
            ddp_find_unused_parameters=False,
        )
        class CapturedTrainingArguments:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        with patch(
            "match.models.transformer.training.TrainingArguments",
            CapturedTrainingArguments,
        ):
            arguments = _training_arguments(
                Path("output"),
                config,
                learning_rate=config.learning_rate,
                weight_decay=config.weight_decay,
            )

        self.assertEqual(arguments.ddp_backend, "nccl")
        self.assertFalse(arguments.ddp_find_unused_parameters)

    def test_distributed_mode_requires_fixed_batch_and_single_hpo_trial(self) -> None:
        with self.assertRaisesRegex(ValueError, "auto_find_batch_size"):
            SequenceClassifierConfig(
                "checkpoint",
                distributed_enabled=True,
                distributed_expected_world_size=2,
            )
        with self.assertRaisesRegex(ValueError, "hpo_trials"):
            SequenceClassifierConfig(
                "checkpoint",
                hpo_trials=2,
                auto_find_batch_size=False,
                distributed_enabled=True,
                distributed_expected_world_size=2,
            )


if __name__ == "__main__":
    unittest.main()
