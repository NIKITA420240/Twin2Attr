import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from match.batch_profiles import (
    BatchMeasurement,
    PerformanceEnvironment,
    PerformanceWorkload,
    build_profile,
    load_profile,
    model_content_hash,
    profile_fingerprint,
    recommended_starting_batch_size,
    update_profile,
)


class BatchPerformanceProfileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.environment = PerformanceEnvironment(
            gpu="NVIDIA H100 80GB",
            vram_bytes=85_899_345_920,
            torch="2.13.0+cu130",
            transformers="4.57.6",
            cuda="13.0",
        )
        self.workload = PerformanceWorkload(
            model_hash="model-hash",
            mode="inference",
            dtype="bfloat16",
            attention="sdpa",
            torch_compile=True,
            max_length=128,
            padding_buckets=(32, 64, 96, 128),
        )

    def test_selects_fastest_batch_and_applies_safety_fraction(self) -> None:
        profile = build_profile(
            self.environment,
            self.workload,
            (
                BatchMeasurement(128, "completed", tokens_per_second=121_000),
                BatchMeasurement(256, "completed", tokens_per_second=187_000),
                BatchMeasurement(384, "oom"),
            ),
        )

        self.assertEqual(profile.best_batch_size, 256)
        self.assertEqual(profile.safe_batch_size, 224)
        self.assertEqual(
            [measurement.status for measurement in profile.measurements],
            ["completed", "completed", "oom"],
        )

    def test_updates_existing_profile_by_batch_size(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = update_profile(
                root,
                self.environment,
                self.workload,
                (BatchMeasurement(128, "completed", tokens_per_second=100.0),),
            )
            second = update_profile(
                root,
                self.environment,
                self.workload,
                (
                    BatchMeasurement(128, "completed", tokens_per_second=120.0),
                    BatchMeasurement(256, "completed", tokens_per_second=150.0),
                ),
            )
            restored = load_profile(root, self.environment, self.workload)

        self.assertIsNotNone(restored)
        self.assertEqual(first.safe_batch_size, 112)
        self.assertEqual(second.best_batch_size, 256)
        self.assertEqual(second.safe_batch_size, 224)
        self.assertEqual(
            [measurement.tokens_per_second for measurement in restored.measurements],
            [120.0, 150.0],
        )

    def test_recommends_only_exact_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            update_profile(
                root,
                self.environment,
                self.workload,
                (BatchMeasurement(256, "completed", tokens_per_second=150.0),),
            )
            self.assertEqual(
                recommended_starting_batch_size(
                    root,
                    self.environment,
                    self.workload,
                ),
                224,
            )
            self.assertIsNone(
                recommended_starting_batch_size(
                    root,
                    self.environment,
                    replace(self.workload, attention="eager"),
                )
            )

    def test_model_hash_tracks_weight_content(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(
                json.dumps({"model_type": "test"}), encoding="utf-8"
            )
            weights = root / "model.safetensors"
            weights.write_bytes(b"first")
            first = model_content_hash(root)
            weights.write_bytes(b"second")
            second = model_content_hash(root)

        self.assertNotEqual(first, second)

    def test_fingerprint_is_stable(self) -> None:
        self.assertEqual(
            profile_fingerprint(self.environment, self.workload),
            profile_fingerprint(self.environment, self.workload),
        )


if __name__ == "__main__":
    unittest.main()
