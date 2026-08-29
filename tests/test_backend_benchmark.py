import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import polars as pl

from match.analysis.backend_benchmark import (
    BenchmarkOverride,
    BenchmarkTest,
    apply_benchmark_test,
    run_backend_benchmark,
)
from match.config import load_benchmark_config_file
from match.paths import PROJECT_ROOT
from match.prepare_data import PreparedCard, PreparedPair


def _pairs() -> list[PreparedPair]:
    cards = [
        PreparedCard(1, "one", "category", (("brand", "a"),)),
        PreparedCard(2, "two", "category", (("brand", "a"),)),
        PreparedCard(3, "three", "category", (("brand", "b"),)),
    ]
    return [
        PreparedPair(cards[0], cards[1], 1, "category"),
        PreparedPair(cards[0], cards[2], 0, "category"),
    ]


class _Predictor:
    def __init__(self, offset: float) -> None:
        self.offset = offset

    def predict_proba(self, batch) -> np.ndarray:
        base = np.linspace(0.2, 0.8, batch.matches.height)
        return base + self.offset


class BackendBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_benchmark_config_file(
            PROJECT_ROOT / "configs" / "benchmark.yaml"
        )

    def test_applies_dotted_override_to_fresh_validated_config(self) -> None:
        test = BenchmarkTest(
            key="batch",
            name="batch",
            overrides=(
                BenchmarkOverride(
                    "inference.transformer.batch_size",
                    2048,
                ),
            ),
        )

        changed = apply_benchmark_test(self.config, test)

        self.assertEqual(changed.inference.transformer.batch_size, 2048)
        self.assertEqual(self.config.inference.transformer.batch_size, 512)

    def test_rejects_unknown_override_path(self) -> None:
        test = BenchmarkTest(
            key="invalid",
            name="invalid",
            overrides=(BenchmarkOverride("inference.transformer.batc_size", 1),),
        )

        with self.assertRaisesRegex(ValueError, "Unknown benchmark override path"):
            apply_benchmark_test(self.config, test)

    def test_runs_selected_tests_and_writes_comparison(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            solution_path = root / "solution.json"
            solution_path.write_text(
                json.dumps(
                    {
                        "predictor": "transformer",
                        "model_directory": "model",
                    }
                ),
                encoding="utf-8",
            )
            benchmark = replace(
                self.config.analysis_models.backend_benchmark,
                selected_tests=("test_1", "test_2"),
                sample_size=2,
                warmup_batches=1,
                measured_runs=2,
                output_dir=root / "output",
            )
            config = replace(
                self.config,
                inference=replace(
                    self.config.inference,
                    solution_path=solution_path,
                ),
                analysis_models=replace(
                    self.config.analysis_models,
                    backend_benchmark=benchmark,
                ),
            )
            items = pl.DataFrame(
                {
                    "id": [1, 2, 3],
                    "name": ["one", "two", "three"],
                    "category": ["category"] * 3,
                    "attributes": ["{}"] * 3,
                }
            )
            matches = pl.DataFrame(
                {"id1": [1, 1], "id2": [2, 3], "target": [1, 0]}
            )
            data_model = Mock()
            data_model.load_inspection_frames.return_value = SimpleNamespace(
                items=items,
                matches=matches,
            )

            def predictor(solution, _root):
                offset = 0.01 if solution["backend"] == "onnxruntime" else 0.0
                self.assertFalse(solution["onnxruntime"]["fallback_to_pytorch"])
                self.assertFalse(solution["tensorrt"]["fallback_to_onnxruntime"])
                return _Predictor(offset)

            with (
                patch(
                    "match.analysis.backend_benchmark.build_data_model",
                    return_value=data_model,
                ),
                patch(
                    "match.analysis.backend_benchmark.prepare_configured_items",
                    return_value=SimpleNamespace(
                        frame=items,
                        attributes_column="attributes",
                    ),
                ),
                patch(
                    "match.analysis.backend_benchmark.prepare_pair_rows",
                    return_value=_pairs(),
                ),
                patch(
                    "match.analysis.backend_benchmark.build_predictor",
                    side_effect=predictor,
                ),
            ):
                result = run_backend_benchmark(config)

            rows = pl.read_parquet(result.output_path).sort("test")
            report = json.loads(result.report_path.read_text(encoding="utf-8"))

        self.assertEqual(result.completed_tests, 1)
        self.assertEqual(result.failed_tests, 1)
        self.assertEqual(
            rows.get_column("status").to_list(),
            ["completed", "failed"],
        )
        self.assertAlmostEqual(
            rows.filter(pl.col("test") == "test_2")
            .get_column("mean_probability_difference")
            .item(),
            0.01,
        )
        self.assertTrue(
            rows.filter(pl.col("test") == "test_1")
            .get_column("quality_gate_passed")
            .item()
        )
        self.assertFalse(
            rows.filter(pl.col("test") == "test_2")
            .get_column("quality_gate_passed")
            .item()
        )
        self.assertIn(
            "prediction quality gate failed",
            rows.filter(pl.col("test") == "test_2")
            .get_column("error")
            .item(),
        )
        self.assertTrue(rows.get_column("pr_auc").is_not_null().all())
        self.assertEqual(report["test_type"], "speed")
        self.assertEqual(report["reference_test"], "test_1")


if __name__ == "__main__":
    unittest.main()
