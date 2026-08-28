import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import polars as pl

from match.benchmarks.suite import apply_benchmark_test, load_benchmark_suite
from match.benchmarks.training_quality import run_training_quality_benchmark
from match.config import load_app_config_file
from match.paths import PROJECT_ROOT


class TrainingQualityBenchmarkTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_app_config_file(PROJECT_ROOT / "configs" / "pipeline.yaml")
        self.tests_path = (
            PROJECT_ROOT
            / "configs"
            / "benchmark_tests"
            / "transitivity_quality.yaml"
        )
        self.codex_tests_path = (
            PROJECT_ROOT
            / "configs"
            / "benchmark_tests"
            / "codex_annotation_quality.yaml"
        )

    def test_codex_suite_compares_equal_size_training_sets(self) -> None:
        suite = load_benchmark_suite(
            self.codex_tests_path,
            allowed_test_types={"training_quality"},
        )
        baseline = apply_benchmark_test(
            self.config,
            suite.tests["human_llm"],
        )
        fixed = apply_benchmark_test(
            self.config,
            suite.tests["human_llm_codex_fixed_budget"],
        )
        fixed_llm = next(
            source
            for source in fixed.data_model_description.mix_dataset_codex.sources
            if source.name == "llm"
        )

        self.assertEqual(suite.reference_test, "human_llm")
        self.assertEqual(
            list(suite.tests),
            ["human_llm", "human_llm_codex_fixed_budget"],
        )
        self.assertEqual(baseline.training.data_model, "mix_dataset")
        self.assertEqual(fixed.training.data_model, "mix_dataset_codex")
        self.assertEqual(fixed_llm.max_rows, 724_772)
        self.assertEqual(suite.seeds, (42,))

    def test_suite_changes_the_intended_weighting_switch(self) -> None:
        suite = load_benchmark_suite(
            self.tests_path,
            allowed_test_types={"training_quality"},
        )
        baseline = apply_benchmark_test(
            self.config,
            suite.tests["transitivity_off"],
        )
        weighted = apply_benchmark_test(
            self.config,
            suite.tests["transitivity_on"],
        )
        baseline_llm = next(
            source
            for source in baseline.data_model_description.mix_dataset.sources
            if source.name == "llm"
        )
        weighted_llm = next(
            source
            for source in weighted.data_model_description.mix_dataset.sources
            if source.name == "llm"
        )

        self.assertEqual(baseline.training.model, "transformer")
        self.assertEqual(baseline.training.data_model, "mix_dataset")
        self.assertFalse(baseline_llm.weight_model.enabled)
        self.assertTrue(weighted_llm.weight_model.enabled)
        self.assertEqual(suite.seeds, (42,))

    def test_writes_paired_metric_delta_and_aggregate(self) -> None:
        def fake_case(_config, test, seed, *, experiments_root):
            del experiments_root
            metric = 0.70 if test.key == "transitivity_off" else 0.73
            return {
                "test": test.key,
                "name": test.name,
                "seed": seed,
                "experiment_name": f"{test.key}-seed-{seed}",
                "status": "completed",
                "validation_macro_pr_auc": metric,
                "delta_validation_macro_pr_auc": None,
                "validation_pairs_hash": "same-split",
                "same_validation_split_as_reference": None,
                "training_seconds": 10.0,
                "total_train_rows": 100,
                "human_train_rows": 20,
                "llm_train_rows": 80,
                "codex_reviewed_train_rows": 0,
                "llm_mean_weight_multiplier": 0.8,
                "llm_downweighted_fraction": 0.2,
                "llm_violating_fraction": 0.1,
                "experiment_path": "experiment",
                "error": None,
            }

        with tempfile.TemporaryDirectory() as directory, patch(
            "match.benchmarks.training_quality._run_case",
            side_effect=fake_case,
        ):
            result = run_training_quality_benchmark(
                self.config,
                tests_path=self.tests_path,
                output_dir=Path(directory) / "run",
            )
            rows = pl.read_parquet(result.results_path)
            aggregates = pl.read_parquet(result.aggregates_path)
            report = json.loads(result.report_path.read_text(encoding="utf-8"))

        weighted = rows.filter(pl.col("test") == "transitivity_on")
        self.assertAlmostEqual(
            weighted.get_column("delta_validation_macro_pr_auc").item(),
            0.03,
        )
        self.assertTrue(
            weighted.get_column("same_validation_split_as_reference").item()
        )
        self.assertAlmostEqual(
            aggregates.filter(pl.col("test") == "transitivity_on")
            .get_column("mean_delta_validation_macro_pr_auc")
            .item(),
            0.03,
        )
        self.assertEqual(
            aggregates.filter(pl.col("test") == "transitivity_on")
            .get_column("total_training_seconds")
            .item(),
            10.0,
        )
        self.assertEqual(
            aggregates.filter(pl.col("test") == "transitivity_on")
            .get_column("mean_llm_train_rows")
            .item(),
            80.0,
        )
        self.assertGreaterEqual(report["total_elapsed_seconds"], 0.0)
        self.assertEqual(report["reference_test"], "transitivity_off")


if __name__ == "__main__":
    unittest.main()
