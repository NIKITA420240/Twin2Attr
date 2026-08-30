import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import polars as pl

from match.benchmarks.suite import apply_benchmark_test, load_benchmark_suite
from match.benchmarks.training_quality import (
    _run_case,
    run_training_quality_benchmark,
)
from match.config import load_app_config_file
from match.models.artifacts import TrainingArtifacts
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
        self.hard_negative_tests_path = (
            PROJECT_ROOT
            / "configs"
            / "benchmark_tests"
            / "hard_negative_quality.yaml"
        )
        self.attribute_word_dropout_tests_path = (
            PROJECT_ROOT
            / "configs"
            / "benchmark_tests"
            / "attribute_word_dropout_quality.yaml"
        )
        self.vote_sampling_tests_path = (
            PROJECT_ROOT
            / "configs"
            / "benchmark_tests"
            / "llm_vote_sampling_quality.yaml"
        )
        self.soft_label_confidence_tests_path = (
            PROJECT_ROOT
            / "configs"
            / "benchmark_tests"
            / "soft_label_confidence_quality.yaml"
        )
        self.neural_relabel_tests_path = (
            PROJECT_ROOT
            / "configs"
            / "benchmark_tests"
            / "neural_relabel_quality.yaml"
        )

    def test_run_case_uses_current_experiment_tracking_contract(self) -> None:
        typed_tests_path = (
            PROJECT_ROOT
            / "configs"
            / "benchmark_tests"
            / "typed_attribute_quality.yaml"
        )
        suite = load_benchmark_suite(
            typed_tests_path,
            allowed_test_types={"training_quality"},
        )

        def fake_train(config, *, experiment_name):
            experiment_dir = config.training.solution_path.parent
            experiment_dir.mkdir(parents=True)
            (experiment_dir / "experiment.json").write_text(
                json.dumps(
                    {
                        "split": {
                            "validation_pairs_hash": "same-split",
                            "train_rows": 10,
                        },
                        "sample_weighting": {},
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(experiment_name, "typed_attributes_off-seed-42")
            return TrainingArtifacts(
                predictor="boosting",
                boosting_dir=config.model_description.boosting.artifact_dir,
                metrics=(("boosting.validation_macro_pr_auc", 0.7),),
            )

        with tempfile.TemporaryDirectory() as directory, patch(
            "match.benchmarks.training_quality.train",
            side_effect=fake_train,
        ):
            row = _run_case(
                self.config,
                suite.tests["typed_attributes_off"],
                42,
                experiments_root=Path(directory),
            )

        self.assertEqual(row["status"], "completed")
        self.assertEqual(row["validation_macro_pr_auc"], 0.7)
        self.assertEqual(row["validation_pairs_hash"], "same-split")

    def test_hard_negative_suite_is_additive_three_epoch_ablation(self) -> None:
        suite = load_benchmark_suite(
            self.hard_negative_tests_path,
            allowed_test_types={"training_quality"},
        )
        configured = {
            key: apply_benchmark_test(self.config, test)
            for key, test in suite.tests.items()
        }

        self.assertEqual(suite.reference_test, "human_llm")
        self.assertEqual(
            list(suite.tests),
            ["human_llm", "human_llm_hard_negative"],
        )
        expected_factors = {
            "human_llm": ("mix_dataset", None),
            "human_llm_hard_negative": (
                "mix_dataset_hard_negative",
                None,
            ),
        }
        self.assertEqual(
            {
                key: (
                    config.training.data_model,
                    config.training.augmentation_model,
                )
                for key, config in configured.items()
            },
            expected_factors,
        )
        experiment = configured["human_llm_hard_negative"]
        self.assertEqual(
            {
                config.model_description.transformer.max_epochs
                for config in configured.values()
            },
            {3},
        )
        hard_source = next(
            source
            for source in experiment.data_model_description
            .mix_dataset_hard_negative.sources
            if source.name == "hard_negative"
        )
        self.assertEqual(hard_source.max_rows, 130_000)
        self.assertEqual(hard_source.weight, 0.5)
        self.assertEqual(suite.seeds, (42,))

    def test_attribute_word_dropout_suite_is_independent_factorial(self) -> None:
        suite = load_benchmark_suite(
            self.attribute_word_dropout_tests_path,
            allowed_test_types={"training_quality"},
        )
        configured = {
            key: apply_benchmark_test(self.config, test)
            for key, test in suite.tests.items()
        }

        self.assertEqual(suite.reference_test, "human_llm")
        self.assertEqual(
            {
                key: (
                    config.training.data_model,
                    config.training.augmentation_model,
                    config.model_description.transformer.max_epochs,
                )
                for key, config in configured.items()
            },
            {
                "human_llm": ("mix_dataset", None, 3),
                "human_llm_word_dropout": (
                    "mix_dataset",
                    "attribute_word_dropout",
                    3,
                ),
                "human_llm_hard_negative": (
                    "mix_dataset_hard_negative",
                    None,
                    3,
                ),
                "human_llm_hard_negative_word_dropout": (
                    "mix_dataset_hard_negative",
                    "attribute_word_dropout",
                    3,
                ),
            },
        )
        self.assertEqual(suite.seeds, (42,))

    def test_neural_relabel_suite_has_three_equal_budget_arms(self) -> None:
        suite = load_benchmark_suite(
            self.neural_relabel_tests_path,
            allowed_test_types={"training_quality"},
        )
        configured = {
            key: apply_benchmark_test(self.config, test)
            for key, test in suite.tests.items()
        }

        self.assertEqual(suite.reference_test, "old_labels_fixed_pairs")
        self.assertEqual(
            list(suite.tests),
            [
                "current_pipeline",
                "old_labels_fixed_pairs",
                "new_labels_fixed_pairs",
            ],
        )
        self.assertEqual(
            configured["current_pipeline"].training.data_model,
            "mix_dataset",
        )
        self.assertEqual(
            {
                config.model_description.transformer.max_epochs
                for config in configured.values()
            },
            {4},
        )
        for key, target_column in (
            ("old_labels_fixed_pairs", "source_score"),
            ("new_labels_fixed_pairs", "llm_score"),
        ):
            config = configured[key]
            self.assertEqual(
                config.training.data_model,
                "mix_dataset_neural_review",
            )
            sources = {
                source.name: source
                for source in config.data_model_description
                .mix_dataset_neural_review.sources
            }
            self.assertEqual(sources["llm"].max_rows, 649_663)
            self.assertEqual(
                sources["neural_review"].target_column,
                target_column,
            )
            self.assertFalse(sources["llm"].weight_model.enabled)
            self.assertFalse(sources["neural_review"].weight_model.enabled)
            self.assertFalse(
                sources["neural_review"].confidence_weighting.enabled
            )
        self.assertEqual(suite.seeds, (42,))

    def test_soft_label_confidence_suite_is_full_factorial(self) -> None:
        suite = load_benchmark_suite(
            self.soft_label_confidence_tests_path,
            allowed_test_types={"training_quality"},
        )
        combinations = {}
        transitivity_settings = set()
        for key, test in suite.tests.items():
            configured = apply_benchmark_test(self.config, test)
            llm = next(
                source
                for source in configured.data_model_description.mix_dataset.sources
                if source.name == "llm"
            )
            combinations[key] = (
                llm.splitter.target_mode,
                llm.confidence_weighting.enabled,
            )
            transitivity_settings.add(
                (
                    llm.weight_model.enabled,
                    llm.weight_model.penalty_strength,
                    llm.weight_model.min_weight_multiplier,
                    llm.weight_model.min_comparable_neighbors,
                    llm.weight_model.confidence_weighted_violations,
                )
            )

        self.assertEqual(suite.reference_test, "hard_no_confidence")
        self.assertEqual(
            combinations,
            {
                "hard_no_confidence": ("hard", False),
                "soft_no_confidence": ("soft", False),
                "hard_with_confidence": ("hard", True),
                "soft_with_confidence": ("soft", True),
            },
        )
        self.assertEqual(len(transitivity_settings), 1)
        self.assertEqual(suite.seeds, (42, 43))

    def test_vote_sampling_suite_keeps_pool_and_budget_paired(self) -> None:
        suite = load_benchmark_suite(
            self.vote_sampling_tests_path,
            allowed_test_types={"training_quality"},
        )
        strategies = {}
        for key, test in suite.tests.items():
            configured = apply_benchmark_test(self.config, test)
            llm = next(
                source
                for source in configured.data_model_description.mix_dataset.sources
                if source.name == "llm"
            )
            self.assertEqual(llm.max_rows, 750_000)
            self.assertEqual(llm.splitter.negative_threshold, 2)
            self.assertEqual(llm.splitter.positive_threshold, 7)
            self.assertEqual(
                configured.model_description.transformer.pretrained_model_path,
                "weights/rubert-tiny2",
            )
            strategies[key] = llm.sampling_strategy

        self.assertEqual(suite.reference_test, "random")
        self.assertEqual(
            strategies,
            {
                "random": "category_target_balanced",
                "confidence_weighted": (
                    "category_target_confidence_weighted"
                ),
                "confidence_priority": (
                    "category_target_confidence_priority"
                ),
            },
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
                "data_model": "mix_dataset",
                "augmentation_model": None,
                "status": "completed",
                "validation_macro_pr_auc": metric,
                "delta_validation_macro_pr_auc": None,
                "validation_pairs_hash": "same-split",
                "same_validation_split_as_reference": None,
                "training_seconds": 10.0,
                "total_train_rows": 100,
                "human_train_rows": 20,
                "llm_train_rows": 80,
                "hard_negative_train_rows": 0,
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
        self.assertEqual(
            aggregates.filter(pl.col("test") == "transitivity_on")
            .get_column("mean_hard_negative_train_rows")
            .item(),
            0.0,
        )
        self.assertGreaterEqual(report["total_elapsed_seconds"], 0.0)
        self.assertEqual(report["reference_test"], "transitivity_off")


if __name__ == "__main__":
    unittest.main()
