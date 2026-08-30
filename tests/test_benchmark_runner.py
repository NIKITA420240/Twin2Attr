import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from match.benchmarks.runner import run_configured_benchmarks
from match.benchmarks.suite import apply_benchmark_test, load_benchmark_suite
from match.config import load_benchmark_config_file
from match.paths import PROJECT_ROOT


class ConfiguredBenchmarkRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_benchmark_config_file(
            PROJECT_ROOT / "configs" / "benchmark.yaml"
        )

    def test_loads_all_jobs_from_one_config(self) -> None:
        settings = self.config.benchmark
        self.assertIsNotNone(settings)
        self.assertEqual(
            settings.base_config_path,
            PROJECT_ROOT / "configs" / "benchmark_pipeline.yaml",
        )
        self.assertEqual(
            [job.name for job in settings.jobs],
            [
                "backends",
                "speed_optimizations",
                "inference_quality",
                "transitivity_quality",
                "typed_attribute_quality",
                "typed_transformer_quality",
                "codex_annotation_quality",
                "hard_negative_quality",
                "attribute_word_dropout_quality",
                "neural_relabel_quality",
                "llm_vote_sampling_quality",
                "soft_label_confidence_quality",
            ],
        )
        self.assertEqual(settings.jobs[-1].type, "training_quality")

    def test_runs_enabled_jobs_and_writes_master_report(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            settings = replace(
                self.config.benchmark,
                output_dir=Path(directory),
                jobs=(
                    replace(self.config.benchmark.jobs[0], enabled=True),
                    replace(self.config.benchmark.jobs[-1], enabled=True),
                ),
            )
            config = replace(self.config, benchmark=settings)

            def inference(*_args, output_dir, **_kwargs):
                output_dir.mkdir(parents=True)
                report = output_dir / "report.json"
                report.write_text("{}", encoding="utf-8")
                return SimpleNamespace(report_path=report)

            def training(*_args, output_dir, **_kwargs):
                output_dir.mkdir(parents=True)
                report = output_dir / "report.json"
                report.write_text("{}", encoding="utf-8")
                return SimpleNamespace(report_path=report)

            with (
                patch(
                    "match.benchmarks.runner._run_inference_job",
                    side_effect=inference,
                ),
                patch(
                    "match.benchmarks.runner.run_training_quality_benchmark",
                    side_effect=training,
                ),
            ):
                result = run_configured_benchmarks(config)

            report = json.loads(result.report_path.read_text(encoding="utf-8"))

        self.assertEqual(result.completed_jobs, 2)
        self.assertEqual(result.failed_jobs, 0)
        self.assertEqual([row["status"] for row in report["jobs"]], [
            "completed",
            "completed",
        ])

    def test_speed_suite_was_rebuilt_as_sdpa_ab_test(self) -> None:
        suite = load_benchmark_suite(
            PROJECT_ROOT
            / "configs"
            / "benchmark_tests"
            / "speed_optimizations.yaml",
            allowed_test_types={"speed"},
        )

        self.assertEqual(tuple(suite.tests), ("eager_reference", "sdpa"))
        self.assertEqual(suite.reference_test, "eager_reference")
        eager = apply_benchmark_test(self.config, suite.tests["eager_reference"])
        sdpa = apply_benchmark_test(self.config, suite.tests["sdpa"])
        self.assertEqual(
            eager.inference.transformer.attention.implementation,
            "eager",
        )
        self.assertEqual(
            sdpa.inference.transformer.attention.implementation,
            "sdpa",
        )

    def test_typed_attribute_ablation_cases_produce_valid_configs(self) -> None:
        suite = load_benchmark_suite(
            PROJECT_ROOT
            / "configs"
            / "benchmark_tests"
            / "typed_attribute_quality.yaml",
            allowed_test_types={"training_quality"},
        )
        expected = {
            "typed_attributes_off": (),
            "code_only": ("CODE",),
            "code_physical": ("CODE", "PHYSICAL"),
            "code_physical_set": ("CODE", "PHYSICAL", "SET"),
            "all_typed_attributes": (
                "CODE",
                "PHYSICAL",
                "NUMERIC",
                "SET",
                "TEXT",
            ),
        }

        for key, enabled_types in expected.items():
            configured = apply_benchmark_test(self.config, suite.tests[key])
            options = configured.pair_features.typed_attributes
            actual = options.enabled_types if options.enabled else ()
            self.assertEqual(actual, enabled_types)
            self.assertEqual(configured.training.model, "boosting")
            self.assertIsNone(configured.training.augmentation_model)

    def test_typed_transformer_cases_preserve_current_nemotron_contract(self) -> None:
        suite = load_benchmark_suite(
            PROJECT_ROOT
            / "configs"
            / "benchmark_tests"
            / "typed_transformer_quality.yaml",
            allowed_test_types={"training_quality"},
        )

        native = apply_benchmark_test(self.config, suite.tests["native_baseline"])
        typed = apply_benchmark_test(
            self.config,
            suite.tests["typed_attribute_fusion"],
        )
        gated = apply_benchmark_test(
            self.config,
            suite.tests["gated_residual_fusion"],
        )

        self.assertEqual(native.model_description.transformer.head.type, "native")
        self.assertFalse(native.pair_features.typed_attributes.enabled)
        self.assertEqual(
            typed.model_description.transformer.head.type,
            "typed_attribute_fusion",
        )
        self.assertTrue(typed.pair_features.typed_attributes.enabled)
        self.assertEqual(
            typed.model_description.transformer.profile,
            "prompted_binary_reranker",
        )
        self.assertEqual(
            gated.model_description.transformer.head.type,
            "gated_residual_fusion",
        )
        self.assertEqual(
            gated.model_description.transformer.head.poolings,
            ("mean", "attention"),
        )
        self.assertTrue(gated.pair_features.typed_attributes.enabled)


if __name__ == "__main__":
    unittest.main()
