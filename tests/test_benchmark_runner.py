import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from match.benchmarks.runner import run_configured_benchmarks
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
                "codex_annotation_quality",
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
                    self.config.benchmark.jobs[0],
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


if __name__ == "__main__":
    unittest.main()
