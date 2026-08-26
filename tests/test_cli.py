import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run as run_module
from run import (
    _predict_solution_path,
    _with_default_command,
    ensure_polars_available,
    ensure_preprocessing_runtime_available,
    parse_args,
)


class UnifiedCliTests(unittest.TestCase):
    def test_evaluator_arguments_default_to_predict(self) -> None:
        arguments = ["--items_path", "items.parquet"]

        self.assertEqual(
            _with_default_command(arguments),
            ["predict", *arguments],
        )

    def test_explicit_command_is_preserved(self) -> None:
        self.assertEqual(
            _with_default_command(
                ["train", "model_description.transformer.max_epochs=1"]
            ),
            ["train", "model_description.transformer.max_epochs=1"],
        )

    def test_parses_evaluator_invocation_as_predict(self) -> None:
        args = parse_args(
            [
                "--items_path",
                "items.parquet",
                "--matches_path",
                "matches.parquet",
                "--output_path",
                "submission.csv",
            ]
        )

        self.assertEqual(args.command, "predict")
        self.assertEqual(args.items_path, "items.parquet")
        self.assertEqual(args.matches_path, "matches.parquet")
        self.assertEqual(args.output_path, "submission.csv")

    def test_parses_training_overrides(self) -> None:
        args = parse_args(
            [
                "train",
                "model_description.transformer.max_epochs=3",
                "split.mode=auto",
            ]
        )

        self.assertEqual(args.command, "train")
        self.assertEqual(
            args.overrides,
            ["model_description.transformer.max_epochs=3", "split.mode=auto"],
        )

    def test_predict_resolves_solution_from_pipeline_config(self) -> None:
        args = parse_args(
            [
                "predict",
                "--items_path",
                "items.parquet",
                "--matches_path",
                "matches.parquet",
                "--output_path",
                "submission.csv",
            ]
        )

        solution_path = _predict_solution_path(args)

        self.assertEqual(
            solution_path,
            run_module.SOURCE_ROOT.parent / "solution.json",
        )

    def test_explicit_solution_overrides_pipeline_config(self) -> None:
        args = parse_args(
            [
                "predict",
                "--items_path",
                "items.parquet",
                "--matches_path",
                "matches.parquet",
                "--output_path",
                "submission.csv",
                "--solution",
                "another-solution.json",
            ]
        )

        self.assertEqual(
            _predict_solution_path(args),
            Path("another-solution.json"),
        )

    def test_parses_initialize_overrides(self) -> None:
        args = parse_args(
            [
                "initialize",
                "model_description.transformer.pair_encoding.max_length=64",
                "model_description.transformer.artifact_dir=models/initialized",
            ]
        )

        self.assertEqual(args.command, "initialize")
        self.assertEqual(
            args.overrides,
            [
                "model_description.transformer.pair_encoding.max_length=64",
                "model_description.transformer.artifact_dir=models/initialized",
            ],
        )

    def test_parses_analysis_overrides(self) -> None:
        args = parse_args(
            [
                "analyze",
                "analysis.data_model=mix_dataset",
                "analysis_models.attribute_importance.sample_size=1000",
            ]
        )

        self.assertEqual(args.command, "analyze")
        self.assertEqual(
            args.overrides,
            [
                "analysis.data_model=mix_dataset",
                "analysis_models.attribute_importance.sample_size=1000",
            ],
        )

    def test_bootstraps_bundled_polars_when_image_does_not_have_it(self) -> None:
        missing = ModuleNotFoundError("No module named 'polars'", name="polars")
        with (
            patch.object(
                run_module.importlib,
                "import_module",
                side_effect=[missing, object()],
            ) as import_module,
            patch.object(run_module.subprocess, "run") as install,
            patch.object(run_module.sys, "path", list(run_module.sys.path)),
            patch.object(
                run_module,
                "__file__",
                str(
                    run_module.SOURCE_ROOT.parent
                    / "build_submission"
                    / "run.py"
                ),
            ),
        ):
            ensure_polars_available()

        command = install.call_args.args[0]
        self.assertIn("--no-index", command)
        self.assertIn("polars==1.43.2", command)
        self.assertEqual(import_module.call_count, 2)

    def test_bootstraps_bundled_preprocessing_runtime(self) -> None:
        missing = ModuleNotFoundError("No module named 'loguru'", name="loguru")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheels = root / "vendor_wheels"
            wheels.mkdir()
            for name in (
                "pymorphy3-2.0.6-py3-none-any.whl",
                "pymorphy3_dicts_ru-2.4.417150.4580142-py2.py3-none-any.whl",
                "dawg2_python-0.9.0-py3-none-any.whl",
                "setuptools-84.0.0-py3-none-any.whl",
                "joblib-1.5.3-py3-none-any.whl",
                "loguru-0.7.3-py3-none-any.whl",
                "Pint-0.25.3-py3-none-any.whl",
                "flexcache-0.3-py3-none-any.whl",
                "flexparser-0.4-py3-none-any.whl",
                "platformdirs-4.11.3-py3-none-any.whl",
                "typing_extensions-4.16.0-py3-none-any.whl",
                "colorama-0.4.6-py2.py3-none-any.whl",
                "win32_setctime-1.2.0-py3-none-any.whl",
                "orjson-3.11.9-cp312-cp312-manylinux2014_x86_64.whl",
            ):
                (wheels / name).write_bytes(b"wheel")
            solution = root / "solution.json"
            manifests = (
                {"features": {"normalization": {"enabled": True}}},
                {"normalization": {"enabled": True}},
            )
            for manifest in manifests:
                with self.subTest(manifest=manifest):
                    solution.write_text(json.dumps(manifest), encoding="utf-8")
                    with (
                        patch.object(
                            run_module.importlib,
                            "import_module",
                            side_effect=[
                                missing,
                                object(),
                                object(),
                                object(),
                                object(),
                                object(),
                            ],
                        ) as import_module,
                        patch.object(run_module.subprocess, "run") as install,
                        patch.object(
                            run_module.sys,
                            "path",
                            list(run_module.sys.path),
                        ),
                        patch.object(run_module, "__file__", str(root / "run.py")),
                    ):
                        ensure_preprocessing_runtime_available(solution)

                    command = install.call_args.args[0]
                    self.assertIn("--no-index", command)
                    self.assertIn("pymorphy3==2.0.6", command)
                    self.assertIn("joblib==1.5.3", command)
                    self.assertIn("loguru==0.7.3", command)
                    self.assertIn("pint==0.25.3", command)
                    self.assertIn("orjson==3.11.9", command)
                    self.assertEqual(import_module.call_count, 6)


if __name__ == "__main__":
    unittest.main()
