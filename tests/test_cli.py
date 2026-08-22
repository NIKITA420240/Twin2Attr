import unittest
from unittest.mock import patch

import run as run_module
from run import _with_default_command, ensure_polars_available, parse_args


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
                ["train", "models_parameters.transformer.max_epochs=1"]
            ),
            ["train", "models_parameters.transformer.max_epochs=1"],
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
                "models_parameters.transformer.max_epochs=3",
                "split.mode=auto",
            ]
        )

        self.assertEqual(args.command, "train")
        self.assertEqual(
            args.overrides,
            ["models_parameters.transformer.max_epochs=3", "split.mode=auto"],
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


if __name__ == "__main__":
    unittest.main()
