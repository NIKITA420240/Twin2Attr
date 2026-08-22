import unittest

from run import _with_default_command, parse_args


class UnifiedCliTests(unittest.TestCase):
    def test_evaluator_arguments_default_to_predict(self) -> None:
        arguments = ["--items_path", "items.parquet"]

        self.assertEqual(
            _with_default_command(arguments),
            ["predict", *arguments],
        )

    def test_explicit_command_is_preserved(self) -> None:
        self.assertEqual(
            _with_default_command(["train", "model.max_epochs=1"]),
            ["train", "model.max_epochs=1"],
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
        args = parse_args(["train", "model.max_epochs=3", "split.mode=auto"])

        self.assertEqual(args.command, "train")
        self.assertEqual(
            args.overrides,
            ["model.max_epochs=3", "split.mode=auto"],
        )


if __name__ == "__main__":
    unittest.main()
