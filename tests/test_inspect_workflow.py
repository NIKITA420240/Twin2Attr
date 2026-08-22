import unittest
from contextlib import nullcontext
from unittest.mock import ANY, patch

from match.config import load_app_config_file
from match.paths import PROJECT_ROOT
from match.workflows.inspect import inspect_max_length


class InspectWorkflowTests(unittest.TestCase):
    def test_runs_explicit_inspection_steps(self) -> None:
        config = load_app_config_file(
            PROJECT_ROOT / "configs" / "pipeline.yaml"
        )
        items = object()
        matches = object()
        pairs = [object(), object()]

        with (
            patch(
                "match.workflows.inspect.workflow_logging",
                return_value=nullcontext(),
            ),
            patch("match.workflows.inspect.check_optional_features"),
            patch(
                "match.workflows.inspect.normalization_enabled",
                return_value=False,
            ),
            patch(
                "match.workflows.inspect.read_parquet",
                side_effect=[items, matches],
            ) as read_parquet,
            patch(
                "match.workflows.inspect.prepare_pair_rows",
                return_value=pairs,
            ) as prepare_pair_rows,
            patch(
                "match.workflows.inspect.resolve_max_length",
                return_value=256,
            ) as resolve_max_length,
        ):
            result = inspect_max_length(config)

        self.assertEqual(result, 256)
        self.assertEqual(read_parquet.call_count, 2)
        prepare_pair_rows.assert_called_once_with(
            items,
            matches,
            "attributes",
            split_name="inspection",
        )
        resolve_max_length.assert_called_once_with(
            ANY,
            pairs,
        )


if __name__ == "__main__":
    unittest.main()
