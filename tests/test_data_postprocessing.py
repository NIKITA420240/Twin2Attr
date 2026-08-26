import tempfile
import unittest
from pathlib import Path

import polars as pl

from match.data_postprocessing import (
    AttributePriorityTable,
    apply_manifest_postprocessing,
)
from match.prepare_data import PreparedCard, PreparedPair


def _write_priorities(path: Path) -> None:
    pl.DataFrame(
        {
            "scope": ["global", "global", "category", "category"],
            "category": [None, None, "phones", "phones"],
            "attribute": ["brand", "memory", "memory", "brand"],
            "priority": [1, 2, 1, 2],
        }
    ).write_parquet(path)


def _pair() -> PreparedPair:
    attributes = (
        ("unknown_first", "value"),
        ("brand", "acme"),
        ("unknown_second", "value"),
        ("memory", "large"),
    )
    return PreparedPair(
        PreparedCard(1, "left", "phones", attributes),
        PreparedCard(2, "right", "phones", tuple(reversed(attributes))),
        None,
        "phones",
    )


class AttributeSortTests(unittest.TestCase):
    def test_uses_category_priority_and_keeps_unknown_source_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "priorities.parquet"
            _write_priorities(path)
            pair = AttributePriorityTable.load(path).sort_pairs([_pair()])[0]

        self.assertEqual(
            [name for name, _ in pair.left.attributes],
            ["memory", "brand", "unknown_first", "unknown_second"],
        )
        self.assertEqual(
            [name for name, _ in pair.right.attributes],
            ["memory", "brand", "unknown_second", "unknown_first"],
        )
        self.assertTrue(pair.preserve_attribute_order)
        self.assertTrue(pair.skip_oversized_attributes)

    def test_manifest_uses_relative_priorities_path(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "analysis" / "priorities.parquet"
            path.parent.mkdir()
            _write_priorities(path)
            solution = {
                "data_postprocessing_model": "attribute_sort",
                "data_postprocessing_models": {
                    "attribute_sort": {
                        "priorities_path": "analysis/priorities.parquet",
                    }
                },
            }

            pairs = apply_manifest_postprocessing([_pair()], solution, root)

        self.assertEqual(pairs[0].left.attributes[0][0], "memory")

    def test_rejects_incomplete_priority_table(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "priorities.parquet"
            pl.DataFrame({"attribute": ["brand"]}).write_parquet(path)

            with self.assertRaisesRegex(ValueError, "missing columns"):
                AttributePriorityTable.load(path)


if __name__ == "__main__":
    unittest.main()
