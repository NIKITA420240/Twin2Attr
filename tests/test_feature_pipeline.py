import unittest

import polars as pl

from match.features.contracts import PreparedItems
from match.features.pipeline import FeaturePipeline


class _RecordingEnricher:
    def __init__(self, name, calls):
        self.name = name
        self.calls = calls

    def enrich(self, items):
        self.calls.append((self.name, items.attributes_column))
        column = f"{self.name}_attributes"
        return PreparedItems(
            items.frame.with_columns(pl.lit("{}").alias(column)),
            column,
        )


class FeaturePipelineTests(unittest.TestCase):
    def test_applies_enrichers_in_declared_order(self) -> None:
        calls = []
        pipeline = FeaturePipeline(
            (
                _RecordingEnricher("ner", calls),
                _RecordingEnricher("physical", calls),
            )
        )
        items = PreparedItems(
            pl.DataFrame({"attributes": ["{}"]}),
            "attributes",
        )

        result = pipeline.enrich(items)

        self.assertEqual(
            calls,
            [("ner", "attributes"), ("physical", "ner_attributes")],
        )
        self.assertEqual(result.attributes_column, "physical_attributes")


if __name__ == "__main__":
    unittest.main()
