import tempfile
import unittest
from pathlib import Path

import polars as pl

from match.data.preprocessing import prepare_manifest_items
from match.features.contracts import PreparedItems
from match.features.normalization import NormalizationItemEnricher
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
                _RecordingEnricher("normalization", calls),
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
            [
                ("normalization", "attributes"),
                ("ner", "normalization_attributes"),
                ("physical", "ner_attributes"),
            ],
        )
        self.assertEqual(result.attributes_column, "physical_attributes")

    def test_normalization_enricher_updates_active_attributes_column(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            synonyms = root / "synonyms.parquet"
            unique_attributes = root / "unique_attributes.parquet"
            pl.DataFrame(
                {"replacer": ["ширина"], "synonyms": [["ширина"]]}
            ).write_parquet(synonyms)
            pl.DataFrame({"attribute": ["Ширина, мм"]}).write_parquet(
                unique_attributes
            )
            enricher = NormalizationItemEnricher(
                synonyms_path=synonyms,
                unique_attributes_path=unique_attributes,
                n_jobs=1,
            )

            result = enricher.enrich(
                PreparedItems(
                    pl.DataFrame({"attributes": ['{"Ширина, см": "10"}']}),
                    "attributes",
                )
            )

        self.assertEqual(result.attributes_column, "normalized_attributes")
        self.assertEqual(
            result.frame["normalized_attributes"][0],
            '{"ширина, мм": "100"}',
        )

    def test_legacy_manifest_normalization_remains_supported(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pl.DataFrame(
                {"replacer": ["ширина"], "synonyms": [["ширина"]]}
            ).write_parquet(root / "synonyms.parquet")
            pl.DataFrame({"attribute": ["Ширина, мм"]}).write_parquet(
                root / "unique.parquet"
            )
            result = prepare_manifest_items(
                pl.DataFrame({"attributes": ['{"Ширина, см": "10"}']}),
                {
                    "normalization": {
                        "enabled": True,
                        "source_column": "attributes",
                        "output_column": "normalized_attributes",
                        "synonyms_path": "synonyms.parquet",
                        "unique_attributes_path": "unique.parquet",
                        "n_jobs": 1,
                        "chunk_size": 10,
                    }
                },
                root,
            )

        self.assertEqual(result.attributes_column, "normalized_attributes")
        self.assertEqual(
            result.frame["normalized_attributes"][0],
            '{"ширина, мм": "100"}',
        )


if __name__ == "__main__":
    unittest.main()
