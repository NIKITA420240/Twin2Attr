import json
import unittest

import polars as pl

from match.features.contracts import ItemEnricher, PreparedItems
from match.features.ner.enrichment import NerItemEnricher
from match.features.ner.entities import NerEntity


class _FakeExtractor:
    def extract(self, texts):
        self.texts = list(texts)
        return [
            [
                NerEntity("бренд", "bosch", 0, 5, 0.95),
                NerEntity("тип", "дрель", 6, 11, 0.90),
            ],
            [NerEntity("бренд", "redmond", 0, 7, 0.93)],
        ]


class NerItemEnricherTests(unittest.TestCase):
    def test_adds_auditable_and_merged_attribute_columns(self) -> None:
        extractor = _FakeExtractor()
        enricher = NerItemEnricher(extractor)
        items = PreparedItems(
            pl.DataFrame(
                {
                    "id": [1, 2],
                    "name": ["bosch дрель", "redmond чайник"],
                    "attributes": [
                        '{"мощность":"600"}',
                        '{"Бренд":"Redmond"}',
                    ],
                }
            ),
            "attributes",
        )

        result = enricher.enrich(items)

        self.assertIsInstance(enricher, ItemEnricher)
        self.assertEqual(result.attributes_column, "enriched_attributes")
        self.assertEqual(extractor.texts, ["bosch дрель", "redmond чайник"])
        self.assertEqual(
            json.loads(result.frame["ner_attributes"][0]),
            {"бренд": "bosch", "тип": "дрель"},
        )
        self.assertEqual(
            json.loads(result.frame["enriched_attributes"][0]),
            {"бренд": "bosch", "тип": "дрель", "мощность": "600"},
        )
        self.assertEqual(
            json.loads(result.frame["enriched_attributes"][1])["Бренд"],
            "Redmond",
        )
        self.assertNotIn(
            "бренд",
            json.loads(result.frame["enriched_attributes"][1]),
        )

    def test_rejects_misaligned_extractor_output(self) -> None:
        class BrokenExtractor:
            def extract(self, texts):
                return []

        enricher = NerItemEnricher(BrokenExtractor())
        items = PreparedItems(
            pl.DataFrame({"name": ["item"], "attributes": ["{}"]}),
            "attributes",
        )

        with self.assertRaisesRegex(RuntimeError, "different number of rows"):
            enricher.enrich(items)


if __name__ == "__main__":
    unittest.main()
