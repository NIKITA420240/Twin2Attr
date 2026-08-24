import json
import tempfile
import unittest
from pathlib import Path

import polars as pl

from match import normalize_attributes, normalize_physical_attributes


class NormalizeAttributesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_directory = tempfile.TemporaryDirectory()
        self.synonyms_path = Path(self.temp_directory.name) / "synonyms.parquet"
        self.unique_attributes_path = (
            Path(self.temp_directory.name) / "unique_attributes.parquet"
        )
        pl.DataFrame(
            {
                "replacer": ["ширина"],
                "synonyms": [["ширина", "ширь"]],
            }
        ).write_parquet(self.synonyms_path)
        pl.DataFrame(
            {
                "attribute": [
                    "Ширина, мм",
                    "Артикул",
                    "длина упаковки, мм",
                    "ширина упаковки, мм",
                    "высота упаковки, мм",
                    "Ширь товара",
                ]
            }
        ).write_parquet(self.unique_attributes_path)

    def tearDown(self) -> None:
        self.temp_directory.cleanup()

    def test_normalizes_units_and_preserves_original_column(self) -> None:
        raw = json.dumps(
            {
                "Ширина, см": "10 см",
                "Артикул": "ABC-10 мм",
            },
            ensure_ascii=False,
        )
        source = pl.DataFrame({"id": [1], "attributes": [raw]})

        result = normalize_attributes(
            source,
            self.synonyms_path,
            self.unique_attributes_path,
            n_jobs=1,
        )
        normalized = json.loads(result["normalized_attributes"][0])

        self.assertEqual(result["attributes"][0], raw)
        self.assertEqual(normalized["ширина, мм"], "100")
        self.assertEqual(normalized["артикул"], "ABC-10 мм")

    def test_splits_multidimensional_attribute(self) -> None:
        raw = json.dumps({"Размер упаковки, см": "10 x 20 x 30"}, ensure_ascii=False)

        result = normalize_attributes(
            pl.DataFrame({"attributes": [raw]}),
            self.synonyms_path,
            self.unique_attributes_path,
            n_jobs=1,
        )
        normalized = json.loads(result["normalized_attributes"][0])

        self.assertEqual(normalized["длина упаковки, мм"], "100")
        self.assertEqual(normalized["ширина упаковки, мм"], "200")
        self.assertEqual(normalized["высота упаковки, мм"], "300")

    def test_preserves_non_finite_physical_values(self) -> None:
        for value in ("nan", "inf", "-inf"):
            with self.subTest(value=value):
                self.assertEqual(
                    normalize_physical_attributes({"Ширина, см": value}),
                    {"Ширина, см": value},
                )

    def test_preserves_value_when_unit_conversion_overflows(self) -> None:
        self.assertEqual(
            normalize_physical_attributes({"Ширина, м": "1e308"}),
            {"Ширина, м": "1e308"},
        )

    def test_loads_synonyms_from_parquet(self) -> None:
        raw = json.dumps({"Ширь товара": "10"}, ensure_ascii=False)

        result = normalize_attributes(
            pl.DataFrame({"attributes": [raw]}),
            self.synonyms_path,
            self.unique_attributes_path,
            n_jobs=1,
        )
        normalized = json.loads(result["normalized_attributes"][0])

        self.assertEqual(normalized, {"ширина товара": "10"})

    def test_supports_custom_columns_and_nulls(self) -> None:
        frame = pl.DataFrame({"raw": [None, "not-json"]})

        result = normalize_attributes(
            frame,
            self.synonyms_path,
            self.unique_attributes_path,
            source_column="raw",
            output_column="clean",
            n_jobs=1,
        )

        self.assertEqual(result["clean"].to_list(), [None, "not-json"])

    def test_requires_polars_dataframe(self) -> None:
        with self.assertRaises(TypeError):
            normalize_attributes(  # type: ignore[arg-type]
                [],
                self.synonyms_path,
                self.unique_attributes_path,
                n_jobs=1,
            )

    def test_parallel_output_matches_sequential_output(self) -> None:
        raw_values = [
            json.dumps({"Ширина, см": str(value)}, ensure_ascii=False)
            for value in range(1, 7)
        ]
        frame = pl.DataFrame({"attributes": raw_values})

        sequential = normalize_attributes(
            frame,
            self.synonyms_path,
            self.unique_attributes_path,
            n_jobs=1,
        )
        parallel = normalize_attributes(
            frame,
            self.synonyms_path,
            self.unique_attributes_path,
            n_jobs=2,
            chunk_size=2,
        )

        self.assertEqual(
            parallel["normalized_attributes"].to_list(),
            sequential["normalized_attributes"].to_list(),
        )


if __name__ == "__main__":
    unittest.main()
