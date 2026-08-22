import unittest

from match.features.physical.normalizer import PhysicalUnitNormalizer
from match.features.physical.parser import PhysicalAttributeParser


class PhysicalAttributeParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = PhysicalAttributeParser(n_jobs=1)

    def test_extracts_and_normalizes_multidimensional_size(self) -> None:
        parsed = self.parser.parse("стол деревянный 120x60x75 см")
        normalized = PhysicalUnitNormalizer().normalize(parsed)

        self.assertEqual(
            normalized,
            {
                "длина, мм": "1200",
                "ширина, мм": "600",
                "высота, мм": "750",
            },
        )

    def test_uses_dimension_keyword_and_common_unit_aliases(self) -> None:
        parsed = self.parser.parse("диаметр 39 сантиметров мощность 600 w")
        normalized = PhysicalUnitNormalizer().normalize(parsed)

        self.assertEqual(normalized["диаметр, мм"], "390")
        self.assertEqual(normalized["мощность, вт"], "600")

    def test_does_not_treat_year_as_grams(self) -> None:
        self.assertNotIn("вес, г", self.parser.parse("модель 2025г"))


if __name__ == "__main__":
    unittest.main()
