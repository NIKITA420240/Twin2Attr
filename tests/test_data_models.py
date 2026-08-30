import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import polars as pl

from match.config import (
    ConfidenceWeightingSettings,
    DatasetOverlapResolutionSettings,
    DatasetSourceSettings,
    DatasetSplitterSettings,
    MixedDatasetSettings,
    SampleWeightModelSettings,
    load_app_config_file,
)
from match.data_models import BaseDatasetModel, build_data_model
from match.data_models.models import MixedDatasetModel
from match.data_models.preparation import (
    prepare_source_matches,
    resolve_source_overlaps,
)
from match.paths import PROJECT_ROOT


def _splitter(score_type: str, *, votes: bool = False) -> DatasetSplitterSettings:
    return DatasetSplitterSettings(
        splitter_type="binary",
        score_type=score_type,
        total_votes=9 if votes else None,
        negative_threshold=2 if votes else 0,
        positive_threshold=7 if votes else 1,
        uncertain_action="drop",
    )


class DataModelTests(unittest.TestCase):
    def test_mixed_dataset_loads_source_specific_card_json_items(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            items_path = root / "items.parquet"
            human_path = root / "human.parquet"
            hard_items_path = root / "hard_items.parquet"
            hard_matches_path = root / "hard_matches.parquet"

            pl.DataFrame(
                {
                    "id": list(range(1, 9)),
                    "name": [f"human {index}" for index in range(1, 9)],
                    "category": ["a"] * 8,
                    "attributes": ["{}"] * 8,
                }
            ).write_parquet(items_path)
            pl.DataFrame(
                {
                    "id1": [1, 3, 5, 7],
                    "id2": [2, 4, 6, 8],
                    "target": [0, 1, 0, 1],
                }
            ).write_parquet(human_path)
            pl.DataFrame(
                {
                    "id": [101, 102, 103, 104],
                    "source_id": [11, 11, 12, 12],
                    "category": ["a"] * 4,
                    "card_json": [
                        '{"Название":"Hard A","Код модели":"M1",'
                        '"категория":"a","объем":"500"}',
                        '{"Название":"Hard A","Код модели":"M1",'
                        '"категория":"a","объем":"750"}',
                        '{"Название":"Hard B","Код модели":"M2",'
                        '"категория":"a","количество":"1"}',
                        '{"Название":"Hard B","Код модели":"M2",'
                        '"категория":"a","количество":"10"}',
                    ],
                }
            ).write_parquet(hard_items_path)
            pl.DataFrame(
                {
                    "id1": [101, 103],
                    "id2": [102, 104],
                    "target": [0, 0],
                }
            ).write_parquet(hard_matches_path)

            settings = MixedDatasetSettings(
                items=items_path,
                sources=(
                    DatasetSourceSettings(
                        name="human",
                        matches=human_path,
                        weight=3.0,
                        max_rows=None,
                        sampling_strategy="random",
                        splitter=_splitter("label"),
                    ),
                    DatasetSourceSettings(
                        name="hard_negative",
                        matches=hard_matches_path,
                        weight=0.5,
                        max_rows=1,
                        sampling_strategy="random",
                        splitter=_splitter("label"),
                        items=hard_items_path,
                    ),
                ),
                validation_source="human",
                validation_fraction=0.5,
                leakage_scope="none",
                candidate_splits=8,
                seed=7,
            )

            result = MixedDatasetModel(settings).load_training_splits()

        hard_rows = result.train_matches.filter(
            pl.col("data_source") == "hard_negative"
        )
        self.assertEqual(hard_rows.height, 1)
        self.assertEqual(hard_rows.get_column("sample_weight").to_list(), [0.5])
        hard_ids = set(hard_rows.select("id1", "id2").row(0))
        hard_items = result.items.filter(pl.col("id").is_in(hard_ids))
        self.assertEqual(hard_items.height, 2)
        self.assertTrue(
            all(name.startswith("Hard ") for name in hard_items["name"])
        )
        self.assertTrue(
            all("Код модели" not in value for value in hard_items["attributes"])
        )

    def test_probability_source_uses_configured_target_column(self) -> None:
        items = pl.DataFrame(
            {"id": list(range(1, 9)), "category": ["a"] * 8}
        )
        matches = pl.DataFrame(
            {
                "id1": [1, 2, 3, 4],
                "id2": [5, 6, 7, 8],
                "source_score": [5 / 9] * 4,
                "llm_score": [0.0, 0.49, 0.5, 0.95],
            }
        )
        source = DatasetSourceSettings(
            name="neural_review",
            matches=Path("matches.parquet"),
            weight=1.0,
            max_rows=None,
            sampling_strategy="random",
            splitter=DatasetSplitterSettings(
                splitter_type="binary",
                score_type="probability",
                total_votes=None,
                negative_threshold=0.5,
                positive_threshold=0.5,
                uncertain_action="keep",
                target_mode="hard",
            ),
            target_column="llm_score",
        )

        prepared = prepare_source_matches(items, matches, source, seed=42)

        self.assertEqual(prepared.height, 4)
        self.assertEqual(
            prepared.get_column("target").to_list(),
            [0, 0, 1, 1],
        )
        self.assertEqual(
            prepared.get_column("training_target").to_list(),
            [0.0, 0.0, 1.0, 1.0],
        )
        self.assertEqual(
            prepared.get_column("sample_weight").to_list(),
            [1.0, 1.0, 1.0, 1.0],
        )

    def test_soft_vote_targets_and_confidence_weights_are_preserved(self) -> None:
        items = pl.DataFrame(
            {"id": list(range(1, 9)), "category": ["a"] * 8}
        )
        matches = pl.DataFrame(
            {
                "id1": [1, 2, 3, 4],
                "id2": [5, 6, 7, 8],
                "target": [0.0, 2 / 9, 7 / 9, 1.0],
            }
        )
        source = DatasetSourceSettings(
            name="llm",
            matches=Path("matches.parquet"),
            weight=2.0,
            max_rows=None,
            sampling_strategy="random",
            splitter=replace(
                _splitter("votes", votes=True),
                target_mode="soft",
            ),
            confidence_weighting=ConfidenceWeightingSettings(
                enabled=True,
                min_weight_multiplier=0.2,
                power=1.0,
            ),
        )

        prepared = prepare_source_matches(items, matches, source, seed=42)

        self.assertEqual(prepared.get_column("target").to_list(), [0, 0, 1, 1])
        for actual, wanted in zip(
            prepared.get_column("training_target"),
            [0.0, 2 / 9, 7 / 9, 1.0],
            strict=True,
        ):
            self.assertAlmostEqual(actual, wanted, places=6)
        expected = [2.0, 2.0 * 5 / 9, 2.0 * 5 / 9, 2.0]
        for actual, wanted in zip(
            prepared.get_column("sample_weight"), expected, strict=True
        ):
            self.assertAlmostEqual(actual, wanted, places=6)

    def test_soft_vote_targets_can_keep_uncertain_rows(self) -> None:
        items = pl.DataFrame(
            {"id": list(range(1, 9)), "category": ["a"] * 8}
        )
        matches = pl.DataFrame(
            {
                "id1": [1, 2, 3, 4],
                "id2": [5, 6, 7, 8],
                "target": [3 / 9, 4 / 9, 5 / 9, 6 / 9],
            }
        )
        source = DatasetSourceSettings(
            name="llm",
            matches=Path("matches.parquet"),
            weight=1.0,
            max_rows=None,
            sampling_strategy="random",
            splitter=replace(
                _splitter("votes", votes=True),
                target_mode="soft",
                uncertain_action="keep",
            ),
        )

        prepared = prepare_source_matches(items, matches, source, seed=42)

        self.assertEqual(prepared.height, 4)
        self.assertEqual(prepared.get_column("target").to_list(), [0, 0, 1, 1])
        for actual, wanted in zip(
            prepared.get_column("training_target"),
            [3 / 9, 4 / 9, 5 / 9, 6 / 9],
            strict=True,
        ):
            self.assertAlmostEqual(actual, wanted, places=6)

    def test_confidence_priority_keeps_most_confident_vote_rows(self) -> None:
        items = pl.DataFrame(
            {
                "id": list(range(1, 17)),
                "category": ["a"] * 16,
            }
        )
        votes = [0, 0, 2, 2, 7, 7, 9, 9]
        matches = pl.DataFrame(
            {
                "id1": list(range(1, 9)),
                "id2": list(range(9, 17)),
                "target": [vote / 9 for vote in votes],
            }
        )
        source = DatasetSourceSettings(
            name="llm",
            matches=Path("matches.parquet"),
            weight=1.0,
            max_rows=4,
            sampling_strategy="category_target_confidence_priority",
            splitter=_splitter("votes", votes=True),
        )

        prepared = prepare_source_matches(items, matches, source, seed=42)

        self.assertEqual(prepared.height, 4)
        self.assertEqual(
            prepared.group_by("target").len().sort("target").rows(),
            [(0, 2), (1, 2)],
        )
        self.assertEqual(set(prepared.get_column("annotation_votes")), {0, 9})

    def test_confidence_weighted_sampling_is_reproducible(self) -> None:
        items = pl.DataFrame(
            {
                "id": list(range(1, 41)),
                "category": ["a"] * 40,
            }
        )
        votes = [0, 1, 2, 7, 8, 9] * 3
        matches = pl.DataFrame(
            {
                "id1": list(range(1, 19)),
                "id2": list(range(21, 39)),
                "target": [vote / 9 for vote in votes],
            }
        )
        source = DatasetSourceSettings(
            name="llm",
            matches=Path("matches.parquet"),
            weight=1.0,
            max_rows=10,
            sampling_strategy="category_target_confidence_weighted",
            splitter=_splitter("votes", votes=True),
            confidence_power=2.0,
        )

        first = prepare_source_matches(items, matches, source, seed=42)
        second = prepare_source_matches(items, matches, source, seed=42)

        self.assertEqual(first.height, 10)
        self.assertEqual(
            first.select("id1", "id2").rows(),
            second.select("id1", "id2").rows(),
        )

    def test_overlap_resolution_keeps_highest_priority_symmetric_pair(self) -> None:
        sources = {
            "human": pl.DataFrame(
                {
                    "id1": [1, 3, 7],
                    "id2": [2, 4, 8],
                    "target": [1, 0, 1],
                }
            ),
            "llm": pl.DataFrame(
                {
                    "id1": [2, 4, 5],
                    "id2": [1, 3, 6],
                    "target": [0, 0, 1],
                }
            ),
        }

        resolved = resolve_source_overlaps(
            sources,
            DatasetOverlapResolutionSettings(
                enabled=True,
                source_priority=("human", "llm"),
            ),
        )

        self.assertEqual(resolved["human"].height, 3)
        self.assertEqual(
            resolved["llm"].select("id1", "id2", "target").rows(),
            [(5, 6, 1)],
        )

    def test_factory_selects_configured_data_model(self) -> None:
        config = load_app_config_file(PROJECT_ROOT / "configs" / "pipeline.yaml")

        self.assertIsInstance(build_data_model(config), MixedDatasetModel)
        base_config = replace(
            config,
            training=replace(
                config.training,
                model="boosting",
                data_model="base_dataset",
            ),
        )
        self.assertIsInstance(build_data_model(base_config), BaseDatasetModel)
        codex_config = replace(
            config,
            training=replace(
                config.training,
                model="transformer",
                data_model="mix_dataset_codex",
            ),
        )
        codex_model = build_data_model(codex_config)
        self.assertIsInstance(codex_model, MixedDatasetModel)
        self.assertEqual(
            [source.name for source in codex_model.settings.sources],
            ["human", "codex_reviewed", "llm"],
        )
        neural_config = replace(
            config,
            training=replace(
                config.training,
                model="transformer",
                data_model="mix_dataset_neural_review",
            ),
        )
        neural_model = build_data_model(neural_config)
        self.assertIsInstance(neural_model, MixedDatasetModel)
        self.assertEqual(
            [source.name for source in neural_model.settings.sources],
            ["human", "neural_review", "llm"],
        )
        all_annotations_model = build_data_model(config)
        self.assertIsInstance(all_annotations_model, MixedDatasetModel)
        self.assertEqual(
            [
                source.name
                for source in all_annotations_model.settings.sources
            ],
            ["human", "codex_reviewed", "neural_review", "llm"],
        )
        self.assertEqual(
            next(
                source
                for source in all_annotations_model.settings.sources
                if source.name == "codex_reviewed"
            ).weight,
            1.5,
        )

    def test_converts_vote_scores_and_applies_source_weight(self) -> None:
        items = pl.DataFrame(
            {
                "id": list(range(1, 9)),
                "category": ["a"] * 4 + ["b"] * 4,
            }
        )
        matches = pl.DataFrame(
            {
                "id1": list(range(1, 9)),
                "id2": [2, 1, 4, 3, 6, 5, 8, 7],
                "target": [0.0, 1 / 9, 2 / 9, 3 / 9, 6 / 9, 7 / 9, 8 / 9, 1.0],
            }
        )
        source = DatasetSourceSettings(
            name="llm",
            matches=Path("matches.parquet"),
            weight=1.5,
            max_rows=None,
            sampling_strategy="category_target_balanced",
            splitter=_splitter("votes", votes=True),
        )

        prepared = prepare_source_matches(items, matches, source, seed=42)

        self.assertEqual(prepared.height, 6)
        self.assertEqual(set(prepared.get_column("target")), {0, 1})
        self.assertEqual(set(prepared.get_column("sample_weight")), {1.5})
        self.assertEqual(set(prepared.get_column("data_source")), {"llm"})

    def test_mixed_dataset_uses_only_human_for_validation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            items_path = root / "items.parquet"
            human_path = root / "human.parquet"
            llm_path = root / "llm.parquet"
            items = pl.DataFrame(
                {
                    "id": list(range(1, 17)),
                    "name": [f"item {index}" for index in range(1, 17)],
                    "category": ["a"] * 8 + ["b"] * 8,
                    "attributes": ["{}"] * 16,
                }
            )
            human = pl.DataFrame(
                {
                    "id1": [1, 2, 3, 4, 9, 10, 11, 12],
                    "id2": [2, 1, 4, 3, 10, 9, 12, 11],
                    "target": [0, 0, 1, 1, 0, 0, 1, 1],
                }
            )
            llm = pl.DataFrame(
                {
                    "id1": [5, 6, 7, 8, 13, 14, 15, 16],
                    "id2": [6, 5, 8, 7, 14, 13, 16, 15],
                    "target": [0.0, 1 / 9, 8 / 9, 1.0, 0.0, 2 / 9, 7 / 9, 1.0],
                }
            )
            items.write_parquet(items_path)
            human.write_parquet(human_path)
            llm.write_parquet(llm_path)
            settings = MixedDatasetSettings(
                items=items_path,
                sources=(
                    DatasetSourceSettings(
                        "human",
                        human_path,
                        3.0,
                        None,
                        "random",
                        _splitter("label"),
                    ),
                    DatasetSourceSettings(
                        "llm",
                        llm_path,
                        1.0,
                        None,
                        "random",
                        _splitter("votes", votes=True),
                    ),
                ),
                validation_source="human",
                validation_fraction=0.5,
                leakage_scope="none",
                candidate_splits=32,
                seed=7,
            )

            result = MixedDatasetModel(settings).load_training_splits()

        self.assertIn("llm", set(result.train_matches.get_column("data_source")))
        self.assertEqual(
            set(result.validation_matches.get_column("data_source")),
            {"human"},
        )
        self.assertEqual(
            set(result.validation_matches.get_column("sample_weight")),
            {1.0},
        )
        source_weights = result.train_matches.group_by("data_source").agg(
            pl.col("sample_weight").unique()
        )
        self.assertEqual(
            source_weights.filter(pl.col("data_source") == "human")
            .item(0, 1)
            .to_list(),
            [3.0],
        )
        self.assertEqual(
            source_weights.filter(pl.col("data_source") == "llm")
            .item(0, 1)
            .to_list(),
            [1.0],
        )

    def test_mixed_dataset_accepts_source_weight_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            items_path = root / "items.parquet"
            human_path = root / "human.parquet"
            llm_path = root / "llm.parquet"
            pl.DataFrame(
                {
                    "id": list(range(1, 12)),
                    "name": [f"item {index}" for index in range(1, 12)],
                    "category": ["a"] * 11,
                    "attributes": ["{}"] * 11,
                }
            ).write_parquet(items_path)
            pl.DataFrame(
                {
                    "id1": [1, 3, 5, 7],
                    "id2": [2, 4, 6, 8],
                    "target": [0, 1, 0, 1],
                }
            ).write_parquet(human_path)
            pl.DataFrame(
                {
                    "id1": [9, 10, 9],
                    "id2": [10, 11, 11],
                    "target": [8 / 9, 8 / 9, 1 / 9],
                }
            ).write_parquet(llm_path)
            settings = MixedDatasetSettings(
                items=items_path,
                sources=(
                    DatasetSourceSettings(
                        "human",
                        human_path,
                        3.0,
                        None,
                        "random",
                        _splitter("label"),
                    ),
                    DatasetSourceSettings(
                        "llm",
                        llm_path,
                        1.0,
                        None,
                        "random",
                        _splitter("votes", votes=True),
                        SampleWeightModelSettings(
                            type="transitivity",
                            enabled=True,
                            min_comparable_neighbors=1,
                        ),
                    ),
                ),
                validation_source="human",
                validation_fraction=0.5,
                leakage_scope="none",
                candidate_splits=8,
                seed=7,
            )

            result = MixedDatasetModel(settings).load_training_splits()

        self.assertTrue(
            {
                "id1",
                "id2",
                "target",
                "sample_weight",
                "data_source",
            }.issubset(result.train_matches.columns)
        )
        self.assertIn("annotation_votes", result.train_matches.columns)
        self.assertEqual(
            set(result.train_matches.get_column("data_source")),
            {"human", "llm"},
        )


if __name__ == "__main__":
    unittest.main()
