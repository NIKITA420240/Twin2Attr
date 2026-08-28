import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import polars as pl

from match.config import (
    DatasetOverlapResolutionSettings,
    DatasetSourceSettings,
    DatasetSplitterSettings,
    MixedDatasetSettings,
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

        self.assertIsInstance(build_data_model(config), BaseDatasetModel)
        mixed_config = replace(
            config,
            training=replace(
                config.training,
                model="boosting",
                data_model="mix_dataset",
            ),
        )
        self.assertIsInstance(build_data_model(mixed_config), MixedDatasetModel)
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


if __name__ == "__main__":
    unittest.main()
