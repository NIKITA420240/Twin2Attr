"""Concrete base and mixed training-data recipes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import orjson
import polars as pl
from loguru import logger

from ..config import (
    BaseDatasetSettings,
    DatasetSourceSettings,
    DatasetSplitterSettings,
    MixedDatasetSettings,
)
from ..data import read_parquet
from ..data_split import DataSplitConfig, split_matches, validate_predefined_split
from .contracts import InspectionFrames, LoadedTrainingSplits
from .preparation import (
    finalize_source_matches,
    prepare_source_labels,
    prepare_source_matches,
    resolve_source_overlaps,
)


def _split_config(
    *,
    validation_fraction: float,
    leakage_scope: str,
    seed: int,
    candidate_splits: int,
) -> DataSplitConfig:
    return DataSplitConfig(
        validation_fraction=validation_fraction,
        leakage_scope=leakage_scope,
        seed=seed,
        candidate_splits=candidate_splits,
    )


def _base_source(path: Path) -> DatasetSourceSettings:
    return DatasetSourceSettings(
        name="base",
        matches=path,
        weight=1.0,
        max_rows=None,
        sampling_strategy="random",
        splitter=DatasetSplitterSettings(
            splitter_type="binary",
            score_type="label",
            total_votes=None,
            negative_threshold=0,
            positive_threshold=1,
            uncertain_action="drop",
        ),
    )


def _item_lookup(path: Path, *, label: str) -> pl.DataFrame:
    return read_parquet(path, label=label, columns=("id", "category"))


def _item_paths(settings: MixedDatasetSettings) -> tuple[Path, ...]:
    paths = [settings.items]
    for source in settings.sources:
        if source.items is not None and source.items not in paths:
            paths.append(source.items)
    return tuple(paths)


def _combined_item_lookup(settings: MixedDatasetSettings) -> pl.DataFrame:
    parts = [
        _item_lookup(path, label=f"mixed dataset item lookup ({path})")
        for path in _item_paths(settings)
    ]
    lookup = pl.concat(parts, how="vertical_relaxed")
    duplicate_count = lookup.height - lookup.get_column("id").n_unique()
    if duplicate_count:
        raise ValueError(
            "mixed dataset item sources contain "
            f"{duplicate_count} duplicate item ids"
        )
    return lookup


def _decode_card_json(row: dict[str, object]) -> dict[str, str]:
    raw = row.get("card_json")
    try:
        card = {} if raw is None else orjson.loads(raw)
    except (orjson.JSONDecodeError, TypeError, UnicodeDecodeError) as error:
        raise ValueError("card_json must contain a JSON object") from error
    if not isinstance(card, dict):
        raise ValueError("card_json must contain a JSON object")

    name = " ".join(
        str(value).strip()
        for value in (card.get("Название"), card.get("Код модели"))
        if value is not None and str(value).strip()
    )
    category = card.get("категория") or row.get("category") or ""
    attributes = dict(card)
    attributes.pop("Название", None)
    attributes.pop("категория", None)
    attributes.pop("Код модели", None)
    return {
        "name": name,
        "category": str(category),
        "attributes": orjson.dumps(attributes).decode("utf-8"),
    }


def _standardize_selected_items(
    path: Path,
    required_ids: pl.DataFrame,
) -> pl.DataFrame:
    schema = pl.scan_parquet(path).collect_schema()
    columns = set(schema.names())
    if {"id", "name", "category", "attributes"}.issubset(columns):
        return (
            pl.scan_parquet(path)
            .select("id", "name", "category", "attributes")
            .join(required_ids.lazy(), on="id", how="inner")
            .collect(engine="streaming")
        )
    if {"id", "category", "card_json"}.issubset(columns):
        selected = (
            pl.scan_parquet(path)
            .select("id", "category", "card_json")
            .join(required_ids.lazy(), on="id", how="inner")
            .collect(engine="streaming")
        )
        if selected.is_empty():
            return pl.DataFrame(
                schema={
                    "id": schema["id"],
                    "name": pl.String,
                    "category": pl.String,
                    "attributes": pl.String,
                }
            )
        decoded_type = pl.Struct(
            {
                "name": pl.String,
                "category": pl.String,
                "attributes": pl.String,
            }
        )
        return (
            selected.with_columns(
                pl.struct("card_json", "category")
                .map_elements(_decode_card_json, return_dtype=decoded_type)
                .alias("_card")
            )
            .drop("card_json", "category")
            .unnest("_card")
            .select("id", "name", "category", "attributes")
        )
    raise ValueError(
        f"items file {path} must contain either id/name/category/attributes "
        "or id/category/card_json"
    )


def _selected_mixed_items(
    settings: MixedDatasetSettings,
    *match_frames: pl.DataFrame,
) -> pl.DataFrame:
    started_at = perf_counter()
    required_ids = pl.concat(
        [
            frame.select(pl.col(column).alias("id"))
            for frame in match_frames
            for column in ("id1", "id2")
        ]
    ).unique()
    items = pl.concat(
        [
            _standardize_selected_items(path, required_ids)
            for path in _item_paths(settings)
        ],
        how="vertical_relaxed",
    )
    if items.height:
        items = items.unique(subset=["id"], keep="first", maintain_order=True)
    if items.height != required_ids.height:
        missing = required_ids.join(items.select("id"), on="id", how="anti")
        raise ValueError(
            f"mixed items files are missing {missing.height} selected ids"
        )
    logger.info(
        "Loaded selected mixed items: paths={}, rows={}, elapsed_seconds={:.3f}",
        [str(path) for path in _item_paths(settings)],
        items.height,
        perf_counter() - started_at,
    )
    return items


def _selected_items(
    path: Path,
    *match_frames: pl.DataFrame,
) -> pl.DataFrame:
    started_at = perf_counter()
    required_ids = pl.concat(
        [
            frame.select(pl.col(column).alias("id"))
            for frame in match_frames
            for column in ("id1", "id2")
        ]
    ).unique()
    items = (
        pl.scan_parquet(path)
        .join(required_ids.lazy(), on="id", how="inner")
        .collect(engine="streaming")
    )
    if items.height != required_ids.height:
        raise ValueError(
            f"items file is missing {required_ids.height - items.height} selected ids"
        )
    logger.info(
        "Loaded selected items: path={!s}, rows={}, elapsed_seconds={:.3f}",
        path,
        items.height,
        perf_counter() - started_at,
    )
    return items


@dataclass(frozen=True, slots=True)
class BaseDatasetModel:
    settings: BaseDatasetSettings
    create_stacking_split: bool = False

    def _load_matches(self) -> tuple[pl.DataFrame, pl.DataFrame]:
        item_lookup = _item_lookup(
            self.settings.items,
            label="base dataset item lookup",
        )
        matches = read_parquet(self.settings.matches, label="base dataset matches")
        return item_lookup, prepare_source_matches(
            item_lookup,
            matches,
            _base_source(self.settings.matches),
            seed=self.settings.seed,
        )

    def load_training_splits(self) -> LoadedTrainingSplits:
        item_lookup, matches = self._load_matches()
        result = split_matches(
            item_lookup,
            matches,
            _split_config(
                validation_fraction=self.settings.validation_fraction,
                leakage_scope=self.settings.leakage_scope,
                seed=self.settings.seed,
                candidate_splits=self.settings.candidate_splits,
            ),
        )
        validation = result.validation_matches.with_columns(
            pl.lit(1.0).cast(pl.Float32).alias("sample_weight")
        )
        train_matches = result.train_matches
        stacking_matches: pl.DataFrame | None = None
        if self.create_stacking_split:
            relative_fraction = self.settings.stacking_train_fraction / (
                1.0 - self.settings.validation_fraction
            )
            stacking_split = split_matches(
                item_lookup,
                train_matches,
                _split_config(
                    validation_fraction=relative_fraction,
                    leakage_scope=self.settings.leakage_scope,
                    seed=self.settings.seed + 1,
                    candidate_splits=self.settings.candidate_splits,
                ),
            )
            train_matches = stacking_split.train_matches
            stacking_matches = stacking_split.validation_matches
            logger.info(
                "Built stacking split: base_train_rows={}, stacking_train_rows={}, "
                "validation_rows={}",
                train_matches.height,
                stacking_matches.height,
                validation.height,
            )
        selected_frames = [train_matches]
        if stacking_matches is not None:
            selected_frames.append(stacking_matches)
        selected_frames.append(validation)
        items = _selected_items(self.settings.items, *selected_frames)
        return LoadedTrainingSplits(
            items,
            train_matches,
            validation,
            stacking_matches,
        )

    def load_inspection_frames(self) -> InspectionFrames:
        _, matches = self._load_matches()
        items = _selected_items(self.settings.items, matches)
        return InspectionFrames(items, matches)


@dataclass(frozen=True, slots=True)
class MixedDatasetModel:
    settings: MixedDatasetSettings

    def _source(self, name: str) -> DatasetSourceSettings:
        return next(source for source in self.settings.sources if source.name == name)

    def _load_source(
        self,
        items: pl.DataFrame,
        source: DatasetSourceSettings,
    ) -> pl.DataFrame:
        matches = read_parquet(
            source.matches,
            label=f"{source.name} dataset matches",
        )
        return prepare_source_matches(
            items,
            matches,
            source,
            seed=self.settings.seed,
        )

    def _load_source_labels(
        self,
        source: DatasetSourceSettings,
    ) -> pl.DataFrame:
        matches = read_parquet(
            source.matches,
            label=f"{source.name} dataset matches",
        )
        return prepare_source_labels(matches, source)

    def load_training_splits(self) -> LoadedTrainingSplits:
        item_lookup = _combined_item_lookup(self.settings)
        labels = {
            source.name: self._load_source_labels(source)
            for source in self.settings.sources
        }
        labels = resolve_source_overlaps(
            labels,
            self.settings.overlap_resolution,
        )
        prepared = {
            source.name: finalize_source_matches(
                item_lookup,
                labels[source.name],
                source,
                seed=self.settings.seed,
            )
            for source in self.settings.sources
        }
        validation_source = prepared.pop(self.settings.validation_source)
        split = split_matches(
            item_lookup,
            validation_source,
            _split_config(
                validation_fraction=self.settings.validation_fraction,
                leakage_scope=self.settings.leakage_scope,
                seed=self.settings.seed,
                candidate_splits=self.settings.candidate_splits,
            ),
        )
        train_parts = [split.train_matches, *prepared.values()]
        train_matches = pl.concat(train_parts, how="diagonal_relaxed")
        validation_matches = split.validation_matches.with_columns(
            pl.lit(1.0).cast(pl.Float32).alias("sample_weight")
        )
        validate_predefined_split(
            item_lookup,
            train_matches,
            validation_matches,
            leakage_scope=self.settings.leakage_scope,
        )
        logger.info(
            "Built mixed dataset: train_rows={}, validation_rows={}, sources={}",
            train_matches.height,
            validation_matches.height,
            [source.name for source in self.settings.sources],
        )
        items = _selected_mixed_items(
            self.settings,
            train_matches,
            validation_matches,
        )
        return LoadedTrainingSplits(items, train_matches, validation_matches)

    def load_inspection_frames(self) -> InspectionFrames:
        item_lookup = _combined_item_lookup(self.settings)
        source = self._source(self.settings.validation_source)
        matches = self._load_source(item_lookup, source)
        return InspectionFrames(
            _selected_mixed_items(self.settings, matches),
            matches,
        )


__all__ = ["BaseDatasetModel", "MixedDatasetModel"]
