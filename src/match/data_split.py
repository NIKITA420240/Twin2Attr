"""Model-independent train/validation splitting for product pairs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl
from loguru import logger
from sklearn.model_selection import GroupShuffleSplit

__all__ = [
    "DataSplitConfig",
    "DataSplitResult",
    "split_matches",
    "validate_predefined_split",
]


@dataclass(frozen=True, slots=True)
class DataSplitConfig:
    """Parameters for an automatic category/target-aware group split.

    ``leakage_scope='pair'`` keeps duplicate and reversed pairs together.
    ``leakage_scope='item'`` keeps entire connected item components together,
    which is stricter but may make exact stratification impossible.
    """

    validation_fraction: float = 0.2
    leakage_scope: str = "pair"
    seed: int = 42
    candidate_splits: int = 128

    def __post_init__(self) -> None:
        if not 0.0 < self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must be between zero and one")
        if self.leakage_scope not in {"none", "pair", "item"}:
            raise ValueError("leakage_scope must be one of: none, pair, item")
        if self.candidate_splits < 1:
            raise ValueError("candidate_splits must be positive")


@dataclass(frozen=True, slots=True)
class DataSplitResult:
    """Generated splits and compact diagnostics for logging/tests."""

    train_matches: pl.DataFrame
    validation_matches: pl.DataFrame
    actual_validation_fraction: float
    missing_eligible_strata: int
    max_stratum_fraction_deviation: float


def _validate_inputs(
    items: pl.DataFrame,
    matches: pl.DataFrame,
    *,
    id_column: str,
    category_column: str,
    left_id_column: str,
    right_id_column: str,
    target_column: str,
) -> None:
    if not isinstance(items, pl.DataFrame) or not isinstance(matches, pl.DataFrame):
        raise TypeError("items and matches must be polars.DataFrame instances")
    missing_item_columns = {id_column, category_column} - set(items.columns)
    if missing_item_columns:
        raise ValueError(f"items frame is missing columns: {sorted(missing_item_columns)}")
    missing_match_columns = {left_id_column, right_id_column, target_column} - set(matches.columns)
    if missing_match_columns:
        raise ValueError(f"matches frame is missing columns: {sorted(missing_match_columns)}")
    if matches.height < 2:
        raise ValueError("at least two matched pairs are required for splitting")
    if items.get_column(id_column).null_count():
        raise ValueError("items frame contains null ids")
    if items.height != items.get_column(id_column).n_unique():
        raise ValueError("items frame contains duplicate ids")
    if items.get_column(category_column).null_count():
        raise ValueError("items frame contains null categories")
    if any(matches.get_column(column).null_count() for column in (left_id_column, right_id_column)):
        raise ValueError("matches frame contains null item ids")
    available_ids = items.select(pl.col(id_column).alias("_item_id"))
    for match_id_column in (left_id_column, right_id_column):
        missing_id_count = (
            matches.select(pl.col(match_id_column).alias("_item_id"))
            .unique()
            .join(available_ids, on="_item_id", how="anti")
            .height
        )
        if missing_id_count:
            raise ValueError(
                f"matches frame references {missing_id_count} unknown ids in "
                f"{match_id_column!r}"
            )
    targets = matches.get_column(target_column)
    if targets.null_count() or not set(targets.unique().to_list()).issubset({0, 1, False, True}):
        raise ValueError("target must contain only 0 and 1")
    if set(int(value) for value in targets.unique()) != {0, 1}:
        raise ValueError("target must contain both classes")


def _enrich_with_category(
    items: pl.DataFrame,
    matches: pl.DataFrame,
    *,
    id_column: str,
    category_column: str,
    left_id_column: str,
) -> pl.DataFrame:
    category_lookup = items.select(
        pl.col(id_column).alias(left_id_column),
        pl.col(category_column).alias("_split_category"),
    )
    enriched = matches.with_row_index("_split_row").join(
        category_lookup,
        on=left_id_column,
        how="left",
        validate="m:1",
    )
    missing_categories = enriched.get_column("_split_category").null_count()
    if missing_categories:
        raise ValueError(f"matches frame references {missing_categories} unknown left item ids")
    return enriched.sort("_split_row")


def _pair_group_ids(
    matches: pl.DataFrame,
    *,
    left_id_column: str,
    right_id_column: str,
) -> np.ndarray:
    pair_keys = matches.select(
        pl.min_horizontal(left_id_column, right_id_column).alias("_pair_left"),
        pl.max_horizontal(left_id_column, right_id_column).alias("_pair_right"),
    ).with_row_index("_split_row")
    unique_pairs = pair_keys.select("_pair_left", "_pair_right").unique(
        maintain_order=True
    ).with_row_index("_split_group")
    return (
        pair_keys.join(
            unique_pairs,
            on=["_pair_left", "_pair_right"],
            how="left",
            validate="m:1",
        )
        .sort("_split_row")
        .get_column("_split_group")
        .to_numpy()
        .astype(np.int64, copy=False)
    )


def _item_component_ids(
    matches: pl.DataFrame,
    *,
    left_id_column: str,
    right_id_column: str,
) -> np.ndarray:
    """Return connected-component ids without materializing a graph library."""
    parent: dict[Any, Any] = {}
    rank: dict[Any, int] = {}

    def find(item: Any) -> Any:
        parent.setdefault(item, item)
        rank.setdefault(item, 0)
        root = item
        while parent[root] != root:
            root = parent[root]
        while parent[item] != item:
            next_item = parent[item]
            parent[item] = root
            item = next_item
        return root

    def union(left: Any, right: Any) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if rank[left_root] < rank[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        if rank[left_root] == rank[right_root]:
            rank[left_root] += 1

    rows = matches.select(left_id_column, right_id_column).iter_rows()
    for left_id, right_id in rows:
        union(left_id, right_id)

    component_by_root: dict[Any, int] = {}
    groups = np.empty(matches.height, dtype=np.int64)
    for index, (left_id, _) in enumerate(
        matches.select(left_id_column, right_id_column).iter_rows()
    ):
        root = find(left_id)
        groups[index] = component_by_root.setdefault(root, len(component_by_root))
    logger.info(
        "Built item-connected components: items={}, components={}",
        len(parent),
        len(component_by_root),
    )
    return groups


def _group_ids(
    matches: pl.DataFrame,
    leakage_scope: str,
    *,
    left_id_column: str,
    right_id_column: str,
) -> np.ndarray:
    if leakage_scope == "none":
        return np.arange(matches.height, dtype=np.int64)
    if leakage_scope == "pair":
        return _pair_group_ids(
            matches,
            left_id_column=left_id_column,
            right_id_column=right_id_column,
        )
    return _item_component_ids(
        matches,
        left_id_column=left_id_column,
        right_id_column=right_id_column,
    )


def _stratum_ids(enriched: pl.DataFrame, *, target_column: str) -> np.ndarray:
    keys = enriched.select("_split_category", target_column).with_row_index(
        "_split_row"
    )
    unique_keys = keys.select("_split_category", target_column).unique(
        maintain_order=True
    ).with_row_index("_split_stratum")
    return (
        keys.join(
            unique_keys,
            on=["_split_category", target_column],
            how="left",
            validate="m:1",
        )
        .sort("_split_row")
        .get_column("_split_stratum")
        .to_numpy()
        .astype(np.int64, copy=False)
    )


def _eligible_strata(
    strata: np.ndarray,
    groups: np.ndarray,
) -> np.ndarray:
    stratum_count = int(strata.max()) + 1
    group_strata = np.unique(np.column_stack((strata, groups)), axis=0)
    group_counts = np.bincount(group_strata[:, 0], minlength=stratum_count)
    return group_counts >= 2


def _split_diagnostics(
    strata: np.ndarray,
    eligible_strata: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    expected_fraction: float,
) -> tuple[int, float, float]:
    stratum_count = int(strata.max()) + 1
    total_counts = np.bincount(strata, minlength=stratum_count)
    validation_counts = np.bincount(
        strata[validation_indices], minlength=stratum_count
    )
    train_counts = total_counts - validation_counts

    missing = int(
        np.sum(
            eligible_strata
            & ((train_counts == 0) | (validation_counts == 0))
        )
    )
    validation_rates = validation_counts / total_counts
    max_deviation = float(np.max(np.abs(validation_rates - expected_fraction)))
    actual_fraction = len(validation_indices) / len(strata)
    return missing, max_deviation, actual_fraction


def _select_rows(frame: pl.DataFrame, indices: np.ndarray) -> pl.DataFrame:
    selected = pl.Series("_selected_row", np.sort(indices).astype(np.int64, copy=False))
    return (
        frame.with_row_index("_selected_row")
        .filter(pl.col("_selected_row").is_in(selected))
        .drop("_selected_row")
    )


def split_matches(
    items: pl.DataFrame,
    matches: pl.DataFrame,
    config: DataSplitConfig,
    *,
    id_column: str = "id",
    category_column: str = "category",
    left_id_column: str = "id1",
    right_id_column: str = "id2",
    target_column: str = "target",
) -> DataSplitResult:
    """Choose the best group split among deterministic random candidates.

    Candidates are ranked first by whether eligible ``category × target``
    strata occur on both sides, then by deviation from the requested number of
    validation rows in each stratum, and finally by total validation size.
    """
    if not isinstance(config, DataSplitConfig):
        raise TypeError("config must be a DataSplitConfig")
    _validate_inputs(
        items,
        matches,
        id_column=id_column,
        category_column=category_column,
        left_id_column=left_id_column,
        right_id_column=right_id_column,
        target_column=target_column,
    )
    enriched = _enrich_with_category(
        items,
        matches,
        id_column=id_column,
        category_column=category_column,
        left_id_column=left_id_column,
    )
    strata = _stratum_ids(enriched, target_column=target_column)
    groups = _group_ids(
        matches,
        config.leakage_scope,
        left_id_column=left_id_column,
        right_id_column=right_id_column,
    )
    unique_group_count = len(np.unique(groups))
    if unique_group_count < 2:
        raise ValueError(f"cannot split {unique_group_count} independent {config.leakage_scope} group")

    splitter = GroupShuffleSplit(
        n_splits=config.candidate_splits,
        test_size=config.validation_fraction,
        random_state=config.seed,
    )
    total_counts = np.bincount(strata)
    expected_validation_counts = total_counts * config.validation_fraction
    eligible_strata = _eligible_strata(strata, groups)
    best: tuple[tuple[int, float, float], np.ndarray, np.ndarray] | None = None
    dummy_features = np.zeros((matches.height, 1), dtype=np.uint8)
    for train_indices, validation_indices in splitter.split(dummy_features, strata, groups):
        missing, _, actual_fraction = _split_diagnostics(
            strata,
            eligible_strata,
            train_indices,
            validation_indices,
            config.validation_fraction,
        )
        validation_counts = np.bincount(strata[validation_indices], minlength=len(total_counts))
        distribution_error = float(np.abs(validation_counts - expected_validation_counts).sum() / matches.height)
        size_error = abs(actual_fraction - config.validation_fraction)
        score = (missing, distribution_error, size_error)
        if best is None or score < best[0]:
            best = (score, train_indices, validation_indices)

    if best is None:
        raise RuntimeError("group split search did not produce a candidate")
    _, train_indices, validation_indices = best
    missing, max_deviation, actual_fraction = _split_diagnostics(
        strata,
        eligible_strata,
        train_indices,
        validation_indices,
        config.validation_fraction,
    )
    targets = matches.get_column(target_column).to_numpy()
    if set(np.unique(targets[train_indices]).tolist()) != {0, 1}:
        raise ValueError("group constraints produced a training split without both target classes")
    if set(np.unique(targets[validation_indices]).tolist()) != {0, 1}:
        raise ValueError("group constraints produced a validation split without both target classes")
    if missing:
        logger.warning(
            "Generated split misses {} eligible category×target strata on one side; "
            "group constraints make a closer candidate unavailable",
            missing,
        )
    logger.info(
        "Generated group-aware split: scope={}, candidates={}, train_rows={}, "
        "validation_rows={}, requested_fraction={:.4f}, actual_fraction={:.4f}, "
        "max_stratum_fraction_deviation={:.4f}",
        config.leakage_scope,
        config.candidate_splits,
        len(train_indices),
        len(validation_indices),
        config.validation_fraction,
        actual_fraction,
        max_deviation,
    )
    return DataSplitResult(
        train_matches=_select_rows(matches, train_indices),
        validation_matches=_select_rows(matches, validation_indices),
        actual_validation_fraction=actual_fraction,
        missing_eligible_strata=missing,
        max_stratum_fraction_deviation=max_deviation,
    )


def validate_predefined_split(
    items: pl.DataFrame,
    train_matches: pl.DataFrame,
    validation_matches: pl.DataFrame,
    *,
    leakage_scope: str = "pair",
    id_column: str = "id",
    category_column: str = "category",
    left_id_column: str = "id1",
    right_id_column: str = "id2",
    target_column: str = "target",
) -> None:
    """Validate labeled predefined splits and reject cross-split leakage."""
    if leakage_scope not in {"none", "pair", "item"}:
        raise ValueError("leakage_scope must be one of: none, pair, item")
    for name, frame in (("train", train_matches), ("validation", validation_matches)):
        try:
            _validate_inputs(
                items,
                frame,
                id_column=id_column,
                category_column=category_column,
                left_id_column=left_id_column,
                right_id_column=right_id_column,
                target_column=target_column,
            )
        except ValueError as error:
            raise ValueError(f"invalid predefined {name} split: {error}") from error

    if leakage_scope == "pair":
        def canonical_pairs(frame: pl.DataFrame) -> pl.DataFrame:
            return frame.select(
                pl.min_horizontal(left_id_column, right_id_column).alias("_left"),
                pl.max_horizontal(left_id_column, right_id_column).alias("_right"),
            ).unique()

        leaked_count = canonical_pairs(train_matches).join(
            canonical_pairs(validation_matches),
            on=["_left", "_right"],
            how="inner",
        ).height
        if leaked_count:
            raise ValueError(
                f"predefined splits contain {leaked_count} duplicate or reversed pairs"
            )
    elif leakage_scope == "item":
        def item_ids(frame: pl.DataFrame) -> pl.DataFrame:
            return pl.concat(
                [
                    frame.select(pl.col(left_id_column).alias("_item_id")),
                    frame.select(pl.col(right_id_column).alias("_item_id")),
                ]
            ).unique()

        leaked_count = item_ids(train_matches).join(
            item_ids(validation_matches), on="_item_id", how="inner"
        ).height
        if leaked_count:
            raise ValueError(
                f"predefined splits share {leaked_count} item ids"
            )

    combined = pl.concat(
        [
            train_matches.select(left_id_column, right_id_column, target_column),
            validation_matches.select(left_id_column, right_id_column, target_column),
        ],
        how="vertical_relaxed",
    )
    train_size = train_matches.height
    enriched = _enrich_with_category(
        items,
        combined,
        id_column=id_column,
        category_column=category_column,
        left_id_column=left_id_column,
    )
    strata = _stratum_ids(enriched, target_column=target_column)
    groups = _group_ids(
        combined,
        leakage_scope,
        left_id_column=left_id_column,
        right_id_column=right_id_column,
    )
    train_indices = np.arange(train_size, dtype=np.int64)
    validation_indices = np.arange(train_size, combined.height, dtype=np.int64)
    actual_fraction = validation_matches.height / combined.height
    missing, max_deviation, _ = _split_diagnostics(
        strata,
        _eligible_strata(strata, groups),
        train_indices,
        validation_indices,
        actual_fraction,
    )
    if missing:
        logger.warning(
            "Predefined split misses {} eligible category×target strata on one side",
            missing,
        )
    logger.info(
        "Validated predefined split: scope={}, train_rows={}, validation_rows={}, "
        "max_stratum_fraction_deviation={:.4f}",
        leakage_scope,
        train_matches.height,
        validation_matches.height,
        max_deviation,
    )
