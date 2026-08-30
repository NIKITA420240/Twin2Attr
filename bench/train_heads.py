"""Train the final bidirectional category-conditioned head."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, pstdev
from time import perf_counter
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np
import polars as pl
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from tqdm.auto import tqdm

from frozen_features import (
    extract_frozen_features,
    load_frozen_backbone,
    parameter_count,
    sha256_file,
)
from heads import BidirectionalCategoryConditionedHead, head_loss
from match.config import load_app_config_file
from match.data.preprocessing import prepare_configured_items
from match.data_split import DataSplitConfig, split_matches
from match.experiments import validation_pairs_hash
from match.pair_encoding import PairEncodingCollator
from match.prepare_data import prepare_pairs


HEAD_NAME = "bidirectional_category_conditioned"


def _project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_config(path: Path) -> dict[str, Any]:
    values = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("benchmark config must contain a mapping")
    return values


def _weights_snapshot() -> dict[str, tuple[int, int]]:
    root = PROJECT_ROOT / "weights"
    return {
        str(path.relative_to(root)): (path.stat().st_size, path.stat().st_mtime_ns)
        for path in root.rglob("*")
        if path.is_file()
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sample_matches(
    matches: pl.DataFrame,
    items_path: Path,
    max_pairs: int | None,
    *,
    seed: int,
) -> pl.DataFrame:
    if max_pairs is None or matches.height <= max_pairs:
        return matches
    categories = (
        pl.scan_parquet(items_path)
        .select(pl.col("id").alias("id1"), "category")
        .join(matches.lazy().select("id1").unique(), on="id1", how="semi")
        .collect()
    )
    enriched = matches.join(categories, on="id1", how="left", validate="m:1")
    if enriched.get_column("category").null_count():
        raise ValueError("cannot sample matches with unknown left item ids")
    fraction = max_pairs / matches.height
    sampled = (
        enriched.with_columns(
            pl.struct("id1", "id2", "target").hash(seed=seed).alias("_hash"),
            pl.len().over("category", "target").alias("_group_rows"),
        )
        .sort("category", "target", "_hash")
        .with_columns(pl.int_range(pl.len()).over("category", "target").alias("_rank"))
        .filter(
            pl.col("_rank")
            < (pl.col("_group_rows") * fraction).ceil().cast(pl.UInt32)
        )
        .drop("category", "_hash", "_group_rows", "_rank")
    )
    return (
        sampled.sample(n=max_pairs, shuffle=True, seed=seed)
        if sampled.height > max_pairs
        else sampled
    )


def _load_required_items(items_path: Path, matches: pl.DataFrame) -> pl.DataFrame:
    required_ids = pl.concat(
        [
            matches.select(pl.col("id1").alias("id")),
            matches.select(pl.col("id2").alias("id")),
        ]
    ).unique()
    items = (
        pl.scan_parquet(items_path)
        .join(required_ids.lazy(), on="id", how="semi")
        .collect()
    )
    if items.height != required_ids.height:
        raise ValueError(
            f"items lookup returned {items.height} rows for {required_ids.height} ids"
        )
    return items


def _prepare_items(
    items: pl.DataFrame,
    config: dict[str, Any],
    *,
    cache_dir: Path,
    cache_tag: str,
) -> tuple[pl.DataFrame, str]:
    data = config["data"]
    normalize = bool(data["normalize_attributes"])
    cache_path = cache_dir / f"items-{cache_tag}-normalized-{int(normalize)}.parquet"
    metadata_path = cache_path.with_suffix(".json")
    if cache_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        return pl.read_parquet(cache_path), str(metadata["attributes_column"])
    if normalize:
        pipeline_config = load_app_config_file(_project_path(data["pipeline_config"]))
        prepared = prepare_configured_items(items, pipeline_config)
        frame = prepared.frame
        attributes_column = prepared.attributes_column
    else:
        frame = items
        attributes_column = "attributes"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(cache_path, compression="zstd")
    metadata_path.write_text(
        json.dumps(
            {
                "attributes_column": attributes_column,
                "rows": frame.height,
                "normalized": normalize,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return frame, attributes_column


def _cache_tag(
    matches: pl.DataFrame,
    checkpoint_sha256: str,
    config: dict[str, Any],
) -> str:
    digest = hashlib.sha256()
    digest.update(checkpoint_sha256.encode())
    digest.update(str(matches.height).encode())
    digest.update(matches.select("id1", "id2", "target").hash_rows().to_numpy().tobytes())
    digest.update(json.dumps(config["encoder"], sort_keys=True).encode())
    digest.update(str(config["data"]["normalize_attributes"]).encode())
    return digest.hexdigest()[:16]


def _macro_average_precision(
    labels: np.ndarray,
    probabilities: np.ndarray,
    categories: np.ndarray,
) -> float:
    scores = []
    for category in np.unique(categories):
        selected = categories == category
        category_labels = labels[selected]
        if len(np.unique(category_labels)) != 2:
            raise ValueError(f"category {category!r} does not contain both classes")
        scores.append(average_precision_score(category_labels, probabilities[selected]))
    return float(np.mean(scores))


def _evaluate(
    model: BidirectionalCategoryConditionedHead,
    features: np.ndarray,
    reverse_features: np.ndarray,
    category_ids: np.ndarray,
    labels: np.ndarray,
    categories: np.ndarray,
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray]:
    model.eval()
    probabilities: list[np.ndarray] = []
    directional_gaps: list[np.ndarray] = []
    with torch.inference_mode():
        for offset in range(0, len(labels), batch_size):
            end = min(offset + batch_size, len(labels))
            forward = torch.from_numpy(
                np.asarray(features[offset:end], dtype=np.float32)
            ).to(device)
            reverse = torch.from_numpy(
                np.asarray(reverse_features[offset:end], dtype=np.float32)
            ).to(device)
            category = torch.from_numpy(
                np.asarray(category_ids[offset:end], dtype=np.int64)
            ).to(device)
            output = model(
                forward,
                reverse_features=reverse,
                category_ids=category,
            )
            probabilities.append(torch.sigmoid(output.logit).cpu().numpy())
            directional_gaps.append(
                torch.abs(output.forward_logit - output.reverse_logit).cpu().numpy()
            )
    scores = np.concatenate(probabilities)
    gaps = np.concatenate(directional_gaps)
    return {
        "macro_pr_auc": _macro_average_precision(labels, scores, categories),
        "global_pr_auc": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "mean_directional_logit_gap": float(gaps.mean()),
    }, scores


def _train_one(
    seed: int,
    *,
    all_features: np.ndarray,
    all_reverse_features: np.ndarray,
    all_category_ids: np.ndarray,
    train_rows: int,
    train_labels: np.ndarray,
    validation_labels: np.ndarray,
    validation_categories: np.ndarray,
    hidden_size: int,
    config: dict[str, Any],
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    _seed_everything(seed)
    head_config = config["head"]
    training = config["training"]
    model = BidirectionalCategoryConditionedHead(
        hidden_size=hidden_size,
        projection_dim=int(head_config["projection_dim"]),
        mlp_dim=int(head_config["mlp_dim"]),
        dropout=float(head_config["dropout"]),
        num_categories=int(np.max(all_category_ids)) + 1,
        category_embedding_dim=int(head_config["category_embedding_dim"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        min_lr=1.0e-5,
    )
    positives = float(train_labels.sum())
    positive_weight = torch.tensor(
        (len(train_labels) - positives) / positives,
        device=device,
    )
    batch_size = int(training["batch_size"])
    max_epochs = int(training["epochs"])
    generator = torch.Generator().manual_seed(seed)
    best_score = -math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    started = perf_counter()

    train_features = all_features[:train_rows]
    validation_features = all_features[train_rows:]
    train_reverse = all_reverse_features[:train_rows]
    validation_reverse = all_reverse_features[train_rows:]
    train_categories = all_category_ids[:train_rows]
    validation_category_ids = all_category_ids[train_rows:]

    for epoch in range(1, max_epochs + 1):
        model.train()
        permutation = torch.randperm(train_rows, generator=generator).numpy()
        total_loss = 0.0
        seen = 0
        for offset in tqdm(
            range(0, train_rows, batch_size),
            desc=f"{HEAD_NAME} seed={seed} epoch={epoch}",
            unit="batch",
            leave=False,
        ):
            indices = permutation[offset : offset + batch_size]
            forward_batch = torch.from_numpy(
                np.asarray(train_features[indices], dtype=np.float32)
            ).to(device)
            reverse_batch = torch.from_numpy(
                np.asarray(train_reverse[indices], dtype=np.float32)
            ).to(device)
            category_batch = torch.from_numpy(
                np.asarray(train_categories[indices], dtype=np.int64)
            ).to(device)
            target_batch = torch.from_numpy(
                train_labels[indices].astype(np.float32)
            ).to(device)
            optimizer.zero_grad(set_to_none=True)
            output = model(
                forward_batch,
                reverse_features=reverse_batch,
                category_ids=category_batch,
            )
            loss = head_loss(
                output,
                target_batch,
                positive_weight=positive_weight,
                consistency_weight=float(head_config["consistency_weight"]),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            current_rows = len(indices)
            total_loss += float(loss.detach()) * current_rows
            seen += current_rows

        validation_metrics, _ = _evaluate(
            model,
            validation_features,
            validation_reverse,
            validation_category_ids,
            validation_labels,
            validation_categories,
            batch_size=max(batch_size, 4096),
            device=device,
        )
        score = validation_metrics["macro_pr_auc"]
        scheduler.step(score)
        record = {
            "epoch": epoch,
            "train_loss": total_loss / seen,
            **validation_metrics,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        print(
            f"{HEAD_NAME} seed={seed} epoch={epoch}: "
            f"loss={record['train_loss']:.5f}, macro_pr_auc={score:.6f}"
        )
        if score > best_score + 1.0e-6:
            best_score = score
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("training did not produce a best head state")
    model.load_state_dict(best_state)
    final_metrics, probabilities = _evaluate(
        model,
        validation_features,
        validation_reverse,
        validation_category_ids,
        validation_labels,
        validation_categories,
        batch_size=max(batch_size, 4096),
        device=device,
    )
    symmetry_rows = min(4096, len(validation_labels))
    with torch.inference_mode():
        forward = torch.from_numpy(
            np.asarray(validation_features[:symmetry_rows], dtype=np.float32)
        ).to(device)
        reverse = torch.from_numpy(
            np.asarray(validation_reverse[:symmetry_rows], dtype=np.float32)
        ).to(device)
        category = torch.from_numpy(
            np.asarray(validation_category_ids[:symmetry_rows], dtype=np.int64)
        ).to(device)
        original = model(
            forward,
            reverse_features=reverse,
            category_ids=category,
        ).logit
        swapped = model(
            reverse,
            reverse_features=forward,
            category_ids=category,
        ).logit
        symmetry_error = float(torch.max(torch.abs(original - swapped)).cpu())

    case_dir = output_dir / HEAD_NAME / f"seed-{seed}"
    case_dir.mkdir(parents=True, exist_ok=False)
    torch.save(
        {
            "head": HEAD_NAME,
            "hidden_size": hidden_size,
            "num_categories": int(np.max(all_category_ids)) + 1,
            "head_config": head_config,
            "state_dict": best_state,
        },
        case_dir / "head.pt",
    )
    np.save(case_dir / "validation_probabilities.npy", probabilities)
    result = {
        "head": HEAD_NAME,
        "seed": seed,
        "parameters": parameter_count(model),
        "best_epoch": best_epoch,
        "trained_epochs": max_epochs,
        **final_metrics,
        "symmetry_max_abs_error": symmetry_error,
        "training_seconds": perf_counter() - started,
    }
    (case_dir / "metrics.json").write_text(
        json.dumps({**result, "history": history}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="bench/config.yaml")
    parser.add_argument("--checkpoint-file", default=None)
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    config = _load_config(_project_path(args.config))
    if args.seeds is not None:
        config["training"]["seeds"] = list(args.seeds)
    weights_before = _weights_snapshot()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    checkpoint_file = _project_path(
        args.checkpoint_file or config["encoder"]["checkpoint_file"]
    )
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"immutable checkpoint does not exist: {checkpoint_file}")
    checkpoint_stat = (checkpoint_file.stat().st_size, checkpoint_file.stat().st_mtime_ns)
    checkpoint_sha256 = sha256_file(checkpoint_file)
    print(f"Device: {device}; checkpoint sha256={checkpoint_sha256}")

    items_path = _project_path(config["data"]["items_path"])
    matches = pl.read_parquet(_project_path(config["data"]["matches_path"])).select(
        "id1",
        "id2",
        pl.col("target").cast(pl.Int8),
    )
    matches = _sample_matches(matches, items_path, args.max_pairs, seed=42)
    raw_items = _load_required_items(items_path, matches)
    split = split_matches(
        raw_items,
        matches,
        DataSplitConfig(
            validation_fraction=float(config["data"]["validation_fraction"]),
            leakage_scope=str(config["data"]["leakage_scope"]),
            seed=42,
            candidate_splits=int(config["data"]["candidate_splits"]),
        ),
    )
    combined_matches = pl.concat(
        [
            split.train_matches.with_columns(pl.lit("train").alias("split")),
            split.validation_matches.with_columns(pl.lit("validation").alias("split")),
        ]
    )
    cache_dir = _project_path(config["output"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_tag = _cache_tag(matches, checkpoint_sha256, config)
    prepared_items, attributes_column = _prepare_items(
        raw_items,
        config,
        cache_dir=cache_dir,
        cache_tag=cache_tag,
    )
    pair_metadata = combined_matches.join(
        prepared_items.select(
            pl.col("id").alias("id1"),
            pl.col("category").alias("category"),
        ),
        on="id1",
        how="left",
        validate="m:1",
    )
    metadata_path = cache_dir / f"pairs-{cache_tag}.parquet"
    pair_metadata.write_parquet(metadata_path)
    features_path = cache_dir / f"frozen-features-{cache_tag}.npy"
    reverse_features_path = cache_dir / f"reverse-frozen-features-{cache_tag}.npy"

    if not features_path.is_file() or not reverse_features_path.is_file():
        pairs = prepare_pairs(
            prepared_items,
            combined_matches.drop("split"),
            attributes_column=attributes_column,
        )
        config_dir = _project_path(config["encoder"]["config_dir"])
        backbone, tokenizer, hidden_size = load_frozen_backbone(
            config_dir,
            checkpoint_file,
            device=device,
        )
        encoder = config["encoder"]
        collator = PairEncodingCollator(
            tokenizer,
            int(encoder["max_length"]),
            use_field_tokens=True,
            max_attribute_value_chars=encoder["max_attribute_value_chars"],
            max_attribute_value_tokens=encoder["max_attribute_value_tokens"],
            include_labels=False,
            batch_fields=bool(encoder["batch_fields"]),
            field_chunk_size=int(encoder["field_chunk_size"]),
        )
        if not features_path.is_file():
            extract_frozen_features(
                pairs,
                backbone=backbone,
                tokenizer=tokenizer,
                collator=collator,
                hidden_size=hidden_size,
                output_path=features_path,
                batch_size=int(encoder["extraction_batch_size"]),
                device=device,
                description="Frozen encoder (A, B)",
            )
        if not reverse_features_path.is_file():
            reverse_pairs = [replace(pair, left=pair.right, right=pair.left) for pair in pairs]
            extract_frozen_features(
                reverse_pairs,
                backbone=backbone,
                tokenizer=tokenizer,
                collator=collator,
                hidden_size=hidden_size,
                output_path=reverse_features_path,
                batch_size=int(encoder["extraction_batch_size"]),
                device=device,
                description="Frozen encoder (B, A)",
            )
        del pairs, backbone
        if device.type == "cuda":
            torch.cuda.empty_cache()

    all_features = np.load(features_path, mmap_mode="r")
    all_reverse_features = np.load(reverse_features_path, mmap_mode="r")
    if all_features.shape != all_reverse_features.shape:
        raise RuntimeError("forward and reverse feature shapes differ")
    if all_features.shape[0] != combined_matches.height:
        raise RuntimeError("cached feature count does not match current pairs")
    hidden_size = int(all_features.shape[-1])
    category_names = sorted(str(value) for value in pair_metadata["category"].unique())
    category_to_id = {value: index for index, value in enumerate(category_names)}
    all_category_ids = np.asarray(
        [category_to_id[str(value)] for value in pair_metadata["category"]],
        dtype=np.int64,
    )

    train_rows = split.train_matches.height
    train_labels = pair_metadata["target"][:train_rows].to_numpy().astype(np.int64)
    validation_labels = pair_metadata["target"][train_rows:].to_numpy().astype(np.int64)
    validation_categories = pair_metadata["category"][train_rows:].to_numpy()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = _project_path(config["output"]["runs_dir"]) / timestamp
    suffix = 1
    while run_dir.exists():
        run_dir = run_dir.with_name(f"{timestamp}-{suffix}")
        suffix += 1
    run_dir.mkdir(parents=True)
    resolved = {
        **config,
        "resolved": {
            "device": str(device),
            "checkpoint_file": str(checkpoint_file),
            "checkpoint_sha256": checkpoint_sha256,
            "feature_path": str(features_path),
            "reverse_feature_path": str(reverse_features_path),
            "pair_metadata_path": str(metadata_path),
            "train_rows": train_rows,
            "validation_rows": len(validation_labels),
            "validation_pairs_hash": validation_pairs_hash(split.validation_matches),
            "hidden_size": hidden_size,
            "category_to_id": category_to_id,
        },
    }
    (run_dir / "resolved_config.json").write_text(
        json.dumps(resolved, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    rows = [
        _train_one(
            int(seed),
            all_features=all_features,
            all_reverse_features=all_reverse_features,
            all_category_ids=all_category_ids,
            train_rows=train_rows,
            train_labels=train_labels,
            validation_labels=validation_labels,
            validation_categories=validation_categories,
            hidden_size=hidden_size,
            config=config,
            device=device,
            output_dir=run_dir,
        )
        for seed in config["training"]["seeds"]
    ]
    macro = [float(row["macro_pr_auc"]) for row in rows]
    aggregates = {
        "head": HEAD_NAME,
        "runs": len(rows),
        "mean_macro_pr_auc": mean(macro),
        "std_macro_pr_auc": pstdev(macro),
        "mean_global_pr_auc": mean(float(row["global_pr_auc"]) for row in rows),
        "mean_roc_auc": mean(float(row["roc_auc"]) for row in rows),
        "parameters": int(rows[0]["parameters"]),
    }
    pl.DataFrame(rows).write_parquet(run_dir / "results.parquet")
    pl.DataFrame([aggregates]).write_parquet(run_dir / "aggregates.parquet")
    (run_dir / "report.json").write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "checkpoint_sha256": checkpoint_sha256,
                "train_rows": train_rows,
                "validation_rows": len(validation_labels),
                "results": rows,
                "aggregates": aggregates,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if _weights_snapshot() != weights_before:
        raise RuntimeError("weights/ changed during training")
    if (checkpoint_file.stat().st_size, checkpoint_file.stat().st_mtime_ns) != checkpoint_stat:
        raise RuntimeError("immutable checkpoint changed during training")
    print("\nFinal aggregate")
    print(pl.DataFrame([aggregates]))
    print(f"Saved to {run_dir}")


if __name__ == "__main__":
    main()
