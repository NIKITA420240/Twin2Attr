"""Versioned experiment directories and a compact CSV experiment registry."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import tempfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import polars as pl

from .config import AppConfig
from .models.artifacts import TrainingArtifacts
from .paths import PROJECT_ROOT

if TYPE_CHECKING:
    from .data_models.contracts import LoadedTrainingSplits


_EXPERIMENT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_REQUIRED_EXPERIMENT_REGISTRY_COLUMNS = (
    "experiment_name",
    "created_at",
    "model",
    "epochs",
    "macro_pr_auc_human",
    "train_data",
    "backbone_lr",
    "embeddings_lr",
    "train_new_token_embeddings_only",
    "head_lr",
    "train_last_n_layers",
    "layerwise_lr_decay",
    "scheduler",
    "head",
    "train_rows",
    "validation_rows",
    "validation_pairs_hash",
    "s3_path",
)
EXPERIMENT_REGISTRY_COLUMNS = (
    *_REQUIRED_EXPERIMENT_REGISTRY_COLUMNS,
    "registry_schema_version",
    "completed_epochs",
    "performance_epochs_observed",
    "train_batch_size",
    "gradient_accumulation_steps",
    "effective_batch_size",
    "max_length",
    "avg_train_epoch_seconds",
    "total_train_seconds",
    "avg_examples_per_second",
    "avg_real_tokens_per_second",
    "avg_padding_efficiency",
    "avg_train_peak_cuda_memory_gib",
    "max_train_peak_cuda_memory_gib",
    "gpu_name",
    "gpu_count",
    "precision",
    "trainable_parameters",
    "warmup_ratio",
    "special_token_initialization",
    "special_token_adaptation_mode",
    "special_token_adaptation_max_optimizer_steps",
    "special_token_adaptation_actual_optimizer_steps",
    "special_token_adaptation_embeddings_lr",
    "special_token_adaptation_full_model_backbone_lr",
    "special_token_adaptation_seconds",
)


def validate_experiment_name(value: str) -> str:
    """Validate a user-owned directory name without inventing a version."""
    name = value.strip()
    if not _EXPERIMENT_NAME.fullmatch(name):
        raise ValueError(
            "EXPERIMENT_NAME must start with a letter or digit and contain "
            "only letters, digits, '.', '_' and '-' (maximum 128 characters)"
        )
    return name


def configure_experiment(
    config: AppConfig,
    experiment_name: str,
    *,
    experiments_root: Path | None = None,
) -> tuple[AppConfig, Path]:
    """Redirect every generated artifact and log into one experiment folder."""
    name = validate_experiment_name(experiment_name)
    root = (experiments_root or PROJECT_ROOT / "experiments").resolve()
    experiment_dir = root / name
    model_root = experiment_dir / "models" / "twin2attr"
    configured = replace(
        config,
        model_description=replace(
            config.model_description,
            transformer=replace(
                config.model_description.transformer,
                artifact_dir=model_root / "transformer",
            ),
            maxpooling=replace(
                config.model_description.maxpooling,
                artifact_path=model_root / "maxpooling.joblib",
            ),
            fusion=replace(
                config.model_description.fusion,
                artifact_path=model_root / "fusion.pt",
            ),
            boosting=replace(
                config.model_description.boosting,
                artifact_dir=model_root / "boosting",
            ),
            stacking=replace(
                config.model_description.stacking,
                artifact_dir=model_root / "stacking",
            ),
        ),
        training=replace(
            config.training,
            resolved_config_path=model_root / "pipeline_config.yaml",
            solution_path=experiment_dir / "solution.json",
        ),
        inference=replace(
            config.inference,
            solution_path=experiment_dir / "solution.json",
        ),
        logging=replace(
            config.logging,
            file=experiment_dir / "logs" / "pipeline.log",
        ),
    )
    return configured, experiment_dir


def _matches_hash(matches: pl.DataFrame, *, include_target: bool) -> str:
    required = [
        "id1",
        "id2",
        *(("target",) if include_target else ()),
    ]
    if any(name not in matches.columns for name in required):
        raise ValueError(f"matches must contain {', '.join(required)}")
    left = pl.col("id1").cast(pl.String).fill_null("<null>")
    right = pl.col("id2").cast(pl.String).fill_null("<null>")
    expressions = [
        pl.min_horizontal(left, right).alias("id1"),
        pl.max_horizontal(left, right).alias("id2"),
    ]
    if include_target:
        expressions.append(
            pl.col("target").cast(pl.String).fill_null("<null>")
        )
    canonical = (
        matches.select(expressions)
        .sort(required)
        .write_csv()
        .encode("utf-8")
    )
    return hashlib.sha256(canonical).hexdigest()


def validation_pairs_hash(matches: pl.DataFrame) -> str:
    """Return a row- and pair-direction-independent validation split hash."""
    return _matches_hash(matches, include_target=True)


def _training_metadata(config: AppConfig) -> dict[str, Any]:
    path = (
        config.model_description.transformer.artifact_dir
        / "training_metadata.json"
    )
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _head_description(config: AppConfig) -> str:
    head = config.model_description.transformer.head
    if head.type == "default":
        return "Default HF: CLS -> Dropout -> Linear(2)"
    if head.type == "hybrid":
        mode = "trainable" if head.train_logit_weights else "fixed"
        return (
            "Hybrid("
            f"native={head.native_logit_weight}, "
            f"attention={head.attention_logit_weight}, weights={mode})"
        )
    poolings = ", ".join(
        "AttentionPool" if pooling == "attention" else pooling.upper()
        for pooling in head.poolings
    )
    hidden = ", ".join(str(value) for value in head.mlp_hidden_dims)
    mlp = f"MLP(hidden=[{hidden}]) -> 2 logits" if hidden else "Linear(2)"
    return f"Concat({poolings}) -> {mlp}, dropout={head.dropout}"


def _selected_split_settings(config: AppConfig) -> Any:
    if config.training.data_model == "base_dataset":
        return config.data_model_description.base_dataset
    return getattr(config.data_model_description, config.training.data_model)


def _training_data_label(config: AppConfig) -> str:
    if config.training.data_model == "base_dataset":
        return "Human"
    settings = _selected_split_settings(config)
    source_names = (
        source.name
        for source in settings.sources
    )
    return " + ".join(
        name.upper() if name.lower() == "llm" else name.replace("_", " ").title()
        for name in source_names
    )


def _sample_weighting_summary(matches: pl.DataFrame) -> dict[str, Any]:
    """Summarize how source and graph weights changed the training loss."""
    if "sample_weight" not in matches.columns:
        return {}
    sources = (
        matches.get_column("data_source").unique().sort().to_list()
        if "data_source" in matches.columns
        else ["all"]
    )
    result: dict[str, Any] = {}
    for source in sources:
        selected = (
            matches
            if source == "all"
            else matches.filter(pl.col("data_source") == source)
        )
        summary: dict[str, Any] = {
            "rows": selected.height,
            "mean_sample_weight": float(
                selected.get_column("sample_weight").mean()
            ),
            "target_counts": {
                str(row["target"]): int(row["len"])
                for row in selected.group_by("target")
                .len()
                .sort("target")
                .iter_rows(named=True)
            },
        }
        if {"id1", "id2"}.issubset(selected.columns):
            summary["pairs_hash"] = _matches_hash(
                selected,
                include_target=False,
            )
            summary["targets_hash"] = _matches_hash(
                selected,
                include_target=True,
            )
        if "annotation_votes" in selected.columns:
            vote_rows = selected.filter(pl.col("annotation_votes").is_not_null())
            if vote_rows.height:
                summary["vote_counts"] = {
                    str(row["annotation_votes"]): int(row["len"])
                    for row in vote_rows.group_by("annotation_votes")
                    .len()
                    .sort("annotation_votes")
                    .iter_rows(named=True)
                }
        if "weight_multiplier" in selected.columns:
            multipliers = selected.get_column("weight_multiplier").drop_nulls()
            if len(multipliers):
                summary.update(
                    {
                        "mean_weight_multiplier": float(multipliers.mean()),
                        "min_weight_multiplier": float(multipliers.min()),
                        "downweighted_fraction": float(
                            (multipliers < 1.0).mean()
                        ),
                    }
                )
        if "transitivity_violations" in selected.columns:
            violations = selected.get_column(
                "transitivity_violations"
            ).drop_nulls()
            if len(violations):
                summary["violating_fraction"] = float(
                    (violations > 0).mean()
                )
        result[str(source)] = summary
    return result


def save_experiment_record(
    config: AppConfig,
    artifacts: TrainingArtifacts,
    splits: "LoadedTrainingSplits",
    *,
    experiment_name: str,
) -> Path:
    """Persist the self-contained record for one successful experiment."""
    name = validate_experiment_name(experiment_name)
    experiment_dir = config.training.solution_path.parent
    experiment_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )
    split_settings = _selected_split_settings(config)
    split_hash = validation_pairs_hash(splits.validation_matches)
    stacking_matches = getattr(splits, "stacking_matches", None)
    metrics = dict(artifacts.metrics)
    metric_name = f"{artifacts.predictor}.validation_macro_pr_auc"
    macro_pr_auc = metrics.get(metric_name)

    transformer = config.model_description.transformer
    training_metadata = _training_metadata(config)
    resolved = training_metadata.get("resolved_config")
    resolved_config = resolved if isinstance(resolved, dict) else {}
    backbone_lr = resolved_config.get("learning_rate", transformer.learning_rate)
    embeddings_lr = resolved_config.get(
        "embeddings_learning_rate",
        transformer.embeddings_learning_rate or transformer.learning_rate,
    )
    head_lr = resolved_config.get(
        "head_learning_rate",
        transformer.head_learning_rate or transformer.learning_rate,
    )
    s3_path = f"experiments/{name}"
    row: dict[str, Any] = {
        "experiment_name": name,
        "created_at": created_at,
        "model": Path(transformer.pretrained_model_path).name,
        "epochs": transformer.max_epochs,
        "macro_pr_auc_human": "" if macro_pr_auc is None else macro_pr_auc,
        "train_data": _training_data_label(config),
        "backbone_lr": backbone_lr,
        "embeddings_lr": embeddings_lr,
        "train_new_token_embeddings_only": (
            transformer.train_new_token_embeddings_only
        ),
        "head_lr": head_lr,
        "train_last_n_layers": (
            ""
            if transformer.train_last_n_layers is None
            else transformer.train_last_n_layers
        ),
        "layerwise_lr_decay": transformer.layerwise_lr_decay,
        "scheduler": transformer.lr_scheduler_type,
        "head": _head_description(config),
        "train_rows": splits.train_matches.height,
        "validation_rows": splits.validation_matches.height,
        "validation_pairs_hash": split_hash,
        "s3_path": s3_path,
    }
    record = {
        "schema_version": 1,
        **row,
        "predictor": artifacts.predictor,
        "metrics": metrics,
        "split": {
            "seed": split_settings.seed,
            "validation_fraction": split_settings.validation_fraction,
            "stacking_train_fraction": getattr(
                split_settings,
                "stacking_train_fraction",
                None,
            ),
            "leakage_scope": split_settings.leakage_scope,
            "candidate_splits": split_settings.candidate_splits,
            "train_rows": splits.train_matches.height,
            "stacking_train_rows": (
                0 if stacking_matches is None else stacking_matches.height
            ),
            "validation_rows": splits.validation_matches.height,
            "validation_pairs_hash": split_hash,
        },
        "sample_weighting": _sample_weighting_summary(splits.train_matches),
    }
    record_path = experiment_dir / "experiment.json"
    record_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return record_path


def _load_experiment_row(record_path: Path) -> dict[str, Any]:
    value = json.loads(record_path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"experiment record must contain an object: {record_path}")
    missing = [
        column
        for column in _REQUIRED_EXPERIMENT_REGISTRY_COLUMNS
        if column not in value
    ]
    if missing:
        raise ValueError(
            f"experiment record is missing registry fields {missing}: {record_path}"
        )
    name = validate_experiment_name(str(value["experiment_name"]))
    if record_path.parent.name != name:
        raise ValueError(
            "experiment directory does not match experiment_name: "
            f"{record_path.parent.name!r} != {name!r}"
        )
    row = {column: value.get(column, "") for column in EXPERIMENT_REGISTRY_COLUMNS}
    if row["registry_schema_version"] == "":
        row["registry_schema_version"] = value.get("schema_version", "")
    return row


def rebuild_experiment_registry(
    experiments_root: str | Path,
    registry_path: str | Path | None = None,
) -> tuple[Path, int]:
    """Atomically rebuild experiments.csv from independent experiment records."""
    root = Path(experiments_root).expanduser().resolve()
    record_paths = sorted(root.glob("*/experiment.json"))
    if not record_paths:
        raise ValueError(f"no experiment.json records found under {root}")

    rows: list[dict[str, Any]] = []
    seen_names: set[str] = set()
    for record_path in record_paths:
        row = _load_experiment_row(record_path)
        name = str(row["experiment_name"])
        if name in seen_names:
            raise ValueError(f"duplicate experiment_name in registry input: {name}")
        seen_names.add(name)
        rows.append(row)
    rows.sort(key=lambda row: (str(row["created_at"]), str(row["experiment_name"])))

    target = (
        Path(registry_path).expanduser().resolve()
        if registry_path is not None
        else root / "experiments.csv"
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            writer = csv.DictWriter(output, fieldnames=EXPERIMENT_REGISTRY_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        temporary_path.replace(target)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return target, len(rows)


__all__ = [
    "EXPERIMENT_REGISTRY_COLUMNS",
    "configure_experiment",
    "rebuild_experiment_registry",
    "save_experiment_record",
    "validate_experiment_name",
    "validation_pairs_hash",
]
