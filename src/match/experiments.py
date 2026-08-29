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
_LEGACY_EXPERIMENT_REGISTRY_COLUMNS = (
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
_PERFORMANCE_REGISTRY_COLUMNS = (
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
)
_INTERMEDIATE_SPECIAL_TOKEN_REGISTRY_COLUMNS = (
    "special_token_initialization",
    "special_token_adaptation_mode",
    "special_token_adaptation_steps",
    "special_token_adaptation_embeddings_lr",
    "special_token_adaptation_full_model_backbone_lr",
)
_SPECIAL_TOKEN_REGISTRY_COLUMNS = (
    "warmup_ratio",
    "special_token_initialization",
    "special_token_adaptation_mode",
    "special_token_adaptation_max_optimizer_steps",
    "special_token_adaptation_actual_optimizer_steps",
    "special_token_adaptation_embeddings_lr",
    "special_token_adaptation_full_model_backbone_lr",
    "special_token_adaptation_seconds",
)
_PRE_SPECIAL_TOKEN_REGISTRY_COLUMNS = (
    *_LEGACY_EXPERIMENT_REGISTRY_COLUMNS,
    *_PERFORMANCE_REGISTRY_COLUMNS,
)
EXPERIMENT_REGISTRY_COLUMNS = (
    *_PRE_SPECIAL_TOKEN_REGISTRY_COLUMNS,
    *_SPECIAL_TOKEN_REGISTRY_COLUMNS,
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
) -> tuple[AppConfig, Path, Path]:
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
    return configured, experiment_dir, root / "experiments.csv"


def validation_pairs_hash(matches: pl.DataFrame) -> str:
    """Return a row- and pair-direction-independent validation split hash."""
    columns = [name for name in ("id1", "id2", "target") if name in matches.columns]
    if columns != ["id1", "id2", "target"]:
        raise ValueError("validation matches must contain id1, id2 and target")
    left = pl.col("id1").cast(pl.String).fill_null("<null>")
    right = pl.col("id2").cast(pl.String).fill_null("<null>")
    canonical = (
        matches.select(
            pl.min_horizontal(left, right).alias("id1"),
            pl.max_horizontal(left, right).alias("id2"),
            pl.col("target").cast(pl.String).fill_null("<null>"),
        )
        .sort(columns)
        .write_csv()
        .encode("utf-8")
    )
    return hashlib.sha256(canonical).hexdigest()


def _training_metadata(config: AppConfig) -> dict[str, Any]:
    path = (
        config.model_description.transformer.artifact_dir
        / "training_metadata.json"
    )
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _migrate_registry_schema(registry_path: Path) -> None:
    if not registry_path.is_file() or registry_path.stat().st_size == 0:
        return
    with registry_path.open("r", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source)
        header = reader.fieldnames or []
        rows = list(reader)
    if header == list(EXPERIMENT_REGISTRY_COLUMNS):
        return
    if tuple(header) not in {
        tuple(_LEGACY_EXPERIMENT_REGISTRY_COLUMNS),
        tuple(_PRE_SPECIAL_TOKEN_REGISTRY_COLUMNS),
        (
            *_PRE_SPECIAL_TOKEN_REGISTRY_COLUMNS,
            *_INTERMEDIATE_SPECIAL_TOKEN_REGISTRY_COLUMNS,
        ),
    }:
        raise ValueError(
            f"experiment registry has an unexpected schema: {registry_path}"
        )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=registry_path.parent,
            prefix=f".{registry_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as output:
            temporary_path = Path(output.name)
            writer = csv.DictWriter(output, fieldnames=EXPERIMENT_REGISTRY_COLUMNS)
            writer.writeheader()
            for legacy_row in rows:
                migrated_row = {
                    key: value
                    for key, value in legacy_row.items()
                    if key in EXPERIMENT_REGISTRY_COLUMNS
                }
                if "special_token_adaptation_steps" in legacy_row:
                    migrated_row[
                        "special_token_adaptation_max_optimizer_steps"
                    ] = legacy_row["special_token_adaptation_steps"]
                migrated_row["registry_schema_version"] = legacy_row.get(
                    "registry_schema_version", 1
                )
                writer.writerow(migrated_row)
        temporary_path.replace(registry_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _mean_history_value(
    history: list[dict[str, Any]],
    key: str,
) -> float | str:
    values = [
        float(record[key])
        for record in history
        if isinstance(record.get(key), (int, float))
    ]
    return "" if not values else sum(values) / len(values)


def _performance_summary(training_metadata: dict[str, Any]) -> dict[str, Any]:
    raw_history = training_metadata.get("performance_history")
    history = (
        [record for record in raw_history if isinstance(record, dict)]
        if isinstance(raw_history, list)
        else []
    )
    raw_resolved = training_metadata.get("resolved_config")
    resolved = raw_resolved if isinstance(raw_resolved, dict) else {}
    raw_runtime = training_metadata.get("runtime")
    runtime = raw_runtime if isinstance(raw_runtime, dict) else {}

    train_batch_size = resolved.get("train_batch_size", "")
    accumulation = resolved.get("gradient_accumulation_steps", "")
    world_size = runtime.get("world_size", "")
    effective_batch_size: int | str = ""
    if all(
        isinstance(value, int) and not isinstance(value, bool) and value > 0
        for value in (train_batch_size, accumulation, world_size)
    ):
        effective_batch_size = train_batch_size * accumulation * world_size

    train_seconds = [
        float(record["train_seconds"])
        for record in history
        if isinstance(record.get("train_seconds"), (int, float))
    ]
    peak_memory = [
        float(record["peak_cuda_memory_gib"])
        for record in history
        if isinstance(record.get("peak_cuda_memory_gib"), (int, float))
    ]
    completed_epochs = training_metadata.get("completed_epochs", "")
    if completed_epochs == "" and history:
        completed_epochs = len(history)

    return {
        "registry_schema_version": 3,
        "completed_epochs": completed_epochs,
        "performance_epochs_observed": len(history) if history else "",
        "train_batch_size": train_batch_size,
        "gradient_accumulation_steps": accumulation,
        "effective_batch_size": effective_batch_size,
        "max_length": resolved.get("max_length", ""),
        "avg_train_epoch_seconds": (
            "" if not train_seconds else sum(train_seconds) / len(train_seconds)
        ),
        "total_train_seconds": "" if not train_seconds else sum(train_seconds),
        "avg_examples_per_second": _mean_history_value(
            history,
            "examples_per_second",
        ),
        "avg_real_tokens_per_second": _mean_history_value(
            history,
            "real_tokens_per_second",
        ),
        "avg_padding_efficiency": _mean_history_value(
            history,
            "padding_efficiency",
        ),
        "avg_train_peak_cuda_memory_gib": (
            "" if not peak_memory else sum(peak_memory) / len(peak_memory)
        ),
        "max_train_peak_cuda_memory_gib": (
            "" if not peak_memory else max(peak_memory)
        ),
        "gpu_name": runtime.get("gpu_name", ""),
        "gpu_count": runtime.get("gpu_count", ""),
        "precision": runtime.get("precision", ""),
        "trainable_parameters": runtime.get("trainable_parameters", ""),
    }


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
    return config.data_model_description.mix_dataset


def _training_data_label(config: AppConfig) -> str:
    if config.training.data_model == "base_dataset":
        return "Human"
    source_names = (
        source.name
        for source in config.data_model_description.mix_dataset.sources
    )
    return " + ".join(
        name.upper() if name.lower() == "llm" else name.title()
        for name in source_names
    )


def save_experiment_record(
    config: AppConfig,
    artifacts: TrainingArtifacts,
    splits: "LoadedTrainingSplits",
    *,
    experiment_name: str,
    registry_path: Path,
) -> tuple[Path, Path]:
    """Persist experiment.json and append one successful run to the CSV registry."""
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
    special_token_adaptation = transformer.special_token_adaptation
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
    performance_summary = _performance_summary(training_metadata)
    raw_adaptation = training_metadata.get("special_token_adaptation")
    adaptation_runtime = (
        raw_adaptation if isinstance(raw_adaptation, dict) else {}
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
        **performance_summary,
        "warmup_ratio": transformer.warmup_ratio,
        "special_token_initialization": (
            transformer.special_token_initialization.method
        ),
        "special_token_adaptation_mode": (
            special_token_adaptation.mode
        ),
        "special_token_adaptation_max_optimizer_steps": (
            special_token_adaptation.max_optimizer_steps
        ),
        "special_token_adaptation_actual_optimizer_steps": (
            adaptation_runtime.get("actual_optimizer_steps", "")
        ),
        "special_token_adaptation_embeddings_lr": (
            ""
            if special_token_adaptation.embeddings_learning_rate is None
            else special_token_adaptation.embeddings_learning_rate
        ),
        "special_token_adaptation_full_model_backbone_lr": (
            ""
            if special_token_adaptation.full_model_backbone_learning_rate is None
            else special_token_adaptation.full_model_backbone_learning_rate
        ),
        "special_token_adaptation_seconds": adaptation_runtime.get(
            "seconds", ""
        ),
    }
    record = {
        "schema_version": 3,
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
    }
    record_path = experiment_dir / "experiment.json"
    record_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    _migrate_registry_schema(registry_path)
    write_header = not registry_path.is_file() or registry_path.stat().st_size == 0
    with registry_path.open("a", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=EXPERIMENT_REGISTRY_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    return record_path, registry_path


__all__ = [
    "EXPERIMENT_REGISTRY_COLUMNS",
    "configure_experiment",
    "save_experiment_record",
    "validate_experiment_name",
    "validation_pairs_hash",
]
