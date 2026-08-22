"""Typed application configuration and the OmegaConf boundary adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from .paths import resolve_project_path


@dataclass(frozen=True, slots=True)
class TrainingSettings:
    model: str

    def __post_init__(self) -> None:
        if self.model not in {"transformer", "maxpooling", "fusion"}:
            raise ValueError(
                "training.model must be one of: transformer, maxpooling, fusion"
            )


@dataclass(frozen=True, slots=True)
class InferenceSettings:
    model: str

    def __post_init__(self) -> None:
        if self.model not in {"transformer", "maxpooling", "fusion"}:
            raise ValueError(
                "inference.model must be one of: transformer, maxpooling, fusion"
            )


@dataclass(frozen=True, slots=True)
class PathSettings:
    items: Path
    train_matches: Path
    validation_matches: Path
    inspect_matches: Path | None


@dataclass(frozen=True, slots=True)
class NormalizationSettings:
    enabled: bool
    source_column: str
    output_column: str
    synonyms_path: Path
    unique_attributes_path: Path
    n_jobs: int
    chunk_size: int


@dataclass(frozen=True, slots=True)
class SplitSettings:
    mode: str
    validation_fraction: float
    leakage_scope: str
    seed: int
    candidate_splits: int
    train_output_path: Path
    validation_output_path: Path


@dataclass(frozen=True, slots=True)
class PairEncodingSettings:
    use_field_tokens: bool
    max_attribute_value_tokens: int | None
    max_length: int | None
    quantile: float
    sample_size: int
    hard_cap: int


@dataclass(frozen=True, slots=True)
class TransformerParameters:
    pretrained_model_path: str
    max_epochs: int
    hpo_trials: int
    learning_rate: float
    weight_decay: float
    hpo_learning_rate_min: float
    hpo_learning_rate_max: float
    hpo_weight_decay_min: float
    hpo_weight_decay_max: float
    train_batch_size: int
    eval_batch_size: int
    gradient_accumulation_steps: int
    warmup_ratio: float
    max_grad_norm: float
    early_stopping_patience: int
    auto_find_batch_size: bool
    batch_size: int

    def __post_init__(self) -> None:
        if not self.pretrained_model_path.strip():
            raise ValueError("transformer.pretrained_model_path must not be empty")
        if self.max_epochs < 1 or self.hpo_trials < 1:
            raise ValueError("transformer epoch and HPO counts must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("transformer optimizer parameters are invalid")
        if not 0.0 < self.hpo_learning_rate_min < self.hpo_learning_rate_max:
            raise ValueError("transformer HPO learning-rate bounds are invalid")
        if not 0.0 <= self.hpo_weight_decay_min < self.hpo_weight_decay_max:
            raise ValueError("transformer HPO weight-decay bounds are invalid")
        if min(
            self.train_batch_size,
            self.eval_batch_size,
            self.gradient_accumulation_steps,
            self.early_stopping_patience,
            self.batch_size,
        ) < 1:
            raise ValueError("transformer batch and patience values must be positive")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("transformer.warmup_ratio must be in [0, 1)")
        if self.max_grad_norm <= 0.0:
            raise ValueError("transformer.max_grad_norm must be positive")


@dataclass(frozen=True, slots=True)
class MaxPoolingParameters:
    vector_size: int
    window: int
    min_count: int
    workers: int
    fasttext_epochs: int
    classifier_epochs: int
    batch_size: int
    validation_fraction: float
    patience: int
    dropout: float
    learning_rate: float
    weight_decay: float


@dataclass(frozen=True, slots=True)
class FusionParameters:
    embedding_batch_size: int
    hidden_dim: int
    dropout: float
    batch_size: int
    max_epochs: int
    patience: int
    learning_rate: float
    weight_decay: float


@dataclass(frozen=True, slots=True)
class ModelsParametersSettings:
    transformer: TransformerParameters
    maxpooling: MaxPoolingParameters
    fusion: FusionParameters


@dataclass(frozen=True, slots=True)
class ArtifactSettings:
    transformer_dir: Path
    maxpooling_path: Path
    fusion_path: Path
    resolved_config_path: Path
    solution_path: Path


@dataclass(frozen=True, slots=True)
class NerSettings:
    enabled: bool
    provider: str | None
    model_dir: Path | None
    cluster_centers_path: Path | None
    source_column: str
    output_column: str
    enriched_column: str
    merge_policy: str
    batch_size: int
    max_length: int
    use_amp: bool
    semantic_cleanup: bool

    def __post_init__(self) -> None:
        if self.provider not in {None, "word_ner"}:
            raise ValueError("features.ner.provider must be 'word_ner' or null")
        if self.merge_policy != "missing_only":
            raise ValueError("features.ner.merge_policy must be 'missing_only'")
        if not self.source_column.strip():
            raise ValueError("features.ner.source_column must not be empty")
        if not self.output_column.strip() or not self.enriched_column.strip():
            raise ValueError("features.ner output columns must not be empty")
        if self.output_column == self.enriched_column:
            raise ValueError("NER output and enriched columns must be different")
        if self.batch_size < 1 or self.max_length < 8:
            raise ValueError("NER batch_size must be positive and max_length at least 8")
        if self.enabled:
            if self.provider != "word_ner":
                raise ValueError("enabled NER requires provider='word_ner'")
            if self.model_dir is None:
                raise ValueError("enabled NER requires features.ner.model_dir")
            if self.semantic_cleanup and self.cluster_centers_path is None:
                raise ValueError(
                    "NER semantic cleanup requires cluster_centers_path"
                )


@dataclass(frozen=True, slots=True)
class PhysicalFeatureSettings:
    enabled: bool
    source_column: str
    output_column: str
    enriched_column: str
    merge_policy: str
    normalize_units: bool
    n_jobs: int
    chunk_size: int

    def __post_init__(self) -> None:
        if not self.source_column.strip():
            raise ValueError("features.physical.source_column must not be empty")
        if not self.output_column.strip() or not self.enriched_column.strip():
            raise ValueError("features.physical output columns must not be empty")
        if self.output_column == self.enriched_column:
            raise ValueError("physical output and enriched columns must be different")
        if self.merge_policy != "missing_only":
            raise ValueError(
                "features.physical.merge_policy must be 'missing_only'"
            )
        if self.n_jobs == 0 or self.chunk_size < 1:
            raise ValueError(
                "features.physical n_jobs must not be zero and chunk_size positive"
            )


@dataclass(frozen=True, slots=True)
class FeatureSettings:
    ner: NerSettings
    physical: PhysicalFeatureSettings


@dataclass(frozen=True, slots=True)
class RuntimeSettings:
    device: str | None
    seed: int


@dataclass(frozen=True, slots=True)
class LoggingSettings:
    level: str
    file: Path | None
    rotation: str


@dataclass(frozen=True, slots=True)
class SubmissionSettings:
    output_path: Path


@dataclass(frozen=True, slots=True)
class AppConfig:
    training: TrainingSettings
    inference: InferenceSettings
    paths: PathSettings
    normalization: NormalizationSettings
    split: SplitSettings
    pair_encoding: PairEncodingSettings
    models_parameters: ModelsParametersSettings
    artifacts: ArtifactSettings
    features: FeatureSettings
    runtime: RuntimeSettings
    logging: LoggingSettings
    submission: SubmissionSettings


ConfigSource = AppConfig | DictConfig | Mapping[str, Any]


def _section(values: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    section = values.get(name)
    if not isinstance(section, Mapping):
        raise ValueError(f"config section {name!r} must be a mapping")
    return section


def _required(values: Mapping[str, Any], name: str) -> Any:
    value = values.get(name)
    if value is None:
        raise ValueError(f"config value {name!r} is required")
    return value


def _path(value: Any, name: str) -> Path:
    if value is None or not str(value).strip():
        raise ValueError(f"config value {name!r} must contain a path")
    return resolve_project_path(str(value))


def _optional_path(value: Any) -> Path | None:
    if value is None or not str(value).strip():
        return None
    return resolve_project_path(str(value))


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    raise ValueError(f"config value {name!r} must be a boolean")


def load_app_config(config: ConfigSource) -> AppConfig:
    """Resolve OmegaConf values once and return immutable typed settings."""
    if isinstance(config, AppConfig):
        return config
    omega = config if isinstance(config, DictConfig) else OmegaConf.create(config)
    resolved = OmegaConf.to_container(omega, resolve=True)
    if not isinstance(resolved, Mapping):
        raise ValueError("application config must be a mapping")

    training = _section(resolved, "training")
    inference_value = resolved.get("inference", training)
    if not isinstance(inference_value, Mapping):
        raise ValueError("config section 'inference' must be a mapping")
    inference = inference_value
    paths = _section(resolved, "paths")
    normalization = _section(resolved, "normalization")
    split = _section(resolved, "split")
    encoding = _section(resolved, "pair_encoding")
    models_parameters = _section(resolved, "models_parameters")
    transformer = _section(models_parameters, "transformer")
    maxpooling = _section(models_parameters, "maxpooling")
    fusion = _section(models_parameters, "fusion")
    artifacts = _section(resolved, "artifacts")
    features = _section(resolved, "features")
    ner = _section(features, "ner")
    physical = _section(features, "physical")
    runtime = _section(resolved, "runtime")
    logging = _section(resolved, "logging")
    submission_value = resolved.get("submission", {})
    if not isinstance(submission_value, Mapping):
        raise ValueError("config section 'submission' must be a mapping")
    submission = submission_value

    return AppConfig(
        training=TrainingSettings(model=str(_required(training, "model"))),
        inference=InferenceSettings(model=str(_required(inference, "model"))),
        paths=PathSettings(
            items=_path(_required(paths, "items"), "paths.items"),
            train_matches=_path(
                _required(paths, "train_matches"),
                "paths.train_matches",
            ),
            validation_matches=_path(
                _required(paths, "validation_matches"),
                "paths.validation_matches",
            ),
            inspect_matches=_optional_path(paths.get("inspect_matches")),
        ),
        normalization=NormalizationSettings(
            enabled=_bool(
                normalization.get("enabled", False),
                "normalization.enabled",
            ),
            source_column=str(_required(normalization, "source_column")),
            output_column=str(_required(normalization, "output_column")),
            synonyms_path=_path(
                _required(normalization, "synonyms_path"),
                "normalization.synonyms_path",
            ),
            unique_attributes_path=_path(
                _required(normalization, "unique_attributes_path"),
                "normalization.unique_attributes_path",
            ),
            n_jobs=int(_required(normalization, "n_jobs")),
            chunk_size=int(_required(normalization, "chunk_size")),
        ),
        split=SplitSettings(
            mode=str(_required(split, "mode")),
            validation_fraction=float(_required(split, "validation_fraction")),
            leakage_scope=str(_required(split, "leakage_scope")),
            seed=int(_required(split, "seed")),
            candidate_splits=int(_required(split, "candidate_splits")),
            train_output_path=_path(
                _required(split, "train_output_path"),
                "split.train_output_path",
            ),
            validation_output_path=_path(
                _required(split, "validation_output_path"),
                "split.validation_output_path",
            ),
        ),
        pair_encoding=PairEncodingSettings(
            use_field_tokens=_bool(
                _required(encoding, "use_field_tokens"),
                "pair_encoding.use_field_tokens",
            ),
            max_attribute_value_tokens=_optional_int(
                encoding.get("max_attribute_value_tokens")
            ),
            max_length=_optional_int(encoding.get("max_length")),
            quantile=float(_required(encoding, "quantile")),
            sample_size=int(_required(encoding, "sample_size")),
            hard_cap=int(_required(encoding, "hard_cap")),
        ),
        models_parameters=ModelsParametersSettings(
            transformer=TransformerParameters(
                pretrained_model_path=str(
                    _required(transformer, "pretrained_model_path")
                ),
                max_epochs=int(_required(transformer, "max_epochs")),
                hpo_trials=int(_required(transformer, "hpo_trials")),
                learning_rate=float(_required(transformer, "learning_rate")),
                weight_decay=float(_required(transformer, "weight_decay")),
                hpo_learning_rate_min=float(
                    _required(transformer, "hpo_learning_rate_min")
                ),
                hpo_learning_rate_max=float(
                    _required(transformer, "hpo_learning_rate_max")
                ),
                hpo_weight_decay_min=float(
                    _required(transformer, "hpo_weight_decay_min")
                ),
                hpo_weight_decay_max=float(
                    _required(transformer, "hpo_weight_decay_max")
                ),
                train_batch_size=int(
                    _required(transformer, "train_batch_size")
                ),
                eval_batch_size=int(
                    _required(transformer, "eval_batch_size")
                ),
                gradient_accumulation_steps=int(
                    _required(transformer, "gradient_accumulation_steps")
                ),
                warmup_ratio=float(_required(transformer, "warmup_ratio")),
                max_grad_norm=float(_required(transformer, "max_grad_norm")),
                early_stopping_patience=int(
                    _required(transformer, "early_stopping_patience")
                ),
                auto_find_batch_size=_bool(
                    _required(transformer, "auto_find_batch_size"),
                    "models_parameters.transformer.auto_find_batch_size",
                ),
                batch_size=int(_required(transformer, "batch_size")),
            ),
            maxpooling=MaxPoolingParameters(
                vector_size=int(_required(maxpooling, "vector_size")),
                window=int(_required(maxpooling, "window")),
                min_count=int(_required(maxpooling, "min_count")),
                workers=int(_required(maxpooling, "workers")),
                fasttext_epochs=int(_required(maxpooling, "fasttext_epochs")),
                classifier_epochs=int(
                    _required(maxpooling, "classifier_epochs")
                ),
                batch_size=int(_required(maxpooling, "batch_size")),
                validation_fraction=float(
                    _required(maxpooling, "validation_fraction")
                ),
                patience=int(_required(maxpooling, "patience")),
                dropout=float(_required(maxpooling, "dropout")),
                learning_rate=float(_required(maxpooling, "learning_rate")),
                weight_decay=float(_required(maxpooling, "weight_decay")),
            ),
            fusion=FusionParameters(
                embedding_batch_size=int(
                    _required(fusion, "embedding_batch_size")
                ),
                hidden_dim=int(_required(fusion, "hidden_dim")),
                dropout=float(_required(fusion, "dropout")),
                batch_size=int(_required(fusion, "batch_size")),
                max_epochs=int(_required(fusion, "max_epochs")),
                patience=int(_required(fusion, "patience")),
                learning_rate=float(_required(fusion, "learning_rate")),
                weight_decay=float(_required(fusion, "weight_decay")),
            ),
        ),
        artifacts=ArtifactSettings(
            transformer_dir=_path(
                _required(artifacts, "transformer_dir"),
                "artifacts.transformer_dir",
            ),
            maxpooling_path=_path(
                _required(artifacts, "maxpooling_path"),
                "artifacts.maxpooling_path",
            ),
            fusion_path=_path(
                _required(artifacts, "fusion_path"),
                "artifacts.fusion_path",
            ),
            resolved_config_path=_path(
                _required(artifacts, "resolved_config_path"),
                "artifacts.resolved_config_path",
            ),
            solution_path=_path(
                _required(artifacts, "solution_path"),
                "artifacts.solution_path",
            ),
        ),
        features=FeatureSettings(
            ner=NerSettings(
                enabled=_bool(
                    ner.get("enabled", False),
                    "features.ner.enabled",
                ),
                provider=None if ner.get("provider") is None else str(ner["provider"]),
                model_dir=_optional_path(ner.get("model_dir")),
                cluster_centers_path=_optional_path(
                    ner.get("cluster_centers_path")
                ),
                source_column=str(ner.get("source_column", "name")),
                output_column=str(ner.get("output_column", "ner_attributes")),
                enriched_column=str(
                    ner.get("enriched_column", "enriched_attributes")
                ),
                merge_policy=str(ner.get("merge_policy", "missing_only")),
                batch_size=int(ner.get("batch_size", 512)),
                max_length=int(ner.get("max_length", 100)),
                use_amp=_bool(
                    ner.get("use_amp", True),
                    "features.ner.use_amp",
                ),
                semantic_cleanup=_bool(
                    ner.get("semantic_cleanup", True),
                    "features.ner.semantic_cleanup",
                ),
            ),
            physical=PhysicalFeatureSettings(
                enabled=_bool(
                    physical.get("enabled", False),
                    "features.physical.enabled",
                ),
                source_column=str(physical.get("source_column", "name")),
                output_column=str(
                    physical.get("output_column", "physical_attributes")
                ),
                enriched_column=str(
                    physical.get("enriched_column", "feature_attributes")
                ),
                merge_policy=str(
                    physical.get("merge_policy", "missing_only")
                ),
                normalize_units=_bool(
                    physical.get("normalize_units", True),
                    "features.physical.normalize_units",
                ),
                n_jobs=int(physical.get("n_jobs", 1)),
                chunk_size=int(physical.get("chunk_size", 10_000)),
            ),
        ),
        runtime=RuntimeSettings(
            device=None if runtime.get("device") is None else str(runtime["device"]),
            seed=int(_required(runtime, "seed")),
        ),
        logging=LoggingSettings(
            level=str(_required(logging, "level")),
            file=_optional_path(logging.get("file")),
            rotation=str(_required(logging, "rotation")),
        ),
        submission=SubmissionSettings(
            output_path=_path(
                submission.get(
                    "output_path",
                    "dist/twin2attr_submission.zip",
                ),
                "submission.output_path",
            ),
        ),
    )


def load_app_config_file(
    path: str | Path,
    overrides: Sequence[str] = (),
) -> AppConfig:
    """Load YAML plus dot-list overrides at the application boundary."""
    config_path = resolve_project_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Pipeline config does not exist: {config_path}")
    config = OmegaConf.load(config_path)
    clean_overrides = [value for value in overrides if value != "--"]
    if clean_overrides:
        config = OmegaConf.merge(
            config,
            OmegaConf.from_dotlist(clean_overrides),
        )
    return load_app_config(config)


def _serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return {key: _serializable(item) for key, item in asdict(value).items()}
    if isinstance(value, Mapping):
        return {key: _serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    return value


def save_app_config(config: AppConfig, path: Path) -> None:
    """Persist the resolved typed configuration as YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(OmegaConf.create(_serializable(config)), path)


__all__ = [
    "AppConfig",
    "ArtifactSettings",
    "ConfigSource",
    "FeatureSettings",
    "FusionParameters",
    "InferenceSettings",
    "LoggingSettings",
    "MaxPoolingParameters",
    "ModelsParametersSettings",
    "NerSettings",
    "NormalizationSettings",
    "PairEncodingSettings",
    "PathSettings",
    "PhysicalFeatureSettings",
    "RuntimeSettings",
    "SplitSettings",
    "SubmissionSettings",
    "TrainingSettings",
    "TransformerParameters",
    "load_app_config",
    "load_app_config_file",
    "save_app_config",
]
