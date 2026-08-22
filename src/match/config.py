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
    batch_size: int


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


@dataclass(frozen=True, slots=True)
class FeatureSettings:
    ner: NerSettings


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
class AppConfig:
    training: TrainingSettings
    paths: PathSettings
    normalization: NormalizationSettings
    split: SplitSettings
    pair_encoding: PairEncodingSettings
    models_parameters: ModelsParametersSettings
    artifacts: ArtifactSettings
    features: FeatureSettings
    runtime: RuntimeSettings
    logging: LoggingSettings


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
    runtime = _section(resolved, "runtime")
    logging = _section(resolved, "logging")

    return AppConfig(
        training=TrainingSettings(model=str(_required(training, "model"))),
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
    "LoggingSettings",
    "MaxPoolingParameters",
    "ModelsParametersSettings",
    "NerSettings",
    "NormalizationSettings",
    "PairEncodingSettings",
    "PathSettings",
    "RuntimeSettings",
    "SplitSettings",
    "TrainingSettings",
    "TransformerParameters",
    "load_app_config",
    "load_app_config_file",
    "save_app_config",
]
