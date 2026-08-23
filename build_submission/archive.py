"""Build an evaluator archive from the typed pipeline configuration."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

from match.config import AppConfig
from match.models.artifacts import (
    TrainingArtifacts,
    build_solution_manifest,
)
from match.paths import PROJECT_ROOT

_REQUIRED_PROJECT_FILES = ("run.py", "metadata.json")
_VENDOR_WHEELS_SOURCE = PurePosixPath("build_submission/vendor_wheels")
_VENDOR_WHEELS_TARGET = PurePosixPath("vendor_wheels")
_PREPROCESSING_WHEEL_PATTERNS = (
    "pymorphy3-*.whl",
    "pymorphy3_dicts_ru-*.whl",
    "dawg2_python-*.whl",
    "setuptools-*.whl",
    "joblib-*.whl",
    "loguru-*.whl",
    "Pint-*.whl",
    "flexcache-*.whl",
    "flexparser-*.whl",
    "platformdirs-*.whl",
    "typing_extensions-*.whl",
    "colorama-*.whl",
    "win32_setctime-*.whl",
)
_SKIPPED_DIRECTORY_NAMES = {".cache", "__pycache__"}
_STORED_SUFFIXES = {
    ".bin",
    ".cbm",
    ".joblib",
    ".pt",
    ".safetensors",
    ".whl",
}


@dataclass(frozen=True, slots=True)
class SubmissionArchive:
    path: Path
    predictor: str
    file_count: int
    uncompressed_bytes: int


@dataclass(frozen=True, slots=True)
class _ArchiveInput:
    source: Path
    archive_path: PurePosixPath


def _selected_artifacts(config: AppConfig) -> TrainingArtifacts:
    predictor = config.inference.model
    if predictor == "transformer":
        return TrainingArtifacts(
            predictor=predictor,
            transformer_dir=config.inference.transformer_dir,
        )
    if predictor == "maxpooling":
        return TrainingArtifacts(
            predictor=predictor,
            maxpooling_path=config.inference.maxpooling_path,
        )
    if predictor == "fusion":
        return TrainingArtifacts(
            predictor=predictor,
            transformer_dir=config.inference.transformer_dir,
            maxpooling_path=config.inference.maxpooling_path,
            fusion_path=config.inference.fusion_path,
        )
    if predictor == "boosting":
        return TrainingArtifacts(
            predictor=predictor,
            boosting_dir=config.inference.boosting_dir,
        )
    if predictor == "cascade":
        return TrainingArtifacts(
            predictor=predictor,
            transformer_dir=config.inference.transformer_dir,
            boosting_dir=config.inference.boosting_dir,
        )
    raise ValueError(f"unsupported submission predictor: {predictor!r}")


def _archive_path(path: Path, *, project_root: Path) -> PurePosixPath:
    resolved = path.expanduser().resolve()
    try:
        relative = resolved.relative_to(project_root)
    except ValueError as error:
        raise ValueError(
            f"submission resource must be inside the project: {resolved}"
        ) from error
    return PurePosixPath(relative.as_posix())


def _required_resources(
    config: AppConfig,
    artifacts: TrainingArtifacts,
) -> tuple[Path, ...]:
    resources: list[Path] = []
    for path in (
        artifacts.transformer_dir,
        artifacts.maxpooling_path,
        artifacts.fusion_path,
        artifacts.boosting_dir,
    ):
        if path is not None:
            resources.append(path)
    if config.normalization.enabled:
        resources.extend(
            (
                config.normalization.synonyms_path,
                config.normalization.unique_attributes_path,
            )
        )
    if config.features.ner.enabled:
        if config.features.ner.model_dir is None:
            raise ValueError("enabled NER requires features.ner.model_dir")
        resources.append(config.features.ner.model_dir)
        if config.features.ner.cluster_centers_path is not None:
            resources.append(config.features.ner.cluster_centers_path)
    return tuple(resources)


def _validate_resource(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(
            "Submission artifact does not exist: "
            f"{path}. Train the selected model or fix pipeline.yaml."
        )
    if path.is_dir() and not any(path.iterdir()):
        raise ValueError(f"submission artifact directory is empty: {path}")
    if not path.is_file() and not path.is_dir():
        raise ValueError(f"unsupported submission artifact: {path}")


def _validate_transformer_artifact(directory: Path) -> None:
    """Reject raw language checkpoints before they reach the evaluator."""
    config_path = directory / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"trained Transformer config does not exist: {config_path}"
        )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"invalid Transformer config: {config_path}") from error
    if not isinstance(config, dict):
        raise ValueError(f"Transformer config must be a JSON object: {config_path}")

    use_field_tokens = config.get("match_use_field_tokens")
    max_length = config.get("match_max_length")
    architectures = config.get("architectures")
    is_sequence_classifier = isinstance(architectures, list) and any(
        "SequenceClassifier" in str(name) or "SequenceClassification" in str(name)
        for name in architectures
    )
    is_pooling_classifier = (
        config.get("model_type") == "match_pooling_sequence_classifier"
        or config.get("match_head_type") == "pooling"
    )
    if (
        not isinstance(use_field_tokens, bool)
        or not isinstance(max_length, int)
        or max_length < 1
        or not (is_sequence_classifier or is_pooling_classifier)
    ):
        raise ValueError(
            "submission Transformer is not a trained Twin2Attr classifier: "
            f"{directory}. Package the output of `run.py train`, not a raw "
            "pretrained language model."
        )

    weight_patterns = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    if not any((directory / name).is_file() for name in weight_patterns):
        raise FileNotFoundError(
            f"trained Transformer weights are missing in {directory}"
        )

    if use_field_tokens:
        from transformers import AutoTokenizer

        from match.pair_encoding import _require_pair_special_tokens

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                directory,
                local_files_only=True,
            )
            _require_pair_special_tokens(tokenizer)
        except (OSError, ValueError) as error:
            raise ValueError(
                "trained Transformer tokenizer is incompatible with its "
                f"pair-encoding metadata: {directory}"
            ) from error


def _skip_file(path: Path, *, source_root: Path) -> bool:
    relative = path.relative_to(source_root)
    if any(part in _SKIPPED_DIRECTORY_NAMES for part in relative.parts):
        return True
    if any(part.startswith("checkpoint-") for part in relative.parts[:-1]):
        return True
    return path.suffix in {".pyc", ".pyo"}


def _files_for_path(
    source: Path,
    archive_path: PurePosixPath,
) -> Iterable[_ArchiveInput]:
    if source.is_file():
        yield _ArchiveInput(source, archive_path)
        return
    for candidate in sorted(source.rglob("*")):
        if not candidate.is_file() or _skip_file(candidate, source_root=source):
            continue
        relative = candidate.relative_to(source)
        yield _ArchiveInput(
            candidate,
            archive_path / PurePosixPath(relative.as_posix()),
        )


def _collect_inputs(
    resources: Iterable[Path],
    *,
    project_root: Path,
    include_catboost: bool,
    include_preprocessing_runtime: bool,
) -> tuple[_ArchiveInput, ...]:
    sources = [project_root / name for name in _REQUIRED_PROJECT_FILES]
    sources.append(project_root / "src" / "match")
    sources.extend(resources)
    entries: dict[PurePosixPath, _ArchiveInput] = {}
    for source in sources:
        _validate_resource(source)
        archive_path = _archive_path(source, project_root=project_root)
        for entry in _files_for_path(source, archive_path):
            previous = entries.get(entry.archive_path)
            if previous is not None and previous.source != entry.source:
                raise ValueError(
                    f"duplicate archive path: {entry.archive_path}"
                )
            entries[entry.archive_path] = entry
    wheels_source = project_root / Path(_VENDOR_WHEELS_SOURCE)
    _validate_polars_wheels(wheels_source)
    wheel_patterns = ["polars-*.whl", "polars_runtime_32-*.whl"]
    if include_preprocessing_runtime:
        _validate_preprocessing_wheels(wheels_source)
        wheel_patterns.extend(_PREPROCESSING_WHEEL_PATTERNS)
    if include_catboost:
        _validate_catboost_wheels(wheels_source)
        wheel_patterns.append("catboost-*.whl")
    for pattern in wheel_patterns:
        for wheel in sorted(wheels_source.glob(pattern)):
            entry = _ArchiveInput(wheel, _VENDOR_WHEELS_TARGET / wheel.name)
            entries[entry.archive_path] = entry
    return tuple(entries[name] for name in sorted(entries, key=str))


def _validate_polars_wheels(directory: Path) -> None:
    if not directory.is_dir():
        raise FileNotFoundError(
            f"bundled wheels directory does not exist: {directory}"
        )
    required_patterns = ("polars-*.whl", "polars_runtime_32-*.whl")
    missing = [
        pattern
        for pattern in required_patterns
        if not any(directory.glob(pattern))
    ]
    if missing:
        raise FileNotFoundError(
            f"bundled Polars wheels are missing in {directory}: {missing}"
        )


def _validate_catboost_wheels(directory: Path) -> None:
    if not any(directory.glob("catboost-*.whl")):
        raise FileNotFoundError(
            f"bundled CatBoost wheel is missing in {directory}"
        )


def _validate_preprocessing_wheels(directory: Path) -> None:
    missing = [
        pattern
        for pattern in _PREPROCESSING_WHEEL_PATTERNS
        if not any(directory.glob(pattern))
    ]
    if missing:
        raise FileNotFoundError(
            f"bundled preprocessing wheels are missing in {directory}: {missing}"
        )


def _compression(path: Path) -> int:
    return ZIP_STORED if path.suffix.lower() in _STORED_SUFFIXES else ZIP_DEFLATED


def build_submission_archive(
    config: AppConfig,
    output_path: str | Path | None = None,
    *,
    project_root: Path = PROJECT_ROOT,
) -> SubmissionArchive:
    """Validate and package the configured predictor and enabled features."""
    root = project_root.expanduser().resolve()
    artifacts = _selected_artifacts(config)
    resources = _required_resources(config, artifacts)
    for resource in resources:
        _validate_resource(resource)
    if artifacts.transformer_dir is not None:
        _validate_transformer_artifact(artifacts.transformer_dir)

    solution = build_solution_manifest(
        config,
        artifacts,
        map_path=lambda path: str(_archive_path(path, project_root=root)),
    )
    inputs = _collect_inputs(
        resources,
        project_root=root,
        include_catboost=artifacts.predictor in {"boosting", "cascade"},
        include_preprocessing_runtime=(
            config.normalization.enabled or config.features.physical.enabled
        ),
    )
    target = Path(output_path or config.submission.output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.unlink(missing_ok=True)
    try:
        with ZipFile(temporary, "w", allowZip64=True) as archive:
            archive.writestr(
                "solution.json",
                json.dumps(solution, ensure_ascii=False, indent=2) + "\n",
                compress_type=ZIP_DEFLATED,
            )
            for entry in inputs:
                archive.write(
                    entry.source,
                    str(entry.archive_path),
                    compress_type=_compression(entry.source),
                )
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise

    return SubmissionArchive(
        path=target,
        predictor=artifacts.predictor,
        file_count=len(inputs) + 1,
        uncompressed_bytes=sum(entry.source.stat().st_size for entry in inputs)
        + len(json.dumps(solution, ensure_ascii=False).encode("utf-8")),
    )


__all__ = ["SubmissionArchive", "build_submission_archive"]
