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

_REQUIRED_PROJECT_FILES = ("run.py",)
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
    "orjson-*.whl",
)
_ONNX_RUNTIME_WHEEL_PATTERNS = (
    "onnxruntime_gpu-*.whl",
    "coloredlogs-*.whl",
    "flatbuffers-*.whl",
    "humanfriendly-*.whl",
)
_TENSORRT_WHEEL_PATTERNS = (
    "tensorrt_cu12_bindings-*.whl",
    "zstandard-*.whl",
)
_TENSORRT_RUNTIME_SOURCE = PurePosixPath("build_submission/tensorrt_runtime")
_TENSORRT_RUNTIME_TARGET = PurePosixPath("tensorrt_runtime")
_TENSORRT_RUNTIME_PATTERN = "tensorrt-runtime-*.tar.zst"
_SKIPPED_DIRECTORY_NAMES = {".cache", "__pycache__"}
_STORED_SUFFIXES = {
    ".bin",
    ".cbm",
    ".joblib",
    ".pt",
    ".safetensors",
    ".whl",
}
_TRANSFORMER_WEIGHT_PATTERNS = (
    "model.safetensors",
    "model.safetensors.index.json",
    "model-*.safetensors",
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
    "pytorch_model-*.bin",
)


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
            transformer_dir=config.model_description.transformer.artifact_dir,
        )
    if predictor == "maxpooling":
        return TrainingArtifacts(
            predictor=predictor,
            maxpooling_path=config.model_description.maxpooling.artifact_path,
        )
    if predictor == "fusion":
        return TrainingArtifacts(
            predictor=predictor,
            transformer_dir=config.model_description.transformer.artifact_dir,
            maxpooling_path=config.model_description.maxpooling.artifact_path,
            fusion_path=config.model_description.fusion.artifact_path,
        )
    if predictor == "boosting":
        return TrainingArtifacts(
            predictor=predictor,
            boosting_dir=config.model_description.boosting.artifact_dir,
        )
    if predictor == "cascade":
        return TrainingArtifacts(
            predictor=predictor,
            transformer_dir=config.model_description.transformer.artifact_dir,
            boosting_dir=config.model_description.boosting.artifact_dir,
        )
    if predictor == "stacking":
        return TrainingArtifacts(
            predictor=predictor,
            transformer_dir=config.model_description.transformer.artifact_dir,
            stacking_dir=config.model_description.stacking.artifact_dir,
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
        artifacts.stacking_dir,
    ):
        if path is not None:
            resources.append(path)
    normalization = config.features.normalization
    if normalization.enabled:
        resources.extend(
            (
                normalization.synonyms_path,
                normalization.unique_attributes_path,
            )
        )
    if config.features.ner.enabled:
        if config.features.ner.model_dir is None:
            raise ValueError("enabled NER requires features.ner.model_dir")
        resources.append(config.features.ner.model_dir)
        if config.features.ner.cluster_centers_path is not None:
            resources.append(config.features.ner.cluster_centers_path)
    if config.inference.data_postprocessing_model == "attribute_sort":
        resources.append(
            config.data_postprocessing_models.attribute_sort.priorities_path
        )
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


def _validate_transformer_artifact(
    directory: Path,
    *,
    require_pytorch_weights: bool = True,
) -> None:
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
    is_causal_zero_shot_reranker = (
        not require_pytorch_weights
        and config.get("match_profile") in {"qwen3_reranker", "mxbai_reranker"}
        and config.get("match_head_type") == "native"
        and config.get("match_initialized_only") is True
        and config.get("match_num_logits") == 1
        and isinstance(config.get("match_no_token_id"), int)
        and isinstance(config.get("match_yes_token_id"), int)
        and config.get("match_no_token_id") != config.get("match_yes_token_id")
    )
    pytorch_quantization = config.get("match_pytorch_quantization")
    is_qwen3_pytorch_int8 = (
        require_pytorch_weights
        and config.get("match_profile") == "qwen3_reranker"
        and config.get("match_head_type") == "native"
        and config.get("match_num_logits") == 1
        and isinstance(config.get("match_no_token_id"), int)
        and isinstance(config.get("match_yes_token_id"), int)
        and isinstance(pytorch_quantization, dict)
        and pytorch_quantization.get("format")
        == "qwen3_weight_only_int8_per_row_v1"
        and pytorch_quantization.get("weights_file") == "model.safetensors"
    )
    if (
        not isinstance(use_field_tokens, bool)
        or not isinstance(max_length, int)
        or max_length < 1
        or not (
            is_sequence_classifier
            or is_pooling_classifier
            or is_causal_zero_shot_reranker
            or is_qwen3_pytorch_int8
        )
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
    if require_pytorch_weights and not any(
        (directory / name).is_file() for name in weight_patterns
    ):
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


def _validate_onnx_artifacts(directory: Path, *, predictor: str) -> None:
    onnx_directory = directory / "onnx"
    required = (
        ("encoder.onnx",)
        if predictor == "fusion"
        else ("classifier.onnx",)
    )
    missing = [name for name in required if not (onnx_directory / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "ONNX Runtime backend requires exported Transformer graphs in "
            f"{onnx_directory}: {missing}"
        )


def _skip_file(
    path: Path,
    *,
    source_root: Path,
    skip_transformer_weights: bool = False,
) -> bool:
    relative = path.relative_to(source_root)
    if any(part in _SKIPPED_DIRECTORY_NAMES for part in relative.parts):
        return True
    if any(part.startswith("checkpoint-") for part in relative.parts[:-1]):
        return True
    if (
        skip_transformer_weights
        and len(relative.parts) == 1
        and any(relative.match(pattern) for pattern in _TRANSFORMER_WEIGHT_PATTERNS)
    ):
        return True
    return path.suffix in {".pyc", ".pyo"}


def _files_for_path(
    source: Path,
    archive_path: PurePosixPath,
    *,
    skip_transformer_weights: bool = False,
) -> Iterable[_ArchiveInput]:
    if source.is_file():
        yield _ArchiveInput(source, archive_path)
        return
    for candidate in sorted(source.rglob("*")):
        if not candidate.is_file() or _skip_file(
            candidate,
            source_root=source,
            skip_transformer_weights=skip_transformer_weights,
        ):
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
    include_onnxruntime: bool,
    include_tensorrt: bool,
    include_runtime: bool = True,
    transformer_dir: Path | None = None,
    skip_transformer_weights: bool = False,
) -> tuple[_ArchiveInput, ...]:
    sources = [project_root / name for name in _REQUIRED_PROJECT_FILES]
    sources.append(project_root / "src" / "match")
    sources.extend(resources)
    entries: dict[PurePosixPath, _ArchiveInput] = {}
    for source in sources:
        _validate_resource(source)
        archive_path = _archive_path(source, project_root=project_root)
        omit_weights = (
            skip_transformer_weights
            and transformer_dir is not None
            and source.resolve() == transformer_dir.resolve()
        )
        for entry in _files_for_path(
            source,
            archive_path,
            skip_transformer_weights=omit_weights,
        ):
            previous = entries.get(entry.archive_path)
            if previous is not None and previous.source != entry.source:
                raise ValueError(
                    f"duplicate archive path: {entry.archive_path}"
                )
            entries[entry.archive_path] = entry
    wheels_source = project_root / Path(_VENDOR_WHEELS_SOURCE)
    wheel_patterns: list[str] = []
    if include_runtime:
        _validate_polars_wheels(wheels_source)
        wheel_patterns.extend(("polars-*.whl", "polars_runtime_32-*.whl"))
    if include_runtime and include_preprocessing_runtime:
        _validate_preprocessing_wheels(wheels_source)
        wheel_patterns.extend(_PREPROCESSING_WHEEL_PATTERNS)
    if include_runtime and include_catboost:
        _validate_catboost_wheels(wheels_source)
        wheel_patterns.append("catboost-*.whl")
    if include_runtime and include_onnxruntime:
        _validate_onnxruntime_wheels(wheels_source)
        wheel_patterns.extend(_ONNX_RUNTIME_WHEEL_PATTERNS)
    if include_runtime and include_tensorrt:
        _validate_tensorrt_wheels(wheels_source)
        wheel_patterns.extend(_TENSORRT_WHEEL_PATTERNS)
    for pattern in wheel_patterns:
        for wheel in sorted(wheels_source.glob(pattern)):
            entry = _ArchiveInput(wheel, _VENDOR_WHEELS_TARGET / wheel.name)
            entries[entry.archive_path] = entry
    if include_runtime and include_tensorrt:
        runtime_source = project_root / Path(_TENSORRT_RUNTIME_SOURCE)
        for payload in sorted(runtime_source.iterdir()):
            if not payload.is_file():
                continue
            entry = _ArchiveInput(
                payload,
                _TENSORRT_RUNTIME_TARGET / payload.name,
            )
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


def _validate_onnxruntime_wheels(directory: Path) -> None:
    missing = [
        pattern
        for pattern in _ONNX_RUNTIME_WHEEL_PATTERNS
        if not any(directory.glob(pattern))
    ]
    if missing:
        raise FileNotFoundError(
            f"bundled ONNX Runtime wheels are missing in {directory}: {missing}"
        )
    runtime_wheels = sorted(directory.glob("onnxruntime_gpu-*.whl"))
    if len(runtime_wheels) != 1:
        raise ValueError(
            "expected exactly one bundled ONNX Runtime GPU wheel in "
            f"{directory}, found {len(runtime_wheels)}"
        )


def _validate_tensorrt_wheels(directory: Path) -> None:
    missing = [
        pattern for pattern in _TENSORRT_WHEEL_PATTERNS if not any(directory.glob(pattern))
    ]
    if missing:
        raise FileNotFoundError(
            f"bundled TensorRT wheels are missing in {directory}: {missing}"
        )
    runtime_directory = directory.parent / "tensorrt_runtime"
    payloads = sorted(runtime_directory.glob(_TENSORRT_RUNTIME_PATTERN))
    if len(payloads) != 1:
        raise FileNotFoundError(
            "expected exactly one compressed TensorRT runtime payload in "
            f"{runtime_directory}, found {len(payloads)}"
        )
    if not (runtime_directory / "LICENSE.txt").is_file():
        raise FileNotFoundError(
            f"TensorRT runtime license is missing: {runtime_directory / 'LICENSE.txt'}"
        )


def _compression(path: Path) -> int:
    return (
        ZIP_STORED
        if path.suffix.lower() in _STORED_SUFFIXES | {".zst"}
        else ZIP_DEFLATED
    )


def _load_image_weights_manifest(
    manifest_path: str | Path,
    *,
    transformer_dir: Path,
    project_root: Path,
) -> tuple[dict[str, object], frozenset[PurePosixPath]]:
    path = Path(manifest_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"image weights manifest does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as error:
        raise ValueError(f"invalid image weights manifest: {path}") from error
    if not isinstance(value, dict):
        raise ValueError("image weights manifest must contain a JSON object")
    image_directory = value.get("image_directory")
    files = value.get("files")
    if not isinstance(image_directory, str) or not image_directory.startswith("/"):
        raise ValueError("image weights manifest requires an absolute image_directory")
    if not isinstance(files, list) or not files:
        raise ValueError("image weights manifest requires a non-empty files array")

    archive_root = _archive_path(transformer_dir, project_root=project_root)
    packaged_entries: list[dict[str, object]] = []
    excluded: set[PurePosixPath] = set()
    seen: set[str] = set()
    for entry in files:
        if not isinstance(entry, dict):
            raise ValueError("image weight entries must be JSON objects")
        name = entry.get("name")
        size = entry.get("size")
        relative = PurePosixPath(str(name))
        if (
            not isinstance(name, str)
            or not name
            or len(relative.parts) != 1
            or relative.name != name
            or name in seen
        ):
            raise ValueError(f"invalid or duplicate image weight name: {name!r}")
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError(f"invalid image weight size for {name!r}: {size!r}")
        source = transformer_dir / "onnx" / name
        if not source.is_file() or source.stat().st_size != size:
            raise ValueError(
                f"image weight source does not match manifest: {source}"
            )
        seen.add(name)
        packaged_entries.append({"name": name, "size": size})
        excluded.add(archive_root / "onnx" / relative)
    return (
        {
            "image_directory": image_directory,
            "files": packaged_entries,
        },
        frozenset(excluded),
    )


def build_submission_archive(
    config: AppConfig,
    output_path: str | Path | None = None,
    *,
    project_root: Path = PROJECT_ROOT,
    image_weights_manifest: str | Path | None = None,
) -> SubmissionArchive:
    """Validate and package the configured predictor and enabled features."""
    root = project_root.expanduser().resolve()
    artifacts = _selected_artifacts(config)
    resources = _required_resources(config, artifacts)
    backend = config.inference.transformer.backend
    uses_ort_tensorrt = (
        backend == "onnxruntime"
        and config.inference.transformer.onnxruntime.provider == "tensorrt"
    )
    onnx_only = backend in {"tensorrt", "adaptive"} or (
        backend == "onnxruntime"
        and not config.inference.transformer.onnxruntime.fallback_to_pytorch
    )
    for resource in resources:
        _validate_resource(resource)
    if artifacts.transformer_dir is not None:
        _validate_transformer_artifact(
            artifacts.transformer_dir,
            require_pytorch_weights=not onnx_only,
        )
        if backend in {"onnxruntime", "tensorrt", "adaptive"}:
            _validate_onnx_artifacts(
                artifacts.transformer_dir,
                predictor=artifacts.predictor,
            )

    solution = build_solution_manifest(
        config,
        artifacts,
        map_path=lambda path: str(_archive_path(path, project_root=root)),
    )
    excluded_image_weights: frozenset[PurePosixPath] = frozenset()
    if image_weights_manifest is not None:
        if artifacts.transformer_dir is None:
            raise ValueError("image weights require a Transformer artifact")
        if not config.submission.use_custom_image:
            raise ValueError("image weights require submission.use_custom_image=true")
        external_weights, excluded_image_weights = _load_image_weights_manifest(
            image_weights_manifest,
            transformer_dir=artifacts.transformer_dir,
            project_root=root,
        )
        solution["external_weights"] = external_weights
    inputs = _collect_inputs(
        resources,
        project_root=root,
        include_catboost=artifacts.predictor in {"boosting", "cascade", "stacking"},
        include_preprocessing_runtime=(
            config.features.normalization.enabled
            or config.features.ner.enabled
            or config.features.physical.enabled
        ),
        include_onnxruntime=(
            backend in {"onnxruntime", "adaptive"}
            or (
                backend == "tensorrt"
                and config.inference.transformer.tensorrt.fallback_to_onnxruntime
            )
        ),
        include_tensorrt=(
            backend in {"tensorrt", "adaptive"} or uses_ort_tensorrt
        ),
        include_runtime=not config.submission.use_custom_image,
        transformer_dir=artifacts.transformer_dir,
        skip_transformer_weights=onnx_only,
    )
    if excluded_image_weights:
        available = {entry.archive_path for entry in inputs}
        missing = excluded_image_weights - available
        if missing:
            raise ValueError(
                "image weight entries are absent from submission inputs: "
                f"{sorted(map(str, missing))}"
            )
        inputs = tuple(
            entry
            for entry in inputs
            if entry.archive_path not in excluded_image_weights
        )
    target = Path(output_path or config.submission.output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    temporary.unlink(missing_ok=True)
    metadata = {
        "image": config.submission.image,
        "entry_point": config.submission.entry_point,
    }
    solution_json = json.dumps(solution, ensure_ascii=False, indent=2) + "\n"
    metadata_json = json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    try:
        with ZipFile(temporary, "w", allowZip64=True) as archive:
            archive.writestr(
                "solution.json",
                solution_json,
                compress_type=ZIP_DEFLATED,
            )
            archive.writestr(
                "metadata.json",
                metadata_json,
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
        file_count=len(inputs) + 2,
        uncompressed_bytes=sum(entry.source.stat().st_size for entry in inputs)
        + len(solution_json.encode("utf-8"))
        + len(metadata_json.encode("utf-8")),
    )


__all__ = ["SubmissionArchive", "build_submission_archive"]
