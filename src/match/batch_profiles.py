"""Persistent hardware/workload profiles for batch-size tuning."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import transformers


PROFILE_SCHEMA_VERSION = 1
_MODEL_FILES = ("config.json", "model.safetensors", "pytorch_model.bin")


@dataclass(frozen=True, slots=True)
class PerformanceEnvironment:
    gpu: str
    vram_bytes: int
    torch: str
    transformers: str
    cuda: str | None


@dataclass(frozen=True, slots=True)
class PerformanceWorkload:
    model_hash: str
    mode: str
    dtype: str
    attention: str
    torch_compile: bool
    max_length: int | None
    padding_buckets: tuple[int, ...]
    length_quantiles: dict[str, int] | None = None


@dataclass(frozen=True, slots=True)
class BatchMeasurement:
    batch_size: int
    status: str
    tokens_per_second: float | None = None
    pairs_per_second: float | None = None
    peak_vram_gib: float | None = None
    p95_step_ms: float | None = None
    measured_at: str | None = None

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive")
        if self.status not in {"completed", "oom", "failed"}:
            raise ValueError("measurement status must be completed, oom or failed")
        for name in (
            "tokens_per_second",
            "pairs_per_second",
            "peak_vram_gib",
            "p95_step_ms",
        ):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class BatchPerformanceProfile:
    environment: PerformanceEnvironment
    workload: PerformanceWorkload
    measurements: tuple[BatchMeasurement, ...]
    best_batch_size: int | None
    safe_batch_size: int | None
    fingerprint: str
    schema_version: int = PROFILE_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["workload"]["padding_buckets"] = list(
            self.workload.padding_buckets
        )
        return result

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> BatchPerformanceProfile:
        environment = PerformanceEnvironment(**value["environment"])
        workload_value = dict(value["workload"])
        workload_value["padding_buckets"] = tuple(
            int(item) for item in workload_value.get("padding_buckets", ())
        )
        workload = PerformanceWorkload(**workload_value)
        measurements = tuple(
            BatchMeasurement(**measurement)
            for measurement in value.get("measurements", ())
        )
        return cls(
            environment=environment,
            workload=workload,
            measurements=measurements,
            best_batch_size=value.get("best_batch_size"),
            safe_batch_size=value.get("safe_batch_size"),
            fingerprint=str(value["fingerprint"]),
            schema_version=int(value.get("schema_version", 1)),
        )


def current_environment() -> PerformanceEnvironment:
    """Describe the runtime that determines whether a profile can be reused."""
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(torch.cuda.current_device())
        gpu = str(properties.name)
        vram_bytes = int(properties.total_memory)
    else:
        gpu = "cpu"
        vram_bytes = 0
    return PerformanceEnvironment(
        gpu=gpu,
        vram_bytes=vram_bytes,
        torch=str(torch.__version__),
        transformers=str(transformers.__version__),
        cuda=None if torch.version.cuda is None else str(torch.version.cuda),
    )


def _hash_file(path: Path, digest: Any) -> None:
    digest.update(path.name.encode("utf-8"))
    digest.update(str(path.stat().st_size).encode("ascii"))
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)


def model_content_hash(path: str | Path) -> str:
    """Hash model configuration and weights without depending on path or mtime."""
    model_path = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    if model_path.is_file():
        _hash_file(model_path, digest)
        return digest.hexdigest()
    if not model_path.is_dir():
        digest.update(f"missing:{model_path}".encode("utf-8"))
        return digest.hexdigest()

    selected: list[Path] = []
    for name in _MODEL_FILES:
        candidate = model_path / name
        if candidate.is_file():
            selected.append(candidate)
    selected.extend(sorted(model_path.glob("model-*.safetensors")))
    selected.extend(sorted(model_path.glob("pytorch_model-*.bin")))
    unique = sorted(set(selected), key=lambda item: item.name)
    if not unique:
        digest.update(f"empty:{model_path}".encode("utf-8"))
    for candidate in unique:
        _hash_file(candidate, digest)
    return digest.hexdigest()


def profile_fingerprint(
    environment: PerformanceEnvironment,
    workload: PerformanceWorkload,
) -> str:
    """Return an exact, stable environment plus workload fingerprint."""
    payload = {
        "environment": asdict(environment),
        "workload": asdict(workload),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _throughput(measurement: BatchMeasurement) -> float:
    if measurement.tokens_per_second is not None:
        return measurement.tokens_per_second
    return measurement.pairs_per_second or 0.0


def _rounded_safe_batch(
    best_batch_size: int,
    *,
    safe_batch_fraction: float,
    batch_size_multiple: int,
) -> int:
    requested = int(math.floor(best_batch_size * safe_batch_fraction))
    rounded = requested - requested % batch_size_multiple
    return max(1, rounded)


def build_profile(
    environment: PerformanceEnvironment,
    workload: PerformanceWorkload,
    measurements: tuple[BatchMeasurement, ...],
    *,
    safe_batch_fraction: float = 0.875,
    batch_size_multiple: int = 8,
) -> BatchPerformanceProfile:
    if not 0.0 < safe_batch_fraction <= 1.0:
        raise ValueError("safe_batch_fraction must be in (0, 1]")
    if batch_size_multiple < 1:
        raise ValueError("batch_size_multiple must be positive")
    completed = [
        measurement
        for measurement in measurements
        if measurement.status == "completed" and _throughput(measurement) > 0.0
    ]
    best = max(completed, key=_throughput) if completed else None
    best_batch_size = None if best is None else best.batch_size
    safe_batch_size = (
        None
        if best_batch_size is None
        else _rounded_safe_batch(
            best_batch_size,
            safe_batch_fraction=safe_batch_fraction,
            batch_size_multiple=batch_size_multiple,
        )
    )
    return BatchPerformanceProfile(
        environment=environment,
        workload=workload,
        measurements=tuple(sorted(measurements, key=lambda item: item.batch_size)),
        best_batch_size=best_batch_size,
        safe_batch_size=safe_batch_size,
        fingerprint=profile_fingerprint(environment, workload),
    )


def profile_path(directory: Path, profile: BatchPerformanceProfile) -> Path:
    return directory / profile.workload.mode / f"{profile.fingerprint}.json"


def load_profile(
    directory: Path,
    environment: PerformanceEnvironment,
    workload: PerformanceWorkload,
) -> BatchPerformanceProfile | None:
    fingerprint = profile_fingerprint(environment, workload)
    path = directory / workload.mode / f"{fingerprint}.json"
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"batch profile must contain an object: {path}")
    profile = BatchPerformanceProfile.from_dict(value)
    if profile.fingerprint != fingerprint:
        raise ValueError(f"batch profile fingerprint mismatch: {path}")
    return profile


def save_profile(directory: Path, profile: BatchPerformanceProfile) -> Path:
    """Atomically persist a profile so interrupted benchmarks cannot corrupt it."""
    path = profile_path(directory, profile)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        profile.to_dict(),
        ensure_ascii=False,
        indent=2,
        allow_nan=False,
    ) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def update_profile(
    directory: Path,
    environment: PerformanceEnvironment,
    workload: PerformanceWorkload,
    new_measurements: tuple[BatchMeasurement, ...],
    *,
    safe_batch_fraction: float = 0.875,
    batch_size_multiple: int = 8,
) -> BatchPerformanceProfile:
    """Merge measurements by batch size and return the new recommendation."""
    existing = load_profile(directory, environment, workload)
    merged = {
        measurement.batch_size: measurement
        for measurement in (() if existing is None else existing.measurements)
    }
    merged.update(
        {measurement.batch_size: measurement for measurement in new_measurements}
    )
    profile = build_profile(
        environment,
        workload,
        tuple(merged.values()),
        safe_batch_fraction=safe_batch_fraction,
        batch_size_multiple=batch_size_multiple,
    )
    save_profile(directory, profile)
    return profile


def recommended_starting_batch_size(
    directory: Path,
    environment: PerformanceEnvironment,
    workload: PerformanceWorkload,
) -> int | None:
    profile = load_profile(directory, environment, workload)
    return None if profile is None else profile.safe_batch_size


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "BatchMeasurement",
    "BatchPerformanceProfile",
    "PerformanceEnvironment",
    "PerformanceWorkload",
    "build_profile",
    "current_environment",
    "load_profile",
    "model_content_hash",
    "profile_fingerprint",
    "recommended_starting_batch_size",
    "save_profile",
    "update_profile",
    "utc_now",
]
