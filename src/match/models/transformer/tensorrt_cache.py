"""Persistent native TensorRT engine and timing caches."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .onnx_export import ONNX_DIRECTORY_NAME
from .tensorrt_common import TensorRTInitializationError


@dataclass(frozen=True, slots=True)
class TensorRTEngineCache:
    model_directory: Path
    enabled: bool = True
    directory: Path | None = None
    timing_enabled: bool = True
    timing_path: Path | None = None

    def _directory(self) -> Path:
        return self.directory or (
            self.model_directory / ONNX_DIRECTORY_NAME / "native_trt_cache"
        )

    def _ensure_parent(self, path: Path) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise TensorRTInitializationError(
                f"failed to create TensorRT cache directory for {path}: {error}"
            ) from error

    def engine_path(
        self,
        onnx_path: Path,
        kind: str,
        identity: Mapping[str, Any],
    ) -> Path | None:
        if not self.enabled:
            return None
        digest = hashlib.sha256()
        try:
            with onnx_path.open("rb") as source:
                for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                    digest.update(chunk)
        except OSError as error:
            raise TensorRTInitializationError(
                f"failed to fingerprint ONNX graph {onnx_path}: {error}"
            ) from error
        digest.update(json.dumps(identity, sort_keys=True).encode())
        path = self._directory() / f"{kind}-{digest.hexdigest()[:20]}.engine"
        self._ensure_parent(path)
        return path

    def load_engine(self, path: Path | None) -> bytes | None:
        if path is None or not path.is_file():
            return None
        try:
            return path.read_bytes()
        except OSError as error:
            raise TensorRTInitializationError(
                f"failed to read TensorRT engine cache {path}: {error}"
            ) from error

    def store_engine(self, path: Path | None, serialized: bytes) -> None:
        if path is None:
            return
        self._atomic_write(path, serialized, "engine")

    def load_timing(self) -> bytes:
        path = self.resolved_timing_path()
        if path is None or not path.is_file():
            return b""
        try:
            return path.read_bytes()
        except OSError as error:
            raise TensorRTInitializationError(
                f"failed to read TensorRT timing cache {path}: {error}"
            ) from error

    def store_timing(self, serialized: bytes) -> None:
        path = self.resolved_timing_path()
        if path is not None:
            self._atomic_write(path, serialized, "timing")

    def resolved_timing_path(self) -> Path | None:
        if not self.timing_enabled:
            return None
        path = self.timing_path or self._directory() / "timing.cache"
        self._ensure_parent(path)
        return path

    def _atomic_write(self, path: Path, serialized: bytes, label: str) -> None:
        self._ensure_parent(path)
        temporary = path.with_name(f".{path.name}.tmp")
        try:
            temporary.write_bytes(serialized)
            temporary.replace(path)
        except OSError as error:
            raise TensorRTInitializationError(
                f"failed to write TensorRT {label} cache {path}: {error}"
            ) from error


__all__ = ["TensorRTEngineCache"]
