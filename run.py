"""Unified entry point for training, inspection and competition inference."""

import argparse
import ctypes
import importlib
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any


def _configure_huggingface_cache() -> None:
    """Keep Transformers dynamic modules off a potentially read-only home."""
    cache_root = Path(tempfile.gettempdir()) / "twin2attr_huggingface"
    os.environ.setdefault("HF_HOME", str(cache_root))
    os.environ.setdefault("HF_MODULES_CACHE", str(cache_root / "modules"))


# Nemotron ships custom ``trust_remote_code`` modules.  Transformers resolves
# its module-cache location during import, so configure a writable default
# before any command imports the model stack.
_configure_huggingface_cache()

# The evaluator limits the total number of processes/threads. Normalization
# already uses multiple worker processes, so allowing NumPy/OpenBLAS to create
# another full thread pool in every process can exhaust that limit before the
# Transformer is even imported.
for _thread_environment_variable in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
):
    os.environ[_thread_environment_variable] = "1"
os.environ["POLARS_MAX_THREADS"] = "8"

if TYPE_CHECKING:
    from match.config import AppConfig


SOURCE_ROOT = Path(__file__).resolve().parent / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

COMMANDS = {"train", "predict", "inspect", "initialize", "analyze"}
DEFAULT_CONFIG = "configs/pipeline.yaml"
POLARS_VERSION = "1.43.2"
CATBOOST_VERSION = "1.2.10"
PYMORPHY3_VERSION = "2.0.6"
JOBLIB_VERSION = "1.5.3"
LOGURU_VERSION = "0.7.3"
PINT_VERSION = "0.25.3"
ORJSON_VERSION = "3.11.9"
TENSORRT_VERSION = "10.9.0.34"
ZSTANDARD_VERSION = "0.25.0"
TENSORRT_RUNTIME_LIBRARIES = (
    "libnvinfer.so.10",
    "libnvinfer_plugin.so.10",
    "libnvonnxparser.so.10",
    "libnvinfer_builder_resource.so.10.9.0",
)


def ensure_polars_available() -> None:
    """Load Polars or install its bundled wheels into a temporary directory."""
    try:
        importlib.import_module("polars")
        return
    except ModuleNotFoundError as error:
        if error.name != "polars":
            raise

    wheels_dir = Path(__file__).resolve().parent / "vendor_wheels"
    if not any(wheels_dir.glob("polars-*.whl")) or not any(
        wheels_dir.glob("polars_runtime_32-*.whl")
    ):
        raise RuntimeError(
            "Polars is not installed and its bundled wheels are missing: "
            f"{wheels_dir}"
        )
    install_dir = (
        Path(tempfile.gettempdir())
        / f"twin2attr_polars_{POLARS_VERSION}_"
        f"{sys.version_info.major}{sys.version_info.minor}"
    )
    install_dir.mkdir(parents=True, exist_ok=True)
    print(f"Polars is absent; installing bundled wheels into {install_dir}")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-index",
            "--find-links",
            str(wheels_dir),
            "--upgrade",
            "--target",
            str(install_dir),
            f"polars=={POLARS_VERSION}",
        ],
        check=True,
    )
    sys.path.insert(0, str(install_dir))
    importlib.invalidate_caches()
    importlib.import_module("polars")


def ensure_preprocessing_runtime_available(
    solution_path: str | Path | None = None,
) -> None:
    """Install the complete bundled runtime needed by item preprocessing."""
    import json

    path = (
        Path(solution_path)
        if solution_path is not None
        else Path(__file__).resolve().parent / "solution.json"
    ).expanduser().resolve()
    if not path.is_file():
        return
    solution = json.loads(path.read_text(encoding="utf-8"))
    features = solution.get("features")
    feature_values = features if isinstance(features, dict) else {}
    normalization = feature_values.get(
        "normalization",
        solution.get("normalization"),
    )
    physical = feature_values.get("physical")
    ner = feature_values.get("ner")
    needs_preprocessing_runtime = (
        isinstance(normalization, dict)
        and bool(normalization.get("enabled", False))
    ) or (
        isinstance(physical, dict)
        and bool(physical.get("enabled", False))
    ) or (
        isinstance(ner, dict)
        and bool(ner.get("enabled", False))
    )
    if not needs_preprocessing_runtime:
        return

    runtime_modules = ("loguru", "joblib", "pint", "pymorphy3", "orjson")
    for module_name in runtime_modules:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            if error.name != module_name:
                raise
            break
    else:
        return

    wheels_dir = Path(__file__).resolve().parent / "vendor_wheels"
    required_patterns = (
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
    missing = [
        pattern for pattern in required_patterns if not any(wheels_dir.glob(pattern))
    ]
    if missing:
        raise RuntimeError(
            "item preprocessing dependencies are unavailable and bundled "
            "wheels are missing from "
            f"{wheels_dir}: {missing}"
        )
    tag = f"{sys.version_info.major}{sys.version_info.minor}"
    install_dir = (
        Path(tempfile.gettempdir())
        / f"twin2attr_preprocessing_1_{tag}"
    )
    install_dir.mkdir(parents=True, exist_ok=True)
    print(
        "Item preprocessing dependencies are absent; "
        f"installing bundled wheels into {install_dir}"
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-index",
            "--find-links",
            str(wheels_dir),
            "--upgrade",
            "--target",
            str(install_dir),
            f"pymorphy3=={PYMORPHY3_VERSION}",
            f"joblib=={JOBLIB_VERSION}",
            f"loguru=={LOGURU_VERSION}",
            f"pint=={PINT_VERSION}",
            f"orjson=={ORJSON_VERSION}",
        ],
        check=True,
    )
    sys.path.insert(0, str(install_dir))
    importlib.invalidate_caches()
    for module_name in runtime_modules:
        importlib.import_module(module_name)


def ensure_catboost_available(solution_path: str | Path | None = None) -> None:
    """Install the bundled CatBoost wheel only for predictors that need it."""
    import json

    path = (
        Path(solution_path)
        if solution_path is not None
        else Path(__file__).resolve().parent / "solution.json"
    ).expanduser().resolve()
    if not path.is_file():
        return
    solution = json.loads(path.read_text(encoding="utf-8"))
    if solution.get("predictor") not in {"boosting", "cascade", "stacking"}:
        return
    try:
        importlib.import_module("catboost")
        return
    except ModuleNotFoundError as error:
        if error.name != "catboost":
            raise

    wheels_dir = Path(__file__).resolve().parent / "vendor_wheels"
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    wheels = sorted(wheels_dir.glob(f"catboost-*-{tag}-{tag}-*.whl"))
    if not wheels:
        raise RuntimeError(
            f"CatBoost is not installed and no bundled wheel supports Python {tag}"
        )
    install_dir = Path(tempfile.gettempdir()) / f"twin2attr_catboost_{CATBOOST_VERSION}_{tag}"
    install_dir.mkdir(parents=True, exist_ok=True)
    print(f"CatBoost is absent; installing bundled wheel into {install_dir}")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-index",
            "--no-deps",
            "--upgrade",
            "--target",
            str(install_dir),
            str(wheels[-1]),
        ],
        check=True,
    )
    sys.path.insert(0, str(install_dir))
    importlib.invalidate_caches()
    importlib.import_module("catboost")


def ensure_onnxruntime_available(
    solution_path: str | Path | None = None,
    *,
    backend_override: str | None = None,
) -> None:
    """Install the bundled GPU runtime for an ONNX submission."""
    import json

    path = (
        Path(solution_path)
        if solution_path is not None
        else Path(__file__).resolve().parent / "solution.json"
    ).expanduser().resolve()
    if not path.is_file():
        return
    solution = json.loads(path.read_text(encoding="utf-8"))
    backend = backend_override or solution.get("backend")
    native_fallback = bool(
        solution.get("tensorrt", {}).get("fallback_to_onnxruntime", True)
    )
    if backend != "onnxruntime" and not (
        backend == "tensorrt" and native_fallback
    ):
        return

    provider = str(solution.get("onnxruntime", {}).get("provider", "cuda"))
    required_provider = {
        "cuda": "CUDAExecutionProvider",
        "tensorrt": "TensorrtExecutionProvider",
    }.get(provider)
    try:
        runtime = importlib.import_module("onnxruntime")
    except ModuleNotFoundError as error:
        if error.name != "onnxruntime":
            raise
    else:
        if required_provider and required_provider not in runtime.get_available_providers():
            raise RuntimeError(
                f"installed ONNX Runtime does not expose {required_provider}; "
                f"available providers: {runtime.get_available_providers()}"
            )
        return

    wheels_dir = Path(__file__).resolve().parent / "vendor_wheels"
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    runtime_wheels = sorted(
        wheels_dir.glob(f"onnxruntime_gpu-*-{tag}-{tag}-*.whl")
    )
    if len(runtime_wheels) != 1:
        raise RuntimeError(
            "expected exactly one bundled ONNX Runtime GPU wheel for the "
            f"evaluator's Python version ({tag}), found {len(runtime_wheels)}"
        )
    runtime_wheel = runtime_wheels[0]
    runtime_cache_key = runtime_wheel.name.split(f"-{tag}-{tag}-", 1)[0]
    install_dir = (
        Path(tempfile.gettempdir())
        / f"twin2attr_{runtime_cache_key}_{tag}"
    )
    install_dir.mkdir(parents=True, exist_ok=True)
    if not (install_dir / "onnxruntime").is_dir():
        dependencies = [
            *sorted(wheels_dir.glob("coloredlogs-*.whl")),
            *sorted(wheels_dir.glob("flatbuffers-*.whl")),
            *sorted(wheels_dir.glob("humanfriendly-*.whl")),
        ]
        if len(dependencies) != 3:
            raise RuntimeError("bundled ONNX Runtime dependencies are incomplete")
        print(
            "Installing bundled ONNX Runtime GPU into "
            f"{install_dir}"
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-index",
                "--no-deps",
                "--upgrade",
                "--target",
                str(install_dir),
                str(runtime_wheel),
                *(str(wheel) for wheel in dependencies),
            ],
            check=True,
        )
    sys.path.insert(0, str(install_dir))
    importlib.invalidate_caches()
    runtime = importlib.import_module("onnxruntime")
    if required_provider and required_provider not in runtime.get_available_providers():
        raise RuntimeError(
            f"bundled ONNX Runtime does not expose {required_provider}; "
            f"available providers: {runtime.get_available_providers()}"
        )


def _requires_tensorrt_runtime(
    solution: dict[str, Any],
    backend_override: str | None = None,
) -> bool:
    backend = backend_override or solution.get("backend")
    provider = str(solution.get("onnxruntime", {}).get("provider", "cuda"))
    return backend == "tensorrt" or (
        backend == "onnxruntime" and provider.lower() == "tensorrt"
    )


def ensure_tensorrt_available(
    solution_path: str | Path | None = None,
    *,
    backend_override: str | None = None,
) -> None:
    """Install bundled TensorRT bindings and libraries when selected."""
    import json

    path = (
        Path(solution_path)
        if solution_path is not None
        else Path(__file__).resolve().parent / "solution.json"
    ).expanduser().resolve()
    if not path.is_file():
        return
    solution = json.loads(path.read_text(encoding="utf-8"))
    if not _requires_tensorrt_runtime(solution, backend_override):
        return
    try:
        importlib.import_module("tensorrt")
        return
    except ModuleNotFoundError as error:
        if error.name != "tensorrt":
            raise
    wheels_dir = Path(__file__).resolve().parent / "vendor_wheels"
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    bindings = sorted(
        wheels_dir.glob(f"tensorrt_cu12_bindings-*-{tag}-*-manylinux*.whl")
    )
    if not bindings:
        raise RuntimeError(
            f"TensorRT is not bundled for the evaluator's Python version ({tag})"
        )
    zstandard_wheels = sorted(
        wheels_dir.glob(f"zstandard-*-{tag}-{tag}-manylinux*.whl")
    )
    if len(zstandard_wheels) != 1:
        raise RuntimeError(
            "expected exactly one bundled zstandard wheel for the evaluator's "
            f"Python version ({tag}), found {len(zstandard_wheels)}"
        )
    runtime_payloads = sorted(
        (Path(__file__).resolve().parent / "tensorrt_runtime").glob(
            "tensorrt-runtime-*.tar.zst"
        )
    )
    if len(runtime_payloads) != 1:
        raise RuntimeError(
            "expected exactly one bundled TensorRT runtime payload, found "
            f"{len(runtime_payloads)}"
        )
    install_dir = (
        Path(tempfile.gettempdir()) / f"twin2attr_tensorrt_{TENSORRT_VERSION}_{tag}"
    )
    install_dir.mkdir(parents=True, exist_ok=True)
    if not (install_dir / "tensorrt_bindings").is_dir() or not (
        install_dir / "zstandard"
    ).is_dir():
        print(f"Installing bundled TensorRT into {install_dir}")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-index",
                "--no-deps",
                "--upgrade",
                "--target",
                str(install_dir),
                str(bindings[-1]),
                str(zstandard_wheels[0]),
            ],
            check=True,
        )
    sys.path.insert(0, str(install_dir))
    importlib.invalidate_caches()
    zstandard = importlib.import_module("zstandard")

    libraries_dir = install_dir / "lib"
    marker = libraries_dir / ".complete"
    if not marker.is_file() or any(
        not (libraries_dir / name).is_file()
        for name in TENSORRT_RUNTIME_LIBRARIES
    ):
        staging = install_dir / "lib.staging"
        shutil.rmtree(staging, ignore_errors=True)
        staging.mkdir(parents=True)
        print(f"Extracting bundled TensorRT runtime into {libraries_dir}")
        with runtime_payloads[0].open("rb") as compressed:
            with zstandard.ZstdDecompressor().stream_reader(compressed) as reader:
                with tarfile.open(fileobj=reader, mode="r|") as archive:
                    for member in archive:
                        if not member.isfile():
                            continue
                        name = Path(member.name).name
                        if name not in TENSORRT_RUNTIME_LIBRARIES:
                            raise RuntimeError(
                                f"unexpected file in TensorRT runtime: {member.name}"
                            )
                        source = archive.extractfile(member)
                        if source is None:
                            raise RuntimeError(
                                f"cannot extract TensorRT runtime file: {member.name}"
                            )
                        with (staging / name).open("wb") as target:
                            shutil.copyfileobj(source, target, length=8 * 1024**2)
        missing_libraries = [
            name
            for name in TENSORRT_RUNTIME_LIBRARIES
            if not (staging / name).is_file()
        ]
        if missing_libraries:
            raise RuntimeError(
                f"TensorRT runtime payload is incomplete: {missing_libraries}"
            )
        shutil.rmtree(libraries_dir, ignore_errors=True)
        staging.replace(libraries_dir)
        marker.write_text(TENSORRT_VERSION, encoding="utf-8")

    load_mode = getattr(ctypes, "RTLD_GLOBAL", 0)
    for name in TENSORRT_RUNTIME_LIBRARIES:
        ctypes.CDLL(str(libraries_dir / name), mode=load_mode)
    bindings_module = importlib.import_module("tensorrt_bindings")
    sys.modules.setdefault("tensorrt", bindings_module)


def _with_default_command(arguments: Sequence[str]) -> list[str]:
    """Treat the evaluator's argument-only invocation as ``predict``."""
    values = list(arguments)
    if values and values[0] in COMMANDS:
        return values
    return ["predict", *values]


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Path to the base YAML configuration file",
    )
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help=(
            "Hydra-style overrides such as "
            "model_description.transformer.max_epochs=3"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Twin2Attr training and inference")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="Train configured model stages")
    _add_config_arguments(train)

    initialize = commands.add_parser(
        "initialize",
        help="Save an inference artifact with an untrained Transformer head",
    )
    _add_config_arguments(initialize)

    inspect = commands.add_parser(
        "inspect",
        help="Inspect pair lengths without training or prediction",
    )
    _add_config_arguments(inspect)

    analyze = commands.add_parser(
        "analyze",
        help="Calculate configured offline model statistics",
    )
    _add_config_arguments(analyze)

    predict = commands.add_parser("predict", help="Create evaluator-compatible CSV")
    predict.add_argument(
        "--items_path",
        "--items-path",
        "-i",
        dest="items_path",
        required=True,
    )
    predict.add_argument(
        "--matches_path",
        "--matches-path",
        "-m",
        dest="matches_path",
        required=True,
    )
    predict.add_argument(
        "--output_path",
        "--output-path",
        "-o",
        dest="output_path",
        required=True,
    )
    predict.add_argument(
        "--solution",
        default=None,
        help=(
            "Optional path to solution.json; otherwise use inference.solution_path "
            "from the local pipeline config or solution.json beside run.py"
        ),
    )
    predict.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Optional local pipeline config used to resolve inference.solution_path",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse an explicit argument list or the current process arguments."""
    arguments = sys.argv[1:] if argv is None else list(argv)
    return build_parser().parse_args(_with_default_command(arguments))


def _load_workflow_config(
    config_path: str,
    overrides: Sequence[str],
) -> "AppConfig":
    from match.config import load_app_config_file

    return load_app_config_file(config_path, overrides)


def _predict_solution_path(args: argparse.Namespace) -> Path | None:
    if args.solution:
        path = Path(args.solution).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Solution manifest does not exist: {path}")
        return path
    config_path = Path(args.config).expanduser()
    if config_path.is_file():
        path = _load_workflow_config(
            str(config_path), ()
        ).inference.solution_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Solution manifest does not exist: {path}")
        return path
    packaged_solution = Path(__file__).resolve().parent / "solution.json"
    if packaged_solution.is_file():
        return packaged_solution
    raise FileNotFoundError(
        "Solution manifest was not provided, the configured manifest is "
        f"unavailable, and packaged manifest does not exist: {packaged_solution}"
    )


def _adaptive_backend_for_input(
    solution_path: Path | None,
    matches_path: str | Path,
) -> str | None:
    if solution_path is None or not solution_path.is_file():
        return None
    import json

    solution = json.loads(solution_path.read_text(encoding="utf-8"))
    import polars as pl
    from match.backend_routing import resolve_effective_backend

    pair_count = int(
        pl.scan_parquet(matches_path).select(pl.len()).collect().item()
    )
    selected = resolve_effective_backend(solution, pair_count)
    print(
        "Resolved inference backend: "
        f"backend={selected}; pairs={pair_count}; "
        f"configured_backend={solution.get('backend', 'pytorch')}"
    )
    return selected


def run_predict(args: argparse.Namespace) -> None:
    """Create a validated evaluator-compatible prediction CSV."""
    solution_path = _predict_solution_path(args)
    ensure_polars_available()
    selected_backend = _adaptive_backend_for_input(
        solution_path,
        args.matches_path,
    )
    ensure_tensorrt_available(
        solution_path,
        backend_override=selected_backend,
    )
    ensure_onnxruntime_available(
        solution_path,
        backend_override=selected_backend,
    )
    print(f"Validated inference runtime: backend={selected_backend}")
    ensure_preprocessing_runtime_available(solution_path)
    ensure_catboost_available(solution_path)

    from match.submission import create_submission

    result = create_submission(
        items_path=args.items_path,
        matches_path=args.matches_path,
        output_path=args.output_path,
        solution_path=solution_path,
        backend_override=selected_backend,
    )
    print(f"Submission saved to {args.output_path}; rows={result.height}")


def run_train(args: argparse.Namespace) -> None:
    """Load the training configuration and execute its explicit workflow."""
    from match.experiments import configure_experiment
    from match.workflows.train import train

    experiment_name = os.environ.get("EXPERIMENT_NAME")
    if experiment_name is None:
        raise ValueError("EXPERIMENT_NAME environment variable is required")
    config = _load_workflow_config(
        args.config,
        args.overrides,
    )
    config, experiment_dir, registry_path = configure_experiment(
        config,
        experiment_name,
    )
    artifacts = train(
        config,
        experiment_name=experiment_name,
        experiment_registry_path=registry_path,
    )
    print(
        f"Experiment {experiment_name!r} saved to {experiment_dir}; "
        f"predictor={artifacts.predictor}"
    )


def run_initialize(args: argparse.Namespace) -> None:
    """Create a Transformer inference artifact without fitting it."""
    from match.workflows.initialize import initialize

    config = _load_workflow_config(args.config, args.overrides)
    artifacts = initialize(config)
    print(
        "Untrained Transformer artifact initialized at "
        f"{artifacts.transformer_dir}"
    )


def run_inspect(args: argparse.Namespace) -> None:
    """Inspect the configured data and print the recommended pair length."""
    from match.workflows.inspect import inspect_max_length

    config = _load_workflow_config(
        args.config,
        args.overrides,
    )
    result = inspect_max_length(config)
    print(f"Recommended max_length: {result}")


def run_analyze(args: argparse.Namespace) -> None:
    """Calculate and persist configured offline model statistics."""
    from match.workflows.analyze import analyze

    config = _load_workflow_config(args.config, args.overrides)
    result = analyze(config)
    print(
        f"Attribute importance saved to {result.output_path}; "
        f"sample_rows={result.sample_rows}, attribute_rows={result.attribute_rows}"
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    if args.command == "predict":
        run_predict(args)
        return

    if args.command == "train":
        run_train(args)
        return

    if args.command == "initialize":
        run_initialize(args)
        return

    if args.command == "inspect":
        run_inspect(args)
        return

    if args.command == "analyze":
        run_analyze(args)
        return

    raise ValueError(f"Unsupported command: {args.command!r}")


if __name__ == "__main__":
    main()
