"""Unified entry point for training, inspection and competition inference."""

import argparse
import importlib
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from match.config import AppConfig


SOURCE_ROOT = Path(__file__).resolve().parent / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

COMMANDS = {"train", "predict", "inspect"}
DEFAULT_CONFIG = "configs/pipeline.yaml"
POLARS_VERSION = "1.43.2"
CATBOOST_VERSION = "1.2.10"
PYMORPHY3_VERSION = "2.0.6"
JOBLIB_VERSION = "1.5.3"
LOGURU_VERSION = "0.7.3"
PINT_VERSION = "0.25.3"


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
    normalization = solution.get("normalization")
    features = solution.get("features")
    physical = features.get("physical") if isinstance(features, dict) else None
    needs_preprocessing_runtime = (
        isinstance(normalization, dict)
        and bool(normalization.get("enabled", False))
    ) or (
        isinstance(physical, dict)
        and bool(physical.get("enabled", False))
    )
    if not needs_preprocessing_runtime:
        return

    runtime_modules = ("loguru", "joblib", "pint", "pymorphy3")
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
    if solution.get("predictor") not in {"boosting", "cascade"}:
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
            "models_parameters.transformer.max_epochs=3"
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Twin2Attr training and inference")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="Train configured model stages")
    _add_config_arguments(train)

    inspect = commands.add_parser(
        "inspect",
        help="Inspect pair lengths without training or prediction",
    )
    _add_config_arguments(inspect)

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
        help="Optional path to solution.json; defaults to the project root",
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


def run_predict(args: argparse.Namespace) -> None:
    """Create a validated evaluator-compatible prediction CSV."""
    ensure_polars_available()
    ensure_preprocessing_runtime_available(args.solution)
    ensure_catboost_available(args.solution)

    from match.submission import create_submission

    result = create_submission(
        items_path=args.items_path,
        matches_path=args.matches_path,
        output_path=args.output_path,
        solution_path=Path(args.solution) if args.solution else None,
    )
    print(f"Submission saved to {args.output_path}; rows={result.height}")


def run_train(args: argparse.Namespace) -> None:
    """Load the training configuration and execute its explicit workflow."""
    from match.workflows.train import train

    config = _load_workflow_config(
        args.config,
        args.overrides,
    )
    train(config)


def run_inspect(args: argparse.Namespace) -> None:
    """Inspect the configured data and print the recommended pair length."""
    from match.workflows.inspect import inspect_max_length

    config = _load_workflow_config(
        args.config,
        args.overrides,
    )
    result = inspect_max_length(config)
    print(f"Recommended max_length: {result}")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    if args.command == "predict":
        run_predict(args)
        return

    if args.command == "train":
        run_train(args)
        return

    if args.command == "inspect":
        run_inspect(args)
        return

    raise ValueError(f"Unsupported command: {args.command!r}")


if __name__ == "__main__":
    main()
