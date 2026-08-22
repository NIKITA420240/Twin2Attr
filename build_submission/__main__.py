"""Command-line entry point for building the evaluator ZIP archive."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m build_submission",
        description="Build a Twin2Attr evaluator archive from pipeline.yaml",
    )
    parser.add_argument(
        "--config",
        default="configs/pipeline.yaml",
        help="Path to pipeline YAML (default: configs/pipeline.yaml)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Override submission.output_path from the config",
    )
    parser.add_argument(
        "overrides",
        nargs=argparse.REMAINDER,
        help="OmegaConf overrides, for example inference.model=fusion",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)

    from match.config import load_app_config_file

    from .archive import build_submission_archive

    config = load_app_config_file(args.config, args.overrides)
    try:
        result = build_submission_archive(config, args.output)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"Cannot build submission archive: {error}") from error
    size_mib = result.path.stat().st_size / (1024 * 1024)
    print(
        f"Submission archive saved to {result.path}; "
        f"predictor={result.predictor}; files={result.file_count}; "
        f"size={size_mib:.1f} MiB"
    )


if __name__ == "__main__":
    main()
