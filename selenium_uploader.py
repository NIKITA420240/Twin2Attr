"""Command-line entry point for the ODS Selenium uploader."""

from __future__ import annotations

import argparse
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from selenium import webdriver
from selenium.common.exceptions import SessionNotCreatedException
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.edge.options import Options as EdgeOptions

from ods_submission import DEFAULT_URL, run_submission


LOGGER = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_LOG_PATH = PROJECT_DIR / "logs" / "selenium_uploader.log"

SUPPORTED_ARCHIVE_SUFFIXES = (
    ".zip",
    ".7z",
    ".rar",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".tar.bz2",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload an archive to the ODS E-Cup submission page."
    )
    parser.add_argument("archive", type=Path, help="Path to the archive")
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"Page to open (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--browser",
        choices=("chrome", "edge"),
        default="chrome",
        help="Browser to use (default: chrome)",
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        help="Persistent browser profile directory",
    )
    parser.add_argument(
        "--no-submit",
        action="store_true",
        help="Attach the archive but do not click 'Отправить решение'",
    )
    parser.add_argument(
        "--upload-timeout",
        type=int,
        default=21_600,
        metavar="SECONDS",
        help="Maximum upload/result wait time (default: 21600, six hours)",
    )
    parser.add_argument(
        "--evaluation-timeout",
        type=int,
        default=21_600,
        metavar="SECONDS",
        help="Maximum evaluation wait time (default: 21600, six hours)",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help=f"Log path (default: {DEFAULT_LOG_PATH})",
    )
    return parser.parse_args()


def configure_logging(log_path: Path) -> Path:
    resolved_path = log_path.expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    file_handler = RotatingFileHandler(
        resolved_path,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logging.basicConfig(
        level=logging.INFO,
        handlers=(console_handler, file_handler),
        force=True,
    )
    return resolved_path


def resolve_archive(path: Path) -> Path:
    archive = path.expanduser().resolve()
    if not archive.is_file():
        raise SystemExit(f"Archive not found: {archive}")

    lowercase_name = archive.name.lower()
    if not lowercase_name.endswith(SUPPORTED_ARCHIVE_SUFFIXES):
        raise SystemExit(
            "Unsupported archive type. Expected one of: "
            + ", ".join(SUPPORTED_ARCHIVE_SUFFIXES)
        )
    return archive


def default_profile_dir() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", Path.home()))
    return base / "Twin2Attr" / "selenium-profile"


def create_driver(browser: str, profile_dir: Path) -> webdriver.Remote:
    """Start a detached browser using the persistent uploader profile."""
    profile_dir.mkdir(parents=True, exist_ok=True)

    if browser == "edge":
        options = EdgeOptions()
        options.add_argument(f"--user-data-dir={profile_dir}")
        options.add_argument("--profile-directory=Default")
        options.add_argument("--start-maximized")
        options.add_experimental_option("detach", True)
        return webdriver.Edge(options=options)

    options = ChromeOptions()
    options.add_argument(f"--user-data-dir={profile_dir}")
    options.add_argument("--profile-directory=Default")
    options.add_argument("--start-maximized")
    options.add_experimental_option("detach", True)
    return webdriver.Chrome(options=options)


def main() -> int:
    args = parse_args()
    if args.upload_timeout <= 0 or args.evaluation_timeout <= 0:
        raise SystemExit("Upload and evaluation timeouts must be greater than zero.")

    archive = resolve_archive(args.archive)
    profile_dir = (args.profile_dir or default_profile_dir()).expanduser().resolve()
    log_path = configure_logging(args.log_file)

    LOGGER.info("Archive ready: %s (%d bytes)", archive, archive.stat().st_size)
    LOGGER.info("Browser profile: %s", profile_dir)
    LOGGER.info("Log file: %s", log_path)
    LOGGER.info("Starting detached browser...")

    try:
        driver = create_driver(args.browser, profile_dir)
    except SessionNotCreatedException:
        LOGGER.exception(
            "Could not start the browser. Close the existing Selenium browser "
            "that uses this profile and try again: %s",
            profile_dir,
        )
        return 2

    try:
        result = run_submission(
            driver,
            archive,
            url=args.url,
            submit=not args.no_submit,
            timeout_seconds=args.upload_timeout,
            evaluation_timeout_seconds=args.evaluation_timeout,
        )
    except Exception:
        LOGGER.exception("Submission workflow failed. The browser will remain open.")
        return 1

    if result is not None:
        LOGGER.info(
            "Final result: status=%s; metric=%s; error_log=%s",
            result.status,
            result.metric or "—",
            "captured" if result.error_log else "none",
        )
    LOGGER.info("Workflow finished. The browser will remain open; close it manually.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
