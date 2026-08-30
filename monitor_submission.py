"""Monitor the newest ODS E-Cup submission without uploading another archive."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from selenium.common.exceptions import SessionNotCreatedException

from ods_submission import (
    DEFAULT_URL,
    dismiss_cookie_banner,
    login_if_needed,
    monitor_latest_submission,
)
from selenium_uploader import DEFAULT_LOG_PATH, configure_logging, create_driver


LOGGER = logging.getLogger(__name__)


def default_monitor_profile_dir() -> Path:
    base = Path(os.environ.get("LOCALAPPDATA", Path.home()))
    return base / "Twin2Attr" / "monitor-profile"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Monitor the newest ODS E-Cup submission."
    )
    parser.add_argument(
        "--archive-name",
        default="twin2attr_submission.zip",
        help="Expected archive in the newest row",
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="ODS submissions page")
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=default_monitor_profile_dir(),
        help="Persistent profile used only by the monitor",
    )
    parser.add_argument(
        "--evaluation-timeout",
        type=int,
        default=21_600,
        metavar="SECONDS",
        help="Maximum evaluation wait time (default: 21600, six hours)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=15,
        metavar="SECONDS",
        help="Page refresh interval (default: 15 seconds)",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_LOG_PATH,
        help=f"Log path (default: {DEFAULT_LOG_PATH})",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.evaluation_timeout <= 0 or args.poll_interval <= 0:
        raise SystemExit("Timeout and poll interval must be greater than zero.")

    log_path = configure_logging(args.log_file)
    profile_dir = args.profile_dir.expanduser().resolve()
    LOGGER.info("Starting standalone evaluation monitor.")
    LOGGER.info("Monitor profile: %s", profile_dir)
    LOGGER.info("Log file: %s", log_path)

    try:
        driver = create_driver("chrome", profile_dir)
    except SessionNotCreatedException:
        LOGGER.exception(
            "Could not start the monitor browser. Close the browser using this "
            "profile and try again: %s",
            profile_dir,
        )
        return 2

    try:
        driver.get(args.url)
        dismiss_cookie_banner(driver)
        login_if_needed(driver)
        dismiss_cookie_banner(driver)
        result = monitor_latest_submission(
            driver,
            archive_name=args.archive_name,
            timeout_seconds=args.evaluation_timeout,
            poll_interval_seconds=args.poll_interval,
        )
    except Exception:
        LOGGER.exception("Evaluation monitoring failed. The browser will remain open.")
        return 1

    LOGGER.info(
        "Final result: status=%s; metric=%s; error_log=%s",
        result.status,
        result.metric or "—",
        "captured" if result.error_log else "none",
    )
    LOGGER.info("Monitoring finished. The browser will remain open.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
