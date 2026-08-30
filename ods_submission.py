"""ODS authentication and competition submission workflow."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import keyring
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait


LOGGER = logging.getLogger(__name__)

DEFAULT_URL = "https://ods.ai/competitions/e-cup-2026-matching/submissions"
CREDENTIAL_SERVICE = "Twin2Attr ODS uploader"
ARCHIVE_INPUT_XPATH = (
    '//*[@id="__next"]/div[1]/div/div[2]/div[4]/div/div/div[1]/div/'
    "form/div/div[1]/div/input"
)
SUBMISSION_FORM_XPATH = f"{ARCHIVE_INPUT_XPATH}/ancestor::form[1]"
SUBMIT_BUTTON_XPATH = (
    '//button[@type="submit" and normalize-space()="Отправить решение"]'
)
COOKIE_ACCEPT_BUTTON_XPATH = (
    '//div[contains(concat(" ", normalize-space(@class), " "), " CookieBanner ")]'
    '//button[normalize-space()="Принять"]'
)
FORM_ERROR_XPATH = (
    f'{SUBMISSION_FORM_XPATH}//*[contains(concat(" ", normalize-space(@class), '
    '" "), " error ")]'
)
UPLOAD_PROGRESS_XPATH = (
    '//*[contains(normalize-space(.), "Не обновляйте и не закрывайте страницу") '
    'and not(.//*[contains(normalize-space(.), '
    '"Не обновляйте и не закрывайте страницу")])]'
)
LATEST_SUBMISSION_ROW_XPATH = (
    '//*[@id="__next"]/div/div/div[2]/div[5]/div/div/form/table/tbody/tr[1]'
)
ERROR_LOG_MODAL_XPATH = "/html/body/div[2]/div/div/div[2]/div"
PENDING_STATUSES = {"queued", "qued", "running", "pending"}
NO_METRIC_VALUES = {"", "-", "—"}


class SubmissionFailed(RuntimeError):
    """Raised when ODS reports a submission error."""


@dataclass(frozen=True)
class EvaluationResult:
    status: str
    metric: str | None = None
    error_log: str | None = None


def load_credentials() -> tuple[str, str]:
    """Read ODS credentials from environment variables or Windows Credential Manager."""
    email = os.environ.get("ODS_EMAIL") or keyring.get_password(
        CREDENTIAL_SERVICE, "email"
    )
    password = os.environ.get("ODS_PASSWORD") or keyring.get_password(
        CREDENTIAL_SERVICE, "password"
    )
    if not email or not password:
        raise SubmissionFailed(
            "ODS credentials were not found in Windows Credential Manager."
        )
    return email, password


def dismiss_cookie_banner(driver: webdriver.Remote) -> None:
    """Accept the ODS cookie banner when it is displayed."""
    try:
        accept_button = WebDriverWait(driver, 5).until(
            EC.element_to_be_clickable((By.XPATH, COOKIE_ACCEPT_BUTTON_XPATH))
        )
    except TimeoutException:
        return

    accept_button.click()
    WebDriverWait(driver, 5).until(
        EC.invisibility_of_element_located((By.XPATH, COOKIE_ACCEPT_BUTTON_XPATH))
    )
    LOGGER.info("Cookie banner accepted.")


def login_if_needed(driver: webdriver.Remote) -> None:
    """Log in when ODS redirects the browser to its authentication form."""
    wait = WebDriverWait(driver, 20)
    wait.until(
        lambda browser: browser.execute_script("return document.readyState")
        == "complete"
    )

    try:
        username = WebDriverWait(driver, 5).until(
            EC.visibility_of_element_located((By.XPATH, '//*[@id="username"]'))
        )
        password_input = WebDriverWait(driver, 5).until(
            EC.visibility_of_element_located((By.XPATH, '//*[@id="password"]'))
        )
    except TimeoutException:
        LOGGER.info("Saved session found; login is not required.")
        return

    LOGGER.info("Login form found. Authenticating automatically.")
    email, password = load_credentials()

    username.clear()
    username.send_keys(email)
    password_input.clear()
    password_input.send_keys(password)

    if username.get_attribute("value") != email:
        raise SubmissionFailed("Email was not entered into the login form.")
    if password_input.get_attribute("value") != password:
        raise SubmissionFailed("Password was not entered into the login form.")

    time.sleep(1)
    driver.find_element(By.CSS_SELECTOR, 'button[type="submit"]').click()

    try:
        wait.until(EC.invisibility_of_element_located((By.ID, "password")))
    except TimeoutException as error:
        raise SubmissionFailed(
            "Automatic login was not confirmed. Check the open browser for "
            "an error, CAPTCHA, or an additional verification step."
        ) from error
    LOGGER.info("Login completed; cookies are stored in the Selenium profile.")


def _visible_texts(driver: webdriver.Remote, xpath: str) -> list[str]:
    texts: list[str] = []
    for element in driver.find_elements(By.XPATH, xpath):
        if element.is_displayed() and element.text.strip():
            texts.append(" ".join(element.text.split()))
    return list(dict.fromkeys(texts))


def wait_for_submission_result(
    driver: webdriver.Remote, *, timeout_seconds: int
) -> None:
    """Wait until ODS confirms success or displays a form error."""
    deadline = time.monotonic() + timeout_seconds
    last_progress = ""
    LOGGER.info("Waiting for ODS to upload and register the solution...")

    while time.monotonic() < deadline:
        errors = _visible_texts(driver, FORM_ERROR_XPATH)
        if errors:
            raise SubmissionFailed("ODS reported an error: " + " | ".join(errors))

        archive_inputs = driver.find_elements(By.XPATH, ARCHIVE_INPUT_XPATH)
        if archive_inputs and not (archive_inputs[0].get_attribute("value") or ""):
            LOGGER.info("ODS confirmed the submission; the form was cleared.")
            return

        progress_messages = _visible_texts(driver, UPLOAD_PROGRESS_XPATH)
        progress = " | ".join(progress_messages)
        if progress and progress != last_progress:
            LOGGER.info("ODS progress: %s", progress)
            last_progress = progress

        time.sleep(2)

    raise SubmissionFailed(
        f"ODS did not confirm the submission within {timeout_seconds} seconds."
    )


def _metric_or_none(text: str) -> str | None:
    metric = " ".join(text.split())
    return None if metric in NO_METRIC_VALUES else metric


def _read_error_log(driver: webdriver.Remote, row) -> str:
    try:
        show_button = row.find_element(
            By.XPATH, './/td[4]//button[normalize-space()="Показать"]'
        )
    except Exception as error:
        raise SubmissionFailed(
            "ODS reported a container error, but its log button was not found."
        ) from error

    show_button.click()
    time.sleep(1)
    modal = WebDriverWait(driver, 20).until(
        EC.visibility_of_element_located((By.XPATH, ERROR_LOG_MODAL_XPATH))
    )
    error_log = modal.text.strip()
    if not error_log:
        raise SubmissionFailed("The ODS error-log window opened, but it was empty.")
    return error_log


def monitor_latest_submission(
    driver: webdriver.Remote,
    *,
    archive_name: str | None = None,
    timeout_seconds: int = 21_600,
    poll_interval_seconds: int = 15,
    timeout_metric_grace_seconds: int = 300,
) -> EvaluationResult:
    """Monitor the newest table row until evaluation reaches a final status."""
    deadline = time.monotonic() + timeout_seconds
    timeout_metric_deadline: float | None = None
    last_status = ""
    last_metric: str | None = None

    LOGGER.info("Monitoring the newest ODS submission row...")
    while time.monotonic() < deadline:
        driver.refresh()
        WebDriverWait(driver, 30).until(
            lambda browser: browser.execute_script("return document.readyState")
            == "complete"
        )

        try:
            row = WebDriverWait(driver, 30).until(
                EC.presence_of_element_located(
                    (By.XPATH, LATEST_SUBMISSION_ROW_XPATH)
                )
            )
        except TimeoutException:
            LOGGER.info("The newest submission row is not visible yet.")
            time.sleep(poll_interval_seconds)
            continue

        if archive_name and archive_name.lower() not in row.text.lower():
            LOGGER.info(
                "Waiting for %s to appear as the newest table row.", archive_name
            )
            time.sleep(poll_interval_seconds)
            continue

        status_cell = row.find_element(By.XPATH, "./td[4]")
        metric_cell = row.find_element(By.XPATH, "./td[5]")
        status = status_cell.text.splitlines()[0].strip()
        metric = _metric_or_none(metric_cell.text)
        normalized_status = status.lower()

        if status != last_status or metric != last_metric:
            LOGGER.info("Evaluation status: %s; metric: %s", status, metric or "—")
            last_status = status
            last_metric = metric

        if normalized_status == "success":
            if metric:
                LOGGER.info("Evaluation succeeded. Metric: %s", metric)
                return EvaluationResult(status=status, metric=metric)
            LOGGER.info("Success is visible; waiting for the metric.")

        elif "container did not finish in time" in normalized_status:
            if metric:
                LOGGER.warning(
                    "The container ran out of time, but a metric is available: %s",
                    metric,
                )
                return EvaluationResult(status=status, metric=metric)
            if timeout_metric_deadline is None:
                timeout_metric_deadline = (
                    time.monotonic() + timeout_metric_grace_seconds
                )
                LOGGER.warning(
                    "The container ran out of time. Waiting up to %d seconds "
                    "for a possible metric.",
                    timeout_metric_grace_seconds,
                )
            elif time.monotonic() >= timeout_metric_deadline:
                LOGGER.warning("The container ran out of time; no metric appeared.")
                return EvaluationResult(status=status)

        elif "container finished with non-zero exit code" in normalized_status:
            error_log = _read_error_log(driver, row)
            LOGGER.error("Container finished with a non-zero exit code.\n%s", error_log)
            return EvaluationResult(
                status=status,
                metric=metric,
                error_log=error_log,
            )

        elif normalized_status.startswith("error"):
            buttons = row.find_elements(
                By.XPATH, './/td[4]//button[normalize-space()="Показать"]'
            )
            error_log = _read_error_log(driver, row) if buttons else None
            LOGGER.error(
                "Evaluation failed: %s; metric: %s%s",
                status,
                metric or "—",
                f"\n{error_log}" if error_log else "",
            )
            return EvaluationResult(
                status=status,
                metric=metric,
                error_log=error_log,
            )

        elif normalized_status not in PENDING_STATUSES:
            LOGGER.info("Waiting on the unrecognized intermediate status: %s", status)

        time.sleep(poll_interval_seconds)

    raise SubmissionFailed(
        f"ODS evaluation did not finish within {timeout_seconds} seconds."
    )


def upload_archive(
    driver: webdriver.Remote,
    archive: Path,
    *,
    submit: bool = True,
    timeout_seconds: int = 21_600,
    evaluation_timeout_seconds: int = 21_600,
) -> EvaluationResult | None:
    """Attach the archive, optionally submit it, and wait for the ODS result."""
    wait = WebDriverWait(driver, 30)
    archive_input = wait.until(
        EC.presence_of_element_located((By.XPATH, ARCHIVE_INPUT_XPATH))
    )
    archive_input.send_keys(str(archive))

    selected_file = archive_input.get_attribute("value") or ""
    if not selected_file.lower().endswith(archive.name.lower()):
        raise SubmissionFailed("The archive was not attached to the submission form.")
    LOGGER.info("Archive attached: %s", archive.name)

    if not submit:
        LOGGER.info("--no-submit is active; the submit button was not clicked.")
        return None

    dismiss_cookie_banner(driver)
    submit_button = wait.until(
        EC.element_to_be_clickable((By.XPATH, SUBMIT_BUTTON_XPATH))
    )
    driver.execute_script(
        'arguments[0].scrollIntoView({block: "center", inline: "nearest"});',
        submit_button,
    )
    submit_button.click()
    LOGGER.info("The 'Отправить решение' button was clicked.")
    wait_for_submission_result(driver, timeout_seconds=timeout_seconds)
    LOGGER.info("Upload completed. Waiting 2 seconds for the table to update.")
    time.sleep(2)
    return monitor_latest_submission(
        driver,
        archive_name=archive.name,
        timeout_seconds=evaluation_timeout_seconds,
    )


def run_submission(
    driver: webdriver.Remote,
    archive: Path,
    *,
    url: str = DEFAULT_URL,
    submit: bool = True,
    timeout_seconds: int = 21_600,
    evaluation_timeout_seconds: int = 21_600,
) -> EvaluationResult | None:
    """Run the complete ODS login and submission workflow."""
    driver.get(url)
    dismiss_cookie_banner(driver)
    login_if_needed(driver)
    dismiss_cookie_banner(driver)
    return upload_archive(
        driver,
        archive,
        submit=submit,
        timeout_seconds=timeout_seconds,
        evaluation_timeout_seconds=evaluation_timeout_seconds,
    )
