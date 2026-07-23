"""Data-driven checks for Serenova dialogue regressions.

Run the deterministic cases with pytest.  Live cases are opt-in because they
call a configured provider and intentionally use a dedicated test account.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import HTTPCookieProcessor, Request, build_opener

from langdetect import DetectorFactory, LangDetectException, detect


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CASE_PATH = BASE_DIR / "evaluation" / "dialogue_regression.json"

# langdetect is probabilistic by default; regression decisions must be stable.
DetectorFactory.seed = 0


def load_cases(path: Path = DEFAULT_CASE_PATH) -> list[dict[str, Any]]:
    """Load and validate the portable JSON regression suite."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if payload.get("version") != 1 or not isinstance(cases, list):
        raise ValueError("Unsupported dialogue regression suite format.")
    identifiers = [str(case.get("id", "")).strip() for case in cases]
    if not all(identifiers) or len(identifiers) != len(set(identifiers)):
        raise ValueError("Dialogue regression case IDs must be unique and non-empty.")
    return cases


def validate_reply(case: dict[str, Any], reply: str) -> list[str]:
    """Return user-readable failures without relying on an exact model reply."""
    checks = case.get("checks", {})
    text = (reply or "").strip()
    failures: list[str] = []
    if not text:
        return ["reply was empty"]
    if len(text) < int(checks.get("min_characters", 1)):
        failures.append(f"reply was shorter than {checks['min_characters']} characters")
    normalized = text.casefold()
    for expected in checks.get("must_contain", []):
        if str(expected).casefold() not in normalized:
            failures.append(f"reply did not contain {expected!r}")
    for forbidden in checks.get("must_not_contain", []):
        if str(forbidden).casefold() in normalized:
            failures.append(f"reply contained forbidden text {forbidden!r}")
    expected_language = checks.get("reply_language")
    if expected_language:
        try:
            detected = detect(text)
        except LangDetectException:
            detected = ""
        if expected_language == "zh":
            matches = detected.startswith("zh") or any("\u4e00" <= char <= "\u9fff" for char in text)
        else:
            matches = detected == expected_language
        if not matches:
            failures.append(f"reply language was {detected or 'undetermined'}, expected {expected_language}")
    return failures


def _json_request(opener: Any, url: str, payload: dict[str, Any], headers: dict[str, str] | None = None) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with opener.open(request, timeout=90) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Unable to reach API: {exc.reason}") from exc


def run_live_cases(
    base_url: str,
    user_id: str,
    access_key: str,
    *,
    provider: str | None = None,
    model: str | None = None,
    include_optional: bool = False,
) -> list[dict[str, Any]]:
    """Run opt-in provider checks against a dedicated regression account."""
    root = base_url.rstrip("/")
    cookies = CookieJar()
    opener = build_opener(HTTPCookieProcessor(cookies))
    _json_request(opener, f"{root}/api/v1/auth/login", {"user_id": user_id, "access_key": access_key})
    csrf_name = os.getenv("DIALOGUE_REGRESSION_CSRF_COOKIE_NAME", "chatbot_csrf")
    csrf_token = next((cookie.value for cookie in cookies if cookie.name == csrf_name), "")
    if not csrf_token:
        raise RuntimeError(f"API login did not set the expected {csrf_name} cookie.")

    report: list[dict[str, Any]] = []
    for case in load_cases():
        if not case.get("live") or (case.get("optional") and not include_optional):
            continue
        request = {"message": case["input"], **case.get("request", {})}
        if provider:
            request["provider"] = provider
        if model:
            request["model"] = model
        try:
            response = _json_request(opener, f"{root}/api/v1/chat", request, {"X-CSRF-Token": csrf_token})
            reply = str(response.get("reply", ""))
            failures = validate_reply(case, reply)
            report.append({"id": case["id"], "passed": not failures, "failures": failures, "reply": reply})
        except RuntimeError as exc:
            report.append({"id": case["id"], "passed": False, "failures": [str(exc)], "reply": ""})
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run opt-in Serenova live dialogue regression cases.")
    parser.add_argument("--base-url", default=os.getenv("DIALOGUE_REGRESSION_URL", ""))
    parser.add_argument("--user-id", default=os.getenv("DIALOGUE_REGRESSION_USER_ID", ""))
    parser.add_argument("--access-key", default=os.getenv("DIALOGUE_REGRESSION_ACCESS_KEY", ""))
    parser.add_argument("--provider", default=os.getenv("DIALOGUE_REGRESSION_PROVIDER", ""))
    parser.add_argument("--model", default=os.getenv("DIALOGUE_REGRESSION_MODEL", ""))
    parser.add_argument("--include-optional", action="store_true")
    args = parser.parse_args()
    if not args.base_url or not args.user_id or not args.access_key:
        parser.error("Set --base-url, --user-id, and --access-key for a dedicated regression account.")
    report = run_live_cases(
        args.base_url, args.user_id, args.access_key,
        provider=args.provider or None, model=args.model or None,
        include_optional=args.include_optional,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report and all(item["passed"] for item in report) else 1


if __name__ == "__main__":
    sys.exit(main())
