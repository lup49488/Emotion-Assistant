"""Small in-process abuse limiter for authentication; never stores secrets."""
from __future__ import annotations

import threading
import time
from collections import deque

from config import API_AUTH_MAX_ATTEMPTS, API_AUTH_WINDOW_SECONDS


_LOCK = threading.Lock()
_FAILURES: dict[str, deque[float]] = {}

LOGIN_SCOPE = "login"
EMAIL_VERIFY_SCOPE = "email-verify"


def _key(scope: str, client_ip: str, identifier: str) -> str:
    # The scope keeps separate limits apart even when their identifiers collide.
    return f"{scope}|{client_ip[:128]}|{identifier.strip().casefold()[:128]}"


def _recent(key: str, now: float) -> deque[float]:
    """Return this key's live failure timestamps, dropping the entry once empty.

    Reads never create a key: a plain ``dict`` is used instead of ``defaultdict``
    so that probing with many distinct IP/user pairs cannot grow the map
    unbounded. Empty deques are removed so successful and expired attempts leave
    no residue.
    """
    attempts = _FAILURES.get(key)
    if attempts is None:
        return deque()
    cutoff = now - API_AUTH_WINDOW_SECONDS
    while attempts and attempts[0] < cutoff:
        attempts.popleft()
    if not attempts:
        _FAILURES.pop(key, None)
    return attempts


def attempt_allowed(scope: str, client_ip: str, identifier: str, max_attempts: int) -> tuple[bool, int]:
    now = time.monotonic()
    with _LOCK:
        attempts = _recent(_key(scope, client_ip, identifier), now)
        if len(attempts) < max_attempts:
            return True, 0
        return False, max(1, int(API_AUTH_WINDOW_SECONDS - (now - attempts[0])))


def record_attempt_failure(scope: str, client_ip: str, identifier: str) -> None:
    now = time.monotonic()
    with _LOCK:
        key = _key(scope, client_ip, identifier)
        _recent(key, now)
        _FAILURES.setdefault(key, deque()).append(now)


def clear_attempt_failures(scope: str, client_ip: str, identifier: str) -> None:
    with _LOCK:
        _FAILURES.pop(_key(scope, client_ip, identifier), None)


def login_allowed(client_ip: str, user_id: str) -> tuple[bool, int]:
    # Read the limit at call time so configuration and tests can adjust it.
    return attempt_allowed(LOGIN_SCOPE, client_ip, user_id, API_AUTH_MAX_ATTEMPTS)


def record_login_failure(client_ip: str, user_id: str) -> None:
    record_attempt_failure(LOGIN_SCOPE, client_ip, user_id)


def clear_login_failures(client_ip: str, user_id: str) -> None:
    clear_attempt_failures(LOGIN_SCOPE, client_ip, user_id)
