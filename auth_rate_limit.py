"""Small in-process login abuse limiter; never stores passwords."""
from __future__ import annotations

import threading
import time
from collections import deque

from config import API_AUTH_MAX_ATTEMPTS, API_AUTH_WINDOW_SECONDS


_LOCK = threading.Lock()
_FAILURES: dict[str, deque[float]] = {}


def _key(client_ip: str, user_id: str) -> str:
    return f"{client_ip[:128]}:{user_id.strip().casefold()[:128]}"


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


def login_allowed(client_ip: str, user_id: str) -> tuple[bool, int]:
    now = time.monotonic()
    with _LOCK:
        attempts = _recent(_key(client_ip, user_id), now)
        if len(attempts) < API_AUTH_MAX_ATTEMPTS:
            return True, 0
        return False, max(1, int(API_AUTH_WINDOW_SECONDS - (now - attempts[0])))


def record_login_failure(client_ip: str, user_id: str) -> None:
    now = time.monotonic()
    with _LOCK:
        key = _key(client_ip, user_id)
        _recent(key, now)
        _FAILURES.setdefault(key, deque()).append(now)


def clear_login_failures(client_ip: str, user_id: str) -> None:
    with _LOCK:
        _FAILURES.pop(_key(client_ip, user_id), None)
