"""Small, privacy-conscious runtime observability helpers."""
from __future__ import annotations

import contextvars
import threading
import time
from collections import Counter
from typing import Any


_started_at = time.monotonic()
_request_id: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")
_lock = threading.Lock()
_metrics: Counter[str] = Counter()
_active_requests = 0


def set_request_id(value: str) -> contextvars.Token[str]:
    return _request_id.set(value)


def reset_request_id(token: contextvars.Token[str]) -> None:
    _request_id.reset(token)


def get_request_id() -> str:
    return _request_id.get()


def request_started() -> None:
    global _active_requests
    with _lock:
        _metrics["http_requests_total"] += 1
        _active_requests += 1


def request_finished(status_code: int, duration_ms: int) -> None:
    global _active_requests
    with _lock:
        _active_requests = max(0, _active_requests - 1)
        _metrics[f"http_status_{status_code}"] += 1
        _metrics["http_duration_ms_total"] += max(0, duration_ms)


def chat_finished(success: bool, duration_ms: int, streaming: bool) -> None:
    with _lock:
        _metrics["chat_requests_total"] += 1
        _metrics["chat_success_total" if success else "chat_failure_total"] += 1
        _metrics["chat_stream_total" if streaming else "chat_sync_total"] += 1
        _metrics["chat_duration_ms_total"] += max(0, duration_ms)


def runtime_metrics() -> dict[str, Any]:
    with _lock:
        request_count = _metrics["http_requests_total"]
        chat_count = _metrics["chat_requests_total"]
        return {
            "uptime_seconds": int(time.monotonic() - _started_at),
            "active_requests": _active_requests,
            "http_requests_total": request_count,
            "http_failures_total": sum(value for key, value in _metrics.items() if key.startswith("http_status_") and int(key.rsplit("_", 1)[1]) >= 500),
            "http_average_duration_ms": round(_metrics["http_duration_ms_total"] / request_count, 1) if request_count else 0.0,
            "chat_requests_total": chat_count,
            "chat_failures_total": _metrics["chat_failure_total"],
            "chat_average_duration_ms": round(_metrics["chat_duration_ms_total"] / chat_count, 1) if chat_count else 0.0,
        }
