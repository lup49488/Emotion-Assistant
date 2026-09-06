"""Long-term, privacy-minimized HTTP observability storage."""
from __future__ import annotations

import json
import threading
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from config import BASE_DIR, OBSERVABILITY_RETENTION_DAYS
from sqlite_store import connection, sqlite_enabled


_JSON_PATH = BASE_DIR / "data" / "observability_events.json"
_LOCK = threading.RLock()
_MAX_JSON_EVENTS = 10_000
_HEALTH_PATHS = {"/health", "/health/live", "/health/ready", "/api/v1/status"}


def _now() -> datetime:
    return datetime.now()


def _read_json_events() -> list[dict[str, Any]]:
    try:
        payload = json.loads(_JSON_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return payload if isinstance(payload, list) else []


def _write_json_events(events: list[dict[str, Any]]) -> None:
    _JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary = _JSON_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(events[-_MAX_JSON_EVENTS:], ensure_ascii=False), encoding="utf-8")
    temporary.replace(_JSON_PATH)


def record_http_event(*, request_id: str, method: str, path: str, status_code: int, duration_ms: int) -> None:
    """Persist only route metadata, never query strings, payloads, or identities."""
    event = {
        "request_id": str(request_id)[:128], "method": str(method).upper()[:12],
        "path": str(path)[:512], "status_code": int(status_code),
        "duration_ms": max(0, int(duration_ms)), "created_at": _now().isoformat(timespec="seconds"),
    }
    with _LOCK:
        if sqlite_enabled():
            with connection() as conn:
                conn.execute(
                    "INSERT INTO observability_events(request_id, method, path, status_code, duration_ms, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                    tuple(event.values()),
                )
                cutoff = (_now() - timedelta(days=OBSERVABILITY_RETENTION_DAYS)).isoformat(timespec="seconds")
                conn.execute("DELETE FROM observability_events WHERE created_at < ?", (cutoff,))
            return
        events = _read_json_events()
        events.append(event)
        cutoff = _now() - timedelta(days=OBSERVABILITY_RETENTION_DAYS)
        retained = [item for item in events if _parse_time(item.get("created_at")) >= cutoff]
        _write_json_events(retained)


def _parse_time(value: Any) -> datetime:
    try:
        return datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return datetime.min


def _events_since(days: int) -> list[dict[str, Any]]:
    cutoff = _now() - timedelta(days=max(1, int(days)))
    with _LOCK:
        if sqlite_enabled():
            with connection() as conn:
                rows = conn.execute(
                    "SELECT method, path, status_code, duration_ms, created_at FROM observability_events WHERE created_at >= ? ORDER BY id DESC",
                    (cutoff.isoformat(timespec="seconds"),),
                ).fetchall()
            return [dict(row) for row in rows]
        return [item for item in _read_json_events() if _parse_time(item.get("created_at")) >= cutoff]


def _traffic_kind(path: str) -> str:
    if path in _HEALTH_PATHS:
        return "probe"
    if path in {"/api/v1/chat", "/api/v1/chat/stream"}:
        return "model"
    return "api" if path.startswith("/api/") else "other"


def _percentile(values: list[int], percentile: int) -> float:
    """Nearest-rank percentile: the smallest value at or above the given rank.

    ``(n * p + 99) // 100`` is the ceiling of ``n * p / 100``, and the ``- 1``
    turns that 1-based rank into a list index. No interpolation, so the result
    is always an observed duration.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, (len(ordered) * percentile + 99) // 100 - 1))
    return float(ordered[index])


def _traffic_summary(events: list[dict[str, Any]]) -> dict[str, dict[str, int | float]]:
    grouped: dict[str, list[dict[str, Any]]] = {kind: [] for kind in ("probe", "api", "model", "other")}
    for event in events:
        grouped[_traffic_kind(str(event.get("path", "")))].append(event)
    summary: dict[str, dict[str, int | float]] = {}
    for kind, items in grouped.items():
        durations = [max(0, int(item.get("duration_ms", 0))) for item in items]
        failures = sum(int(item.get("status_code", 0)) >= 500 for item in items)
        total = len(items)
        summary[kind] = {
            "requests": total,
            "failures": failures,
            "failure_rate": round(failures / total * 100, 1) if total else 0.0,
            "average_duration_ms": round(sum(durations) / total, 1) if total else 0.0,
            "p50_duration_ms": _percentile(durations, 50),
            "p95_duration_ms": _percentile(durations, 95),
        }
    return summary


def observability_summary(*, days: int = 7) -> dict[str, Any]:
    events = _events_since(days)
    total = len(events)
    failures = [item for item in events if int(item.get("status_code", 0)) >= 500]
    paths = Counter(str(item.get("path", "")) for item in events)
    statuses = Counter(str(item.get("status_code", "")) for item in events)
    traffic = _traffic_summary(events)
    return {
        "days": max(1, int(days)), "retention_days": OBSERVABILITY_RETENTION_DAYS,
        "requests": total, "failures": len(failures),
        "failure_rate": round(len(failures) / total * 100, 1) if total else 0.0,
        "average_duration_ms": round(sum(int(item.get("duration_ms", 0)) for item in events) / total, 1) if total else 0.0,
        "top_paths": [{"path": path, "requests": count} for path, count in paths.most_common(10)],
        "statuses": dict(statuses),
        "traffic": traffic,
    }
