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


def observability_summary(*, days: int = 7) -> dict[str, Any]:
    events = _events_since(days)
    total = len(events)
    failures = [item for item in events if int(item.get("status_code", 0)) >= 500]
    paths = Counter(str(item.get("path", "")) for item in events)
    statuses = Counter(str(item.get("status_code", "")) for item in events)
    return {
        "days": max(1, int(days)), "retention_days": OBSERVABILITY_RETENTION_DAYS,
        "requests": total, "failures": len(failures),
        "failure_rate": round(len(failures) / total * 100, 1) if total else 0.0,
        "average_duration_ms": round(sum(int(item.get("duration_ms", 0)) for item in events) / total, 1) if total else 0.0,
        "top_paths": [{"path": path, "requests": count} for path, count in paths.most_common(10)],
        "statuses": dict(statuses),
    }
