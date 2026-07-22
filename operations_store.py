"""Administrator-only operational aggregates and durable alert state."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

from config import (
    OPS_ALERT_AVERAGE_LATENCY_MS, OPS_ALERT_HTTP_FAILURE_RATE,
    OPS_ALERT_JOB_FAILURES, OPS_ALERT_MIN_REQUESTS, OPS_ALERT_PROVIDER_FAILURES,
)
from observability_store import observability_summary
from sqlite_store import connection, sqlite_enabled


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _provider_failures(days: int) -> list[dict[str, Any]]:
    if not sqlite_enabled():
        return []
    cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
    with connection() as conn:
        rows = conn.execute(
            """SELECT provider, error_kind, COUNT(*) AS failures
               FROM api_usage_events WHERE success = 0 AND created_at >= ?
               GROUP BY provider, error_kind ORDER BY failures DESC LIMIT 10""",
            (cutoff,),
        ).fetchall()
    return [dict(row) for row in rows]


def _job_summary(days: int) -> dict[str, Any]:
    if not sqlite_enabled():
        return {"counts": {}, "recent_failures": []}
    cutoff = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")
    with connection() as conn:
        rows = conn.execute(
            "SELECT status, COUNT(*) AS count FROM background_jobs WHERE created_at >= ? GROUP BY status", (cutoff,)
        ).fetchall()
        failures = conn.execute(
            """SELECT kind, error, finished_at FROM background_jobs
               WHERE status = 'failed' AND created_at >= ? ORDER BY finished_at DESC LIMIT 8""",
            (cutoff,),
        ).fetchall()
    return {"counts": {str(row["status"]): int(row["count"]) for row in rows}, "recent_failures": [dict(row) for row in failures]}


def _rules(http: dict[str, Any], provider_failures: list[dict[str, Any]], jobs: dict[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    if http["requests"] >= OPS_ALERT_MIN_REQUESTS and http["failure_rate"] >= OPS_ALERT_HTTP_FAILURE_RATE:
        candidates.append({"fingerprint": "http_failure_rate", "severity": "critical", "message": f"HTTP 5xx failure rate is {http['failure_rate']}%.", "metadata": {"failure_rate": http["failure_rate"], "requests": http["requests"]}})
    if http["requests"] >= OPS_ALERT_MIN_REQUESTS and http["average_duration_ms"] >= OPS_ALERT_AVERAGE_LATENCY_MS:
        candidates.append({"fingerprint": "http_latency", "severity": "warning", "message": f"Average HTTP latency is {http['average_duration_ms']} ms.", "metadata": {"average_duration_ms": http["average_duration_ms"]}})
    provider_total = sum(int(item["failures"]) for item in provider_failures)
    if provider_total >= OPS_ALERT_PROVIDER_FAILURES:
        candidates.append({"fingerprint": "provider_failures", "severity": "critical", "message": f"Provider failures reached {provider_total} in the selected window.", "metadata": {"failures": provider_total}})
    failed_jobs = int(jobs["counts"].get("failed", 0))
    if failed_jobs >= OPS_ALERT_JOB_FAILURES:
        candidates.append({"fingerprint": "background_job_failures", "severity": "warning", "message": f"Background jobs failed {failed_jobs} times in the selected window.", "metadata": {"failures": failed_jobs}})
    return candidates


def _persist_alerts(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not sqlite_enabled():
        return [{**item, "status": "active", "first_seen_at": None, "last_seen_at": None} for item in candidates]
    now = _now()
    active = {item["fingerprint"] for item in candidates}
    with connection() as conn:
        for item in candidates:
            conn.execute(
                """INSERT INTO operations_alert_events(fingerprint, severity, message, status, metadata_json, first_seen_at, last_seen_at, resolved_at)
                   VALUES (?, ?, ?, 'active', ?, ?, ?, NULL)
                   ON CONFLICT(fingerprint) DO UPDATE SET severity=excluded.severity, message=excluded.message,
                   status='active', metadata_json=excluded.metadata_json, last_seen_at=excluded.last_seen_at, resolved_at=NULL""",
                (item["fingerprint"], item["severity"], item["message"], json.dumps(item["metadata"]), now, now),
            )
        rows = conn.execute("SELECT fingerprint FROM operations_alert_events WHERE status = 'active'").fetchall()
        for row in rows:
            if str(row["fingerprint"]) not in active:
                conn.execute("UPDATE operations_alert_events SET status='resolved', resolved_at=? WHERE fingerprint=?", (now, row["fingerprint"]))
        results = conn.execute(
            "SELECT fingerprint, severity, message, status, metadata_json, first_seen_at, last_seen_at, resolved_at FROM operations_alert_events ORDER BY status='active' DESC, last_seen_at DESC LIMIT 20"
        ).fetchall()
    alerts: list[dict[str, Any]] = []
    for row in results:
        record = dict(row)
        # 用解析后的 metadata 替换内部的原始 JSON 列，避免冗余字段泄漏进 API 响应。
        record["metadata"] = json.loads(record.pop("metadata_json"))
        alerts.append(record)
    return alerts


def operations_dashboard(*, days: int = 7) -> dict[str, Any]:
    http = observability_summary(days=days)
    provider_failures = _provider_failures(days)
    jobs = _job_summary(days)
    alerts = _persist_alerts(_rules(http, provider_failures, jobs))
    return {"window_days": days, "generated_at": _now(), "http": http, "provider_failures": provider_failures, "jobs": jobs, "alerts": alerts}
