"""Persistent, serialized background jobs for long-running maintenance work."""
from __future__ import annotations

import json
import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from config import BASE_DIR
from json_utils import load_json, save_json
from sqlite_store import connection, ensure_user, sqlite_enabled


JOB_STATUSES = {"queued", "running", "succeeded", "failed", "cancelled"}
_JSON_PATH = BASE_DIR / "data" / "background_jobs.json"
_LOCK = threading.RLock()
logger = logging.getLogger(__name__)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _public_job(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if key != "payload"}


def _row_to_job(row: Any) -> dict[str, Any]:
    return {
        "id": str(row["id"]), "user_id": str(row["user_id"]), "kind": str(row["kind"]),
        "status": str(row["status"]), "progress": int(row["progress"]),
        "message": str(row["message"] or ""),
        "result": json.loads(row["result_json"]) if row["result_json"] else None,
        "error": str(row["error"] or "") or None,
        "created_at": str(row["created_at"]), "started_at": row["started_at"],
        "finished_at": row["finished_at"], "payload": json.loads(row["payload_json"] or "{}"),
    }


def _json_records() -> list[dict[str, Any]]:
    return [item for item in load_json(_JSON_PATH) if isinstance(item, dict)]


def _save_json_records(records: list[dict[str, Any]]) -> None:
    _JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_json(_JSON_PATH, records[-200:])


def create_job(user_id: str, kind: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    record = {
        "id": uuid.uuid4().hex, "user_id": user_id, "kind": kind, "status": "queued",
        "progress": 0, "message": "Queued", "result": None, "error": None,
        "created_at": _now(), "started_at": None, "finished_at": None, "payload": payload or {},
    }
    with _LOCK:
        if sqlite_enabled():
            with connection() as conn:
                ensure_user(conn, user_id)
                conn.execute(
                    """
                    INSERT INTO background_jobs(
                        id, user_id, kind, status, progress, message, payload_json, result_json,
                        error, created_at, started_at, finished_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (record["id"], user_id, kind, "queued", 0, record["message"],
                     json.dumps(record["payload"], ensure_ascii=False), None, None,
                     record["created_at"], None, None),
                )
        else:
            records = _json_records()
            records.append(record)
            _save_json_records(records)
    return _public_job(record)


def get_job(job_id: str, user_id: str | None = None) -> dict[str, Any] | None:
    with _LOCK:
        if sqlite_enabled():
            query = "SELECT * FROM background_jobs WHERE id = ?"
            params: tuple[Any, ...] = (job_id,)
            if user_id is not None:
                query += " AND user_id = ?"
                params = (job_id, user_id)
            with connection() as conn:
                row = conn.execute(query, params).fetchone()
            return _public_job(_row_to_job(row)) if row else None
        for item in reversed(_json_records()):
            if item.get("id") == job_id and (user_id is None or item.get("user_id") == user_id):
                return _public_job(item)
    return None


def list_jobs(user_id: str, limit: int = 30) -> list[dict[str, Any]]:
    with _LOCK:
        if sqlite_enabled():
            with connection() as conn:
                rows = conn.execute(
                    "SELECT * FROM background_jobs WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                    (user_id, limit),
                ).fetchall()
            return [_public_job(_row_to_job(row)) for row in rows]
        records = [item for item in _json_records() if item.get("user_id") == user_id]
        records.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
        return [_public_job(item) for item in records[:limit]]


def _update_job(job_id: str, **changes: Any) -> dict[str, Any] | None:
    allowed = {"status", "progress", "message", "result", "error", "started_at", "finished_at"}
    updates = {key: value for key, value in changes.items() if key in allowed}
    if not updates:
        return get_job(job_id)
    with _LOCK:
        if sqlite_enabled():
            columns: list[str] = []
            values: list[Any] = []
            for key, value in updates.items():
                column = "result_json" if key == "result" else key
                columns.append(f"{column} = ?")
                values.append(json.dumps(value, ensure_ascii=False) if key == "result" and value is not None else value)
            values.append(job_id)
            with connection() as conn:
                conn.execute(f"UPDATE background_jobs SET {', '.join(columns)} WHERE id = ?", values)
            return get_job(job_id)
        records = _json_records()
        for item in records:
            if item.get("id") == job_id:
                item.update(updates)
                _save_json_records(records)
                return _public_job(item)
    return None


def mark_interrupted_jobs() -> int:
    """Make stale queued/running records truthful after a process restart."""
    with _LOCK:
        if sqlite_enabled():
            with connection() as conn:
                cursor = conn.execute(
                    """
                    UPDATE background_jobs
                    SET status = 'failed', error = 'Worker stopped before completion.',
                        message = 'Interrupted by server restart.', finished_at = ?
                    WHERE status IN ('queued', 'running')
                    """,
                    (_now(),),
                )
                return int(cursor.rowcount)
        records = _json_records()
        changed = 0
        for item in records:
            if item.get("status") in {"queued", "running"}:
                item.update({"status": "failed", "error": "Worker stopped before completion.", "message": "Interrupted by server restart.", "finished_at": _now()})
                changed += 1
        if changed:
            _save_json_records(records)
        return changed


class JobContext:
    def __init__(self, job_id: str) -> None:
        self.job_id = job_id

    def progress(self, value: int, message: str) -> None:
        _update_job(self.job_id, progress=max(0, min(100, int(value))), message=message)


JobWorker = Callable[[JobContext], dict[str, Any] | None]


class BackgroundJobManager:
    """A single worker prevents overlapping RAG index mutations."""

    def __init__(self) -> None:
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="serenova-jobs")

    def submit(self, user_id: str, kind: str, worker: JobWorker, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        job = create_job(user_id, kind, payload)
        self._executor.submit(self._run, job["id"], worker)
        return job

    @staticmethod
    def _run(job_id: str, worker: JobWorker) -> None:
        job = get_job(job_id)
        logger.info("event=background_job_started job_id=%s kind=%s", job_id, job.get("kind") if job else "unknown")
        _update_job(job_id, status="running", progress=1, message="Running", started_at=_now(), error=None)
        context = JobContext(job_id)
        try:
            result = worker(context) or {}
        except Exception as exc:
            _update_job(job_id, status="failed", message="Failed", error=str(exc)[:2_000], finished_at=_now())
            logger.exception("event=background_job_failed job_id=%s", job_id)
            return
        _update_job(job_id, status="succeeded", progress=100, message="Completed", result=result, finished_at=_now())
        logger.info("event=background_job_completed job_id=%s", job_id)


job_manager = BackgroundJobManager()
