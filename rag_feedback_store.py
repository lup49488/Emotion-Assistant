"""Durable, user-owned feedback for citations shown with RAG-assisted replies."""
from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from config import BASE_DIR
from json_utils import load_json, save_json
from session_store import validate_user_id
from sqlite_store import connection, ensure_user, sqlite_enabled


_JSON_PATH = BASE_DIR / "data" / "rag_feedback.json"
_LOCK = threading.RLock()


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _citations(items: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    cleaned: list[dict[str, Any]] = []
    for item in items or []:
        source = str(item.get("source", "")).strip()
        if not source:
            continue
        cleaned.append({
            "source": source[:240],
            "chunk_index": max(0, int(item.get("chunk_index", 0))),
            "score": round(float(item.get("score", 0.0)), 3),
            "excerpt": str(item.get("excerpt", "")).strip()[:280],
        })
    return cleaned[:12]


def _records() -> list[dict[str, Any]]:
    data = load_json(_JSON_PATH)
    return data if isinstance(data, list) else []


def _save(records: list[dict[str, Any]]) -> None:
    _JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_json(_JSON_PATH, records[-5_000:])


def create_citation_trace(user_id: str, conversation_id: str | None, citations: list[dict[str, Any]]) -> str | None:
    user_id = validate_user_id(user_id)
    sources = _citations(citations)
    if not sources:
        return None
    trace_id = uuid.uuid4().hex
    record = {
        "id": trace_id,
        "user_id": user_id,
        "conversation_id": (conversation_id or "").strip()[:64],
        "citations": sources,
        "created_at": _now(),
    }
    with _LOCK:
        if sqlite_enabled():
            with connection() as conn:
                ensure_user(conn, user_id)
                conn.execute(
                    "INSERT INTO rag_citation_traces(id, user_id, conversation_id, citations_json, created_at) VALUES (?, ?, ?, ?, ?)",
                    (trace_id, user_id, record["conversation_id"], json.dumps(sources, ensure_ascii=False), record["created_at"]),
                )
        else:
            records = _records()
            records.append(record)
            _save(records)
    return trace_id


def submit_feedback(user_id: str, trace_id: str, helpful: bool, comment: str = "") -> dict[str, Any]:
    user_id = validate_user_id(user_id)
    trace_id = (trace_id or "").strip()
    if not trace_id:
        raise ValueError("Citation trace is required.")
    record = {"helpful": bool(helpful), "comment": (comment or "").strip()[:1_000], "created_at": _now()}
    with _LOCK:
        if sqlite_enabled():
            with connection() as conn:
                trace = conn.execute("SELECT 1 FROM rag_citation_traces WHERE id = ? AND user_id = ?", (trace_id, user_id)).fetchone()
                if trace is None:
                    raise LookupError("Citation trace was not found.")
                conn.execute(
                    "INSERT INTO rag_feedback(trace_id, helpful, comment, created_at) VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(trace_id) DO UPDATE SET helpful=excluded.helpful, comment=excluded.comment, created_at=excluded.created_at",
                    (trace_id, int(record["helpful"]), record["comment"], record["created_at"]),
                )
        else:
            records = _records()
            trace = next((item for item in records if item.get("id") == trace_id and item.get("user_id") == user_id), None)
            if trace is None:
                raise LookupError("Citation trace was not found.")
            trace["feedback"] = record
            _save(records)
    return {"trace_id": trace_id, **record}


def feedback_summary(limit: int = 12) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with _LOCK:
        if sqlite_enabled():
            with connection() as conn:
                rows = [dict(row) for row in conn.execute(
                    "SELECT t.citations_json, f.helpful FROM rag_feedback f JOIN rag_citation_traces t ON t.id = f.trace_id ORDER BY f.created_at DESC"
                ).fetchall()]
        else:
            rows = [{"citations_json": json.dumps(item.get("citations", [])), "helpful": item["feedback"]["helpful"]} for item in _records() if isinstance(item.get("feedback"), dict)]
    total = len(rows)
    helpful = sum(1 for row in rows if bool(row["helpful"]))
    source_feedback: dict[str, dict[str, int]] = {}
    for row in rows:
        for citation in json.loads(row["citations_json"] or "[]"):
            source = str(citation.get("source", "")).strip()
            if source:
                tally = source_feedback.setdefault(source, {"feedback": 0, "unhelpful": 0})
                tally["feedback"] += 1
                tally["unhelpful"] += int(not bool(row["helpful"]))
    sources = [{"source": source, **tally} for source, tally in source_feedback.items()]
    sources.sort(key=lambda item: (-item["unhelpful"], -item["feedback"], item["source"]))
    return {"total": total, "helpful": helpful, "helpful_rate": round((helpful / total) * 100, 1) if total else None, "sources": sources[:max(1, min(limit, 50))]}
