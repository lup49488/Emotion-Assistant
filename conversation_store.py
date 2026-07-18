from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from session_store import user_dir, user_file_lock, validate_user_id
from sqlite_store import connection as sqlite_connection, ensure_user, sqlite_enabled


logger = logging.getLogger(__name__)
_MAX_TITLE_LENGTH = 80


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _json_path(user_id: str) -> Path:
    return user_dir(user_id) / "conversations.json"


def _load_json_conversations(user_id: str) -> list[dict[str, Any]]:
    path = _json_path(user_id)
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("Unable to read conversation archive for user %s.", user_id)
        return []
    return raw if isinstance(raw, list) else []


def _save_json_conversations(user_id: str, conversations: list[dict[str, Any]]) -> None:
    path = _json_path(user_id)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(conversations, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def _title_from_text(text: str) -> str:
    compact = " ".join((text or "").split())
    return (compact[:_MAX_TITLE_LENGTH] or "New conversation")


def _summary(record: dict[str, Any]) -> dict[str, Any]:
    messages = record.get("messages", [])
    return {
        "id": str(record.get("id", "")),
        "title": str(record.get("title", "New conversation")),
        "created_at": str(record.get("created_at", "")),
        "updated_at": str(record.get("updated_at", "")),
        "message_count": len(messages) if isinstance(messages, list) else 0,
    }


def create_conversation(user_id: str, title: str = "New conversation") -> dict[str, Any]:
    user_id = validate_user_id(user_id)
    timestamp = _now()
    record = {
        "id": uuid.uuid4().hex,
        "title": _title_from_text(title),
        "created_at": timestamp,
        "updated_at": timestamp,
        "messages": [],
    }
    if sqlite_enabled():
        with sqlite_connection() as conn:
            ensure_user(conn, user_id)
            conn.execute(
                """
                INSERT INTO conversations(id, user_id, title, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (record["id"], user_id, record["title"], timestamp, timestamp),
            )
        return _summary(record)
    with user_file_lock(user_id):
        conversations = _load_json_conversations(user_id)
        conversations.append(record)
        _save_json_conversations(user_id, conversations)
    return _summary(record)


def list_conversations(user_id: str) -> list[dict[str, Any]]:
    user_id = validate_user_id(user_id)
    if sqlite_enabled():
        with sqlite_connection() as conn:
            rows = conn.execute(
                """
                SELECT c.id, c.title, c.created_at, c.updated_at, COUNT(m.id) AS message_count
                FROM conversations c
                LEFT JOIN conversation_messages m ON m.conversation_id = c.id
                WHERE c.user_id = ?
                GROUP BY c.id
                ORDER BY c.updated_at DESC
                """,
                (user_id,),
            ).fetchall()
        return [dict(row) for row in rows]
    with user_file_lock(user_id):
        conversations = _load_json_conversations(user_id)
    return sorted((_summary(item) for item in conversations), key=lambda item: item["updated_at"], reverse=True)


def get_conversation(user_id: str, conversation_id: str) -> dict[str, Any] | None:
    user_id = validate_user_id(user_id)
    conversation_id = (conversation_id or "").strip()
    if not conversation_id:
        return None
    if sqlite_enabled():
        with sqlite_connection() as conn:
            record = conn.execute(
                "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ? AND user_id = ?",
                (conversation_id, user_id),
            ).fetchone()
            if record is None:
                return None
            messages = conn.execute(
                """
                SELECT role, content, created_at FROM conversation_messages
                WHERE conversation_id = ? ORDER BY position, id
                """,
                (conversation_id,),
            ).fetchall()
        result = dict(record)
        result["messages"] = [dict(message) for message in messages]
        return result
    with user_file_lock(user_id):
        for record in _load_json_conversations(user_id):
            if str(record.get("id")) == conversation_id:
                return dict(record)
    return None


def ensure_conversation(user_id: str, conversation_id: str | None, first_message: str = "") -> dict[str, Any]:
    existing = get_conversation(user_id, conversation_id or "")
    return existing if existing is not None else create_conversation(user_id, _title_from_text(first_message))


def append_exchange(user_id: str, conversation_id: str | None, user_text: str, assistant_text: str) -> str:
    record = ensure_conversation(user_id, conversation_id, user_text)
    conversation_id = str(record["id"])
    timestamp = _now()
    messages = [
        {"role": "user", "content": str(user_text), "created_at": timestamp},
        {"role": "assistant", "content": str(assistant_text), "created_at": timestamp},
    ]
    if sqlite_enabled():
        with sqlite_connection() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(position), -1) AS position FROM conversation_messages WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()
            start = int(row["position"]) + 1
            conn.executemany(
                """
                INSERT INTO conversation_messages(conversation_id, position, role, content, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                [(conversation_id, start + index, item["role"], item["content"], timestamp) for index, item in enumerate(messages)],
            )
            conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (timestamp, conversation_id))
        return conversation_id
    with user_file_lock(user_id):
        conversations = _load_json_conversations(user_id)
        for item in conversations:
            if str(item.get("id")) == conversation_id:
                item.setdefault("messages", []).extend(messages)
                item["updated_at"] = timestamp
                break
        _save_json_conversations(user_id, conversations)
    return conversation_id


def delete_conversation(user_id: str, conversation_id: str) -> bool:
    user_id = validate_user_id(user_id)
    if sqlite_enabled():
        with sqlite_connection() as conn:
            result = conn.execute("DELETE FROM conversations WHERE id = ? AND user_id = ?", (conversation_id, user_id))
        return result.rowcount > 0
    with user_file_lock(user_id):
        conversations = _load_json_conversations(user_id)
        remaining = [item for item in conversations if str(item.get("id")) != conversation_id]
        if len(remaining) == len(conversations):
            return False
        _save_json_conversations(user_id, remaining)
    return True
