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


def _legacy_message_id(conversation_id: str, position: int, message: dict[str, Any]) -> str:
    fingerprint = ":".join(
        (conversation_id, str(position), str(message.get("role", "")), str(message.get("created_at", "")), str(message.get("content", "")))
    )
    return uuid.uuid5(uuid.NAMESPACE_URL, f"serenova-message:{fingerprint}").hex


def _normalized_message(conversation_id: str, position: int, message: dict[str, Any]) -> dict[str, Any]:
    result = dict(message)
    result["id"] = str(result.get("id") or _legacy_message_id(conversation_id, position, result))
    reply_to = str(result.get("reply_to_message_id") or "").strip()
    if reply_to:
        result["reply_to_message_id"] = reply_to
    else:
        result.pop("reply_to_message_id", None)
    return result


def _normalized_messages(conversation_id: str, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [_normalized_message(conversation_id, position, message) for position, message in enumerate(messages)]


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
                SELECT message_id AS id, role, content, created_at, reply_to_message_id FROM conversation_messages
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
                result = dict(record)
                raw_messages = result.get("messages", [])
                result["messages"] = _normalized_messages(conversation_id, raw_messages if isinstance(raw_messages, list) else [])
                return result
    return None


def ensure_conversation(user_id: str, conversation_id: str | None, first_message: str = "") -> dict[str, Any]:
    existing = get_conversation(user_id, conversation_id or "")
    return existing if existing is not None else create_conversation(user_id, _title_from_text(first_message))


def append_exchange(
    user_id: str,
    conversation_id: str | None,
    user_text: str,
    assistant_text: str,
    *,
    reply_to_message_id: str | None = None,
) -> str:
    record = ensure_conversation(user_id, conversation_id, user_text)
    conversation_id = str(record["id"])
    timestamp = _now()
    quote_id = str(reply_to_message_id or "").strip() or None
    messages = [
        {
            "id": uuid.uuid4().hex,
            "role": "user",
            "content": str(user_text),
            "created_at": timestamp,
            "reply_to_message_id": quote_id,
        },
        {"id": uuid.uuid4().hex, "role": "assistant", "content": str(assistant_text), "created_at": timestamp},
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
                INSERT INTO conversation_messages(conversation_id, message_id, position, role, content, created_at, reply_to_message_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (conversation_id, item["id"], start + index, item["role"], item["content"], timestamp, item.get("reply_to_message_id"))
                    for index, item in enumerate(messages)
                ],
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


def get_conversation_message(user_id: str, conversation_id: str | None, message_id: str | None) -> dict[str, Any] | None:
    """Return one owned message for a quote, never trusting client-provided text."""
    user_id = validate_user_id(user_id)
    conversation_id = (conversation_id or "").strip()
    message_id = (message_id or "").strip()
    if not conversation_id or not message_id:
        return None
    if sqlite_enabled():
        with sqlite_connection() as conn:
            row = conn.execute(
                """
                SELECT m.message_id AS id, m.role, m.content, m.created_at, m.reply_to_message_id
                FROM conversation_messages m
                JOIN conversations c ON c.id = m.conversation_id
                WHERE c.user_id = ? AND m.conversation_id = ? AND m.message_id = ?
                """,
                (user_id, conversation_id, message_id),
            ).fetchone()
        return dict(row) if row is not None else None
    record = get_conversation(user_id, conversation_id)
    if record is None:
        return None
    return next((message for message in record["messages"] if message["id"] == message_id), None)


def last_exchange_message_ids(user_id: str, conversation_id: str | None, user_text: str) -> dict[str, str] | None:
    """Return the IDs just archived for one turn so streaming clients can quote it immediately."""
    record = get_conversation(user_id, conversation_id or "")
    messages = record.get("messages", []) if record else []
    if len(messages) < 2:
        return None
    user_message, assistant_message = messages[-2:]
    if user_message.get("role") != "user" or assistant_message.get("role") != "assistant":
        return None
    if str(user_message.get("content", "")) != user_text:
        return None
    return {"user_message_id": str(user_message["id"]), "assistant_message_id": str(assistant_message["id"])}


def remove_last_exchange(user_id: str, conversation_id: str | None, user_text: str) -> bool:
    """Remove the last matching user/assistant pair before a retry."""
    user_id = validate_user_id(user_id)
    conversation_id = (conversation_id or "").strip()
    if not conversation_id:
        return False
    if sqlite_enabled():
        with sqlite_connection() as conn:
            rows = conn.execute(
                """
                SELECT m.id, m.role, m.content FROM conversation_messages m
                JOIN conversations c ON c.id = m.conversation_id
                WHERE m.conversation_id = ? AND c.user_id = ?
                ORDER BY m.position DESC, m.id DESC LIMIT 2
                """,
                (conversation_id, user_id),
            ).fetchall()
            if len(rows) != 2 or rows[0]["role"] != "assistant" or rows[1]["role"] != "user":
                return False
            if str(rows[1]["content"]) != user_text:
                return False
            conn.executemany("DELETE FROM conversation_messages WHERE id = ?", [(rows[0]["id"],), (rows[1]["id"],)])
        return True
    with user_file_lock(user_id):
        conversations = _load_json_conversations(user_id)
        for record in conversations:
            if str(record.get("id")) != conversation_id:
                continue
            messages = record.get("messages", [])
            if not isinstance(messages, list) or len(messages) < 2:
                return False
            user_message, assistant_message = messages[-2:]
            if user_message.get("role") != "user" or assistant_message.get("role") != "assistant":
                return False
            if str(user_message.get("content", "")) != user_text:
                return False
            record["messages"] = messages[:-2]
            record["updated_at"] = _now()
            _save_json_conversations(user_id, conversations)
            return True
    return False


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


def rename_conversation(user_id: str, conversation_id: str, title: str) -> dict[str, Any] | None:
    """Rename one conversation owned by the authenticated user."""
    user_id = validate_user_id(user_id)
    conversation_id = (conversation_id or "").strip()
    new_title = _title_from_text(title)
    if not conversation_id:
        return None
    timestamp = _now()
    if sqlite_enabled():
        with sqlite_connection() as conn:
            result = conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ? AND user_id = ?",
                (new_title, timestamp, conversation_id, user_id),
            )
        if result.rowcount <= 0:
            return None
        return get_conversation(user_id, conversation_id)
    with user_file_lock(user_id):
        conversations = _load_json_conversations(user_id)
        for record in conversations:
            if str(record.get("id")) == conversation_id:
                record["title"] = new_title
                record["updated_at"] = timestamp
                _save_json_conversations(user_id, conversations)
                return _summary(record)
    return None


def restore_conversations(user_id: str, records: list[dict[str, Any]], *, mode: str) -> int:
    """Restore validated exported conversations, preserving their timestamps and messages."""
    user_id = validate_user_id(user_id)
    if mode not in {"merge", "replace"}:
        raise ValueError("Import mode must be merge or replace.")
    restored = [
        {
            "id": str(item["id"]),
            "title": _title_from_text(str(item.get("title", "New conversation"))),
            "created_at": str(item.get("created_at") or _now()),
            "updated_at": str(item.get("updated_at") or _now()),
            "messages": _normalized_messages(str(item["id"]), [dict(message) for message in item.get("messages", [])]),
        }
        for item in records
    ]
    if sqlite_enabled():
        with user_file_lock(user_id):
            with sqlite_connection() as conn:
                ensure_user(conn, user_id)
                if mode == "replace":
                    conn.execute("DELETE FROM conversations WHERE user_id = ?", (user_id,))
                existing = {
                    row["id"] for row in conn.execute(
                        "SELECT id FROM conversations WHERE user_id = ?", (user_id,)
                    ).fetchall()
                }
                added = 0
                for record in restored:
                    if record["id"] in existing:
                        continue
                    conn.execute(
                        "INSERT INTO conversations(id, user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
                        (record["id"], user_id, record["title"], record["created_at"], record["updated_at"]),
                    )
                    conn.executemany(
                        "INSERT INTO conversation_messages(conversation_id, message_id, position, role, content, created_at, reply_to_message_id) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        [
                            (record["id"], message["id"], position, message["role"], message["content"], message["created_at"], message.get("reply_to_message_id"))
                            for position, message in enumerate(record["messages"])
                        ],
                    )
                    existing.add(record["id"])
                    added += 1
        return added

    with user_file_lock(user_id):
        existing_records = [] if mode == "replace" else _load_json_conversations(user_id)
        existing_ids = {str(item.get("id", "")) for item in existing_records}
        additions = [record for record in restored if record["id"] not in existing_ids]
        _save_json_conversations(user_id, [*existing_records, *additions])
    return len(additions)
