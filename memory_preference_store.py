"""Durable per-user policy for automatic memory writes."""
from __future__ import annotations

import json
from datetime import datetime

from memory_store import DEFAULT_MEMORY_SAVE_MODE, MEMORY_SAVE_MODES
from session_store import user_dir, user_file_lock, validate_user_id
from sqlite_store import connection as sqlite_connection, ensure_user, sqlite_enabled


MEMORY_SAVE_MODE_KEY = "memory_save_mode"


def _json_path(user_id: str):
    return user_dir(user_id) / "preferences.json"


def _read_json_preferences(user_id: str) -> dict[str, object]:
    path = _json_path(user_id)
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def normalize_memory_save_mode(value: str | None) -> str:
    candidate = (value or "").strip().lower()
    return candidate if candidate in MEMORY_SAVE_MODES else DEFAULT_MEMORY_SAVE_MODE


def get_memory_save_mode(user_id: str) -> str:
    user_id = validate_user_id(user_id)
    if sqlite_enabled():
        with user_file_lock(user_id):
            with sqlite_connection() as conn:
                row = conn.execute(
                    "SELECT value FROM user_preferences WHERE user_id = ? AND preference_key = ?",
                    (user_id, MEMORY_SAVE_MODE_KEY),
                ).fetchone()
        return normalize_memory_save_mode(row["value"] if row else None)

    with user_file_lock(user_id):
        value = _read_json_preferences(user_id).get(MEMORY_SAVE_MODE_KEY)
    return normalize_memory_save_mode(value if isinstance(value, str) else None)


def set_memory_save_mode(user_id: str, value: str | None) -> str:
    user_id = validate_user_id(user_id)
    mode = normalize_memory_save_mode(value)
    now = datetime.now().isoformat(timespec="seconds")
    if sqlite_enabled():
        with user_file_lock(user_id):
            with sqlite_connection() as conn:
                ensure_user(conn, user_id)
                conn.execute(
                    """
                    INSERT INTO user_preferences(user_id, preference_key, value, updated_at)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(user_id, preference_key) DO UPDATE SET
                        value = excluded.value, updated_at = excluded.updated_at
                    """,
                    (user_id, MEMORY_SAVE_MODE_KEY, mode, now),
                )
        return mode

    with user_file_lock(user_id):
        path = _json_path(user_id)
        preferences = _read_json_preferences(user_id)
        preferences[MEMORY_SAVE_MODE_KEY] = mode
        path.write_text(json.dumps(preferences, ensure_ascii=False, indent=2), encoding="utf-8")
    return mode
