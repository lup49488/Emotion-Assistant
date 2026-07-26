"""Durable per-user style preference used to filter the style corpus."""
from __future__ import annotations

import json
from datetime import datetime

from session_store import user_dir, user_file_lock, validate_user_id
from sqlite_store import connection as sqlite_connection, ensure_user, sqlite_enabled
from style_store import style_prefixes


STYLE_PREFERENCE_KEY = "style_prefix"


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


def normalize_style_prefix(value: str | None) -> str:
    """Return a stored prefix only when it still matches an available style family.

    An empty string means "use the whole style corpus", which is also the
    fallback when a previously selected style no longer has documents.
    """
    candidate = (value or "").strip()
    if not candidate:
        return ""
    available = {prefix.casefold(): prefix for prefix in style_prefixes()}
    return available.get(candidate.casefold(), "")


def get_style_prefix(user_id: str) -> str:
    user_id = validate_user_id(user_id)
    if sqlite_enabled():
        with user_file_lock(user_id):
            with sqlite_connection() as conn:
                row = conn.execute(
                    "SELECT value FROM user_preferences WHERE user_id = ? AND preference_key = ?",
                    (user_id, STYLE_PREFERENCE_KEY),
                ).fetchone()
        return normalize_style_prefix(row["value"] if row else "")

    with user_file_lock(user_id):
        stored = _read_json_preferences(user_id).get(STYLE_PREFERENCE_KEY)
    return normalize_style_prefix(stored if isinstance(stored, str) else "")


def set_style_prefix(user_id: str, value: str | None) -> str:
    """Persist the selected style family; an unknown or empty value clears it."""
    user_id = validate_user_id(user_id)
    prefix = normalize_style_prefix(value)
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
                    (user_id, STYLE_PREFERENCE_KEY, prefix, now),
                )
        return prefix

    with user_file_lock(user_id):
        path = _json_path(user_id)
        preferences = _read_json_preferences(user_id)
        preferences[STYLE_PREFERENCE_KEY] = prefix
        path.write_text(json.dumps(preferences, ensure_ascii=False, indent=2), encoding="utf-8")
    return prefix
