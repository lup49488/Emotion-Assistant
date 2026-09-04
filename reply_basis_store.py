from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from session_store import user_dir, user_file_lock, validate_user_id
from sqlite_store import connection as sqlite_connection, ensure_user, sqlite_enabled


REPLY_BASIS_PREFERENCE_KEY = "reply_basis"
REPLY_BASIS_MODES = {"supportive", "steady", "encouraging"}
DEFAULT_REPLY_BASIS_PREFERENCE = {"enabled": False, "correction": None}


def _normalize(value: Any) -> dict[str, bool | str | None]:
    raw = value if isinstance(value, dict) else {}
    correction = raw.get("correction")
    return {
        "enabled": bool(raw.get("enabled", False)),
        "correction": correction if correction in REPLY_BASIS_MODES else None,
    }


def _read_json_preferences(user_id: str) -> dict[str, Any]:
    path = user_dir(user_id) / "preferences.json"
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def get_reply_basis_preference(user_id: str) -> dict[str, bool | str | None]:
    user_id = validate_user_id(user_id)
    if sqlite_enabled():
        with user_file_lock(user_id):
            with sqlite_connection() as conn:
                row = conn.execute(
                    "SELECT value FROM user_preferences WHERE user_id = ? AND preference_key = ?",
                    (user_id, REPLY_BASIS_PREFERENCE_KEY),
                ).fetchone()
        try:
            return _normalize(json.loads(row["value"]) if row else None)
        except (TypeError, json.JSONDecodeError):
            return dict(DEFAULT_REPLY_BASIS_PREFERENCE)

    try:
        with user_file_lock(user_id):
            return _normalize(_read_json_preferences(user_id).get(REPLY_BASIS_PREFERENCE_KEY))
    except OSError:
        # A reply preference is optional; a legacy read-only user directory must
        # never prevent a chat response from being generated.
        return dict(DEFAULT_REPLY_BASIS_PREFERENCE)


def set_reply_basis_preference(
    user_id: str, enabled: bool, correction: str | None = None,
) -> dict[str, bool | str | None]:
    user_id = validate_user_id(user_id)
    preference = _normalize({"enabled": enabled, "correction": correction})
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
                    (REPLY_BASIS_PREFERENCE_KEY, json.dumps(preference), now),
                )
        return preference

    with user_file_lock(user_id):
        path = user_dir(user_id) / "preferences.json"
        preferences = _read_json_preferences(user_id)
        preferences[REPLY_BASIS_PREFERENCE_KEY] = preference
        path.write_text(json.dumps(preferences, ensure_ascii=False, indent=2), encoding="utf-8")
    return preference
