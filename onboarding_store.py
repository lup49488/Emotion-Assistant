from __future__ import annotations

import json
from datetime import datetime

from session_store import user_dir, user_file_lock, validate_user_id
from sqlite_store import connection as sqlite_connection, ensure_user, sqlite_enabled


ONBOARDING_COMPLETED_KEY = "onboarding_completed"


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


def onboarding_completed(user_id: str) -> bool:
    user_id = validate_user_id(user_id)
    if sqlite_enabled():
        with user_file_lock(user_id):
            with sqlite_connection() as conn:
                row = conn.execute(
                    "SELECT value FROM user_preferences WHERE user_id = ? AND preference_key = ?",
                    (user_id, ONBOARDING_COMPLETED_KEY),
                ).fetchone()
        return bool(row and row["value"] == "true")

    with user_file_lock(user_id):
        return _read_json_preferences(user_id).get(ONBOARDING_COMPLETED_KEY) is True


def mark_onboarding_completed(user_id: str) -> None:
    user_id = validate_user_id(user_id)
    if sqlite_enabled():
        with user_file_lock(user_id):
            with sqlite_connection() as conn:
                ensure_user(conn, user_id)
                conn.execute(
                    """
                    INSERT INTO user_preferences(user_id, preference_key, value, updated_at)
                    VALUES (?, ?, 'true', ?)
                    ON CONFLICT(user_id, preference_key) DO UPDATE SET
                        value = excluded.value, updated_at = excluded.updated_at
                    """,
                    (user_id, ONBOARDING_COMPLETED_KEY, datetime.now().isoformat(timespec="seconds")),
                )
        return

    with user_file_lock(user_id):
        path = _json_path(user_id)
        preferences = _read_json_preferences(user_id)
        preferences[ONBOARDING_COMPLETED_KEY] = True
        path.write_text(json.dumps(preferences, ensure_ascii=False, indent=2), encoding="utf-8")
