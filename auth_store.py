from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any

from session_store import user_dir, user_file_lock, validate_user_id
from sqlite_store import (
    append_auth_event as append_sqlite_auth_event,
    connection as sqlite_connection,
    get_auth_credential,
    legacy_import_completed,
    mark_legacy_import_completed,
    save_auth_credential,
    sqlite_enabled,
)

PBKDF2_ITERATIONS = 200_000
MIN_PASSPHRASE_LENGTH = 8
MIN_ADMIN_RECOVERY_KEY_LENGTH = 20
ADMIN_RECOVERY_KEY_ENV = "CHATBOT_ADMIN_RECOVERY_KEY"


class AccessKeyRecordError(ValueError):
    pass


def _auth_path(user_id: str) -> Path:
    return user_dir(user_id) / "access_key.json"


def _auth_audit_path(user_id: str) -> Path:
    return user_dir(user_id) / "auth_audit.json"


def _hash_passphrase(passphrase: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac(
        "sha256", passphrase.encode("utf-8"), salt, PBKDF2_ITERATIONS
    ).hex()


def _read_record(user_id: str) -> dict[str, Any] | None:
    path = _auth_path(user_id)
    if not path.exists():
        return None
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AccessKeyRecordError("访问密钥记录已损坏，请联系管理员处理。") from exc
    if not isinstance(record, dict):
        raise AccessKeyRecordError("访问密钥记录已损坏，请联系管理员处理。")
    return record


def _write_record(user_id: str, record: dict[str, Any]) -> None:
    _auth_path(user_id).write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _append_audit_event(user_id: str, action: str) -> None:
    path = _auth_audit_path(user_id)
    events: list[dict[str, str]] = []
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, list):
                events = [item for item in raw if isinstance(item, dict)][-99:]
        except (OSError, json.JSONDecodeError):
            events = []
    events.append({
        "time": datetime.now().isoformat(),
        "action": action,
    })
    path.write_text(
        json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def _read_legacy_audit_events(user_id: str) -> list[dict[str, str]]:
    path = _auth_audit_path(user_id)
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(raw, list):
        return []
    return [
        {"time": str(item.get("time") or ""), "action": str(item.get("action") or "")}
        for item in raw
        if isinstance(item, dict) and str(item.get("action") or "").strip()
    ]


def _sqlite_record(user_id: str) -> dict[str, Any] | None:
    with sqlite_connection() as conn:
        _migrate_legacy_auth_unlocked(conn, user_id)
        record = get_auth_credential(conn, user_id)
    if record is None:
        return None
    return {
        "salt": record["salt"],
        "hash": record["password_hash"],
        "created_at": record["created_at"],
        "updated_at": record["updated_at"],
        "changed_at": record.get("changed_at"),
    }


def _migrate_legacy_auth_unlocked(conn: Any, user_id: str) -> bool:
    if legacy_import_completed(conn, user_id, "auth_json"):
        return False
    if get_auth_credential(conn, user_id) is not None:
        mark_legacy_import_completed(conn, user_id, "auth_json")
        return False
    record = _read_record(user_id)
    if record is not None:
        try:
            bytes.fromhex(str(record["salt"]))
            str(record["hash"])
            str(record["created_at"])
            str(record["updated_at"])
        except (KeyError, ValueError) as exc:
            raise AccessKeyRecordError("访问密钥记录已损坏，请联系管理员处理。") from exc
        save_auth_credential(conn, user_id, record)
        for event in _read_legacy_audit_events(user_id):
            append_sqlite_auth_event(
                conn, user_id, event["action"], created_at=event["time"] or None
            )
    mark_legacy_import_completed(conn, user_id, "auth_json")
    return record is not None


def migrate_legacy_auth(user_id: str) -> bool:
    """Import one user's legacy auth JSON exactly once when SQLite is enabled."""
    user_id = validate_user_id(user_id)
    if not sqlite_enabled():
        return False
    with user_file_lock(user_id):
        with sqlite_connection() as conn:
            return _migrate_legacy_auth_unlocked(conn, user_id)


def _new_record(passphrase: str, *, previous_created_at: str | None = None) -> dict[str, Any]:
    salt = secrets.token_bytes(16)
    now = datetime.now().isoformat()
    record = {
        "salt": salt.hex(),
        "hash": _hash_passphrase(passphrase, salt),
        "created_at": previous_created_at or now,
        "updated_at": now,
    }
    if previous_created_at:
        record["changed_at"] = now
    return record


def _validate_passphrase_length(passphrase: str) -> str | None:
    if len(passphrase) < MIN_PASSPHRASE_LENGTH:
        return f"访问密码需要至少 {MIN_PASSPHRASE_LENGTH} 位。"
    return None


def has_access_key(user_id: str) -> bool:
    user_id = validate_user_id(user_id)
    if sqlite_enabled():
        with user_file_lock(user_id):
            return _sqlite_record(user_id) is not None
    return _auth_path(user_id).exists()


def verify_access(user_id: str, passphrase: str) -> tuple[bool, str]:
    """
    首次访问某个 user_id 时会用当前输入的密码为它建立访问密钥；
    此后必须提供同一个密码才能再次读写该用户的数据，
    防止任意输入别人的用户名就看到其对话、情绪和 Mood 记录。
    """
    user_id = validate_user_id(user_id)
    passphrase = (passphrase or "").strip()

    if sqlite_enabled():
        return _verify_access_sqlite(user_id, passphrase)

    with user_file_lock(user_id):
        try:
            record = _read_record(user_id)
        except AccessKeyRecordError as exc:
            return False, str(exc)
        if record is None:
            length_error = _validate_passphrase_length(passphrase)
            if length_error:
                return False, f"首次使用该用户名，请设置至少 {MIN_PASSPHRASE_LENGTH} 位的访问密码。"
            _write_record(user_id, _new_record(passphrase))
            return True, "已为该用户名设置访问密码，请妥善保管，之后需要同样的密码才能访问。"

        try:
            salt = bytes.fromhex(record["salt"])
            expected = str(record["hash"])
        except (KeyError, ValueError):
            return False, "访问密钥记录已损坏，请联系管理员处理。"

        if not passphrase:
            return False, "请输入访问密码。"
        if hmac.compare_digest(_hash_passphrase(passphrase, salt), expected):
            return True, "验证通过。"
        return False, "用户名或访问密码不正确。"


def change_access_key(
    user_id: str,
    current_passphrase: str,
    new_passphrase: str,
) -> tuple[bool, str]:
    user_id = validate_user_id(user_id)
    current_passphrase = (current_passphrase or "").strip()
    new_passphrase = (new_passphrase or "").strip()

    if not current_passphrase:
        return False, "请输入当前访问密码。"
    length_error = _validate_passphrase_length(new_passphrase)
    if length_error:
        return False, length_error
    if current_passphrase == new_passphrase:
        return False, "新访问密码不能与当前访问密码相同。"

    if sqlite_enabled():
        return _change_access_key_sqlite(user_id, current_passphrase, new_passphrase)

    with user_file_lock(user_id):
        try:
            record = _read_record(user_id)
        except AccessKeyRecordError as exc:
            return False, str(exc)
        if record is None:
            return False, "该用户名尚未设置访问密码，请先保存当前访问密码。"

        try:
            salt = bytes.fromhex(record["salt"])
            expected = str(record["hash"])
        except (KeyError, ValueError):
            return False, "访问密钥记录已损坏，请联系管理员处理。"

        if not hmac.compare_digest(_hash_passphrase(current_passphrase, salt), expected):
            return False, "当前访问密码不正确。"

        _write_record(
            user_id,
            _new_record(new_passphrase, previous_created_at=str(record.get("created_at") or "")),
        )
        return True, "访问密码已修改，请使用新密码继续访问。"


def admin_recovery_enabled() -> bool:
    recovery_key = os.getenv(ADMIN_RECOVERY_KEY_ENV, "")
    return len(recovery_key) >= MIN_ADMIN_RECOVERY_KEY_LENGTH


def admin_reset_access_key(
    user_id: str,
    admin_recovery_key: str,
    new_passphrase: str,
) -> tuple[bool, str]:
    """使用仅由服务器环境变量提供的恢复密钥重置已有用户密码。"""
    user_id = validate_user_id((user_id or "").strip())
    submitted_key = admin_recovery_key or ""
    new_passphrase = (new_passphrase or "").strip()
    expected_key = os.getenv(ADMIN_RECOVERY_KEY_ENV, "")

    if len(expected_key) < MIN_ADMIN_RECOVERY_KEY_LENGTH:
        return False, "管理员恢复模式未启用，请先在服务器环境变量中配置恢复密钥。"
    if not hmac.compare_digest(submitted_key, expected_key):
        return False, "管理员恢复密钥不正确。"
    length_error = _validate_passphrase_length(new_passphrase)
    if length_error:
        return False, length_error

    if sqlite_enabled():
        return _admin_reset_access_key_sqlite(user_id, new_passphrase)

    with user_file_lock(user_id):
        try:
            record = _read_record(user_id)
        except AccessKeyRecordError as exc:
            return False, str(exc)
        if record is None:
            return False, "该用户尚未设置访问密码，不能执行管理员恢复。"

        _write_record(
            user_id,
            _new_record(new_passphrase, previous_created_at=str(record.get("created_at") or "")),
        )
        _append_audit_event(user_id, "admin_password_reset")
        return True, "管理员恢复成功，访问密码已重置。"


def _verify_access_sqlite(user_id: str, passphrase: str) -> tuple[bool, str]:
    with user_file_lock(user_id):
        record = _sqlite_record(user_id)
        if record is None:
            length_error = _validate_passphrase_length(passphrase)
            if length_error:
                return False, f"首次使用该用户名，请设置至少 {MIN_PASSPHRASE_LENGTH} 位的访问密码。"
            new_record = _new_record(passphrase)
            with sqlite_connection() as conn:
                save_auth_credential(conn, user_id, new_record)
            return True, "已为该用户名设置访问密码，请妥善保管，之后需要同样的密码才能访问。"

        try:
            salt = bytes.fromhex(record["salt"])
            expected = str(record["hash"])
        except (KeyError, ValueError):
            return False, "访问密钥记录已损坏，请联系管理员处理。"
        if not passphrase:
            return False, "请输入访问密码。"
        if hmac.compare_digest(_hash_passphrase(passphrase, salt), expected):
            return True, "验证通过。"
        return False, "用户名或访问密码不正确。"


def _change_access_key_sqlite(
    user_id: str, current_passphrase: str, new_passphrase: str
) -> tuple[bool, str]:
    with user_file_lock(user_id):
        record = _sqlite_record(user_id)
        if record is None:
            return False, "该用户名尚未设置访问密码，请先保存当前访问密码。"
        try:
            salt = bytes.fromhex(record["salt"])
            expected = str(record["hash"])
        except (KeyError, ValueError):
            return False, "访问密钥记录已损坏，请联系管理员处理。"
        if not hmac.compare_digest(_hash_passphrase(current_passphrase, salt), expected):
            return False, "当前访问密码不正确。"
        with sqlite_connection() as conn:
            save_auth_credential(
                conn,
                user_id,
                _new_record(new_passphrase, previous_created_at=str(record.get("created_at") or "")),
            )
        return True, "访问密码已修改，请使用新密码继续访问。"


def _admin_reset_access_key_sqlite(user_id: str, new_passphrase: str) -> tuple[bool, str]:
    with user_file_lock(user_id):
        record = _sqlite_record(user_id)
        if record is None:
            return False, "该用户尚未设置访问密码，不能执行管理员恢复。"
        with sqlite_connection() as conn:
            save_auth_credential(
                conn,
                user_id,
                _new_record(new_passphrase, previous_created_at=str(record.get("created_at") or "")),
            )
            append_sqlite_auth_event(conn, user_id, "admin_password_reset")
        return True, "管理员恢复成功，访问密码已重置。"
