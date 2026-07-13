from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from datetime import datetime
from pathlib import Path
from typing import Any

from session_store import user_dir, user_file_lock, validate_user_id

PBKDF2_ITERATIONS = 200_000
MIN_PASSPHRASE_LENGTH = 8


class AccessKeyRecordError(ValueError):
    pass


def _auth_path(user_id: str) -> Path:
    return user_dir(user_id) / "access_key.json"


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
    return _auth_path(validate_user_id(user_id)).exists()


def verify_access(user_id: str, passphrase: str) -> tuple[bool, str]:
    """
    首次访问某个 user_id 时会用当前输入的密码为它建立访问密钥；
    此后必须提供同一个密码才能再次读写该用户的数据，
    防止任意输入别人的用户名就看到其对话、情绪和 Mood 记录。
    """
    user_id = validate_user_id(user_id)
    passphrase = (passphrase or "").strip()

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
