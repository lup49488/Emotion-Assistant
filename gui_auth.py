from __future__ import annotations

from auth_store import (
    MIN_PASSPHRASE_LENGTH,
    admin_recovery_enabled,
    admin_reset_access_key,
    change_access_key,
    has_access_key,
    verify_access,
)
from gui_i18n import localize_status_text


AUTH_REQUIRED_MESSAGE = (
    "请输入 User ID 和访问密码后再访问当前用户的数据。"
    f"首次使用某个 User ID 时，请输入至少 {MIN_PASSPHRASE_LENGTH} 位访问密码完成初始化。"
)


def _localized(message: str | None, locale: str | None) -> str | None:
    if message is None or locale is None:
        return message
    return localize_status_text(message, locale)


def authorize(user_id: str, access_key: str, locale: str | None = None) -> tuple[str, str | None]:
    """校验 user_id + 访问密码；首次使用某个用户名会用输入的密码建立密钥。"""
    user_id = (user_id or "").strip()
    if not user_id:
        return "", _localized(AUTH_REQUIRED_MESSAGE, locale)
    try:
        ok, message = verify_access(user_id, access_key)
    except Exception as exc:
        return user_id, _localized(f"访问验证失败：{exc}", locale)
    return user_id, (None if ok else _localized(message, locale))


def authorize_or_message(
    user_id: str, access_key: str, locale: str | None = None
) -> tuple[str | None, str | None]:
    user_id = (user_id or "").strip()
    if not user_id or not (access_key or "").strip():
        return None, _localized(AUTH_REQUIRED_MESSAGE, locale)
    authorized_user, auth_error = authorize(user_id, access_key, locale)
    if auth_error:
        return None, auth_error
    return authorized_user, None


def save_or_verify_access_key(user_id: str, access_key: str) -> str:
    user_id = (user_id or "").strip()
    if not user_id:
        return AUTH_REQUIRED_MESSAGE
    try:
        already_exists = has_access_key(user_id)
        ok, message = verify_access(user_id, access_key)
    except Exception as exc:
        return f"访问密码保存/验证失败：{exc}"
    if not ok:
        return message
    if already_exists:
        return f"用户 {user_id} 的访问密码验证通过。"
    return f"已为用户 {user_id} 保存访问密码。之后访问该用户数据时需要输入同样的密码。"


def change_saved_access_key(user_id: str, current_access_key: str, new_access_key: str) -> str:
    user_id = (user_id or "").strip()
    if not user_id:
        return AUTH_REQUIRED_MESSAGE
    try:
        ok, message = change_access_key(user_id, current_access_key, new_access_key)
    except Exception as exc:
        return f"修改访问密码失败：{exc}"
    return message


def admin_recover_access_key(
    user_id: str,
    admin_recovery_key: str,
    new_access_key: str,
) -> tuple[bool, str]:
    user_id = (user_id or "").strip()
    if not user_id:
        return False, "请输入需要恢复的 User ID。"
    try:
        return admin_reset_access_key(user_id, admin_recovery_key, new_access_key)
    except Exception as exc:
        return False, f"管理员恢复失败：{exc}"


def admin_recovery_status() -> str:
    if admin_recovery_enabled():
        return "管理员恢复模式已启用。"
    return "管理员恢复模式未启用。"
