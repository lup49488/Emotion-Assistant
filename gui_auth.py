from __future__ import annotations

from auth_store import MIN_PASSPHRASE_LENGTH, change_access_key, has_access_key, verify_access


AUTH_REQUIRED_MESSAGE = (
    "请输入 User ID 和访问密码后再访问当前用户的数据。"
    f"首次使用某个 User ID 时，请输入至少 {MIN_PASSPHRASE_LENGTH} 位访问密码完成初始化。"
)


def authorize(user_id: str, access_key: str) -> tuple[str, str | None]:
    """校验 user_id + 访问密码；首次使用某个用户名会用输入的密码建立密钥。"""
    user_id = user_id or "local"
    try:
        ok, message = verify_access(user_id, access_key)
    except Exception as exc:
        return user_id, f"访问验证失败：{exc}"
    return user_id, (None if ok else message)


def authorize_or_message(user_id: str, access_key: str) -> tuple[str | None, str | None]:
    user_id = user_id or "local"
    if not (access_key or "").strip():
        return None, AUTH_REQUIRED_MESSAGE
    authorized_user, auth_error = authorize(user_id, access_key)
    if auth_error:
        return None, auth_error
    return authorized_user, None


def save_or_verify_access_key(user_id: str, access_key: str) -> str:
    user_id = user_id or "local"
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
    user_id = user_id or "local"
    try:
        ok, message = change_access_key(user_id, current_access_key, new_access_key)
    except Exception as exc:
        return f"修改访问密码失败：{exc}"
    return message
