from __future__ import annotations

import session_store
from auth_store import admin_reset_access_key, change_access_key, has_access_key, verify_access


def test_first_access_registers_passphrase(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")

    ok, message = verify_access("alice", "secret123")

    assert ok is True
    assert "已为该用户名设置" in message
    assert has_access_key("alice") is True


def test_first_access_rejects_short_passphrase(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")

    ok, message = verify_access("alice", "abc1234")

    assert ok is False
    assert has_access_key("alice") is False


def test_correct_passphrase_is_accepted_on_repeat_access(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")
    verify_access("alice", "secret123")

    ok, message = verify_access("alice", "secret123")

    assert ok is True
    assert message == "验证通过。"


def test_passphrase_supports_symbols_and_unicode(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")
    passphrase = "Aa1!@#中文_+-=安全"

    ok, _ = verify_access("alice", passphrase)
    repeat_ok, message = verify_access("alice", passphrase)

    assert ok is True
    assert repeat_ok is True
    assert message == "验证通过。"


def test_wrong_passphrase_is_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")
    verify_access("alice", "secret123")

    ok, message = verify_access("alice", "wrong-pass")

    assert ok is False
    assert "不正确" in message


def test_change_access_key_requires_current_passphrase(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")
    verify_access("alice", "secret123")

    ok, message = change_access_key("alice", "wrong-pass", "better-pass!")

    assert ok is False
    assert "当前访问密码不正确" in message
    assert verify_access("alice", "secret123")[0] is True


def test_change_access_key_replaces_old_passphrase(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")
    verify_access("alice", "secret123")

    ok, message = change_access_key("alice", "secret123", "better-pass!")

    assert ok is True
    assert "已修改" in message
    assert verify_access("alice", "secret123")[0] is False
    assert verify_access("alice", "better-pass!")[0] is True


def test_change_access_key_rejects_short_new_passphrase(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")
    verify_access("alice", "secret123")

    ok, message = change_access_key("alice", "secret123", "short")

    assert ok is False
    assert "至少" in message
    assert verify_access("alice", "secret123")[0] is True


def test_other_user_cannot_read_alice_data_without_her_passphrase(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")
    verify_access("alice", "alice-secret")

    ok, _ = verify_access("alice", "someone-else-guess")

    assert ok is False


def test_corrupt_access_key_record_does_not_allow_reset(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")
    verify_access("alice", "alice-secret")
    access_path = session_store.user_dir("alice") / "access_key.json"
    access_path.write_text("{broken json", encoding="utf-8")

    ok, message = verify_access("alice", "new-secret")

    assert ok is False
    assert "已损坏" in message
    assert access_path.read_text(encoding="utf-8") == "{broken json"


def test_admin_recovery_is_disabled_without_server_key(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")
    monkeypatch.delenv("CHATBOT_ADMIN_RECOVERY_KEY", raising=False)
    verify_access("alice", "alice-secret")

    ok, message = admin_reset_access_key("alice", "guess", "new-secret!")

    assert ok is False
    assert "未启用" in message
    assert verify_access("alice", "alice-secret")[0] is True


def test_admin_recovery_rejects_wrong_key(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")
    monkeypatch.setenv("CHATBOT_ADMIN_RECOVERY_KEY", "server-recovery-key-12345")
    verify_access("alice", "alice-secret")

    ok, message = admin_reset_access_key("alice", "wrong-key", "new-secret!")

    assert ok is False
    assert "不正确" in message
    assert verify_access("alice", "alice-secret")[0] is True


def test_admin_recovery_resets_existing_password_and_writes_safe_audit(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")
    recovery_key = "server-recovery-key-12345"
    monkeypatch.setenv("CHATBOT_ADMIN_RECOVERY_KEY", recovery_key)
    verify_access("alice", "alice-secret")

    ok, message = admin_reset_access_key("alice", recovery_key, "new-secret!")

    assert ok is True
    assert "已重置" in message
    assert verify_access("alice", "alice-secret")[0] is False
    assert verify_access("alice", "new-secret!")[0] is True
    audit_text = (session_store.user_dir("alice") / "auth_audit.json").read_text(encoding="utf-8")
    assert "admin_password_reset" in audit_text
    assert recovery_key not in audit_text
    assert "new-secret!" not in audit_text


def test_admin_recovery_does_not_create_unknown_account(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")
    recovery_key = "server-recovery-key-12345"
    monkeypatch.setenv("CHATBOT_ADMIN_RECOVERY_KEY", recovery_key)

    ok, message = admin_reset_access_key("missing", recovery_key, "new-secret!")

    assert ok is False
    assert "尚未设置" in message
    assert has_access_key("missing") is False
