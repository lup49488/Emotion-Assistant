from __future__ import annotations

import pytest

import config
import session_store
from auth_store import verify_access
from email_auth_store import EmailAuthError, EmailAuthService
from sqlite_store import connection


def _service(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")
    monkeypatch.setenv("STORAGE_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DATABASE_PATH", str(tmp_path / "data" / "chatbot.db"))
    monkeypatch.setattr(config, "EMAIL_AUTH_ENABLED", True)
    monkeypatch.setattr(config, "EMAIL_AUTH_PROVIDER", "resend")
    monkeypatch.setattr(config, "EMAIL_AUTH_FROM", "Serenova <no-reply@example.test>")
    monkeypatch.setattr(config, "EMAIL_AUTH_RESEND_API_KEY", "test-email-secret")
    monkeypatch.setattr(config, "EMAIL_AUTH_TURNSTILE_REQUIRED", False)
    sent: list[tuple[str, str]] = []
    return EmailAuthService(b"unit-test-intent-secret", send_code=lambda email, code: sent.append((email, code))), sent


def test_email_registration_creates_an_opaque_account_and_one_time_password_login(tmp_path, monkeypatch):
    service, sent = _service(tmp_path, monkeypatch)

    challenge_id = service.start_challenge("Student@Example.Test", "registration", "", "127.0.0.1", "register-1")
    intent = service.verify_challenge(challenge_id, sent[-1][1], "registration", "register-1")
    user_id = service.register(intent, "correct-horse-battery", "register-1")

    assert user_id.startswith("usr_")
    assert service.login_password("student@example.test", "correct-horse-battery", "login-1") == user_id
    assert service.credential_version(user_id)
    with pytest.raises(EmailAuthError, match="expired"):
        service.register(intent, "another-correct-password", "register-2")


def test_email_password_accepts_eight_characters_and_rejects_shorter_values(tmp_path, monkeypatch):
    service, _sent = _service(tmp_path, monkeypatch)

    assert service._new_password_hash("eight123")
    with pytest.raises(EmailAuthError, match="between 8 and 512"):
        service._new_password_hash("seven77")


def test_legacy_migration_binds_existing_user_without_replacing_its_id(tmp_path, monkeypatch):
    service, sent = _service(tmp_path, monkeypatch)
    assert verify_access("legacy-user", "legacy-access-key")[0] is True

    challenge_id = service.start_challenge("legacy@example.test", "legacy_migration", "", "127.0.0.1", "migrate-1")
    intent = service.verify_challenge(challenge_id, sent[-1][1], "legacy_migration", "migrate-1")
    migrated = service.migrate_legacy(intent, "legacy-user", "legacy-access-key", "new-email-password", "migrate-1")

    assert migrated == "legacy-user"
    assert service.login_password("legacy@example.test", "new-email-password", "login-1") == "legacy-user"
    assert verify_access("legacy-user", "legacy-access-key")[0] is True
    with connection() as conn:
        identity = conn.execute("SELECT email_normalized, legacy_migrated_at FROM email_identities WHERE user_id = ?", ("legacy-user",)).fetchone()
    assert identity["email_normalized"] == "legacy@example.test"
    assert identity["legacy_migrated_at"]


def test_migration_rejects_wrong_legacy_key_without_consuming_the_verified_email(tmp_path, monkeypatch):
    service, sent = _service(tmp_path, monkeypatch)
    assert verify_access("legacy-user", "legacy-access-key")[0] is True
    challenge_id = service.start_challenge("legacy@example.test", "legacy_migration", "", "127.0.0.1", "migrate-1")
    intent = service.verify_challenge(challenge_id, sent[-1][1], "legacy_migration", "migrate-1")

    with pytest.raises(EmailAuthError, match="could not complete"):
        service.migrate_legacy(intent, "legacy-user", "wrong-key", "new-email-password", "migrate-1")
    assert service.migrate_legacy(intent, "legacy-user", "legacy-access-key", "new-email-password", "migrate-2") == "legacy-user"


def test_email_identity_cannot_be_bound_to_two_accounts(tmp_path, monkeypatch):
    service, sent = _service(tmp_path, monkeypatch)
    first = service.start_challenge("only@example.test", "registration", "", "127.0.0.1", "one")
    first_intent = service.verify_challenge(first, sent[-1][1], "registration", "one")
    service.register(first_intent, "correct-horse-battery", "one")

    second = service.start_challenge("only@example.test", "registration", "", "127.0.0.1", "two")
    second_intent = service.verify_challenge(second, sent[-1][1], "registration", "two")
    with pytest.raises(EmailAuthError, match="could not create"):
        service.register(second_intent, "another-correct-pass", "two")


def test_turnstile_failure_prevents_code_delivery(tmp_path, monkeypatch):
    service, sent = _service(tmp_path, monkeypatch)
    monkeypatch.setattr(config, "EMAIL_AUTH_TURNSTILE_REQUIRED", True)
    monkeypatch.setattr(config, "TURNSTILE_SITE_KEY", "test-site-key")
    monkeypatch.setattr(config, "TURNSTILE_SECRET_KEY", "test-turnstile-secret")
    protected = EmailAuthService(b"unit-test-intent-secret", send_code=lambda email, code: sent.append((email, code)), validate_turnstile=lambda _token, _ip: False)

    with pytest.raises(EmailAuthError, match="could not be completed"):
        protected.start_challenge("student@example.test", "registration", "bad-token", "127.0.0.1", "turnstile-1")
    assert sent == []
