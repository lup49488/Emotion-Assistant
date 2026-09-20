from __future__ import annotations

from fastapi.testclient import TestClient

import api_server
import config
import session_store


def test_email_registration_flow_issues_the_existing_signed_session(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")
    monkeypatch.setenv("STORAGE_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DATABASE_PATH", str(tmp_path / "data" / "chatbot.db"))
    monkeypatch.setattr(config, "EMAIL_AUTH_ENABLED", True)
    monkeypatch.setattr(config, "EMAIL_AUTH_PROVIDER", "resend")
    monkeypatch.setattr(config, "EMAIL_AUTH_FROM", "Serenova <no-reply@example.test>")
    monkeypatch.setattr(config, "EMAIL_AUTH_RESEND_API_KEY", "test-email-secret")
    monkeypatch.setattr(config, "EMAIL_AUTH_TURNSTILE_REQUIRED", False)
    sent: list[str] = []
    monkeypatch.setattr(api_server.EMAIL_AUTH_SERVICE, "_send_code", lambda _email, code: sent.append(code))

    with TestClient(api_server.app, base_url="https://testserver") as client:
        config_response = client.get("/api/v1/auth/config")
        start = client.post("/api/v1/auth/email/start", json={"email": "student@example.test", "purpose": "registration", "turnstile_token": ""})
        verified = client.post("/api/v1/auth/email/verify", json={"challenge_id": start.json()["challenge_id"], "code": sent[-1], "purpose": "registration"})
        registered = client.post("/api/v1/auth/register", json={"verified_intent": verified.json()["verified_intent"], "password": "eight123"})
        session = client.get("/api/v1/auth/session")

    assert config_response.json()["email_auth_enabled"] is True
    assert start.status_code == 202
    assert registered.status_code == 200
    assert registered.json()["user_id"].startswith("usr_")
    assert session.status_code == 200
    assert session.json()["user_id"] == registered.json()["user_id"]
