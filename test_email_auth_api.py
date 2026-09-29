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


def test_password_reset_endpoint_invalidates_existing_session(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")
    monkeypatch.setenv("STORAGE_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DATABASE_PATH", str(tmp_path / "data" / "chatbot.db"))
    monkeypatch.setattr(config, "EMAIL_AUTH_ENABLED", True)
    monkeypatch.setattr(config, "EMAIL_AUTH_PROVIDER", "resend")
    monkeypatch.setattr(config, "EMAIL_AUTH_FROM", "Serenova <no-reply@example.test>")
    monkeypatch.setattr(config, "EMAIL_AUTH_RESEND_API_KEY", "test-email-secret")
    monkeypatch.setattr(config, "EMAIL_AUTH_TURNSTILE_REQUIRED", False)
    sent = []
    monkeypatch.setattr(api_server.EMAIL_AUTH_SERVICE, "_send_code", lambda _email, code: sent.append(code))

    with TestClient(api_server.app, base_url="https://testserver") as client:
        registration = client.post("/api/v1/auth/email/start", json={"email": "student@example.test", "purpose": "registration"})
        verified = client.post("/api/v1/auth/email/verify", json={"challenge_id": registration.json()["challenge_id"], "code": sent[-1], "purpose": "registration"})
        registered = client.post("/api/v1/auth/register", json={"verified_intent": verified.json()["verified_intent"], "password": "old-password"})
        assert client.get("/api/v1/auth/session").status_code == 200
        reset = client.post("/api/v1/auth/email/start", json={"email": "student@example.test", "purpose": "password_reset"})
        reset_verified = client.post("/api/v1/auth/email/verify", json={"challenge_id": reset.json()["challenge_id"], "code": sent[-1], "purpose": "password_reset"})
        changed = client.post("/api/v1/auth/password/reset", json={"verified_intent": reset_verified.json()["verified_intent"], "password": "new-password"})
        assert changed.status_code == 204
        assert client.get("/api/v1/auth/session").status_code == 401
        login = client.post("/api/v1/auth/password/login", json={"email": "student@example.test", "password": "new-password"})
        assert login.status_code == 200
        assert login.json()["user_id"] == registered.json()["user_id"]


def _verify_client(tmp_path, monkeypatch, *, limit: int):
    import api_routes.auth as auth_routes
    import auth_rate_limit

    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")
    monkeypatch.setenv("STORAGE_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DATABASE_PATH", str(tmp_path / "data" / "chatbot.db"))
    monkeypatch.setattr(config, "EMAIL_AUTH_ENABLED", True)
    monkeypatch.setattr(config, "EMAIL_AUTH_PROVIDER", "resend")
    monkeypatch.setattr(config, "EMAIL_AUTH_FROM", "Serenova <no-reply@example.test>")
    monkeypatch.setattr(config, "EMAIL_AUTH_RESEND_API_KEY", "test-email-secret")
    monkeypatch.setattr(config, "EMAIL_AUTH_TURNSTILE_REQUIRED", False)
    # Distinct callers are told apart by the proxy header, as in the deployment.
    monkeypatch.setattr(api_server, "API_TRUST_PROXY_HEADERS", True)
    monkeypatch.setattr(auth_routes, "EMAIL_AUTH_VERIFY_MAX_FAILURES_PER_IP", limit)
    # The limiter is process-wide; start every test from an empty table.
    monkeypatch.setattr(auth_rate_limit, "_FAILURES", {})
    sent: list[str] = []
    monkeypatch.setattr(api_server.EMAIL_AUTH_SERVICE, "_send_code", lambda _email, code: sent.append(code))
    return sent


def _verify(client, challenge_id: str, code: str, ip: str):
    return client.post(
        "/api/v1/auth/email/verify",
        json={"challenge_id": challenge_id, "code": code, "purpose": "registration"},
        headers={"CF-Connecting-IP": ip},
    )


def test_verify_is_limited_per_address_across_challenges(tmp_path, monkeypatch):
    sent = _verify_client(tmp_path, monkeypatch, limit=3)

    with TestClient(api_server.app, base_url="https://testserver") as client:
        start = client.post("/api/v1/auth/email/start", json={"email": "student@example.test", "purpose": "registration", "turnstile_token": ""})
        real = start.json()["challenge_id"]
        # Cycling through made-up challenges is the pattern the database's
        # per-challenge cap cannot see.
        failures = [_verify(client, f"made-up-challenge-{index:04d}", "00000000", "203.0.113.7").status_code for index in range(3)]
        blocked = _verify(client, real, sent[-1], "203.0.113.7")
        other_address = _verify(client, real, sent[-1], "198.51.100.4")

    assert failures == [400, 400, 400]
    assert blocked.status_code == 429
    assert "Try again in" in blocked.json()["message"]
    # A different caller is unaffected, and the blocked request never reached the
    # challenge: the correct code still verifies.
    assert other_address.status_code == 200
    assert other_address.json()["verified_intent"]


def test_a_successful_verification_does_not_reset_the_failure_budget(tmp_path, monkeypatch):
    sent = _verify_client(tmp_path, monkeypatch, limit=3)

    with TestClient(api_server.app, base_url="https://testserver") as client:
        for index in range(2):
            _verify(client, f"made-up-challenge-{index:04d}", "00000000", "203.0.113.7")
        start = client.post("/api/v1/auth/email/start", json={"email": "own@example.test", "purpose": "registration", "turnstile_token": ""})
        own = _verify(client, start.json()["challenge_id"], sent[-1], "203.0.113.7")
        third_failure = _verify(client, "made-up-challenge-0002", "00000000", "203.0.113.7")
        after = _verify(client, "made-up-challenge-0003", "00000000", "203.0.113.7")

    assert own.status_code == 200
    assert third_failure.status_code == 400
    assert after.status_code == 429


def test_email_service_outages_do_not_count_against_the_caller(tmp_path, monkeypatch):
    from email_auth_store import EmailAuthUnavailable

    _verify_client(tmp_path, monkeypatch, limit=2)

    def unavailable(*_args, **_kwargs):
        raise EmailAuthUnavailable("Email verification is temporarily unavailable.")

    monkeypatch.setattr(api_server.EMAIL_AUTH_SERVICE, "verify_challenge", unavailable)

    with TestClient(api_server.app, base_url="https://testserver") as client:
        statuses = [_verify(client, "unknown-challenge-0000", "00000000", "203.0.113.7").status_code for _ in range(4)]

    assert statuses == [503, 503, 503, 503]
