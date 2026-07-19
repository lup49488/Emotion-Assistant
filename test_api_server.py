from __future__ import annotations

from fastapi.testclient import TestClient

import api_server
import session_store
from auth_store import change_access_key


def _login(client, monkeypatch, tmp_path, user_id="api-alice", access_key="api-secret"):
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")
    login = client.post("/api/v1/auth/login", json={"user_id": user_id, "access_key": access_key})
    assert login.status_code == 200
    assert login.json()["token_type"] == "cookie"
    assert login.cookies.get(api_server.API_SESSION_COOKIE_NAME)
    assert "HttpOnly" in login.headers["set-cookie"]
    csrf_token = client.cookies.get(api_server.API_CSRF_COOKIE_NAME)
    assert csrf_token
    return {"X-CSRF-Token": csrf_token}


def test_health_reports_storage_backend():
    with TestClient(api_server.app, base_url="https://testserver") as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_login_sets_signed_cookie_and_memory_quality_uses_it(monkeypatch, tmp_path):
    with TestClient(api_server.app, base_url="https://testserver") as client:
        _login(client, monkeypatch, tmp_path)

        response = client.get("/api/v1/memory/quality")

    assert response.status_code == 200
    assert "report" in response.json()


def test_chat_requires_signed_session_cookie():
    with TestClient(api_server.app, base_url="https://testserver") as client:
        response = client.post("/api/v1/chat", json={"message": "hello"})

    assert response.status_code == 401


def test_state_changing_api_requests_require_csrf_header(monkeypatch, tmp_path):
    with TestClient(api_server.app, base_url="https://testserver") as client:
        _login(client, monkeypatch, tmp_path)
        response = client.post("/api/v1/chat", json={"message": "hello"})

    assert response.status_code == 403


def test_logout_clears_signed_session_cookie(monkeypatch, tmp_path):
    with TestClient(api_server.app, base_url="https://testserver") as client:
        headers = _login(client, monkeypatch, tmp_path)
        response = client.post("/api/v1/auth/logout", headers=headers)
        assert response.status_code == 204

        response = client.get("/api/v1/memory/quality")

    assert response.status_code == 401


def test_password_change_invalidates_existing_signed_cookie(monkeypatch, tmp_path):
    with TestClient(api_server.app, base_url="https://testserver") as client:
        _login(client, monkeypatch, tmp_path)
        assert change_access_key("api-alice", "api-secret", "new-api-secret")[0]

        response = client.get("/api/v1/memory/quality")

    assert response.status_code == 401


def test_chat_stream_emits_chunks_receipt_and_done(monkeypatch, tmp_path):
    monkeypatch.setattr(api_server, "_chat_chunks", lambda user_id, request: iter(["你好", "，", "在的"]))

    with TestClient(api_server.app, base_url="https://testserver") as client:
        headers = _login(client, monkeypatch, tmp_path)
        with client.stream(
            "POST", "/api/v1/chat/stream", json={"message": "hi"}, headers=headers
        ) as response:
            assert response.status_code == 200
            body = "".join(response.iter_text())

    assert "event: chunk" in body
    assert "在的" in body
    assert "event: receipt" in body
    assert body.rstrip().endswith("event: done\ndata: {}")


def test_mood_checkin_rejects_invalid_date_with_422(monkeypatch, tmp_path):
    with TestClient(api_server.app, base_url="https://testserver") as client:
        headers = _login(client, monkeypatch, tmp_path)
        # A malformed date passes pydantic but raises ValueError inside add_mood_checkin,
        # exercising the handler's 422 path.
        response = client.post(
            "/api/v1/mood/checkins",
            json={"mood": "开心", "intensity": 3, "checkin_date": "2026/07/18"},
            headers=headers,
        )

    assert response.status_code == 422
    assert response.json()["code"] == "validation_error"
    assert "YYYY-MM-DD" in response.json()["message"]


def test_mood_checkin_response_matches_contract(monkeypatch, tmp_path):
    with TestClient(api_server.app, base_url="https://testserver") as client:
        headers = _login(client, monkeypatch, tmp_path)
        response = client.post(
            "/api/v1/mood/checkins",
            json={"mood": "calm", "intensity": 3, "checkin_date": "2026-07-18"},
            headers=headers,
        )

    assert response.status_code == 200
    assert response.json()["record"]["source"] == "checkin"


def test_conversation_and_memory_api_use_signed_current_user(monkeypatch, tmp_path):
    with TestClient(api_server.app, base_url="https://testserver") as client:
        headers = _login(client, monkeypatch, tmp_path)
        created = client.post("/api/v1/conversations", json={"title": "API chat"}, headers=headers)
        assert created.status_code == 200
        conversation_id = created.json()["conversation"]["id"]

        listed = client.get("/api/v1/conversations")
        detail = client.get(f"/api/v1/conversations/{conversation_id}")
        memory = client.get("/api/v1/memory")

    assert listed.status_code == 200
    assert listed.json()["conversations"][0]["id"] == conversation_id
    assert detail.status_code == 200
    assert memory.status_code == 200
    assert "stable_profile" in memory.json()


def test_usage_export_and_privacy_api_are_authenticated(monkeypatch, tmp_path):
    with TestClient(api_server.app, base_url="https://testserver") as client:
        headers = _login(client, monkeypatch, tmp_path)
        usage = client.get("/api/v1/usage")
        events = client.get("/api/v1/usage/events?limit=10")
        export = client.get("/api/v1/export")
        privacy = client.get("/api/v1/privacy")
        invalid_delete = client.request(
            "DELETE", "/api/v1/privacy/data", json={"confirmation": "NO"}, headers=headers,
        )

    assert usage.status_code == 200
    assert events.status_code == 200
    assert export.status_code == 200
    assert export.json()["user_id"] == "api-alice"
    assert privacy.status_code == 200
    assert invalid_delete.status_code == 422


def test_rag_api_exposes_status_quality_and_protected_search(monkeypatch, tmp_path):
    with TestClient(api_server.app, base_url="https://testserver") as client:
        headers = _login(client, monkeypatch, tmp_path)
        status_response = client.get("/api/v1/rag/status")
        quality = client.get("/api/v1/rag/quality")
        latest = client.get("/api/v1/rag/evaluations/latest")
        search_without_csrf = client.post("/api/v1/rag/search", json={"query": "test"})

    assert status_response.status_code == 200
    assert quality.status_code == 200
    assert latest.status_code == 200
    assert search_without_csrf.status_code == 403
