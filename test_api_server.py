from __future__ import annotations

from fastapi.testclient import TestClient

import api_server
import session_store


def _login(client, monkeypatch, tmp_path, user_id="api-alice", access_key="api-secret"):
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")
    api_server._sessions.clear()
    login = client.post("/api/v1/auth/login", json={"user_id": user_id, "access_key": access_key})
    assert login.status_code == 200
    return {"Authorization": f"Bearer {login.json()['access_token']}"}


def test_health_reports_storage_backend():
    with TestClient(api_server.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_login_and_memory_quality_require_a_bearer_session():
    api_server._sessions.clear()
    with TestClient(api_server.app) as client:
        login = client.post("/api/v1/auth/login", json={"user_id": "api-alice", "access_key": "api-secret"})
        assert login.status_code == 200

        response = client.get(
            "/api/v1/memory/quality",
            headers={"Authorization": f"Bearer {login.json()['access_token']}"},
        )

    assert response.status_code == 200
    assert "report" in response.json()


def test_chat_requires_bearer_session():
    with TestClient(api_server.app) as client:
        response = client.post("/api/v1/chat", json={"message": "hello"})

    assert response.status_code == 401


def test_chat_stream_emits_chunks_receipt_and_done(monkeypatch, tmp_path):
    monkeypatch.setattr(api_server, "_chat_chunks", lambda user_id, request: iter(["你好", "，", "在的"]))

    with TestClient(api_server.app) as client:
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
    with TestClient(api_server.app) as client:
        headers = _login(client, monkeypatch, tmp_path)
        # A malformed date passes pydantic but raises ValueError inside add_mood_checkin,
        # exercising the handler's 422 path.
        response = client.post(
            "/api/v1/mood/checkins",
            json={"mood": "开心", "intensity": 3, "checkin_date": "2026/07/18"},
            headers=headers,
        )

    assert response.status_code == 422
    assert "YYYY-MM-DD" in response.json()["detail"]
