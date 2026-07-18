from __future__ import annotations

from fastapi.testclient import TestClient

import api_server


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
