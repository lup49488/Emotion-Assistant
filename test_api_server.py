from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi.testclient import TestClient

# Tests should not start real model downloads or loads in background threads.
os.environ.setdefault("API_PRELOAD_MODELS", "false")

import api_server
import job_store
import rag_feedback_store
import style_store
from service_errors import ServiceError
import model_warmup
import session_store
from auth_store import change_access_key
from conversation_store import append_exchange


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
        response = client.get("/health", headers={"X-Request-ID": "health-check-1"})

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["components"]["storage"]["status"] == "ok"
    assert "metrics" in response.json()
    assert response.headers["X-Request-ID"] == "health-check-1"


def test_versioned_status_exposes_safe_runtime_metrics():
    with TestClient(api_server.app, base_url="https://testserver") as client:
        response = client.get("/api/v1/status")

    payload = response.json()
    assert response.status_code == 200
    assert payload["api_version"] == "v1"
    assert "api_key" not in str(payload).lower()
    assert payload["metrics"]["http_requests_total"] >= 1


def test_api_lifespan_starts_warmup_once(monkeypatch):
    calls = []
    monkeypatch.setattr(api_server, "start_api_background_warmup", lambda: calls.append(True))

    with TestClient(api_server.app, base_url="https://testserver"):
        pass

    assert calls == [True]


def test_api_warmup_loads_support_models_only(monkeypatch):
    calls = []
    monkeypatch.setattr(model_warmup.goemotions, "predict_emotion", lambda text: calls.append(("emotion", text)))
    monkeypatch.setattr(model_warmup, "get_embedding_model", lambda: calls.append(("embedding", None)))

    model_warmup.warmup_api_models()

    assert calls == [("emotion", "hello"), ("embedding", None)]


def test_api_warmup_optionally_loads_semantic_safety_model(monkeypatch):
    calls = []
    monkeypatch.setattr(model_warmup.goemotions, "predict_emotion", lambda text: calls.append(("emotion", text)))
    monkeypatch.setattr(model_warmup, "get_embedding_model", lambda: calls.append(("embedding", None)))
    monkeypatch.setattr(model_warmup, "semantic_safety_preload_enabled", lambda: True)
    monkeypatch.setattr(model_warmup, "get_semantic_classifier", lambda: calls.append(("safety_semantic", None)))

    model_warmup.warmup_api_models()

    assert calls == [("emotion", "hello"), ("embedding", None), ("safety_semantic", None)]


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
    monkeypatch.setattr(api_server, "_chat_chunks", lambda user_id, request, knowledge_context=None: iter(["你好", "，", "在的"]))

    with TestClient(api_server.app, base_url="https://testserver") as client:
        headers = _login(client, monkeypatch, tmp_path)
        headers["X-Request-ID"] = "stream-check-1"
        with client.stream(
            "POST", "/api/v1/chat/stream", json={"message": "hi"}, headers=headers
        ) as response:
            assert response.status_code == 200
            assert response.headers["X-Request-ID"] == "stream-check-1"
            body = "".join(response.iter_text())

    assert "event: chunk" in body
    assert "在的" in body
    assert "event: receipt" in body
    assert body.rstrip().endswith("event: done\ndata: {}")


def test_chat_stream_emits_real_rag_citations_and_accepts_owned_feedback(monkeypatch, tmp_path):
    monkeypatch.setattr(rag_feedback_store, "_JSON_PATH", tmp_path / "rag_feedback.json")
    monkeypatch.setattr(api_server, "build_knowledge_bundle", lambda _: {
        "context": "[source 1 | guide.md]\nDeployment guide",
        "citations": [{"source": "guide.md", "chunk_index": 0, "score": 0.91, "excerpt": "Deployment guide"}],
    })
    monkeypatch.setattr(api_server, "_chat_chunks", lambda user_id, request, knowledge_context=None: iter(["Answer"]))
    with TestClient(api_server.app, base_url="https://testserver") as client:
        headers = _login(client, monkeypatch, tmp_path)
        with client.stream("POST", "/api/v1/chat/stream", json={"message": "How to deploy?", "use_knowledge": True}, headers=headers) as response:
            body = "".join(response.iter_text())
        citation_data = next(line[6:] for line in body.splitlines() if line.startswith("data: {") and "trace_id" in line)
        trace_id = json.loads(citation_data)["trace_id"]
        feedback = client.post("/api/v1/rag/feedback", json={"trace_id": trace_id, "helpful": True}, headers=headers)

    assert "event: citations" in body
    assert '"source": "guide.md"' in body
    assert feedback.status_code == 200
    assert feedback.json()["helpful"] is True


def test_chat_stream_emits_structured_retryable_service_error(monkeypatch, tmp_path):
    def failing_chunks(*_):
        raise ServiceError("provider_timeout", "provider timed out", retryable=True)
        yield "unreachable"

    monkeypatch.setattr(api_server, "_chat_chunks", failing_chunks)
    with TestClient(api_server.app, base_url="https://testserver") as client:
        headers = _login(client, monkeypatch, tmp_path)
        with client.stream("POST", "/api/v1/chat/stream", json={"message": "hi"}, headers=headers) as response:
            body = "".join(response.iter_text())

    assert 'event: error' in body
    assert '"code":"provider_timeout"' in body
    assert '"retryable":true' in body


def test_chat_stream_emits_retryable_empty_model_response(monkeypatch, tmp_path):
    def empty_response(*_):
        raise ServiceError("empty_model_response", "no displayable text", retryable=True)
        yield "unreachable"

    monkeypatch.setattr(api_server, "_chat_chunks", empty_response)
    with TestClient(api_server.app, base_url="https://testserver") as client:
        headers = _login(client, monkeypatch, tmp_path)
        with client.stream("POST", "/api/v1/chat/stream", json={"message": "hi"}, headers=headers) as response:
            body = "".join(response.iter_text())

    assert '"code":"empty_model_response"' in body
    assert '"retryable":true' in body


def test_chat_retry_replaces_the_last_saved_exchange(monkeypatch, tmp_path):
    def fake_stream(user_id, message, **kwargs):
        reply = "Retried reply"
        append_exchange(user_id, kwargs.get("conversation_id"), message, reply)
        yield reply

    monkeypatch.setattr(api_server, "handle_user_message_stream", fake_stream)
    with TestClient(api_server.app, base_url="https://testserver") as client:
        headers = _login(client, monkeypatch, tmp_path)
        created = client.post("/api/v1/conversations", json={"title": "Retry"}, headers=headers)
        conversation_id = created.json()["conversation"]["id"]
        first = client.post(
            "/api/v1/chat",
            json={"message": "Try again", "conversation_id": conversation_id, "show_memory_receipt": False},
            headers=headers,
        )
        retried = client.post(
            "/api/v1/chat",
            json={
                "message": "Try again", "conversation_id": conversation_id,
                "retry_last_response": True, "show_memory_receipt": False,
            },
            headers=headers,
        )
        detail = client.get(f"/api/v1/conversations/{conversation_id}")

    assert first.status_code == 200
    assert retried.status_code == 200
    assert [message["content"] for message in detail.json()["conversation"]["messages"]] == ["Try again", "Retried reply"]


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


def test_conversation_rename_delete_and_long_term_memory_edit_are_protected(monkeypatch, tmp_path):
    with TestClient(api_server.app, base_url="https://testserver") as client:
        headers = _login(client, monkeypatch, tmp_path)
        created = client.post("/api/v1/conversations", json={"title": "Before"}, headers=headers)
        conversation_id = created.json()["conversation"]["id"]

        renamed = client.put(
            f"/api/v1/conversations/{conversation_id}", json={"title": "After"}, headers=headers,
        )
        updated_memory = client.put(
            "/api/v1/memory/long-term",
            json={"items": [{"text": "I am a student.", "kind": "manual"}]},
            headers=headers,
        )
        removed = client.delete(f"/api/v1/conversations/{conversation_id}", headers=headers)

    assert renamed.status_code == 200
    assert renamed.json()["conversation"]["title"] == "After"
    assert updated_memory.status_code == 200
    assert updated_memory.json()["long_memory"][0]["text"] == "I am a student."
    assert updated_memory.json()["memory_events"][-1]["section"] == "long"
    assert removed.status_code == 204


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
    monkeypatch.setattr(api_server, "API_RAG_ADMIN_USER_IDS", "api-alice")
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


def test_rag_document_management_uses_csrf_and_supported_uploads(monkeypatch, tmp_path):
    monkeypatch.setattr(api_server, "API_RAG_ADMIN_USER_IDS", "api-alice")
    monkeypatch.setattr(api_server, "copy_document_to_store", lambda _: Path("notes.md"))
    def rebuild_gate(*, mutate=None):
        document = mutate() if mutate else None
        return {"documents": 1, "chunks": 2, "errors": [], "published": True, "document": getattr(document, "name", None)}

    monkeypatch.setattr(api_server, "rebuild_with_release_gate", rebuild_gate)
    monkeypatch.setattr(api_server, "delete_document_from_store", lambda name: Path(name))
    submitted = []

    class Context:
        def progress(self, *_):
            pass

    def submit(user_id, kind, worker, *, payload=None):
        submitted.append((user_id, kind, payload, worker(Context())))
        return {"job": {"id": f"job-{len(submitted)}", "kind": kind, "status": "succeeded", "progress": 100, "message": "Completed", "result": submitted[-1][3], "error": None, "created_at": "2026-07-21T00:00:00", "started_at": None, "finished_at": "2026-07-21T00:00:01"}}

    monkeypatch.setattr(api_server, "_submit_rag_job", submit)

    with TestClient(api_server.app, base_url="https://testserver") as client:
        headers = _login(client, monkeypatch, tmp_path)
        upload = client.post(
            "/api/v1/rag/documents",
            files={"file": ("notes.md", b"# A note", "text/markdown")},
            headers=headers,
        )
        remove = client.delete("/api/v1/rag/documents/notes.md", headers=headers)

    assert upload.status_code == 202
    assert upload.json()["job"]["kind"] == "rag_document_upload"
    assert remove.status_code == 202
    assert remove.json()["job"]["kind"] == "rag_document_delete"
    assert submitted[0][3]["uploaded"] == "notes.md"
    assert submitted[1][3]["deleted"] == "notes.md"


def test_rag_rebuild_and_evaluation_return_background_jobs(monkeypatch, tmp_path):
    monkeypatch.setattr(api_server, "API_RAG_ADMIN_USER_IDS", "api-alice")
    def submit(user_id, kind, worker, *, payload=None):
        return {"job": {"id": "job-1", "kind": kind, "status": "queued", "progress": 0, "message": "Queued", "result": None, "error": None, "created_at": "2026-07-21T00:00:00", "started_at": None, "finished_at": None}}

    monkeypatch.setattr(api_server, "_submit_rag_job", submit)
    with TestClient(api_server.app, base_url="https://testserver") as client:
        headers = _login(client, monkeypatch, tmp_path)
        rebuild = client.post("/api/v1/rag/rebuild", headers=headers)
        evaluation = client.post("/api/v1/rag/evaluations/run", json={}, headers=headers)

    assert rebuild.status_code == 202
    assert rebuild.json()["job"]["kind"] == "rag_rebuild"
    assert evaluation.status_code == 202
    assert evaluation.json()["job"]["kind"] == "rag_evaluation"


def test_rag_management_requires_knowledge_administrator(monkeypatch, tmp_path):
    monkeypatch.setattr(api_server, "API_RAG_ADMIN_USER_IDS", "rag-admin")
    with TestClient(api_server.app, base_url="https://testserver") as client:
        headers = _login(client, monkeypatch, tmp_path)
        session = client.get("/api/v1/auth/session")
        rebuild = client.post("/api/v1/rag/rebuild", headers=headers)
        evaluation = client.get("/api/v1/rag/evaluations/latest")

    assert session.status_code == 200
    assert session.json()["can_manage_knowledge"] is False
    assert rebuild.status_code == 403
    assert evaluation.status_code == 403


def test_background_jobs_hide_internal_user_id_and_match_contract(monkeypatch, tmp_path):
    with TestClient(api_server.app, base_url="https://testserver") as client:
        _login(client, monkeypatch, tmp_path)
        job_store.create_job("api-alice", "rag_rebuild")
        response = client.get("/api/v1/jobs")

    assert response.status_code == 200
    assert response.json()["jobs"][0]["kind"] == "rag_rebuild"
    assert "user_id" not in response.json()["jobs"][0]


def test_observability_summary_requires_session_and_returns_persistent_metrics(monkeypatch, tmp_path):
    monkeypatch.setattr(api_server, "API_OPERATIONS_USER_IDS", "api-alice")
    monkeypatch.setattr(api_server, "observability_summary", lambda *, days: {
        "days": days, "retention_days": 90, "requests": 12, "failures": 1,
        "failure_rate": 8.3, "average_duration_ms": 42.0,
        "top_paths": [{"path": "/api/v1/chat", "requests": 5}], "statuses": {"200": 11, "500": 1},
    })
    with TestClient(api_server.app, base_url="https://testserver") as client:
        unauthenticated = client.get("/api/v1/observability/summary")
        headers = _login(client, monkeypatch, tmp_path)
        response = client.get("/api/v1/observability/summary?days=30", headers=headers)

    assert unauthenticated.status_code == 401
    assert response.status_code == 200
    assert response.json()["days"] == 30
    assert response.json()["top_paths"][0]["path"] == "/api/v1/chat"


def test_operations_dashboard_is_restricted_to_configured_users(monkeypatch, tmp_path):
    monkeypatch.setattr(api_server, "API_OPERATIONS_USER_IDS", "api-alice")
    monkeypatch.setattr(api_server, "operations_dashboard", lambda *, days: {
        "window_days": days, "generated_at": "2026-07-21T12:00:00",
        "http": {"requests": 8}, "provider_failures": [],
        "jobs": {"counts": {}}, "alerts": [],
    })
    with TestClient(api_server.app, base_url="https://testserver") as client:
        headers = _login(client, monkeypatch, tmp_path)
        response = client.get("/api/v1/operations/dashboard?days=14", headers=headers)

    assert response.status_code == 200
    assert response.json()["window_days"] == 14


def test_session_reports_operations_access_from_server_whitelist(monkeypatch, tmp_path):
    monkeypatch.setattr(api_server, "API_OPERATIONS_USER_IDS", "api-alice, api-operator")
    with TestClient(api_server.app, base_url="https://testserver") as client:
        headers = _login(client, monkeypatch, tmp_path)
        response = client.get("/api/v1/auth/session", headers=headers)

    assert response.status_code == 200
    assert response.json()["can_access_operations"] is True


def test_public_api_sets_security_headers_and_rejects_oversized_body(monkeypatch):
    monkeypatch.setattr(api_server, "API_MAX_REQUEST_BYTES", 10)
    with TestClient(api_server.app, base_url="https://testserver") as client:
        health = client.get("/health")
        oversized = client.post("/api/v1/auth/login", content="x" * 20, headers={"content-length": "20"})

    assert health.headers["x-content-type-options"] == "nosniff"
    assert health.headers["x-frame-options"] == "DENY"
    assert health.headers["content-security-policy"] == "default-src 'none'; frame-ancestors 'none'"
    assert oversized.status_code == 413


def test_memory_and_export_return_dict_interest_memory(monkeypatch, tmp_path):
    # interest_memory 是字典列表；合约曾误写为 list[list]，一旦用户有兴趣记忆就 500。
    with TestClient(api_server.app, base_url="https://testserver") as client:
        _login(client, monkeypatch, tmp_path)
        import chatbot

        with chatbot.session_store.session("api-alice") as state:
            state.interest_store.append({"text": "喜欢跑步", "time": "2026-07-19T10:00:00"})

        memory = client.get("/api/v1/memory")
        export = client.get("/api/v1/export")

    assert memory.status_code == 200
    assert memory.json()["interest_memory"] == [{"text": "喜欢跑步", "time": "2026-07-19T10:00:00"}]
    assert export.status_code == 200
    assert export.json()["interest_memory"][0]["text"] == "喜欢跑步"


def test_chat_provider_defaults_to_server_configured_provider(monkeypatch, tmp_path):
    captured = {}

    def fake_make_model_config(**kwargs):
        captured.update(kwargs)
        return object()

    def fake_stream(user_id, message, **kwargs):
        yield "ok"

    monkeypatch.setattr(api_server, "make_model_config", fake_make_model_config)
    monkeypatch.setattr(api_server, "handle_user_message_stream", fake_stream)
    monkeypatch.setattr(api_server, "DEFAULT_LLM_PROVIDER", "deepseek")

    with TestClient(api_server.app, base_url="https://testserver") as client:
        headers = _login(client, monkeypatch, tmp_path)
        response = client.post(
            "/api/v1/chat",
            json={"message": "hi", "show_memory_receipt": False},
            headers=headers,
        )

    assert response.status_code == 200
    assert response.json()["reply"] == "ok"
    # 前端未指定 provider 时，应遵循服务器 .env 配置而不是硬编码 local_hf。
    assert captured["provider"] == "deepseek"


def test_retry_cannot_remove_messages_from_another_users_conversation(monkeypatch, tmp_path):
    from conversation_store import append_exchange, get_conversation

    monkeypatch.setenv("STORAGE_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DATABASE_PATH", str(tmp_path / "retry.db"))
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")

    from conversation_store import create_conversation

    victim_conversation = create_conversation("victim-user", "Victim chat")
    append_exchange("victim-user", victim_conversation["id"], "秘密消息", "受害者的回复")

    from conversation_store import remove_last_exchange

    # 攻击者知道会话 ID 和消息原文，但不是会话所有者。
    assert remove_last_exchange("attacker-user", victim_conversation["id"], "秘密消息") is False
    record = get_conversation("victim-user", victim_conversation["id"])
    assert [message["content"] for message in record["messages"]] == ["秘密消息", "受害者的回复"]

    # 所有者本人可以正常执行重试删除。
    assert remove_last_exchange("victim-user", victim_conversation["id"], "秘密消息") is True


def test_retry_rolls_back_short_term_memory(monkeypatch, tmp_path):
    import chatbot

    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")

    def fake_stream(user_id, message, **kwargs):
        yield "ok"

    monkeypatch.setattr(api_server, "handle_user_message_stream", fake_stream)

    with TestClient(api_server.app, base_url="https://testserver") as client:
        headers = _login(client, monkeypatch, tmp_path)
        with chatbot.session_store.session("api-alice") as state:
            state.history.extend([
                {"role": "user", "content": "重试这条"},
                {"role": "assistant", "content": "模型调用失败：boom"},
            ])

        response = client.post(
            "/api/v1/chat",
            json={"message": "重试这条", "retry_last_response": True, "show_memory_receipt": False},
            headers=headers,
        )
        with chatbot.session_store.session("api-alice") as state:
            history_after = list(state.history)

    assert response.status_code == 200
    # 失败的一轮已从短期记忆移除，重试上下文不再包含旧的错误回复。
    assert {"role": "assistant", "content": "模型调用失败：boom"} not in history_after


def test_style_preference_round_trip_and_chat_fallback(monkeypatch, tmp_path):
    import style_preference_store

    monkeypatch.setattr(style_store, "list_documents", lambda: ["温柔型.md", "专业型.md"])
    captured = {}

    def fake_stream(user_id, message, **kwargs):
        captured.update(kwargs)
        yield "ok"

    monkeypatch.setattr(api_server, "handle_user_message_stream", fake_stream)

    with TestClient(api_server.app, base_url="https://testserver") as client:
        headers = _login(client, monkeypatch, tmp_path)
        initial = client.get("/api/v1/style/preference")
        saved = client.put("/api/v1/style/preference", json={"style_prefix": "温柔型"}, headers=headers)
        rejected = client.put("/api/v1/style/preference", json={"style_prefix": "不存在型"}, headers=headers)
        client.put("/api/v1/style/preference", json={"style_prefix": "温柔型"}, headers=headers)
        # 请求未指定 style_prefix 时应回落到已保存的偏好。
        client.post("/api/v1/chat", json={"message": "hi", "show_memory_receipt": False}, headers=headers)
        fallback_prefix = captured.get("style_prefix")
        # 请求显式指定时以请求为准。
        client.post(
            "/api/v1/chat",
            json={"message": "hi", "style_prefix": "专业型", "show_memory_receipt": False},
            headers=headers,
        )
        explicit_prefix = captured.get("style_prefix")

    assert initial.json() == {"style_prefix": "", "available": ["专业型", "温柔型"]}
    assert saved.json()["style_prefix"] == "温柔型"
    # 未知风格被规范化为空，避免检索出零结果。
    assert rejected.json()["style_prefix"] == ""
    assert fallback_prefix == "温柔型"
    assert explicit_prefix == "专业型"


def test_style_preference_update_requires_csrf(monkeypatch, tmp_path):
    with TestClient(api_server.app, base_url="https://testserver") as client:
        _login(client, monkeypatch, tmp_path)
        response = client.put("/api/v1/style/preference", json={"style_prefix": "温柔型"})

    assert response.status_code == 403
