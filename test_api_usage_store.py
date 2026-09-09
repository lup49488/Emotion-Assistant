from datetime import datetime, timedelta

import pytest

import api_usage_store
import session_store
from sqlite_store import connection, ensure_user


def _configure_json(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")
    monkeypatch.setenv("STORAGE_BACKEND", "json")


def _configure_sqlite(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")
    monkeypatch.setenv("STORAGE_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DATABASE_PATH", str(tmp_path / "data" / "chatbot.db"))


def test_usage_summary_tracks_success_failure_and_estimated_cost_in_json(tmp_path, monkeypatch):
    _configure_json(tmp_path, monkeypatch)
    monkeypatch.setattr(api_usage_store, "API_INPUT_COST_PER_1M_TOKENS", 2.0)
    monkeypatch.setattr(api_usage_store, "API_OUTPUT_COST_PER_1M_TOKENS", 4.0)
    now = datetime.now()

    api_usage_store.record_usage(
        "alice", provider="openai", model="test", input_tokens=1000, output_tokens=500,
        duration_ms=120, success=True, created_at=now,
    )
    api_usage_store.record_usage(
        "alice", provider="openai", model="test", input_tokens=100, output_tokens=0,
        duration_ms=80, success=False, error_kind="timeout", created_at=now,
    )

    summary = api_usage_store.usage_summary("alice", now=now)

    assert summary["today"]["requests"] == 2
    assert summary["today"]["failures"] == 1
    assert summary["today"]["input_tokens"] == 1100
    assert summary["today"]["estimated_cost_usd"] == 0.0042


def test_usage_summary_works_with_sqlite(tmp_path, monkeypatch):
    _configure_sqlite(tmp_path, monkeypatch)
    now = datetime.now()
    api_usage_store.record_usage(
        "alice", provider="deepseek", model="chat", input_tokens=10, output_tokens=20,
        duration_ms=50, success=True, created_at=now,
    )

    events = api_usage_store.list_usage_events("alice")
    summary = api_usage_store.usage_summary("alice", now=now)

    assert len(events) == 1
    assert events[0]["model"] == "chat"
    assert summary["month"]["requests"] == 1


def test_request_limit_blocks_before_remote_call(tmp_path, monkeypatch):
    _configure_json(tmp_path, monkeypatch)
    monkeypatch.setattr(api_usage_store, "API_MAX_REQUESTS_PER_MINUTE", 1)
    api_usage_store.record_usage(
        "alice", provider="openai", model="test", input_tokens=1, output_tokens=1,
        duration_ms=1, success=True, created_at=datetime.now() - timedelta(seconds=2),
    )

    try:
        api_usage_store.check_request_allowed("alice", projected_input_tokens=1, projected_output_tokens=1)
    except RuntimeError as exc:
        assert "次/分钟" in str(exc)
    else:
        raise AssertionError("Expected the configured request limit to block the call")


@pytest.mark.parametrize("backend", ["json", "sqlite"])
def test_monthly_budget_includes_usage_beyond_recent_event_limit(tmp_path, monkeypatch, backend):
    (_configure_json if backend == "json" else _configure_sqlite)(tmp_path, monkeypatch)
    now = datetime(2026, 9, 7, 12)
    monkeypatch.setattr(api_usage_store, "_now", lambda: now)
    monkeypatch.setattr(api_usage_store, "API_INPUT_COST_PER_1M_TOKENS", 1.0)
    monkeypatch.setattr(api_usage_store, "API_OUTPUT_COST_PER_1M_TOKENS", 1.0)
    monkeypatch.setattr(api_usage_store, "API_MONTHLY_BUDGET_USD", 10.0)
    monkeypatch.setattr(api_usage_store, "API_DAILY_BUDGET_USD", 0.0)
    monkeypatch.setattr(api_usage_store, "API_MAX_REQUESTS_PER_MINUTE", 0)
    expensive = {
        "provider": "test", "model": "test", "input_tokens": 11_000_000,
        "output_tokens": 0, "estimated_cost_usd": 11.0, "duration_ms": 1,
        "success": True, "error_kind": "", "created_at": "2026-09-01T12:00:00",
    }
    recent = {**expensive, "input_tokens": 0, "estimated_cost_usd": 0.0, "created_at": "2026-09-07T11:00:00"}
    events = [expensive, *[dict(recent) for _ in range(5000)]]
    if backend == "json":
        api_usage_store._write_json_events("alice", events)
    else:
        with connection() as conn:
            ensure_user(conn, "alice")
            conn.executemany(
                """INSERT INTO api_usage_events(user_id, provider, model, input_tokens,
                   output_tokens, estimated_cost_usd, duration_ms, success, error_kind, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                [("alice", *event.values()) for event in events],
            )

    summary = api_usage_store.usage_summary("alice", now=now)

    assert summary["month"]["requests"] == 5001
    assert summary["month"]["estimated_cost_usd"] == 11.0
    assert summary["today"]["requests"] == 5000
    assert len(api_usage_store.list_usage_events("alice")) == 5000
    with pytest.raises(RuntimeError, match="10.0000"):
        api_usage_store.check_request_allowed("alice", projected_input_tokens=0, projected_output_tokens=0)
