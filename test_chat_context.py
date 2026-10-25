"""Conversation boundaries and regeneration use the real offline chat chain."""
from __future__ import annotations

import pytest

import chatbot
import api_server
import conversation_store as archive
import session_store
from sqlite_store import connection


@pytest.fixture(params=["json", "sqlite"])
def chat_user(request, monkeypatch, tmp_path):
    monkeypatch.setenv("STORAGE_BACKEND", request.param)
    monkeypatch.setenv("SQLITE_DATABASE_PATH", str(tmp_path / "chat.db"))
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")
    monkeypatch.setattr(chatbot, "USERS_DIR", tmp_path / "users")
    monkeypatch.setattr(chatbot, "safe_analyze", lambda *_: ("neutral", 0.0))
    monkeypatch.setattr(chatbot, "check_crisis", lambda *_: None)
    monkeypatch.setattr(chatbot, "smart_memory_filter", lambda *_, **__: None)
    return "context-user"


def turn(user, conversation, text, **kwargs):
    return chatbot.handle_user_message_stream(
        user, text, conversation_id=conversation, use_knowledge=False, use_style=False, **kwargs,
    )


def test_selected_thread_supplies_history_after_switch_and_reload(chat_user, monkeypatch):
    prompts = []

    def model(messages, _config):
        prompts.append(messages)
        yield "offline reply"

    monkeypatch.setattr(chatbot, "stream_model_response", model)
    first = archive.create_conversation(chat_user, "A")["id"]
    second = archive.create_conversation(chat_user, "B")["id"]
    list(turn(chat_user, first, "A private topic"))
    list(turn(chat_user, second, "B separate topic"))
    assert prompts[1][1:-1] == []
    chatbot.session_store._sessions.clear()
    list(turn(chat_user, first, "Continue A"))
    assert prompts[2][1:-1] == [
        {"role": "user", "content": "A private topic"},
        {"role": "assistant", "content": "offline reply"},
    ]
    with chatbot.session_store.session(chat_user) as state:
        assert all("B separate topic" != item["content"] for item in state.history)


@pytest.mark.parametrize("outcome", ["failed", "cancelled", "empty", "completed"])
def test_regeneration_preserves_original_until_success(chat_user, monkeypatch, outcome):
    cid = archive.create_conversation(chat_user)["id"]
    archive.append_exchange(chat_user, cid, "earlier", "earlier reply")
    archive.append_exchange(chat_user, cid, "regenerate me", "original reply")
    original = archive.get_conversation(chat_user, cid)["messages"]
    with chatbot.session_store.session(chat_user) as state:
        state.history = [{"role": item["role"], "content": item["content"]} for item in original]
    original_history = list(state.history)

    def model(messages, _config):
        assert messages[1:-1] == original_history[:-2]
        if outcome == "empty":
            return
        yield "replacement"
        if outcome == "failed":
            raise RuntimeError("offline failure")

    monkeypatch.setattr(chatbot, "stream_model_response", model)
    stream = turn(chat_user, cid, "regenerate me", retry_last_response=True)
    if outcome == "empty":
        with pytest.raises(chatbot.ServiceError, match="没有返回"):
            list(stream)
    else:
        assert next(stream) == "replacement"
        assert archive.get_conversation(chat_user, cid)["messages"] == original
        if outcome == "cancelled":
            stream.close()
        elif outcome == "failed":
            with pytest.raises(RuntimeError, match="offline failure"):
                list(stream)
        else:
            assert list(stream) == []
    stored = archive.get_conversation(chat_user, cid)["messages"]
    with chatbot.session_store.session(chat_user) as state:
        if outcome == "completed":
            assert [item["content"] for item in stored] == ["earlier", "earlier reply", "regenerate me", "replacement"]
            assert state.history[-1]["content"] == "replacement"
        else:
            assert stored == original
            assert state.history == original_history


def test_replacement_storage_failure_keeps_original(chat_user, monkeypatch):
    cid = archive.create_conversation(chat_user)["id"]
    archive.append_exchange(chat_user, cid, "question", "original reply")
    original = archive.get_conversation(chat_user, cid)["messages"]
    if archive.sqlite_enabled():
        with connection() as conn:
            conn.execute("""CREATE TRIGGER reject_replacement BEFORE INSERT ON conversation_messages
                BEGIN SELECT RAISE(ABORT, 'offline storage failure'); END""")
    else:
        def fail_save(*_):
            raise RuntimeError("offline storage failure")
        monkeypatch.setattr(archive, "_save_json_conversations", fail_save)
    monkeypatch.setattr(chatbot, "stream_model_response", lambda *_: iter(["replacement"]))
    with pytest.raises(chatbot.ConversationArchiveError, match="could not be saved") as error:
        list(turn(chat_user, cid, "question", retry_last_response=True))
    assert error.value.__cause__ is not None
    assert "offline storage failure" in str(error.value.__cause__)
    assert archive.get_conversation(chat_user, cid)["messages"] == original


def test_retry_configuration_failure_does_not_delete_archive(chat_user, monkeypatch):
    cid = archive.create_conversation(chat_user)["id"]
    archive.append_exchange(chat_user, cid, "question", "original reply")
    original = archive.get_conversation(chat_user, cid)["messages"]

    def fail_config(**_):
        raise RuntimeError("offline configuration failure")

    monkeypatch.setattr(api_server, "make_model_config", fail_config)
    request = api_server.ChatRequest(message="question", conversation_id=cid, retry_last_response=True)
    with pytest.raises(RuntimeError, match="offline configuration failure"):
        list(api_server._chat_chunks(chat_user, request))
    assert archive.get_conversation(chat_user, cid)["messages"] == original


@pytest.mark.parametrize("crisis", [False, True])
def test_evidence_refusal_retry_replaces_only_after_answer_is_ready(chat_user, monkeypatch, crisis):
    cid = archive.create_conversation(chat_user)["id"]
    archive.append_exchange(chat_user, cid, "question", "original reply")
    monkeypatch.setattr(chatbot, "check_crisis", lambda *_: "crisis answer" if crisis else None)
    request = api_server.ChatRequest(message="question", conversation_id=cid, retry_last_response=True)
    reply, from_crisis_layer = api_server._resolve_insufficient_evidence(chat_user, request)
    assert from_crisis_layer is crisis
    assert [item["content"] for item in archive.get_conversation(chat_user, cid)["messages"]] == ["question", reply]
