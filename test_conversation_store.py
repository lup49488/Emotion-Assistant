from __future__ import annotations

from unittest.mock import patch

import chatbot
import conversation_store
import gui_conversations
import session_store
from session_store import SessionState


def _use_json_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_BACKEND", "json")
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")


def _use_sqlite_backend(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DATABASE_PATH", str(tmp_path / "data" / "chatbot.db"))
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")


def _exercise_archive():
    conversation = conversation_store.create_conversation("archive-user", "First topic")
    conversation_store.append_exchange("archive-user", conversation["id"], "hello", "hi")
    summaries = conversation_store.list_conversations("archive-user")
    stored = conversation_store.get_conversation("archive-user", conversation["id"])
    assert summaries[0]["message_count"] == 2
    assert [item["content"] for item in stored["messages"]] == ["hello", "hi"]
    assert conversation_store.delete_conversation("archive-user", conversation["id"]) is True
    assert conversation_store.list_conversations("archive-user") == []


def test_json_conversation_archive(tmp_path, monkeypatch):
    _use_json_backend(tmp_path, monkeypatch)
    _exercise_archive()


def test_sqlite_conversation_archive(tmp_path, monkeypatch):
    _use_sqlite_backend(tmp_path, monkeypatch)
    _exercise_archive()


def _exercise_quote_archive():
    conversation = conversation_store.create_conversation("quote-user", "Quoted topic")
    conversation_store.append_exchange("quote-user", conversation["id"], "first question", "first answer")
    source = conversation_store.get_conversation("quote-user", conversation["id"])["messages"][1]
    conversation_store.append_exchange(
        "quote-user", conversation["id"], "please expand that", "expanded answer", reply_to_message_id=source["id"],
    )
    stored = conversation_store.get_conversation("quote-user", conversation["id"])

    assert all(message["id"] for message in stored["messages"])
    assert stored["messages"][2]["reply_to_message_id"] == source["id"]
    assert conversation_store.get_conversation_message("quote-user", conversation["id"], source["id"])["content"] == "first answer"
    assert conversation_store.get_conversation_message("another-user", conversation["id"], source["id"]) is None
    assert conversation_store.last_exchange_message_ids("quote-user", conversation["id"], "please expand that") == {
        "user_message_id": stored["messages"][2]["id"],
        "assistant_message_id": stored["messages"][3]["id"],
    }


def test_json_quote_archive(tmp_path, monkeypatch):
    _use_json_backend(tmp_path, monkeypatch)
    _exercise_quote_archive()


def test_sqlite_quote_archive(tmp_path, monkeypatch):
    _use_sqlite_backend(tmp_path, monkeypatch)
    _exercise_quote_archive()


def test_chat_archives_normal_reply_to_selected_conversation(tmp_path, monkeypatch):
    _use_json_backend(tmp_path, monkeypatch)
    conversation = conversation_store.create_conversation("archive-user", "Selected")
    state = SessionState(user_id="archive-user")
    with patch.object(chatbot, "safe_analyze", return_value=("neutral", 0.0)), \
        patch.object(chatbot, "smart_memory_filter", return_value="discard"), \
        patch.object(chatbot, "stream_model_response", return_value=iter(["archive reply"])):
        assert "".join(chatbot.chat(state, "archive prompt", conversation_id=conversation["id"], use_style=False)) == "archive reply"

    stored = conversation_store.get_conversation("archive-user", conversation["id"])
    assert [item["content"] for item in stored["messages"]] == ["archive prompt", "archive reply"]


def test_loading_conversation_context_replaces_short_term_history(tmp_path, monkeypatch):
    _use_json_backend(tmp_path, monkeypatch)
    gui_conversations._set_short_term_context(
        "archive-user",
        [{"role": "user", "content": "old"}, {"role": "assistant", "content": "new"}],
    )
    with chatbot.session_store.session("archive-user") as state:
        assert state.history == [
            {"role": "user", "content": "old"},
            {"role": "assistant", "content": "new"},
        ]
