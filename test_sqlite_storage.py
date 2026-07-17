from __future__ import annotations

import session_store
import knowledge_store
import style_store
from auth_store import has_access_key, verify_access
from export_store import build_user_export_payload
from mood_store import add_mood_checkin, delete_mood_checkin, load_mood_checkins
from onboarding_store import mark_onboarding_completed, onboarding_completed
from session_store import SessionState, load_state, persist_state
from sqlite_store import database_path
from unittest.mock import patch


def _enable_sqlite(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")
    monkeypatch.setenv("STORAGE_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DATABASE_PATH", str(tmp_path / "data" / "chatbot.db"))


def test_sqlite_registers_auth_without_creating_legacy_json(tmp_path, monkeypatch):
    _enable_sqlite(tmp_path, monkeypatch)

    ok, _ = verify_access("alice", "secret123")

    assert ok is True
    assert has_access_key("alice") is True
    assert database_path().exists()
    assert not (session_store.user_dir("alice") / "access_key.json").exists()


def test_sqlite_persists_onboarding_completion(tmp_path, monkeypatch):
    _enable_sqlite(tmp_path, monkeypatch)

    mark_onboarding_completed("alice")

    assert onboarding_completed("alice") is True


def test_sqlite_imports_existing_auth_json_once(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")
    monkeypatch.setenv("STORAGE_BACKEND", "json")
    assert verify_access("alice", "legacy-secret")[0] is True

    monkeypatch.setenv("STORAGE_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DATABASE_PATH", str(tmp_path / "data" / "chatbot.db"))

    assert verify_access("alice", "legacy-secret") == (True, "验证通过。")
    assert verify_access("alice", "wrong-secret")[0] is False


def test_sqlite_imports_mood_json_then_uses_database_as_source_of_truth(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")
    monkeypatch.setenv("STORAGE_BACKEND", "json")
    add_mood_checkin("alice", "开心", 4, "legacy", "2026-07-10")

    monkeypatch.setenv("STORAGE_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DATABASE_PATH", str(tmp_path / "data" / "chatbot.db"))
    assert load_mood_checkins("alice")[0]["note"] == "legacy"

    updated = add_mood_checkin("alice", "平静", 2, "database", "2026-07-10")
    assert updated["mood"] == "平静"
    assert load_mood_checkins("alice") == [updated]
    assert delete_mood_checkin("alice", "2026-07-10") is True
    assert load_mood_checkins("alice") == []


def test_sqlite_mood_update_preserves_created_at(tmp_path, monkeypatch):
    _enable_sqlite(tmp_path, monkeypatch)
    first = add_mood_checkin("alice", "一般", 3, "morning", "2026-07-12")
    updated = add_mood_checkin("alice", "平静", 2, "evening", "2026-07-12")

    assert updated["created_at"] == first["created_at"]
    assert updated["note"] == "evening"


def test_sqlite_imports_legacy_session_state_once_then_uses_database(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")
    monkeypatch.setenv("STORAGE_BACKEND", "json")
    legacy = SessionState(user_id="alice")
    legacy.history = [{"role": "user", "content": "legacy chat"}]
    legacy.emotion_memory = [{"text": "felt calm", "time": "2026-07-12T09:00:00"}]
    legacy.long_memory = [{"text": "legacy long", "time": "2026-07-12T09:00:00"}]
    legacy.stable_profile = [{"text": "我是学生", "kind": "profile"}]
    legacy.interest_store.replace_all([{"text": "喜欢编程"}])
    legacy.memory_events = [{"section": "stable", "action": "added", "text": "我是学生"}]
    persist_state(legacy)

    monkeypatch.setenv("STORAGE_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DATABASE_PATH", str(tmp_path / "data" / "chatbot.db"))
    imported = load_state("alice")
    assert imported.history[0]["content"] == "legacy chat"
    assert imported.stable_profile[0]["text"] == "我是学生"
    assert imported.interest_store.items[0]["text"] == "喜欢编程"

    imported.history = [{"role": "user", "content": "database chat"}]
    imported.stable_profile = [{"text": "我喜欢简洁回复", "kind": "profile"}]
    persist_state(imported)
    reloaded = load_state("alice")
    assert reloaded.history[0]["content"] == "database chat"
    assert reloaded.stable_profile[0]["text"] == "我喜欢简洁回复"


def test_sqlite_export_reads_session_state_from_database(tmp_path, monkeypatch):
    _enable_sqlite(tmp_path, monkeypatch)
    state = SessionState(user_id="alice")
    state.history = [{"role": "assistant", "content": "stored in sqlite"}]
    state.stable_profile = [{"text": "我是学生", "kind": "profile"}]
    persist_state(state)
    add_mood_checkin("alice", "开心", 4, "ok", "2026-07-12")

    payload = build_user_export_payload("alice")

    assert payload["history"][0]["content"] == "stored in sqlite"
    assert payload["stable_profile"][0]["text"] == "我是学生"
    assert payload["mood_checkins"][0]["mood"] == "开心"


def test_sqlite_imports_knowledge_metadata_and_uses_database_chunks(tmp_path, monkeypatch):
    _enable_sqlite(tmp_path, monkeypatch)
    base = tmp_path / "knowledge"
    docs_dir = base / "documents"
    docs_dir.mkdir(parents=True)
    (docs_dir / "guide.txt").write_text("legacy source", encoding="utf-8")
    chunks_path = base / "chunks.json"
    chunks_path.write_text(
        '[{"id":"guide.txt#0","source":"guide.txt","text":"legacy chunk","chunk_index":0,"created_at":"2026-07-16"}]',
        encoding="utf-8",
    )

    with patch.object(knowledge_store, "KNOWLEDGE_DIR", base), \
         patch.object(knowledge_store, "KNOWLEDGE_DOCS_DIR", docs_dir), \
         patch.object(knowledge_store, "KNOWLEDGE_CHUNKS_PATH", chunks_path):
        assert knowledge_store.load_chunks()[0]["text"] == "legacy chunk"
        details = knowledge_store.list_document_details()
        assert details == [
            {
                "name": "guide.txt",
                "size_bytes": len("legacy source".encode("utf-8")),
                "modified_at": details[0]["modified_at"],
                "created_at": "",
                "chunks": 1,
            }
        ]
        knowledge_store.save_chunks([
            {"id": "guide.txt#0", "source": "guide.txt", "text": "database chunk", "chunk_index": 0, "created_at": "2026-07-16"}
        ])
        chunks_path.write_text("[]", encoding="utf-8")
        assert knowledge_store.load_chunks()[0]["text"] == "database chunk"


def test_sqlite_imports_style_metadata(tmp_path, monkeypatch):
    _enable_sqlite(tmp_path, monkeypatch)
    base = tmp_path / "style"
    docs_dir = base / "documents"
    docs_dir.mkdir(parents=True)
    (docs_dir / "tone.md").write_text("# tone", encoding="utf-8")
    chunks_path = base / "chunks.json"
    chunks_path.write_text(
        '[{"id":"tone.md#0","source":"tone.md","text":"gentle answer","chunk_index":0,"created_at":"2026-07-16"}]',
        encoding="utf-8",
    )

    with patch.object(style_store, "STYLE_DIR", base), \
         patch.object(style_store, "STYLE_DOCS_DIR", docs_dir), \
         patch.object(style_store, "STYLE_CHUNKS_PATH", chunks_path):
        chunks = style_store.load_chunks()

    assert chunks[0]["source"] == "tone.md"
    assert chunks[0]["text"] == "gentle answer"
