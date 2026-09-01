"""Regression tests for restoring a user-owned Serenova export."""
from __future__ import annotations

import io
import json
from datetime import datetime, timedelta
import zipfile
from unittest.mock import patch

import pytest

import conftest  # noqa: F401  (register stubs before chatbot imports)
import chatbot
import config
import conversation_store
import export_store
import mood_store
import session_store
import user_data_import


@pytest.fixture()
def isolated_user(tmp_path, monkeypatch):
    # conftest's autouse fixture clears the process-wide session cache between tests.
    # chatbot.USERS_DIR must be redirected too: _sync_user_paths() copies it back
    # over session_store.USERS_DIR whenever it differs from the configured default.
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")
    monkeypatch.setattr(chatbot, "USERS_DIR", tmp_path / "users")
    return "importer"


def _export_bytes(user_id: str, **overrides) -> bytes:
    payload = {
        "schema_version": 4,
        "user_id": user_id,
        "history": [],
        "conversations": [],
        "emotion_memory": [],
        "long_memory": [],
        "stable_profile": [],
        "interest_memory": [],
        "memory_events": [],
        "pending_memory": [],
        "mood_checkins": [],
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _recent(days_ago: int = 1) -> str:
    """A timestamp inside the long-term retention window.

    clean_long_term() drops memories older than LONG_TERM_EXPIRY_DAYS, so these
    fixtures must stay relative to now instead of pinning a calendar date.
    """
    return (datetime.now() - timedelta(days=days_ago)).isoformat(timespec="seconds")


def _seed(user_id: str) -> None:
    conversation_store.append_exchange(user_id, None, "旧对话", "旧回复")
    mood_store.add_mood_checkin(user_id, "平静", 3, "旧记录", (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d"))
    with chatbot.session_store.session(user_id) as state:
        state.long_memory.append({"text": "旧记忆", "time": _recent(3)})


def test_failed_replace_rolls_back_instead_of_leaving_half_the_account(isolated_user):
    """Replace writes three stores in turn, so a failure must not discard the rest."""
    _seed(isolated_user)
    raw = _export_bytes(isolated_user, long_memory=[{"text": "新记忆", "time": _recent(1)}])

    original_restore_conversations = user_data_import.restore_conversations
    attempts = 0

    def fail_initial_import(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("disk full")
        return original_restore_conversations(*args, **kwargs)

    with patch.object(user_data_import, "restore_conversations", side_effect=fail_initial_import):
        with pytest.raises(RuntimeError):
            user_data_import.import_user_export(isolated_user, raw, mode="replace")

    restored = export_store.build_user_export_payload(isolated_user)
    assert [item["text"] for item in restored["long_memory"]] == ["旧记忆"]
    assert len(restored["conversations"]) == 1
    assert len(restored["mood_checkins"]) == 1


def test_failed_replace_keeps_a_recoverable_backup_when_rollback_also_fails(isolated_user, caplog):
    _seed(isolated_user)
    raw = _export_bytes(isolated_user, long_memory=[{"text": "新记忆", "time": _recent(1)}])

    with patch.object(user_data_import, "restore_conversations", side_effect=RuntimeError("disk full")), \
         patch.object(user_data_import, "_apply_memory_import", side_effect=RuntimeError("still broken")):
        with pytest.raises(RuntimeError):
            user_data_import.import_user_export(isolated_user, raw, mode="replace")

    assert "记忆备份保存在" in caplog.text


def test_imported_history_is_trimmed_to_the_short_term_window(isolated_user):
    """Otherwise the next turn would send the whole imported transcript to the model."""
    history = [
        {"role": "user" if index % 2 == 0 else "assistant", "content": f"第 {index} 条"}
        for index in range(200)
    ]
    user_data_import.import_user_export(
        isolated_user, _export_bytes(isolated_user, history=history), mode="replace",
    )

    with chatbot.session_store.session(isolated_user) as state:
        assert len(state.history) == config.SHORT_TERM_LIMIT
        assert state.history[-1]["content"] == "第 199 条"


def test_imported_pending_candidates_respect_the_queue_limit_and_are_audited(isolated_user):
    pending = [
        {"id": f"p{index}", "section": "long", "candidate": {"text": f"候选 {index}"}, "score": 4.0}
        for index in range(config.MEMORY_PENDING_LIMIT + 5)
    ]
    user_data_import.import_user_export(
        isolated_user, _export_bytes(isolated_user, pending_memory=pending), mode="replace",
    )

    with chatbot.session_store.session(isolated_user) as state:
        assert len(state.pending_memory) == config.MEMORY_PENDING_LIMIT
        expired = [item for item in state.memory_events if item["action"] == "expired"]
        assert [item["text"] for item in expired] == [f"候选 {index}" for index in range(5)]


def test_repeated_merge_reports_zero_additions_across_every_count(isolated_user):
    _seed(isolated_user)
    raw = json.dumps(export_store.build_user_export_payload(isolated_user), ensure_ascii=False).encode("utf-8")

    first = user_data_import.import_user_export(isolated_user, raw, mode="merge")
    second = user_data_import.import_user_export(isolated_user, raw, mode="merge")

    # The file's own contents already exist, so "imported" must read as zero
    # everywhere rather than reporting the file's size for memories alone.
    assert first == {"mode": "merge", "conversations": 0, "mood_checkins": 0, "memories": 0}
    assert second == first


def test_merge_reports_the_number_of_memories_it_actually_added(isolated_user):
    _seed(isolated_user)
    raw = _export_bytes(
        isolated_user,
        long_memory=[{"text": "旧记忆", "time": _recent(3)}, {"text": "新记忆", "time": _recent(1)}],
    )

    result = user_data_import.import_user_export(isolated_user, raw, mode="merge")

    assert result["memories"] == 1


def test_export_from_another_account_is_rejected(isolated_user):
    raw = _export_bytes("someone-else")

    with pytest.raises(ValueError, match="different user ID"):
        user_data_import.import_user_export(isolated_user, raw, mode="merge")


def test_chatgpt_export_is_previewed_and_imported_only_after_profile_selection(isolated_user):
    raw = json.dumps([{
        "title": "Sleep notes", "create_time": "2026-08-20T10:00:00",
        "mapping": {
            "one": {"message": {"author": {"role": "user"}, "content": {"parts": ["I cannot sleep"]}, "create_time": "2026-08-20T10:00:00"}},
            "two": {"message": {"author": {"role": "assistant"}, "content": {"parts": ["Try a calmer evening routine."]}, "create_time": "2026-08-20T10:01:00"}},
        },
    }], ensure_ascii=False).encode("utf-8")

    preview = user_data_import.preview_external_import(raw)
    result = user_data_import.import_external_export(isolated_user, raw, mode="merge")

    assert preview["source"] == "chatgpt"
    assert preview["conversations"] == 1
    assert result["conversations"] == 1
    restored = export_store.build_user_export_payload(isolated_user)
    assert restored["conversations"][0]["title"] == "Sleep notes"


def _zip_export(members: dict[str, str], *, understate: str | None = None) -> bytes:
    """Build a ZIP export, optionally lying about one member's uncompressed size."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, body in members.items():
            archive.writestr(name, body)
    if understate is None:
        return buffer.getvalue()
    data = bytearray(buffer.getvalue())
    central = data.rfind(b"PK\x01\x02")
    data[central + 24:central + 28] = (1024).to_bytes(4, "little")
    local = data.find(b"PK\x03\x04")
    data[local + 22:local + 26] = (1024).to_bytes(4, "little")
    return bytes(data)


CHATGPT_CONVERSATION = [{
    "title": "Sleep notes",
    "mapping": {
        "one": {"message": {"author": {"role": "user"}, "content": {"parts": ["I cannot sleep"]}}},
        "two": {"message": {"author": {"role": "assistant"}, "content": {"parts": ["Try a calmer evening."]}}},
    },
}]


def test_zip_export_is_imported_like_its_plain_json_equivalent(isolated_user):
    raw = _zip_export({"conversations.json": json.dumps(CHATGPT_CONVERSATION, ensure_ascii=False)})

    preview = user_data_import.preview_external_import(raw)
    result = user_data_import.import_external_export(isolated_user, raw, mode="merge")

    assert preview["source"] == "chatgpt"
    assert preview["conversations"] == 1
    assert result["conversations"] == 1


def test_zip_member_larger_than_the_cap_is_rejected(isolated_user):
    oversized = json.dumps(CHATGPT_CONVERSATION) + " " * (user_data_import.MAX_EXTERNAL_UNCOMPRESSED_BYTES + 1024)
    raw = _zip_export({"conversations.json": oversized})

    with pytest.raises(ValueError, match="too much uncompressed data"):
        user_data_import.preview_external_import(raw)


def test_an_understated_member_size_cannot_slip_past_the_cap(isolated_user):
    # ZipInfo.file_size is attacker-controlled, so the cap must come from the
    # bytes actually read rather than from the archive's own header.
    oversized = json.dumps(CHATGPT_CONVERSATION) + " " * (user_data_import.MAX_EXTERNAL_UNCOMPRESSED_BYTES + 1024)
    raw = _zip_export({"conversations.json": oversized}, understate="conversations.json")

    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        assert archive.getinfo("conversations.json").file_size == 1024  # the lie the header tells

    with pytest.raises(ValueError):
        user_data_import.preview_external_import(raw)


def test_claude_chat_messages_export_is_recognised(isolated_user):
    raw = json.dumps({"chat_messages": [
        {"conversation_uuid": "abc", "sender": "human", "text": "I keep overthinking."},
        {"conversation_uuid": "abc", "sender": "assistant", "text": "Let us slow that down."},
    ]}, ensure_ascii=False).encode("utf-8")

    preview = user_data_import.preview_external_import(raw)

    assert preview["source"] == "claude"
    assert preview["conversations"] == 1


def test_profile_fields_are_only_imported_when_selected(isolated_user):
    raw = _zip_export({
        "conversations.json": json.dumps(CHATGPT_CONVERSATION, ensure_ascii=False),
        "user.json": json.dumps({"email": "someone@example.com"}, ensure_ascii=False),
    })

    without = user_data_import.parse_external_export_bytes(raw)[0]
    assert without["stable_profile"] == []

    preview = user_data_import.preview_external_import(raw)
    keys = [field["key"] for field in preview.get("profile_fields", [])]
    if keys:
        with_selection = user_data_import.parse_external_export_bytes(raw, keys)[0]
        assert len(with_selection["stable_profile"]) == len(keys)


def test_unsupported_payloads_are_rejected(isolated_user):
    with pytest.raises(ValueError):
        user_data_import.preview_external_import(b"not json at all")
    with pytest.raises(ValueError):
        user_data_import.preview_external_import(b"")
    with pytest.raises(ValueError, match="No supported conversations"):
        user_data_import.preview_external_import(b"[]")


def test_rollback_restores_what_was_overwritten_not_the_stored_copy(isolated_user):
    """The snapshot must come from the state the import writes through.

    session_store caches a SessionState per user and the import mutates that
    cached object, while build_user_export_payload() reloads from storage. If the
    snapshot is taken from the reloaded copy, a rollback restores whichever
    version storage happened to hold rather than the one that was replaced.
    """
    _seed(isolated_user)
    store = chatbot.session_store
    with store.session(isolated_user):
        pass  # make sure the user is cached
    # Put the cache ahead of storage, the way an in-flight session does.
    store._sessions[isolated_user].long_memory.append(
        {"text": "仅在缓存中的记忆", "time": _recent(2)}
    )

    raw = _export_bytes(isolated_user, long_memory=[{"text": "新记忆", "time": _recent(1)}])
    attempts = 0
    original = user_data_import.restore_conversations

    def fail_once(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("disk full")
        return original(*args, **kwargs)

    with patch.object(user_data_import, "restore_conversations", side_effect=fail_once):
        with pytest.raises(RuntimeError):
            user_data_import.import_user_export(isolated_user, raw, mode="replace")

    restored = [item["text"] for item in export_store.build_user_export_payload(isolated_user)["long_memory"]]
    assert restored == ["旧记忆", "仅在缓存中的记忆"]
