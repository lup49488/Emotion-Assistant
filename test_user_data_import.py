"""Regression tests for restoring a user-owned Serenova export."""
from __future__ import annotations

import json
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


def _seed(user_id: str) -> None:
    conversation_store.append_exchange(user_id, None, "旧对话", "旧回复")
    mood_store.add_mood_checkin(user_id, "平静", 3, "旧记录", "2026-08-01")
    with chatbot.session_store.session(user_id) as state:
        state.long_memory.append({"text": "旧记忆", "time": "2026-08-01T00:00:00"})


def test_failed_replace_rolls_back_instead_of_leaving_half_the_account(isolated_user):
    """Replace writes three stores in turn, so a failure must not discard the rest."""
    _seed(isolated_user)
    raw = _export_bytes(isolated_user, long_memory=[{"text": "新记忆", "time": "2026-08-04T00:00:00"}])

    with patch.object(user_data_import, "restore_conversations", side_effect=RuntimeError("disk full")):
        with pytest.raises(RuntimeError):
            user_data_import.import_user_export(isolated_user, raw, mode="replace")

    restored = export_store.build_user_export_payload(isolated_user)
    assert [item["text"] for item in restored["long_memory"]] == ["旧记忆"]
    assert len(restored["conversations"]) == 1
    assert len(restored["mood_checkins"]) == 1


def test_failed_replace_keeps_a_recoverable_backup_when_rollback_also_fails(isolated_user, caplog):
    _seed(isolated_user)
    raw = _export_bytes(isolated_user, long_memory=[{"text": "新记忆", "time": "2026-08-04T00:00:00"}])

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
        long_memory=[{"text": "旧记忆", "time": "2026-08-01T00:00:00"}, {"text": "新记忆", "time": "2026-08-04T00:00:00"}],
    )

    result = user_data_import.import_user_export(isolated_user, raw, mode="merge")

    assert result["memories"] == 1


def test_export_from_another_account_is_rejected(isolated_user):
    raw = _export_bytes("someone-else")

    with pytest.raises(ValueError, match="different user ID"):
        user_data_import.import_user_export(isolated_user, raw, mode="merge")
