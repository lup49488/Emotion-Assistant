import json
from unittest.mock import Mock, patch

import pytest

import memory_backup
from session_store import SessionState


def _state(user_id: str = "alice") -> SessionState:
    state = SessionState(user_id=user_id)
    state.vector_index = Mock()
    return state


def test_memory_backup_round_trip_replace(tmp_path):
    state = _state()
    state.history = [{"role": "user", "content": "hello"}]
    state.stable_profile = [{"text": "我是学生", "kind": "profile"}]
    state.interest_store.replace_all([{"text": "喜欢 NLP"}])

    with patch.object(memory_backup, "BACKUPS_DIR", tmp_path):
        path = memory_backup.create_memory_backup("alice", state)
        payload = memory_backup.load_memory_backup(path)

    state.history = []
    state.stable_profile = []
    state.interest_store.replace_all([])
    result = memory_backup.restore_memory_payload(state, payload, mode="replace")

    assert result["source_user_id"] == "alice"
    assert state.history[0]["content"] == "hello"
    assert state.stable_profile[0]["text"] == "我是学生"
    assert state.interest_store.items[0]["text"] == "喜欢 NLP"
    state.vector_index.mark_dirty_for_rebuild.assert_called()


def test_memory_restore_merge_deduplicates_exact_items():
    state = _state()
    item = {"text": "喜欢编程"}
    state.interest_store.replace_all([item])
    payload = {key: [] for key in memory_backup.MEMORY_KEYS}
    payload.update({"kind": "memory_backup", "interest_memory": [item, {"text": "喜欢音乐"}]})

    memory_backup.restore_memory_payload(state, payload, mode="merge")

    assert [item["text"] for item in state.interest_store.items] == ["喜欢编程", "喜欢音乐"]


def test_invalid_backup_is_rejected_before_state_changes():
    state = _state()
    state.history = [{"role": "user", "content": "keep"}]
    payload = {key: [] for key in memory_backup.MEMORY_KEYS}
    payload["stable_profile"] = [{"missing": "text"}]

    with pytest.raises(ValueError, match="non-empty|非空"):
        memory_backup.restore_memory_payload(state, payload, mode="replace")

    assert state.history == [{"role": "user", "content": "keep"}]


def test_backup_does_not_include_secrets(tmp_path):
    state = _state()
    with patch.object(memory_backup, "BACKUPS_DIR", tmp_path):
        path = memory_backup.create_memory_backup("alice", state)
    text = path.read_text(encoding="utf-8")

    assert "access_key" not in text
    assert "api_key" not in text.lower()
    assert json.loads(text)["kind"] == "memory_backup"
