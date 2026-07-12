from unittest.mock import Mock, patch

import Web_GUI
from session_store import SessionState


class DummySessionStore:
    def __init__(self, state):
        self.state = state

    def session(self, user_id):
        class Manager:
            def __enter__(inner_self):
                return self.state

            def __exit__(inner_self, exc_type, exc, tb):
                return False

        return Manager()


def test_format_memory_snapshot_includes_all_sections():
    state = SessionState(user_id="alice")
    state.history.append({"role": "user", "content": "你好"})
    state.emotion_memory.append({"label": "joy", "score": 0.9, "time": "2026-01-01T00:00:00"})
    state.long_memory.append({"text": "用户喜欢写代码", "time": "2026-01-02T00:00:00"})
    state.interest_store.append({"text": "喜欢自然语言处理", "time": "2026-01-03T00:00:00"})

    text = Web_GUI.format_memory_snapshot("alice", state)

    assert "用户：alice" in text
    assert "短期对话 1 条" in text
    assert "情绪记忆 1 条" in text
    assert "长期记忆 1 条" in text
    assert "兴趣记忆 1 条" in text
    assert "喜欢自然语言处理" in text


def test_clear_memory_section_clears_selected_interest_memory():
    state = SessionState(user_id="alice")
    state.history.append({"role": "user", "content": "保留这条"})
    state.interest_store.append({"text": "待清理兴趣"})
    state.vector_index = Mock()

    with patch.object(Web_GUI, "session_store", DummySessionStore(state)):
        result = Web_GUI.clear_memory_section("alice", "interest")

    assert "已清理 alice 的兴趣记忆" in result
    assert state.history == [{"role": "user", "content": "保留这条"}]
    assert state.interest_store.items == []
    state.vector_index.mark_dirty_for_rebuild.assert_called_once()


def test_test_model_connection_reports_missing_api_key():
    with patch.dict("os.environ", {
        "DEEPSEEK_API_KEY": "",
        "LLM_API_KEY": "",
    }, clear=False):
        result = Web_GUI.test_model_connection(
            provider="deepseek",
            model="deepseek-chat",
            base_url="https://api.deepseek.com",
            api_key="",
            temperature=0,
            top_p=1,
            max_new_tokens=1,
        )

    assert result.startswith("连接失败：")
    assert "API Key" in result


def test_test_model_connection_api_success():
    fake_client = Mock()
    fake_client.chat.completions.create.return_value = Mock(choices=[Mock()])
    fake_openai = Mock(return_value=fake_client)

    with patch.object(Web_GUI, "require_openai_client", return_value=fake_openai):
        result = Web_GUI.test_model_connection(
            provider="deepseek",
            model="deepseek-chat",
            base_url="https://api.deepseek.com",
            api_key="test-key",
            temperature=0,
            top_p=1,
            max_new_tokens=1,
        )

    assert result == "API 连接成功：deepseek / deepseek-chat"
    fake_openai.assert_called_once_with(api_key="test-key", base_url="https://api.deepseek.com")
    fake_client.chat.completions.create.assert_called_once()
