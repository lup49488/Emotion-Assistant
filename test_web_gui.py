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
    state.history.append({"role": "user", "content": "hello"})
    state.emotion_memory.append({"label": "joy", "score": 0.9, "time": "2026-01-01T00:00:00"})
    state.long_memory.append({"text": "likes coding", "time": "2026-01-02T00:00:00"})
    state.interest_store.append({"text": "likes NLP", "time": "2026-01-03T00:00:00"})

    text = Web_GUI.format_memory_snapshot("alice", state)

    assert "alice" in text
    assert "hello" in text
    assert "joy" in text
    assert "likes coding" in text
    assert "likes NLP" in text


def test_clear_memory_section_clears_selected_interest_memory():
    state = SessionState(user_id="alice")
    state.history.append({"role": "user", "content": "keep me"})
    state.interest_store.append({"text": "remove me"})
    state.vector_index = Mock()

    with patch.object(Web_GUI, "session_store", DummySessionStore(state)):
        result = Web_GUI.clear_memory_section("alice", "interest")

    assert "alice" in result
    assert state.history == [{"role": "user", "content": "keep me"}]
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

    assert result.startswith("[配置错误]")
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

    assert result == "[成功] API 连接成功：deepseek / deepseek-chat"
    fake_openai.assert_called_once_with(api_key="test-key", base_url="https://api.deepseek.com")
    fake_client.chat.completions.create.assert_called_once()


def test_classify_connection_error_groups_network_errors():
    result = Web_GUI.classify_connection_error(TimeoutError("request timeout"), "deepseek")

    assert result.startswith("[网络错误]")
    assert "timeout" in result


def test_save_model_config_to_env_updates_local_env_without_blank_key(tmp_path):
    env_path = tmp_path / ".env.local"
    env_path.write_text("DEEPSEEK_API_KEY=old-key\nLLM_PROVIDER=local_hf\n", encoding="utf-8")

    with patch.object(Web_GUI, "LOCAL_ENV_PATH", env_path):
        result = Web_GUI.save_model_config_to_env(
            provider="deepseek",
            model="deepseek-reasoner",
            base_url="https://api.deepseek.com",
            api_key="",
            temperature=0.2,
            top_p=0.8,
            max_new_tokens=128,
        )

    text = env_path.read_text(encoding="utf-8")
    assert "已保存模型配置" in result
    assert "DEEPSEEK_API_KEY=old-key" in text
    assert "LLM_PROVIDER=deepseek" in text
    assert "LLM_API_MODEL=deepseek-reasoner" in text
    assert "DEEPSEEK_MODEL=deepseek-reasoner" in text
    assert "LLM_MAX_NEW_TOKENS=128" in text


def test_save_model_config_to_env_writes_explicit_api_key(tmp_path):
    env_path = tmp_path / ".env.local"

    with patch.object(Web_GUI, "LOCAL_ENV_PATH", env_path):
        Web_GUI.save_model_config_to_env(
            provider="openai",
            model="gpt-4.1-mini",
            base_url="",
            api_key="new-key",
            temperature=0.5,
            top_p=1,
            max_new_tokens=64,
        )

    text = env_path.read_text(encoding="utf-8")
    assert "OPENAI_API_KEY=new-key" in text
    assert "OPENAI_MODEL=gpt-4.1-mini" in text


def test_weekly_mood_chart_uses_dark_palette():
    points = [
        {
            "date": "2026-07-12",
            "label": "07-12",
            "mood": "开心",
            "intensity": 4,
            "note": "状态很好",
        }
    ]

    chart = Web_GUI._render_weekly_mood_chart(points, "dark")

    assert 'data-chart-theme="dark"' in chart
    assert "background:#111827" in chart
    assert 'fill="#f8fafc"' in chart
    assert 'stroke="#60a5fa"' in chart


def test_weekly_mood_chart_defaults_to_light_palette():
    chart = Web_GUI._render_weekly_mood_chart([])

    assert 'data-chart-theme="light"' in chart
    assert "background:#ffffff" in chart
    assert 'fill="#6b7280"' in chart
