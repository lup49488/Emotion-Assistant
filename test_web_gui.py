from unittest.mock import Mock, patch
import json
import re
from pathlib import Path

import export_store
import gui_knowledge
import gui_memory
import session_store
import chatbot
import Web_GUI
from gui_model_options import PROVIDER_CHOICES
from gui_i18n import EN_TRANSLATIONS, localize_status_text
from gui_auth import authorize_or_message
from onboarding_store import onboarding_completed
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
    state.stable_profile.append({"text": "is a student", "kind": "profile"})
    state.interest_store.append({"text": "likes NLP", "time": "2026-01-03T00:00:00"})

    text = Web_GUI.format_memory_snapshot("alice", state)

    assert "alice" in text
    assert "hello" in text
    assert "joy" in text
    assert "likes coding" in text
    assert "is a student" in text
    assert "likes NLP" in text


def test_format_memory_event_log_explains_action_and_reason():
    state = SessionState(user_id="alice")
    state.memory_events.append({
        "time": "2026-07-15T10:00:00",
        "section": "stable",
        "action": "added",
        "text": "我是学生",
        "reason": "识别到明确身份",
        "score": 1.5,
    })

    text = Web_GUI.format_memory_event_log("alice", state)

    assert "稳定资料 | 新增" in text
    assert "我是学生" in text
    assert "识别到明确身份" in text


def test_memory_quality_report_flags_duplicate_empty_stale_and_invalid_items():
    state = SessionState(user_id="alice")
    state.long_memory.extend([
        {"text": "我是一名学生", "time": "2020-01-01T00:00:00"},
        {"text": "", "time": "not-a-time"},
    ])
    state.interest_store.append({"text": "我是一名学生", "time": ""})

    report = gui_memory.build_memory_quality_report("alice", state)

    assert "记忆质量评分：" in report
    assert "缺少有效内容" in report
    assert "重复" in report
    assert "超过" in report
    assert "无法解析的时间" in report


def test_clear_memory_section_clears_selected_interest_memory():
    state = SessionState(user_id="alice")
    state.history.append({"role": "user", "content": "keep me"})
    state.interest_store.append({"text": "remove me"})
    state.vector_index = Mock()

    with patch.object(gui_memory, "session_store", DummySessionStore(state)):
        with patch.object(gui_memory, "authorize_or_message", return_value=("alice", None)):
            result = Web_GUI.clear_memory_section("alice", "alice-secret", "interest")

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


def test_gui_provider_choices_hide_local_hf():
    assert "local_hf" not in PROVIDER_CHOICES
    assert "deepseek" in PROVIDER_CHOICES
    assert "nvidia_nim" in PROVIDER_CHOICES


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


def test_save_model_config_to_env_writes_nvidia_nim_settings(tmp_path):
    env_path = tmp_path / ".env.local"

    with patch.object(Web_GUI, "LOCAL_ENV_PATH", env_path):
        Web_GUI.save_model_config_to_env(
            provider="nvidia_nim",
            model="openai/gpt-oss-20b",
            base_url="https://integrate.api.nvidia.com/v1",
            api_key="nim-key",
            temperature=0.5,
            top_p=1,
            max_new_tokens=256,
        )

    text = env_path.read_text(encoding="utf-8")
    assert "LLM_PROVIDER=nvidia_nim" in text
    assert "NVIDIA_NIM_API_KEY=nim-key" in text
    assert "NVIDIA_NIM_MODEL=openai/gpt-oss-20b" in text
    assert "NVIDIA_NIM_BASE_URL=https://integrate.api.nvidia.com/v1" in text


def test_save_local_runtime_config_writes_supported_values(tmp_path):
    env_path = tmp_path / ".env.local"

    with patch.object(Web_GUI, "LOCAL_ENV_PATH", env_path):
        result = Web_GUI.save_local_runtime_config_to_env(
            "bfloat16", "sdpa", True, False, 6
        )

    text = env_path.read_text(encoding="utf-8")
    assert "已保存本地模型运行配置" in result
    assert "LOCAL_MODEL_DTYPE=bfloat16" in text
    assert "LOCAL_MODEL_ATTN_IMPLEMENTATION=sdpa" in text
    assert "LOCAL_MODEL_LOW_CPU_MEM_USAGE=true" in text
    assert "LOCAL_MODEL_COMPILE=false" in text
    assert "LOCAL_MODEL_CPU_THREADS=6" in text


def test_connection_test_detail_includes_timing_and_suggestion():
    detail = Web_GUI.build_connection_test_detail(
        "[认证失败] API Key 无效或无权限",
        "deepseek",
        "deepseek-chat",
        "https://api.deepseek.com",
        0.42,
    )

    assert "耗时：0.42 秒" in detail
    assert "API Key、账户权限" in detail
    assert "deepseek-chat" in detail


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


def test_load_theme_and_weekly_chart_requires_access_key(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")
    Web_GUI.submit_mood_checkin("alice", "alice-secret", "2026-07-12", "开心", 4, "")

    theme, chart, summary = Web_GUI.load_theme_and_weekly_chart(
        "alice", "", "2026-07-12", "light"
    )

    assert theme == "light"
    assert "开心" not in chart
    assert "请输入 User ID 和访问密码" in summary


def test_load_theme_and_weekly_chart_allowed_with_correct_passphrase(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")
    Web_GUI.submit_mood_checkin("alice", "alice-secret", "2026-07-12", "开心", 4, "")

    _, chart, summary = Web_GUI.load_theme_and_weekly_chart(
        "alice", "alice-secret", "2026-07-12", "light"
    )

    assert "开心" in chart
    assert "记录天数" in summary


def test_save_access_key_from_gui_reports_saved_password(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")

    result = Web_GUI.save_access_key_from_gui("alice", "alice-secret")

    assert "已为用户 alice 保存访问密码" in result


def test_save_access_key_from_gui_reports_verified_password(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")
    Web_GUI.save_access_key_from_gui("alice", "alice-secret")

    result = Web_GUI.save_access_key_from_gui("alice", "alice-secret")

    assert "验证通过" in result


def test_change_access_key_from_gui_updates_password(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")
    Web_GUI.save_access_key_from_gui("alice", "alice-secret")

    message, _, _ = Web_GUI.change_access_key_from_gui(
        "alice", "alice-secret", "new-secret!"
    )

    assert "已修改" in message
    assert "不正确" in Web_GUI.refresh_mood_panel("alice", "alice-secret")
    assert "还没有 Mood Check-in 记录" in Web_GUI.refresh_mood_panel("alice", "new-secret!")


def test_admin_recover_access_key_from_gui_updates_password(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")
    recovery_key = "server-recovery-key-12345"
    monkeypatch.setenv("CHATBOT_ADMIN_RECOVERY_KEY", recovery_key)
    Web_GUI.save_access_key_from_gui("alice", "alice-secret")

    message, _, _, login_status = Web_GUI.admin_recover_access_key_and_status(
        "alice", recovery_key, "new-secret!"
    )

    assert "恢复成功" in message
    assert login_status == "已验证：alice"
    assert "不正确" in Web_GUI.refresh_mood_panel("alice", "alice-secret")
    assert "还没有 Mood Check-in 记录" in Web_GUI.refresh_mood_panel("alice", "new-secret!")


def test_english_gui_translations_cover_primary_navigation():
    assert EN_TRANSLATIONS["情绪感知对话助手"] == "Emotion-Aware Chat Assistant"
    assert EN_TRANSLATIONS["用户访问"] == "User access"
    assert EN_TRANSLATIONS["管理员恢复"] == "Administrator recovery"
    assert EN_TRANSLATIONS["知识库 / RAG"] == "Knowledge base / RAG"


def test_auth_required_message_switches_between_english_and_chinese():
    english = localize_status_text(Web_GUI.AUTH_REQUIRED_MESSAGE, "en")
    chinese = localize_status_text(english, "zh-CN")

    assert english.startswith("Enter a User ID and access password")
    assert chinese == Web_GUI.AUTH_REQUIRED_MESSAGE


def test_runtime_status_uses_current_gui_locale():
    english = Web_GUI.refresh_status("local_hf", "demo-model", "", "", "en")
    chinese = Web_GUI.refresh_status("local_hf", "demo-model", "", "", "zh-CN")

    assert "Background warm-up:" in english
    assert "Connection test:" in english
    assert "Conversation status:" in english
    assert "后台预热：" in chinese


def test_existing_status_values_are_relocalized_after_language_switch():
    english, = Web_GUI.relocalize_status_values("未验证", "en")
    chinese, = Web_GUI.relocalize_status_values(english, "zh-CN")

    assert english == "Not verified"
    assert chinese == "未验证"


def test_chat_auth_prompt_uses_session_locale():
    response = Web_GUI.respond(
        "hello",
        [],
        "",
        "",
        False,
        False,
        True,
        "local_hf",
        "demo-model",
        "",
        "",
        0.8,
        0.9,
        64,
        "en",
    )

    message, _ = next(response)
    assert message.startswith("Enter a User ID and access password")


def test_locale_sync_callback_persists_browser_locale():
    *localized, locale = Web_GUI.relocalize_status_values_and_locale("未验证", "en")

    assert localized == ["Not verified"]
    assert locale == "en"


def test_interface_mode_visibility_hides_all_advanced_sections_by_default():
    updates = Web_GUI.interface_mode_visibility("simple")

    assert len(updates) == 8
    assert all(update["visible"] is False for update in updates[:7])
    assert updates[7]["selected"] == "chat"


def test_interface_mode_visibility_shows_all_advanced_sections():
    updates = Web_GUI.interface_mode_visibility("advanced")

    assert len(updates) == 8
    assert all(update["visible"] is True for update in updates[:7])
    assert "selected" not in updates[7]


def test_dismiss_onboarding_hides_the_session_guide(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")
    Web_GUI.save_access_key_from_gui("alice", "alice-secret")

    update = Web_GUI.dismiss_onboarding("alice", "alice-secret")

    assert update["visible"] is False
    assert onboarding_completed("alice") is True


def test_saved_onboarding_preference_hides_guide_after_login(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")
    Web_GUI.save_access_key_from_gui("alice", "alice-secret")
    Web_GUI.dismiss_onboarding("alice", "alice-secret")

    update = Web_GUI.onboarding_visibility_after_login("alice", "alice-secret")

    assert update["visible"] is False


def test_onboarding_guide_strings_have_english_translations():
    from gui_onboarding import (
        ONBOARDING_COMPLETE_LABEL,
        ONBOARDING_GUIDE_TEXT,
        ONBOARDING_TITLE,
    )

    assert EN_TRANSLATIONS[ONBOARDING_TITLE] == "First-use guide"
    assert ONBOARDING_GUIDE_TEXT in EN_TRANSLATIONS
    assert EN_TRANSLATIONS[ONBOARDING_GUIDE_TEXT].startswith("1. Set a User ID")
    assert ONBOARDING_COMPLETE_LABEL in EN_TRANSLATIONS


def test_mood_panel_auth_prompt_uses_english_locale():
    result = Web_GUI.refresh_mood_panel("", "", "en")

    assert result.startswith("Enter a User ID and access password")


def test_mood_panel_wrong_passphrase_is_english_under_english_locale(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")
    Web_GUI.refresh_mood_panel("alice", "alice-secret", "zh-CN")

    result = Web_GUI.refresh_mood_panel("alice", "wrong-pass!", "en")

    assert "incorrect" in result
    assert "不正确" not in result


def test_memory_editor_auth_prompt_uses_english_locale():
    editor_text, message = Web_GUI.load_memory_editor("", "", "history", "en")

    assert editor_text == ""
    assert message.startswith("Enter a User ID and access password")


def test_weekly_dashboard_auth_prompt_uses_english_locale():
    theme, chart, summary, analysis = Web_GUI.load_theme_and_weekly_dashboard(
        "", "", "2026-07-15", "light", "en"
    )

    assert theme == "light"
    assert summary.startswith("Enter a User ID and access password")
    assert analysis.startswith("Enter a User ID and access password")


def test_mood_panel_auth_prompt_stays_chinese_by_default():
    result = Web_GUI.refresh_mood_panel("", "")

    assert result.startswith("请输入 User ID 和访问密码")


def test_mood_history_is_english_under_english_locale(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")

    result = Web_GUI.submit_mood_checkin(
        "alice", "alice-secret", "2026-07-15", "开心", 4, "完成了目标", "en"
    )

    assert result.startswith("Saved 2026-07-15 mood check-in.")
    assert "User: alice" in result
    assert "Happy" in result
    assert "intensity 4/5" in result
    assert "已保存" not in result
    assert "用户：" not in result


def test_mood_history_stays_chinese_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")

    result = Web_GUI.submit_mood_checkin(
        "alice", "alice-secret", "2026-07-15", "开心", 4, ""
    )

    assert result.startswith("已保存 2026-07-15 的 Mood Check-in。")
    assert "用户：alice" in result


def test_weekly_summary_and_analysis_are_english_under_english_locale(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")
    Web_GUI.submit_mood_checkin("alice", "alice-secret", "2026-07-14", "平静", 2, "")
    Web_GUI.submit_mood_checkin("alice", "alice-secret", "2026-07-15", "开心", 4, "")

    chart, summary, analysis = Web_GUI.refresh_weekly_mood_dashboard(
        "alice", "alice-secret", "2026-07-15", "light", "en"
    )

    assert "Average intensity:" in summary
    assert "Trend:" in summary
    assert "平均强度" not in summary
    assert "Mood variation analysis" in analysis
    assert "情绪波动" not in analysis
    assert 'aria-label="Weekly Mood Chart"' in chart
    assert ">Intensity</text>" in chart


def test_weekly_dashboard_stays_chinese_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")
    Web_GUI.submit_mood_checkin("alice", "alice-secret", "2026-07-15", "开心", 4, "")

    chart, summary, analysis = Web_GUI.refresh_weekly_mood_dashboard(
        "alice", "alice-secret", "2026-07-15", "light"
    )

    assert "平均强度" in summary
    assert "情绪波动分析" in analysis
    assert ">强度</text>" in chart


def test_save_mood_dashboard_flow_is_english_under_english_locale(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")

    mood_panel, end_date, chart, summary, analysis = (
        Web_GUI.submit_mood_checkin_and_refresh_dashboard(
            "alice", "alice-secret", "2026-07-15", "疲惫", 3, "", "2026-07-15", "light", "en"
        )
    )

    assert mood_panel.startswith("Saved 2026-07-15 mood check-in.")
    assert end_date == "2026-07-15"
    assert "Average intensity:" in summary
    assert "Mood variation analysis" in analysis


def test_login_status_reports_verified_user(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")
    Web_GUI.save_access_key_from_gui("alice", "alice-secret")

    result = Web_GUI.login_status_text("alice", "alice-secret")

    assert result == "已验证：alice"


def test_blank_user_id_does_not_fall_back_to_local():
    user_id, message = authorize_or_message("", "some-password")

    assert user_id is None
    assert "请输入 User ID" in message


def test_export_user_data_from_gui_requires_access_key(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")

    message, file_path = Web_GUI.export_user_data_from_gui("alice", "")

    assert "请输入 User ID 和访问密码" in message
    assert file_path is None


def test_export_user_data_from_gui_writes_private_user_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")
    monkeypatch.setattr(chatbot, "USERS_DIR", tmp_path / "users")
    monkeypatch.setattr(export_store, "EXPORTS_DIR", tmp_path / "exports")
    Web_GUI.submit_mood_checkin("alice", "alice-secret", "2026-07-12", "开心", 4, "ok")
    Web_GUI.add_stable_profile("alice", "alice-secret", "我是学生")

    message, file_path = Web_GUI.export_user_data_from_gui("alice", "alice-secret")

    assert "已导出 alice 的用户数据" in message
    assert file_path is not None
    payload = json.loads((tmp_path / "exports" / Path(file_path).name).read_text(encoding="utf-8"))
    assert payload["user_id"] == "alice"
    assert payload["mood_checkins"][0]["mood"] == "开心"
    assert payload["stable_profile"][0]["text"] == "我是学生"
    assert "access_key" not in json.dumps(payload, ensure_ascii=False)


def test_refresh_mood_panel_registers_passphrase_on_first_use(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")

    result = Web_GUI.refresh_mood_panel("alice", "alice-secret")

    assert "还没有 Mood Check-in 记录" in result


def test_refresh_mood_panel_blocks_wrong_passphrase(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")
    Web_GUI.refresh_mood_panel("alice", "alice-secret")

    result = Web_GUI.refresh_mood_panel("alice", "someone-else-guess")

    assert "不正确" in result
    assert "Mood Check-in" not in result


def test_submit_mood_checkin_blocked_without_correct_passphrase(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")
    Web_GUI.refresh_mood_panel("alice", "alice-secret")

    result = Web_GUI.submit_mood_checkin(
        "alice", "wrong-pass", "2026-07-12", "开心", 4, ""
    )

    assert "不正确" in result
    assert Web_GUI.refresh_mood_panel("alice", "alice-secret") == "用户：alice\n\n还没有 Mood Check-in 记录。"


def test_load_memory_editor_blocked_without_correct_passphrase(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")
    Web_GUI.refresh_mood_panel("alice", "alice-secret")
    state = SessionState(user_id="alice")
    state.history.append({"role": "user", "content": "private"})

    with patch.object(gui_memory, "session_store", DummySessionStore(state)):
        editor_text, message = Web_GUI.load_memory_editor("alice", "wrong-pass", "history")

    assert editor_text == ""
    assert "不正确" in message
    assert "private" not in message


def test_load_memory_editor_allowed_with_correct_passphrase(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")
    Web_GUI.refresh_mood_panel("alice", "alice-secret")
    state = SessionState(user_id="alice")
    state.history.append({"role": "user", "content": "private"})

    with patch.object(gui_memory, "session_store", DummySessionStore(state)):
        editor_text, message = Web_GUI.load_memory_editor("alice", "alice-secret", "history")

    assert "private" in editor_text


def _has_chinese(text: str) -> bool:
    return bool(re.search(r"[一-鿿]", text))


def test_memory_snapshot_is_english_under_english_locale(tmp_path, monkeypatch):
    # A distinct user id keeps this out of the process-wide SessionStore cache,
    # which other tests populate with Chinese user content.
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")
    Web_GUI.save_access_key_from_gui("i18n-snapshot", "alice-secret")

    snapshot = Web_GUI.load_memory_panel("i18n-snapshot", "alice-secret", "en")

    assert "User: i18n-snapshot" in snapshot
    assert "## Recent conversations" in snapshot
    assert "## Stable profile" in snapshot
    assert "No entries" in snapshot
    assert not _has_chinese(snapshot)


def test_memory_snapshot_stays_chinese_by_default(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")
    Web_GUI.save_access_key_from_gui("i18n-snapshot-zh", "alice-secret")

    snapshot = Web_GUI.load_memory_panel("i18n-snapshot-zh", "alice-secret")

    assert "用户：i18n-snapshot-zh" in snapshot
    assert "## 短期对话" in snapshot
    assert "暂无记录" in snapshot


def test_memory_event_log_and_status_are_english_under_english_locale(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")
    user = "i18n-events"
    Web_GUI.save_access_key_from_gui(user, "alice-secret")
    Web_GUI.save_memory_editor(user, "alice-secret", "history", "[]", "en")

    event_log = Web_GUI.load_memory_event_log(user, "alice-secret", "en")
    backup_status, _ = gui_memory.backup_memory_from_gui(user, "alice-secret", "en")

    assert "memory decisions:" in event_log
    assert "Reason:" in event_log
    assert not _has_chinese(event_log)
    assert backup_status.startswith("Backed up all memory:")


def test_memory_quality_report_counts_are_english(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")
    Web_GUI.save_access_key_from_gui("i18n-quality", "alice-secret")

    report = gui_memory.assess_memory_quality("i18n-quality", "alice-secret", "en")

    assert "Recent conversations:" in report
    assert "Stable profile:" in report
    assert not _has_chinese(report)


def test_knowledge_quality_report_is_english_under_english_locale():
    report = gui_knowledge.format_knowledge_quality_report("en")

    assert "Quality level:" in report
    assert "Documents / chunks:" in report
    assert "Findings:" in report
    assert not _has_chinese(report)


def test_knowledge_quality_report_stays_chinese_by_default():
    report = gui_knowledge.format_knowledge_quality_report()

    assert "质量等级：" in report
    assert "检查结果：" in report


def test_knowledge_search_diagnostics_prompt_is_english():
    message = gui_knowledge.format_knowledge_search_diagnostics("", 4, 0.35, 3, "en")

    assert message == "Enter a retrieval question."


def _isolate_rag_evaluation_report(tmp_path, monkeypatch):
    # 评估报告路径指向空的临时文件，避免受本机真实评估结果污染。
    import rag_evaluation_store

    monkeypatch.setattr(
        rag_evaluation_store, "RAG_EVALUATION_REPORTS_PATH", tmp_path / "reports.json"
    )


def test_rag_status_panels_are_english_under_english_locale(tmp_path, monkeypatch):
    _isolate_rag_evaluation_report(tmp_path, monkeypatch)
    knowledge, style, evaluation = Web_GUI.load_rag_status_panels("en")

    assert "Documents: " in knowledge
    assert "Chunks: " in knowledge
    assert "Index status: " in knowledge
    assert "文档数：" not in knowledge
    assert "Style documents: " in style
    assert "Style chunks: " in style
    assert "风格文档数：" not in style
    assert evaluation == "No RAG evaluation has been run."


def test_rag_status_panels_stay_chinese_by_default(tmp_path, monkeypatch):
    _isolate_rag_evaluation_report(tmp_path, monkeypatch)
    knowledge, style, evaluation = Web_GUI.load_rag_status_panels("zh-CN")

    assert "文档数：" in knowledge
    assert "索引状态：" in knowledge
    assert "风格文档数：" in style
    assert evaluation == "尚未运行 RAG 评估。"


def test_style_import_prompt_and_preview_are_english_under_english_locale():
    import_message = Web_GUI.import_style_files(None, "en")
    preview_message = Web_GUI.preview_style_search("", "en")

    assert import_message.startswith("Select style files to import first.")
    assert "请先选择" not in import_message
    assert preview_message == "Enter a retrieval question or the current conversation intent."


def test_chat_error_is_localized_under_english_locale(tmp_path, monkeypatch):
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")
    Web_GUI.save_access_key_from_gui("i18n-chat-err", "alice-secret")

    def fake_stream(*args, **kwargs):
        # chatbot.chat 现在会把模型/限额失败作为异常抛出，而不是产出错误文本。
        raise RuntimeError("boom")
        yield  # pragma: no cover  (makes this a generator)

    with patch.object(Web_GUI, "handle_user_message_stream", fake_stream), \
         patch.object(Web_GUI, "ensure_conversation", return_value={"id": "conv-1"}), \
         patch.object(Web_GUI, "conversation_choices", return_value=[]):
        outputs = [text for text, _ in Web_GUI.respond(
            "hi", [], "i18n-chat-err", "alice-secret", False, False, False,
            "deepseek", "deepseek-chat", "", "key", 0.8, 0.9, 64, "en",
        )]

    # 界面把抛出的异常本地化为英文的“请求失败”提示。
    assert outputs[-1].startswith("Request failed: ")
    assert "请求失败" not in outputs[-1]


def test_provider_dropdown_labels_claude_by_name_without_changing_the_stored_value():
    """The stored value stays "anthropic"; only the label an operator reads changes."""
    from gui_model_options import provider_dropdown_choices

    choices = provider_dropdown_choices()
    labels = {label for label, _ in choices}
    values = [value for _, value in choices]

    assert ("Anthropic (Claude)", "anthropic") in choices
    assert any("Claude" in label for label in labels)
    assert values == PROVIDER_CHOICES
