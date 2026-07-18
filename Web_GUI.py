from __future__ import annotations

import logging
import os
import threading
import time
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import gradio as gr
import gui_runtime_settings as runtime_settings

from app_logging import clear_logs, get_log_text, setup_gui_logging
import goemotions_local as goemotions
from chatbot import (
    DEFAULT_LLM_PROVIDER,
    get_embedding_model,
    get_llm,
    handle_user_message_stream,
    latest_memory_receipt,
    make_model_config,
    session_store,
)
from config import (
    BASE_DIR,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    KNOWLEDGE_CANDIDATE_MULTIPLIER,
    KNOWLEDGE_ENABLED,
    KNOWLEDGE_RETRIEVAL_THRESHOLD,
    KNOWLEDGE_TOP_K,
    LOCAL_MODEL_ATTN_IMPLEMENTATION,
    LOCAL_MODEL_COMPILE,
    LOCAL_MODEL_CPU_THREADS,
    LOCAL_MODEL_DTYPE,
    LOCAL_MODEL_LOW_CPU_MEM_USAGE,
    LOADED_ENV_FILES,
    STYLE_ENABLED,
)
from gui_auth import (
    AUTH_REQUIRED_MESSAGE,
    admin_recover_access_key,
    admin_recovery_status,
    authorize_or_message as _authorize_or_message,
    change_saved_access_key,
    save_or_verify_access_key,
)
from gui_i18n import GUI_I18N, localize_status_text, tr
from gui_model_options import (
    DEFAULT_BASE_URLS,
    DEFAULT_MODELS,
    MODEL_CHOICES,
    PROVIDER_API_KEYS,
    PROVIDER_CHOICES,
)
from gui_theme import LOAD_THEME_JS, REFRESH_CHART_THEME_JS, THEME_CSS
from gui_event_bindings import bind_gui_events
from gui_onboarding import build_onboarding, dismiss_onboarding, onboarding_visibility_after_login
from gui_tabs_advanced import build_advanced_tab
from gui_tabs_data import build_data_tab
from gui_tabs_knowledge import build_knowledge_tab
from export_store import export_user_data
from gui_memory import (
    MEMORY_SECTION_LABELS,
    clear_memory_section,
    clear_memory_section_and_reload,
    clear_stable_profile,
    format_memory_event_log,
    format_memory_snapshot,
    add_stable_profile,
    assess_memory_quality,
    backup_memory_from_gui,
    load_memory_editor,
    load_memory_event_log,
    load_memory_panel,
    load_stable_profile_editor,
    restore_memory_from_gui,
    save_memory_editor,
    save_stable_profile_editor,
)
from gui_knowledge import (
    clear_knowledge_documents,
    delete_knowledge_document,
    format_knowledge_document_list,
    format_knowledge_quality_report,
    format_knowledge_search_diagnostics,
    refresh_knowledge_document_panel,
)
from gui_mood import (
    _render_weekly_mood_chart,
    delete_mood_checkin_and_refresh_dashboard,
    delete_mood_checkin_from_gui,
    load_theme_and_weekly_dashboard,
    load_theme_and_weekly_chart,
    refresh_mood_panel,
    refresh_weekly_mood_dashboard,
    refresh_weekly_mood_chart,
    submit_mood_checkin_and_refresh_dashboard,
    submit_mood_checkin,
)
from knowledge_store import (
    build_knowledge_context,
    import_documents,
    knowledge_status,
    rebuild_knowledge_index,
)
from llm_providers import require_openai_client
from mood_store import MOOD_CHOICES
from style_store import (
    build_style_context,
    import_style_documents,
    rebuild_style_index,
    style_status,
)

setup_gui_logging()
logger = logging.getLogger(__name__)

LOCAL_ENV_PATH = BASE_DIR / ".env.local"

_status_lock = threading.Lock()
_warmup_status = "未启动"
_last_chat_status = "尚未开始对话"
_last_connection_status = "尚未测试连接"

LOCAL_DTYPE_CHOICES = runtime_settings.LOCAL_DTYPE_CHOICES
LOCAL_ATTENTION_CHOICES = [
    (tr("自动 / 默认"), ""),
    ("PyTorch SDPA", "sdpa"),
    ("Flash Attention 2", "flash_attention_2"),
    ("Eager", "eager"),
]

SYNC_LOCALE_JS = """
(...args) => {
    const locale = document.documentElement.lang || navigator.language || "zh-CN";
    const tabLabels = locale.toLowerCase().startsWith("en")
        ? {chat: "Chat", mood: "Mood", data: "My data", knowledge: "Knowledge", advanced: "Advanced settings"}
        : {chat: "对话", mood: "心情", data: "个人数据", knowledge: "知识库", advanced: "高级设置"};
    document.querySelectorAll('[role="tab"][data-tab-id]').forEach((tab) => {
        const label = tabLabels[tab.dataset.tabId];
        if (label && tab.textContent !== label) tab.textContent = label;
    });
    return [...args.slice(0, -1), locale];
}
"""


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def set_warmup_status(status: str) -> None:
    global _warmup_status
    with _status_lock:
        _warmup_status = f"{_now()} {status}"


def set_chat_status(status: str) -> None:
    global _last_chat_status
    with _status_lock:
        _last_chat_status = f"{_now()} {status}"


def set_connection_status(status: str) -> None:
    global _last_connection_status
    with _status_lock:
        _last_connection_status = f"{_now()} {status}"


def get_key_source(provider: str, gui_api_key: str = "") -> str:
    if gui_api_key.strip():
        return "GUI 输入"
    for env_name in PROVIDER_API_KEYS.get(provider, []):
        if os.getenv(env_name):
            return f"环境变量 {env_name}"
    if provider == "local_hf":
        return "本地模型不需要 API Key"
    return "未检测到 API Key"


def get_env_status() -> str:
    if not LOADED_ENV_FILES:
        return "未加载 .env 文件"
    paths = [Path(path).name for path in LOADED_ENV_FILES]
    return "已加载 " + ", ".join(paths)


def build_status_text(
    provider: str,
    model: str,
    base_url: str,
    api_key: str = "",
) -> str:
    with _status_lock:
        warmup = _warmup_status
        chat = _last_chat_status
        connection = _last_connection_status
    base_url_text = base_url.strip() or "默认/本地"
    key_source = get_key_source(provider, api_key)
    preload = os.getenv("GUI_PRELOAD_MODELS", "true")
    return "\n".join([
        f".env：{get_env_status()}",
        f"Provider：{provider or '未选择'}",
        f"Model：{model or '未选择'}",
        f"Base URL：{base_url_text}",
        f"API Key：{key_source}",
        f"后台预热：{warmup}",
        f"预热开关 GUI_PRELOAD_MODELS：{preload}",
        f"连接测试：{connection}",
        f"对话状态：{chat}",
    ])


def respond(
    message: str,
    history: list,
    user_id: str,
    access_key: str,
    use_knowledge: bool,
    use_style: bool,
    show_memory_receipt: bool,
    provider: str,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    locale: str = "zh-CN",
):
    user_id, auth_error = _authorize_or_message(user_id, access_key)
    if auth_error:
        yield localize_status_text(auth_error, locale)
        return

    config = make_model_config(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )
    partial = ""
    yielded_response = False
    logger.info(
        "开始对话请求：user=%s provider=%s model=%s",
        user_id,
        config.normalized_provider(),
        config.resolved_model(),
    )
    set_chat_status(f"正在请求 {config.normalized_provider()} / {config.resolved_model()}")
    try:
        for chunk in handle_user_message_stream(
            user_id,
            message,
            model_config=config,
            use_knowledge=use_knowledge,
            use_style=use_style,
        ):
            if chunk:
                partial += chunk
                yielded_response = True
                yield partial
        if show_memory_receipt:
            with session_store.session(user_id) as state:
                receipt = latest_memory_receipt(state)
            yield f"{partial}\n\n---\n{receipt}"
        elif not yielded_response:
            fallback = "模型本次没有返回可显示的文本，请重试；若问题持续出现，请检查运行日志。"
            logger.warning("GUI 收到空对话流。user=%s provider=%s", user_id, config.normalized_provider())
            set_chat_status("模型未返回文本")
            yield fallback
        set_chat_status("回复完成")
    except Exception as exc:
        set_chat_status(f"请求失败：{exc}")
        yield f"请求失败：{exc}"


def provider_changed(provider: str, api_key: str = "", locale: str = "zh-CN"):
    choices = MODEL_CHOICES.get(provider, MODEL_CHOICES["custom"])
    default_model = DEFAULT_MODELS.get(provider, choices[0])
    default_base_url = DEFAULT_BASE_URLS.get(provider, "")
    return (
        gr.update(choices=choices, value=default_model),
        gr.update(value=default_base_url),
        localize_status_text(
            build_status_text(provider, default_model, default_base_url, api_key), locale
        ),
    )


def refresh_status(
    provider: str,
    model: str,
    base_url: str,
    api_key: str,
    locale: str = "zh-CN",
) -> str:
    return localize_status_text(build_status_text(provider, model, base_url, api_key), locale)


def save_access_key_from_gui(user_id: str, access_key: str) -> str:
    return save_or_verify_access_key(user_id, access_key)


def login_status_text(user_id: str, access_key: str) -> str:
    user_id, auth_error = _authorize_or_message(user_id, access_key)
    if auth_error:
        return f"未验证：{auth_error}"
    return f"已验证：{user_id}"


def save_access_key_and_status(
    user_id: str, access_key: str, locale: str = "zh-CN"
) -> tuple[str, str, Any]:
    message = save_access_key_from_gui(user_id, access_key)
    return (
        localize_status_text(message, locale),
        localize_status_text(login_status_text(user_id, access_key), locale),
        onboarding_visibility_after_login(user_id, access_key),
    )


def change_access_key_from_gui(
    user_id: str,
    current_access_key: str,
    new_access_key: str,
) -> tuple[str, Any, Any]:
    message = change_saved_access_key(user_id, current_access_key, new_access_key)
    if message.startswith("访问密码已修改"):
        return message, gr.update(value=(new_access_key or "").strip()), gr.update(value="")
    return message, gr.update(), gr.update()


def change_access_key_and_status(
    user_id: str,
    current_access_key: str,
    new_access_key: str,
    locale: str = "zh-CN",
) -> tuple[str, Any, Any, str]:
    message, access_key_update, new_key_update = change_access_key_from_gui(
        user_id,
        current_access_key,
        new_access_key,
    )
    if message.startswith("访问密码已修改"):
        return (
            localize_status_text(message, locale),
            access_key_update,
            new_key_update,
            localize_status_text(login_status_text(user_id, new_access_key), locale),
        )
    return (
        localize_status_text(message, locale),
        access_key_update,
        new_key_update,
        localize_status_text(login_status_text(user_id, current_access_key), locale),
    )


def admin_recover_access_key_and_status(
    user_id: str,
    admin_recovery_key: str,
    new_access_key: str,
    locale: str = "zh-CN",
) -> tuple[str, Any, Any, str]:
    ok, message = admin_recover_access_key(user_id, admin_recovery_key, new_access_key)
    if ok:
        normalized_key = (new_access_key or "").strip()
        return (
            localize_status_text(message, locale),
            gr.update(value=normalized_key),
            gr.update(value=""),
            localize_status_text(login_status_text(user_id, normalized_key), locale),
        )
    return localize_status_text(message, locale), gr.update(), gr.update(value=""), localize_status_text("未验证", locale)


def export_user_data_from_gui(
    user_id: str, access_key: str, locale: str = "zh-CN"
) -> tuple[str, str | None]:
    user_id, auth_error = _authorize_or_message(user_id, access_key)
    if auth_error:
        return localize_status_text(auth_error, locale), None
    try:
        path = export_user_data(user_id)
        return localize_status_text(f"已导出 {user_id} 的用户数据：{path}", locale), str(path)
    except Exception as exc:
        return localize_status_text(f"导出用户数据失败：{exc}", locale), None


def relocalize_status_values(*values: Any) -> tuple[str, ...]:
    if not values:
        return ()
    *status_values, locale = values
    return tuple(localize_status_text(str(value or ""), str(locale)) for value in status_values)


def relocalize_status_values_and_locale(*values: Any) -> tuple[str, ...]:
    if not values:
        return ()
    *status_values, locale = values
    localized = relocalize_status_values(*status_values, locale)
    return (*localized, str(locale or "zh-CN"))


def sync_locale_value(locale: str) -> str:
    return str(locale or "zh-CN")


def interface_mode_visibility(mode: str) -> tuple[Any, ...]:
    visible = str(mode or "simple") == "advanced"
    visibility_updates = tuple(gr.update(visible=visible) for _ in range(7))
    tabs_update = gr.update() if visible else gr.update(selected="chat")
    return (*visibility_updates, tabs_update)


_env_quote = runtime_settings.env_quote
_read_env_lines = runtime_settings.read_env_lines
_upsert_env_values = runtime_settings.upsert_env_values
_provider_key_name = runtime_settings.provider_key_name


def save_model_config_to_env(
    provider: str,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> str:
    return runtime_settings.save_model_config_to_env(
        LOCAL_ENV_PATH, provider, model, base_url, api_key,
        temperature, top_p, max_new_tokens,
    )


def save_model_config_and_refresh(
    provider: str,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    locale: str = "zh-CN",
) -> tuple[str, str]:
    result = save_model_config_to_env(
        provider,
        model,
        base_url,
        api_key,
        temperature,
        top_p,
        max_new_tokens,
    )
    return (
        localize_status_text(result, locale),
        localize_status_text(build_status_text(provider, model, base_url, api_key), locale),
    )


_normalize_local_dtype = runtime_settings.normalize_local_dtype


def build_local_runtime_config_text(
    dtype: str,
    attention_implementation: str,
    low_cpu_mem_usage: bool,
    compile_model: bool,
    cpu_threads: int | float,
) -> str:
    return runtime_settings.build_local_runtime_config_text(
        dtype, attention_implementation, low_cpu_mem_usage, compile_model, cpu_threads
    )


def save_local_runtime_config_to_env(
    dtype: str,
    attention_implementation: str,
    low_cpu_mem_usage: bool,
    compile_model: bool,
    cpu_threads: int | float,
) -> str:
    return runtime_settings.save_local_runtime_config_to_env(
        LOCAL_ENV_PATH, dtype, attention_implementation,
        low_cpu_mem_usage, compile_model, cpu_threads,
    )


def save_local_runtime_config_and_refresh(
    dtype: str,
    attention_implementation: str,
    low_cpu_mem_usage: bool,
    compile_model: bool,
    cpu_threads: int | float,
    locale: str = "zh-CN",
) -> tuple[str, str]:
    try:
        message = save_local_runtime_config_to_env(
            dtype,
            attention_implementation,
            low_cpu_mem_usage,
            compile_model,
            cpu_threads,
        )
    except Exception as exc:
        return (
            localize_status_text(f"保存失败：{exc}", locale),
            localize_status_text(
                build_local_runtime_config_text(
                    dtype,
                    attention_implementation,
                    low_cpu_mem_usage,
                    compile_model,
                    cpu_threads,
                ),
                locale,
            ),
        )
    return (
        localize_status_text(message, locale),
        localize_status_text(
            build_local_runtime_config_text(
                dtype,
                attention_implementation,
                low_cpu_mem_usage,
                compile_model,
                cpu_threads,
            ),
            locale,
        ),
    )


def build_local_runtime_config_text_localized(
    dtype: str,
    attention_implementation: str,
    low_cpu_mem_usage: bool,
    compile_model: bool,
    cpu_threads: int | float,
    locale: str = "zh-CN",
) -> str:
    return localize_status_text(
        build_local_runtime_config_text(
            dtype,
            attention_implementation,
            low_cpu_mem_usage,
            compile_model,
            cpu_threads,
        ),
        locale,
    )


_connection_result = runtime_settings.connection_result
classify_connection_error = runtime_settings.classify_connection_error


def test_model_connection(
    provider: str,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> str:
    return runtime_settings.test_model_connection(
        provider, model, base_url, api_key, temperature, top_p, max_new_tokens,
        make_config=make_model_config,
        get_llm_fn=get_llm,
        require_openai_client_fn=require_openai_client,
        status_callback=set_connection_status,
    )


def build_connection_test_detail(
    result: str,
    provider: str,
    model: str,
    base_url: str,
    elapsed_seconds: float,
) -> str:
    return runtime_settings.build_connection_test_detail(
        result, provider, model, base_url, elapsed_seconds
    )


def test_model_connection_and_refresh(
    provider: str,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    locale: str = "zh-CN",
) -> tuple[str, str, str]:
    started_at = time.perf_counter()
    result = test_model_connection(
        provider,
        model,
        base_url,
        api_key,
        temperature,
        top_p,
        max_new_tokens,
    )
    detail = build_connection_test_detail(
        result,
        provider,
        model,
        base_url,
        time.perf_counter() - started_at,
    )
    return (
        localize_status_text(result, locale),
        localize_status_text(detail, locale),
        localize_status_text(build_status_text(provider, model, base_url, api_key), locale),
    )


def _uploaded_file_path(file_obj: Any) -> str:
    if isinstance(file_obj, (str, Path)):
        return str(file_obj)
    path = getattr(file_obj, "name", None) or getattr(file_obj, "path", None)
    if path:
        return str(path)
    raise ValueError(f"无法识别上传文件：{file_obj!r}")


def import_knowledge_files(files: list[Any] | None) -> str:
    if not files:
        return "请先选择要导入的文件。\n\n" + knowledge_status()
    paths = [_uploaded_file_path(file_obj) for file_obj in files]
    try:
        result = import_documents(paths)
    except Exception as exc:
        return f"导入失败：{exc}\n\n{knowledge_status()}"

    lines = [
        "导入完成。",
        f"新增文件：{len(result.get('imported', []))}",
        f"文档数：{result.get('documents', 0)}",
        f"片段数：{result.get('chunks', 0)}",
    ]
    imported = result.get("imported") or []
    if imported:
        lines.append("已导入：" + ", ".join(imported))
    errors = result.get("errors") or []
    if errors:
        lines.append("导入警告：")
        lines.extend(f"- {error}" for error in errors)
    lines.append("")
    lines.append(knowledge_status())
    return "\n".join(lines)


def rebuild_knowledge_panel() -> str:
    try:
        result = rebuild_knowledge_index()
    except Exception as exc:
        return f"重建索引失败：{exc}\n\n{knowledge_status()}"
    lines = [
        "索引重建完成。",
        f"文档数：{result.get('documents', 0)}",
        f"片段数：{result.get('chunks', 0)}",
    ]
    errors = result.get("errors") or []
    if errors:
        lines.append("重建警告：")
        lines.extend(f"- {error}" for error in errors)
    lines.append("")
    lines.append(knowledge_status())
    return "\n".join(lines)


def preview_knowledge_search(
    query: str,
    top_k: int = KNOWLEDGE_TOP_K,
    threshold: float = KNOWLEDGE_RETRIEVAL_THRESHOLD,
    candidate_multiplier: int = KNOWLEDGE_CANDIDATE_MULTIPLIER,
) -> str:
    return format_knowledge_search_diagnostics(query, top_k, threshold, candidate_multiplier)


def refresh_knowledge_documents_from_gui() -> tuple[Any, str]:
    names, document_list = refresh_knowledge_document_panel()
    return gr.update(choices=names, value=None), document_list


def import_knowledge_files_and_refresh(files: list[Any] | None) -> tuple[str, Any, str]:
    status = import_knowledge_files(files)
    selector, document_list = refresh_knowledge_documents_from_gui()
    return status, selector, document_list


def delete_knowledge_document_from_gui(name: str) -> tuple[Any, str, str]:
    names, document_list, status = delete_knowledge_document(name)
    return gr.update(choices=names, value=None), document_list, status


def clear_knowledge_documents_from_gui() -> tuple[Any, str, str]:
    names, document_list, status = clear_knowledge_documents()
    return gr.update(choices=names, value=None), document_list, status


def import_style_files(files: list[Any] | None) -> str:
    if not files:
        return "请先选择要导入的风格文件。\n\n" + style_status()
    paths = [_uploaded_file_path(file_obj) for file_obj in files]
    try:
        result = import_style_documents(paths)
    except Exception as exc:
        return f"导入失败：{exc}\n\n{style_status()}"

    lines = [
        "风格库导入完成。",
        f"新增文件：{len(result.get('imported', []))}",
        f"文档数：{result.get('documents', 0)}",
        f"片段数：{result.get('chunks', 0)}",
    ]
    imported = result.get("imported") or []
    if imported:
        lines.append("已导入：" + ", ".join(imported))
    errors = result.get("errors") or []
    if errors:
        lines.append("导入警告：")
        lines.extend(f"- {error}" for error in errors)
    lines.append("")
    lines.append(style_status())
    return "\n".join(lines)


def rebuild_style_panel() -> str:
    try:
        result = rebuild_style_index()
    except Exception as exc:
        return f"重建风格索引失败：{exc}\n\n{style_status()}"
    lines = [
        "风格索引重建完成。",
        f"文档数：{result.get('documents', 0)}",
        f"片段数：{result.get('chunks', 0)}",
    ]
    errors = result.get("errors") or []
    if errors:
        lines.append("重建警告：")
        lines.extend(f"- {error}" for error in errors)
    lines.append("")
    lines.append(style_status())
    return "\n".join(lines)


def preview_style_search(query: str) -> str:
    query = (query or "").strip()
    if not query:
        return "请输入检索问题或当前对话意图。"
    context = build_style_context(query)
    return context or "没有检索到高相关风格样例。"


def warmup_models(provider: str) -> None:
    try:
        set_warmup_status("正在预热情绪模型...")
        goemotions.predict_emotion_zh("你好")

        set_warmup_status("正在预热记忆向量模型...")
        get_embedding_model()

        if provider == "local_hf" and os.getenv("GUI_PRELOAD_LOCAL_LLM", "false").lower() == "true":
            set_warmup_status("正在预热本地聊天模型...")
            get_llm()
        elif provider == "local_hf":
            set_warmup_status("基础模型预热完成，本地聊天模型将按需加载")
            return

        set_warmup_status("预热完成")
    except Exception as exc:
        set_warmup_status(f"预热失败：{exc}")


def start_background_warmup(provider: str) -> None:
    if os.getenv("GUI_PRELOAD_MODELS", "true").lower() != "true":
        set_warmup_status("已关闭")
        return
    set_warmup_status("后台预热线程已启动")
    thread = threading.Thread(target=warmup_models, args=(provider,), daemon=True)
    thread.start()


def get_warmup_status() -> str:
    with _status_lock:
        return _warmup_status


initial_provider = os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER)
if initial_provider not in PROVIDER_CHOICES:
    initial_provider = DEFAULT_LLM_PROVIDER if DEFAULT_LLM_PROVIDER in PROVIDER_CHOICES else "local_hf"
initial_model_choices = MODEL_CHOICES.get(initial_provider, MODEL_CHOICES["local_hf"])
initial_model = DEFAULT_MODELS.get(initial_provider, initial_model_choices[0])
initial_base_url = DEFAULT_BASE_URLS.get(initial_provider, "")

with gr.Blocks(title=tr("情绪感知对话助手")) as demo:
    locale_probe_input = gr.Textbox(value="zh-CN", visible=False, render=False)
    provider_input = gr.Dropdown(
        label="Provider",
        choices=PROVIDER_CHOICES,
        value=initial_provider,
        render=False,
    )
    model_input = gr.Dropdown(
        label="Model",
        choices=initial_model_choices,
        value=initial_model,
        allow_custom_value=True,
        render=False,
    )
    base_url_input = gr.Textbox(
        label="API Base URL",
        value=initial_base_url,
        placeholder=tr("本地模型可留空；OpenAI 兼容接口可填写服务地址"),
        render=False,
    )
    api_key_input = gr.Textbox(
        label="API Key",
        value="",
        type="password",
        placeholder=tr("可留空并改用 .env，例如 DEEPSEEK_API_KEY"),
        render=False,
    )
    user_id_input = gr.Textbox(
        label="User ID",
        value=os.getenv("CHATBOT_USER_ID", "").strip(),
        placeholder=tr("首次使用请设置用户名"),
        render=False,
    )
    access_key_input = gr.Textbox(
        label=tr("访问密码"),
        value="",
        type="password",
        placeholder=tr("首次使用该用户名请设置密码；之后需输入同样的密码才能访问其数据"),
        render=False,
    )
    save_access_key_button = gr.Button(tr("保存/验证密码"), render=False)
    login_status_box = gr.Textbox(
        label=tr("登录状态"), value=tr("未验证"), lines=2, interactive=False, render=False
    )
    access_key_status = gr.Textbox(
        label=tr("访问密码状态"),
        value=tr("尚未验证访问密码。"),
        lines=3,
        interactive=False,
        render=False,
    )
    use_knowledge_input = gr.Checkbox(
        label=tr("启用知识库"),
        value=KNOWLEDGE_ENABLED,
        render=False,
    )
    use_style_input = gr.Checkbox(
        label=tr("启用风格参考"),
        value=STYLE_ENABLED,
        render=False,
    )
    show_memory_receipt_input = gr.Checkbox(
        label=tr("显示记忆回执"),
        value=True,
        render=False,
    )
    temperature_input = gr.Slider(
        label="Temperature",
        minimum=0.0,
        maximum=2.0,
        value=DEFAULT_TEMPERATURE,
        step=0.05,
        render=False,
    )
    top_p_input = gr.Slider(
        label="Top P",
        minimum=0.0,
        maximum=1.0,
        value=DEFAULT_TOP_P,
        step=0.05,
        render=False,
    )
    max_new_tokens_input = gr.Number(
        label="Max New Tokens",
        value=DEFAULT_MAX_NEW_TOKENS,
        precision=0,
        render=False,
    )
    with gr.Row():
        interface_mode_input = gr.Radio(
            label=tr("界面模式"),
            choices=[(tr("简洁模式"), "simple"), (tr("高级模式"), "advanced")],
            value="simple",
        )
        theme_mode_input = gr.Radio(
            label=tr("页面主题"),
            choices=[(tr("浅色"), "light"), (tr("深色"), "dark")],
            value="light",
            elem_id="theme-mode",
        )

    with gr.Tabs(selected="chat") as main_tabs:
        with gr.Tab(tr("tab_chat"), id="chat"):
            onboarding = build_onboarding(
                tr, show_on_first_use=not bool(os.getenv("CHATBOT_USER_ID", "").strip())
            )
            with gr.Accordion(tr("常用设置"), open=True):
                with gr.Row():
                    user_id_input.render()
                    access_key_input.render()
                save_access_key_button.render()
                with gr.Row():
                    login_status_box.render()
                    access_key_status.render()
                with gr.Row():
                    provider_input.render()
                    model_input.render()
                with gr.Row():
                    use_knowledge_input.render()
                    use_style_input.render()
                    show_memory_receipt_input.render()
            with gr.Column(visible=False) as advanced_chat_settings:
                with gr.Accordion(tr("生成参数"), open=False):
                    base_url_input.render()
                    api_key_input.render()
                    with gr.Row():
                        temperature_input.render()
                        top_p_input.render()
                        max_new_tokens_input.render()
            locale_probe_input.render()
            gr.ChatInterface(
                fn=respond,
                title=tr("情绪感知对话助手"),
                additional_inputs=[
                    user_id_input,
                    access_key_input,
                    use_knowledge_input,
                    use_style_input,
                    show_memory_receipt_input,
                    provider_input,
                    model_input,
                    base_url_input,
                    api_key_input,
                    temperature_input,
                    top_p_input,
                    max_new_tokens_input,
                    locale_probe_input,
                ],
            )

        with gr.Tab(tr("tab_mood"), id="mood"):
            with gr.Accordion("Mood Check-in", open=True):
                with gr.Row():
                    mood_date_input = gr.Textbox(label=tr("日期"), value=date.today().isoformat(), placeholder="YYYY-MM-DD")
                    mood_choice_input = gr.Dropdown(
                        label=tr("今天的心情"),
                        choices=[(tr(mood), mood) for mood in MOOD_CHOICES],
                        value="一般",
                        allow_custom_value=True,
                    )
                    mood_intensity_input = gr.Slider(label=tr("强度"), minimum=1, maximum=5, value=3, step=1)
                mood_note_input = gr.Textbox(
                    label=tr("备注"),
                    placeholder=tr("可以写下触发原因、身体状态、需要被记住的背景。"),
                    lines=3,
                )
                with gr.Row():
                    save_mood_button = gr.Button(tr("保存 Mood"))
                    refresh_mood_button = gr.Button(tr("刷新 Mood"))
                    delete_mood_button = gr.Button(tr("删除当天记录"))
                mood_box = gr.Textbox(
                    label=tr("最近 Mood Check-ins"),
                    value=tr("点击“刷新 Mood”查看当前用户的记录。"),
                    lines=10,
                    interactive=False,
                )
            with gr.Accordion("Weekly Mood Chart", open=True):
                weekly_mood_end_date_input = gr.Textbox(
                    label=tr("结束日期"), value=date.today().isoformat(), placeholder="YYYY-MM-DD"
                )
                refresh_weekly_mood_button = gr.Button(tr("刷新周情绪图"))
                weekly_mood_chart = gr.HTML(value=_render_weekly_mood_chart([], "light"))
                weekly_mood_summary = gr.Textbox(
                    label=tr("一周摘要"), value=tr(AUTH_REQUIRED_MESSAGE), lines=6, interactive=False
                )
                weekly_mood_analysis = gr.Textbox(
                    label=tr("情绪波动分析"), value=tr(AUTH_REQUIRED_MESSAGE), lines=8, interactive=False
                )

        data_tab = build_data_tab(tr, admin_recovery_status)
        new_access_key_input = data_tab.new_access_key_input
        change_access_key_button = data_tab.change_access_key_button
        export_user_data_button = data_tab.export_user_data_button
        export_user_data_status = data_tab.export_user_data_status
        export_user_data_file = data_tab.export_user_data_file
        admin_recovery_group = data_tab.admin_recovery_group
        admin_recovery_key_input = data_tab.admin_recovery_key_input
        admin_new_access_key_input = data_tab.admin_new_access_key_input
        admin_recovery_button = data_tab.admin_recovery_button
        admin_recovery_status_box = data_tab.admin_recovery_status_box
        stable_profile_input = data_tab.stable_profile_input
        add_stable_profile_button = data_tab.add_stable_profile_button
        load_stable_profile_button = data_tab.load_stable_profile_button
        stable_profile_box = data_tab.stable_profile_box
        stable_advanced_controls = data_tab.stable_advanced_controls
        stable_profile_editor = data_tab.stable_profile_editor
        save_stable_profile_button = data_tab.save_stable_profile_button
        clear_stable_profile_button = data_tab.clear_stable_profile_button
        memory_section = data_tab.memory_section
        load_memory_button = data_tab.load_memory_button
        memory_box = data_tab.memory_box
        memory_advanced_controls = data_tab.memory_advanced_controls
        save_memory_button = data_tab.save_memory_button
        clear_memory_button = data_tab.clear_memory_button
        refresh_memory_events_button = data_tab.refresh_memory_events_button
        assess_memory_quality_button = data_tab.assess_memory_quality_button
        memory_editor = data_tab.memory_editor
        memory_event_box = data_tab.memory_event_box
        memory_quality_box = data_tab.memory_quality_box
        backup_memory_button = data_tab.backup_memory_button
        restore_memory_mode = data_tab.restore_memory_mode
        restore_memory_file = data_tab.restore_memory_file
        restore_memory_button = data_tab.restore_memory_button
        memory_backup_status = data_tab.memory_backup_status
        memory_backup_file = data_tab.memory_backup_file
        memory_safety_backup_file = data_tab.memory_safety_backup_file

        knowledge_tab = build_knowledge_tab(
            tr,
            knowledge_top_k=KNOWLEDGE_TOP_K,
            knowledge_threshold=KNOWLEDGE_RETRIEVAL_THRESHOLD,
            knowledge_candidate_multiplier=KNOWLEDGE_CANDIDATE_MULTIPLIER,
            knowledge_status=knowledge_status,
            style_status=style_status,
            refresh_document_panel=refresh_knowledge_document_panel,
            format_document_list=format_knowledge_document_list,
        )
        advanced_tab = build_advanced_tab(
            tr,
            initial_provider=initial_provider,
            initial_model=initial_model,
            initial_base_url=initial_base_url,
            local_dtype_choices=LOCAL_DTYPE_CHOICES,
            local_attention_choices=LOCAL_ATTENTION_CHOICES,
            local_dtype=LOCAL_MODEL_DTYPE,
            local_attention=LOCAL_MODEL_ATTN_IMPLEMENTATION or "",
            local_low_cpu_memory=LOCAL_MODEL_LOW_CPU_MEM_USAGE,
            local_compile=LOCAL_MODEL_COMPILE,
            local_cpu_threads=LOCAL_MODEL_CPU_THREADS,
            build_status_text=build_status_text,
            build_local_runtime_text=build_local_runtime_config_text,
            get_log_text=get_log_text,
        )

    locale_status_timer = gr.Timer(value=0.75)
    ui = SimpleNamespace(
        demo=demo,
        main_tabs=main_tabs,
        interface_mode_input=interface_mode_input,
        theme_mode_input=theme_mode_input,
        locale_probe_input=locale_probe_input,
        locale_status_timer=locale_status_timer,
        advanced_chat_settings=advanced_chat_settings,
        provider_input=provider_input,
        model_input=model_input,
        base_url_input=base_url_input,
        api_key_input=api_key_input,
        user_id_input=user_id_input,
        access_key_input=access_key_input,
        save_access_key_button=save_access_key_button,
        login_status_box=login_status_box,
        access_key_status=access_key_status,
        temperature_input=temperature_input,
        top_p_input=top_p_input,
        max_new_tokens_input=max_new_tokens_input,
        mood_date_input=mood_date_input,
        mood_choice_input=mood_choice_input,
        mood_intensity_input=mood_intensity_input,
        mood_note_input=mood_note_input,
        save_mood_button=save_mood_button,
        refresh_mood_button=refresh_mood_button,
        delete_mood_button=delete_mood_button,
        mood_box=mood_box,
        weekly_mood_end_date_input=weekly_mood_end_date_input,
        refresh_weekly_mood_button=refresh_weekly_mood_button,
        weekly_mood_chart=weekly_mood_chart,
        weekly_mood_summary=weekly_mood_summary,
        weekly_mood_analysis=weekly_mood_analysis,
        onboarding_guide=onboarding.guide,
        onboarding_complete_button=onboarding.complete_button,
        **data_tab.__dict__,
        **knowledge_tab.__dict__,
        **advanced_tab.__dict__,
    )
    bind_gui_events(
        ui,
        globals(),
        sync_locale_js=SYNC_LOCALE_JS,
        refresh_chart_theme_js=REFRESH_CHART_THEME_JS,
        load_theme_js=LOAD_THEME_JS,
    )

if __name__ == "__main__":
    start_background_warmup(initial_provider)
    demo.launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        share=os.getenv("GRADIO_SHARE", "false").lower() == "true",
        css=THEME_CSS,
        i18n=GUI_I18N,
        footer_links=["settings"],
    )
