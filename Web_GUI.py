from __future__ import annotations

import logging
import os
import threading
import time
from datetime import date, datetime
from pathlib import Path
from typing import Any

import gradio as gr

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
from export_store import export_user_data
from gui_memory import (
    MEMORY_SECTION_LABELS,
    clear_memory_section,
    clear_memory_section_and_reload,
    clear_stable_profile,
    format_memory_event_log,
    format_memory_snapshot,
    add_stable_profile,
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
from llm_providers import ModelRuntimeConfig, require_openai_client
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

LOCAL_DTYPE_CHOICES = ["auto", "bfloat16", "float16", "float32"]
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
            partial += chunk
            yield partial
        if show_memory_receipt:
            with session_store.session(user_id) as state:
                receipt = latest_memory_receipt(state)
            yield f"{partial}\n\n---\n{receipt}"
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
) -> tuple[str, str]:
    message = save_access_key_from_gui(user_id, access_key)
    return (
        localize_status_text(message, locale),
        localize_status_text(login_status_text(user_id, access_key), locale),
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


def _env_quote(value: str) -> str:
    if value == "":
        return ""
    if any(char.isspace() or char in '#"\\' for char in value):
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return value


def _read_env_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").splitlines()


def _upsert_env_values(path: Path, updates: dict[str, str]) -> None:
    lines = _read_env_lines(path)
    remaining = dict(updates)
    new_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        prefix = ""
        candidate = stripped
        if candidate.startswith("export "):
            prefix = "export "
            candidate = candidate[7:].lstrip()
        if not candidate or candidate.startswith("#") or "=" not in candidate:
            new_lines.append(line)
            continue

        key = candidate.split("=", 1)[0].strip()
        if key in remaining:
            new_lines.append(f"{prefix}{key}={_env_quote(remaining.pop(key))}")
        else:
            new_lines.append(line)

    if remaining and new_lines and new_lines[-1].strip():
        new_lines.append("")
    for key, value in remaining.items():
        new_lines.append(f"{key}={_env_quote(value)}")

    path.write_text("\n".join(new_lines).rstrip() + "\n", encoding="utf-8")


def _provider_key_name(provider: str) -> str:
    if provider == "deepseek":
        return "DEEPSEEK_API_KEY"
    if provider == "openai":
        return "OPENAI_API_KEY"
    if provider == "openrouter":
        return "OPENROUTER_API_KEY"
    return "LLM_API_KEY"


def save_model_config_to_env(
    provider: str,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> str:
    provider = (provider or "local_hf").strip()
    model = (model or "").strip()
    base_url = (base_url or "").strip()

    updates = {
        "LLM_PROVIDER": provider,
        "LLM_TEMPERATURE": str(float(temperature)),
        "LLM_TOP_P": str(float(top_p)),
        "LLM_MAX_NEW_TOKENS": str(int(max_new_tokens)),
    }

    if provider == "local_hf":
        updates["CHAT_MODEL_NAME"] = model
    else:
        updates["LLM_API_MODEL"] = model
        if base_url:
            updates["LLM_API_BASE_URL"] = base_url
        if provider == "deepseek":
            updates["DEEPSEEK_MODEL"] = model
        elif provider == "openai":
            updates["OPENAI_MODEL"] = model
        elif provider == "openrouter":
            updates["OPENROUTER_MODEL"] = model

    key_saved = False
    if api_key and api_key.strip():
        updates[_provider_key_name(provider)] = api_key.strip()
        key_saved = True

    _upsert_env_values(LOCAL_ENV_PATH, updates)
    os.environ.update(updates)
    logger.info("模型配置已保存：provider=%s model=%s path=%s", provider, model, LOCAL_ENV_PATH.name)

    message = f"已保存模型配置到 {LOCAL_ENV_PATH.name}。"
    if key_saved:
        message += " API Key 已写入本机 .env.local，请不要提交该文件。"
    else:
        message += " API Key 输入为空，未改动已有密钥。"
    return message


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


def _normalize_local_dtype(dtype: str) -> str:
    normalized = (dtype or "auto").strip().lower()
    aliases = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in LOCAL_DTYPE_CHOICES:
        raise ValueError("模型精度仅支持 auto、bfloat16、float16 或 float32。")
    return normalized


def build_local_runtime_config_text(
    dtype: str,
    attention_implementation: str,
    low_cpu_mem_usage: bool,
    compile_model: bool,
    cpu_threads: int | float,
) -> str:
    attention = (attention_implementation or "").strip() or "自动 / 默认"
    threads = int(cpu_threads or 0)
    return "\n".join([
        "本地 Hugging Face 模型运行配置",
        f"模型精度：{_normalize_local_dtype(dtype)}",
        f"注意力实现：{attention}",
        f"低内存加载：{'开启' if low_cpu_mem_usage else '关闭'}",
        f"torch.compile：{'开启' if compile_model else '关闭'}",
        f"CPU 线程：{'使用 PyTorch 默认值' if threads == 0 else threads}",
        "提示：配置保存后需重启应用，下一次加载本地模型时才会生效。",
    ])


def save_local_runtime_config_to_env(
    dtype: str,
    attention_implementation: str,
    low_cpu_mem_usage: bool,
    compile_model: bool,
    cpu_threads: int | float,
) -> str:
    normalized_dtype = _normalize_local_dtype(dtype)
    attention = (attention_implementation or "").strip()
    if "\r" in attention or "\n" in attention:
        raise ValueError("注意力实现不能包含换行符。")
    threads = int(cpu_threads or 0)
    if not 0 <= threads <= 512:
        raise ValueError("CPU 线程数应在 0 到 512 之间。")

    updates = {
        "LOCAL_MODEL_DTYPE": normalized_dtype,
        "LOCAL_MODEL_ATTN_IMPLEMENTATION": attention,
        "LOCAL_MODEL_LOW_CPU_MEM_USAGE": str(bool(low_cpu_mem_usage)).lower(),
        "LOCAL_MODEL_COMPILE": str(bool(compile_model)).lower(),
        "LOCAL_MODEL_CPU_THREADS": str(threads),
    }
    _upsert_env_values(LOCAL_ENV_PATH, updates)
    os.environ.update(updates)
    logger.info("本地模型运行配置已保存：path=%s", LOCAL_ENV_PATH.name)
    return f"已保存本地模型运行配置到 {LOCAL_ENV_PATH.name}。请重启应用后再加载本地模型。"


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


def _connection_result(level: str, message: str, detail: str = "") -> str:
    parts = [f"[{level}] {message}"]
    if detail:
        parts.append(f"详情：{detail}")
    return "\n".join(parts)


def classify_connection_error(exc: Exception, provider: str) -> str:
    text = str(exc)
    type_name = type(exc).__name__.lower()
    lowered = text.lower()
    provider = (provider or "").lower()

    if "api key" in lowered or "缺少" in text:
        return _connection_result("配置错误", "缺少 API Key", text)
    if "module" in lowered or "缺少依赖" in text or "modulenotfounderror" in type_name:
        return _connection_result("依赖缺失", "缺少运行依赖", text)
    if "401" in lowered or "unauthorized" in lowered or "authentication" in lowered:
        return _connection_result("认证失败", "API Key 无效或无权限", text)
    if "429" in lowered or "rate limit" in lowered or "quota" in lowered or "insufficient" in lowered:
        return _connection_result("限流/额度", "请求频率或账户额度受限", text)
    if any(word in lowered for word in ["timeout", "timed out", "connection", "network", "proxy", "dns", "ssl"]):
        return _connection_result("网络错误", "无法稳定连接模型服务", text)
    if provider == "local_hf":
        return _connection_result("本地模型错误", "本地模型加载或推理失败", text)
    return _connection_result("响应异常", "模型服务返回异常响应", text)


def _test_local_connection(config: ModelRuntimeConfig) -> str:
    tokenizer, model = get_llm()
    model_name = getattr(getattr(model, "config", None), "name_or_path", None)
    tokenizer_name = getattr(tokenizer, "name_or_path", None)
    loaded_name = model_name or tokenizer_name or config.resolved_model()
    return _connection_result("成功", f"本地模型加载成功：{loaded_name}")


def _test_api_connection(config: ModelRuntimeConfig) -> str:
    api_key = config.resolved_api_key()
    if not api_key:
        raise RuntimeError("当前 Provider 缺少 API Key，请在 GUI 输入或写入 .env.local")

    OpenAI = require_openai_client()
    client_kwargs: dict[str, Any] = {"api_key": api_key}
    base_url = config.resolved_base_url()
    if base_url:
        client_kwargs["base_url"] = base_url

    client = OpenAI(**client_kwargs)
    response = client.chat.completions.create(
        model=config.resolved_model(),
        messages=[{"role": "user", "content": "ping"}],
        max_tokens=1,
        temperature=0,
        stream=False,
    )
    if not response.choices:
        raise RuntimeError("接口返回成功，但没有 choices 字段")
    return _connection_result(
        "成功",
        f"API 连接成功：{config.normalized_provider()} / {config.resolved_model()}",
    )


def test_model_connection(
    provider: str,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> str:
    config = make_model_config(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        top_p=top_p,
        max_new_tokens=max_new_tokens,
    )
    set_connection_status(f"正在测试 {config.normalized_provider()} / {config.resolved_model()}")
    try:
        if config.normalized_provider() == "local_hf":
            result = _test_local_connection(config)
        else:
            result = _test_api_connection(config)
        set_connection_status(result)
        return result
    except Exception as exc:
        result = classify_connection_error(exc, config.normalized_provider())
        set_connection_status(result)
        return result


def build_connection_test_detail(
    result: str,
    provider: str,
    model: str,
    base_url: str,
    elapsed_seconds: float,
) -> str:
    level = result.split("]", 1)[0].lstrip("[") if result.startswith("[") else "未知"
    suggestions = {
        "成功": "连接可用，可以开始对话。",
        "配置错误": "检查 Provider、模型名和 API Key 是否填写完整。",
        "认证失败": "检查 API Key、账户权限和所选模型是否可用。",
        "限流/额度": "稍后重试，或检查服务商账户余额与请求额度。",
        "网络错误": "检查网络、代理设置、Base URL 和证书环境。",
        "依赖缺失": "安装提示中提到的 Python 依赖后重启应用。",
        "本地模型错误": "检查模型名称、显存/内存和本地运行配置。",
        "响应异常": "检查服务商返回信息、模型名称和接口兼容性。",
    }
    endpoint = base_url.strip() or ("本地 Hugging Face" if provider == "local_hf" else "服务商默认地址")
    return "\n".join([
        f"测试状态：{level}",
        f"Provider：{provider or '未选择'}",
        f"模型：{model or '未选择'}",
        f"目标：{endpoint}",
        f"耗时：{elapsed_seconds:.2f} 秒",
        f"建议：{suggestions.get(level, '请根据下方详情检查配置。')}",
        "",
        result,
    ])


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

        if provider == "local_hf":
            set_warmup_status("正在预热本地聊天模型...")
            get_llm()

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
    local_dtype_input = gr.Dropdown(
        label=tr("模型精度"),
        choices=LOCAL_DTYPE_CHOICES,
        value=LOCAL_MODEL_DTYPE if LOCAL_MODEL_DTYPE in LOCAL_DTYPE_CHOICES else "auto",
        render=False,
    )
    local_attention_input = gr.Dropdown(
        label=tr("注意力实现"),
        choices=LOCAL_ATTENTION_CHOICES,
        value=LOCAL_MODEL_ATTN_IMPLEMENTATION or "",
        allow_custom_value=True,
        render=False,
    )
    local_low_cpu_mem_input = gr.Checkbox(
        label=tr("低内存加载"),
        value=LOCAL_MODEL_LOW_CPU_MEM_USAGE,
        render=False,
    )
    local_compile_input = gr.Checkbox(
        label=tr("启用 torch.compile"),
        value=LOCAL_MODEL_COMPILE,
        render=False,
    )
    local_cpu_threads_input = gr.Number(
        label=tr("CPU 线程数（0 为默认）"),
        value=LOCAL_MODEL_CPU_THREADS,
        minimum=0,
        maximum=512,
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

        with gr.Tab(tr("tab_data"), id="data"):
            with gr.Accordion(tr("用户访问"), open=True):
                new_access_key_input = gr.Textbox(
                    label=tr("新访问密码"),
                    value="",
                    type="password",
                    placeholder=tr("修改密码时填写，至少 8 位，可包含特殊符号"),
                )
                with gr.Row():
                    change_access_key_button = gr.Button(tr("修改密码"))
                    export_user_data_button = gr.Button(tr("导出用户数据"))
                export_user_data_status = gr.Textbox(
                    label=tr("数据导出状态"), value=tr("尚未导出。"), lines=3, interactive=False
                )
                export_user_data_file = gr.File(label=tr("导出的用户数据"), interactive=False)
                with gr.Column(visible=False) as admin_recovery_group:
                    with gr.Accordion(tr("管理员恢复"), open=False):
                        admin_recovery_key_input = gr.Textbox(
                            label=tr("管理员恢复密钥"), value="", type="password",
                            placeholder=tr("仅从服务器环境变量读取的恢复密钥"),
                        )
                        admin_new_access_key_input = gr.Textbox(
                            label=tr("恢复后的新密码"), value="", type="password",
                            placeholder=tr("修改密码时填写，至少 8 位，可包含特殊符号"),
                        )
                        admin_recovery_button = gr.Button(tr("重置访问密码"), variant="stop")
                        admin_recovery_status_box = gr.Textbox(
                            label=tr("管理员恢复状态"), value=tr(admin_recovery_status()), lines=3, interactive=False
                        )

            with gr.Accordion(tr("稳定资料"), open=False):
                stable_profile_input = gr.Textbox(
                    label=tr("新增稳定资料"),
                    placeholder=tr("例如：我是学生；我喜欢被简洁地称呼；我来自北京"),
                    lines=2,
                )
                with gr.Row():
                    add_stable_profile_button = gr.Button(tr("添加资料"))
                    load_stable_profile_button = gr.Button(tr("查看资料"))
                stable_profile_box = gr.Textbox(
                    label=tr("稳定资料状态"),
                    value=tr("点击“查看资料”加载当前用户的稳定资料。"),
                    lines=10,
                    interactive=False,
                )
                with gr.Column(visible=False) as stable_advanced_controls:
                    stable_profile_editor = gr.Textbox(
                        label=tr("稳定资料 JSON 编辑区"), value="[]", lines=10, interactive=True
                    )
                    with gr.Row():
                        save_stable_profile_button = gr.Button(tr("保存编辑"))
                        clear_stable_profile_button = gr.Button(tr("清空稳定资料"), variant="stop")

            with gr.Accordion(tr("记忆管理"), open=False):
                memory_section = gr.Radio(
                    label=tr("清理范围"),
                    choices=[
                        (tr("短期对话"), "history"), (tr("情绪记忆"), "emotion"),
                        (tr("长期记忆"), "long"), (tr("兴趣记忆"), "interest"),
                        (tr("稳定资料"), "stable"), (tr("全部记忆"), "all"),
                    ],
                    value="history",
                )
                load_memory_button = gr.Button(tr("查看记忆"))
                memory_box = gr.Textbox(
                    label=tr("当前用户记忆"),
                    value=tr("点击“查看记忆”加载当前用户的记忆。"),
                    lines=18,
                    interactive=False,
                )
                with gr.Column(visible=False) as memory_advanced_controls:
                    with gr.Row():
                        save_memory_button = gr.Button(tr("保存修改"))
                        clear_memory_button = gr.Button(tr("清理所选记忆"))
                        refresh_memory_events_button = gr.Button(tr("查看写入记录"))
                    memory_editor = gr.Textbox(label=tr("记忆 JSON 编辑区"), value="[]", lines=14, interactive=True)
                    memory_event_box = gr.Textbox(
                        label=tr("记忆写入记录"),
                        value=tr("点击“查看写入记录”加载最近的记忆判断。"),
                        lines=14,
                        interactive=False,
                    )
                    with gr.Accordion(tr("记忆备份与恢复"), open=False):
                        with gr.Row():
                            backup_memory_button = gr.Button(tr("备份全部记忆"))
                            restore_memory_mode = gr.Radio(
                                choices=[(tr("合并并去重"), "merge"), (tr("覆盖当前记忆"), "replace")],
                                value="merge", label=tr("恢复模式"),
                            )
                        restore_memory_file = gr.File(
                            label=tr("选择记忆备份 JSON"), file_types=[".json"], type="filepath"
                        )
                        restore_memory_button = gr.Button(tr("恢复记忆"), variant="primary")
                        memory_backup_status = gr.Textbox(
                            label=tr("备份 / 恢复状态"), value=tr("尚未执行备份或恢复。"), lines=4, interactive=False
                        )
                        memory_backup_file = gr.File(label=tr("生成的备份文件"), interactive=False)
                        memory_safety_backup_file = gr.File(label=tr("恢复前安全备份"), interactive=False)

        with gr.Tab(tr("tab_knowledge"), id="knowledge"):
            with gr.Accordion(tr("知识库 / RAG"), open=True):
                knowledge_files = gr.File(
                    label=tr("导入资料"), file_count="multiple",
                    file_types=[".txt", ".md", ".markdown", ".csv", ".json", ".pdf", ".docx"],
                )
                with gr.Row():
                    import_knowledge_button = gr.Button(tr("导入并重建索引"))
                    refresh_knowledge_button = gr.Button(tr("查看知识库状态"))
                knowledge_query = gr.Textbox(
                    label=tr("检索预览问题"),
                    placeholder=tr("输入一个问题，查看会被检索进 Prompt 的资料片段"),
                )
                knowledge_box = gr.Textbox(
                    label=tr("知识库状态 / 检索结果"), value=knowledge_status, lines=16, interactive=False
                )
                with gr.Column(visible=False) as knowledge_advanced_controls:
                    with gr.Row():
                        knowledge_top_k = gr.Slider(1, 10, value=KNOWLEDGE_TOP_K, step=1, label=tr("返回片段数"))
                        knowledge_threshold = gr.Slider(
                            0, 1, value=KNOWLEDGE_RETRIEVAL_THRESHOLD, step=0.05, label=tr("相关度阈值")
                        )
                        knowledge_candidate_multiplier = gr.Slider(
                            1, 8, value=KNOWLEDGE_CANDIDATE_MULTIPLIER, step=1, label=tr("候选池倍数")
                        )
                    with gr.Row():
                        rebuild_knowledge_button = gr.Button(tr("重建索引"))
                        preview_knowledge_button = gr.Button(tr("检索质量诊断"))
                        inspect_knowledge_quality_button = gr.Button(tr("检查知识库质量"))
                    with gr.Accordion(tr("RAG 文档管理"), open=False):
                        knowledge_document_selector = gr.Dropdown(
                            label=tr("已入库文档"), choices=refresh_knowledge_document_panel()[0],
                            value=None, interactive=True,
                        )
                        with gr.Row():
                            refresh_knowledge_documents_button = gr.Button(tr("刷新文档列表"))
                            delete_knowledge_document_button = gr.Button(tr("删除所选文档"), variant="stop")
                            clear_knowledge_documents_button = gr.Button(tr("清空全部文档"), variant="stop")
                        knowledge_document_box = gr.Textbox(
                            label=tr("文档清单"), value=format_knowledge_document_list, lines=10, interactive=False
                        )

            with gr.Accordion(tr("风格库 / Style RAG"), open=False):
                style_files = gr.File(
                    label=tr("导入风格样例"), file_count="multiple",
                    file_types=[".txt", ".md", ".markdown", ".csv", ".json", ".jsonl"],
                )
                with gr.Row():
                    import_style_button = gr.Button(tr("导入并重建风格索引"))
                    refresh_style_button = gr.Button(tr("查看风格库状态"))
                style_query = gr.Textbox(
                    label=tr("风格检索预览"),
                    placeholder=tr("输入当前问题或场景，查看会被参考的回复风格样例"),
                )
                preview_style_button = gr.Button(tr("风格检索预览"))
                style_box = gr.Textbox(
                    label=tr("风格库状态 / 检索结果"), value=style_status, lines=16, interactive=False
                )
                with gr.Column(visible=False) as style_advanced_controls:
                    rebuild_style_button = gr.Button(tr("重建风格索引"))

        with gr.Tab(tr("tab_advanced"), id="advanced", visible=False) as advanced_tab:
            with gr.Accordion(tr("运行状态"), open=True):
                status_box = gr.Textbox(
                    label=tr("状态"),
                    value=build_status_text(initial_provider, initial_model, initial_base_url),
                    lines=9,
                    interactive=False,
                )
                with gr.Row():
                    refresh_button = gr.Button(tr("刷新状态"))
                    connection_button = gr.Button(tr("测试连接"))
                    save_model_config_button = gr.Button(tr("保存模型配置"))
                connection_result = gr.Textbox(label=tr("连接测试摘要"), interactive=False)
                connection_detail_result = gr.Textbox(
                    label=tr("连接测试详情"), value=tr("尚未测试连接。"), lines=9, interactive=False
                )
                save_model_config_result = gr.Textbox(label=tr("模型配置保存结果"), interactive=False)
                status_timer = gr.Timer(value=2.0)
            with gr.Accordion(tr("本地模型运行配置"), open=False):
                with gr.Row():
                    local_dtype_input.render()
                    local_attention_input.render()
                with gr.Row():
                    local_low_cpu_mem_input.render()
                    local_compile_input.render()
                    local_cpu_threads_input.render()
                with gr.Row():
                    save_local_runtime_button = gr.Button(tr("保存本地运行配置"))
                    refresh_local_runtime_button = gr.Button(tr("查看当前配置"))
                local_runtime_status = gr.Textbox(
                    label=tr("本地模型配置状态"),
                    value=lambda: build_local_runtime_config_text(
                        LOCAL_MODEL_DTYPE, LOCAL_MODEL_ATTN_IMPLEMENTATION or "",
                        LOCAL_MODEL_LOW_CPU_MEM_USAGE, LOCAL_MODEL_COMPILE, LOCAL_MODEL_CPU_THREADS,
                    ),
                    lines=8,
                    interactive=False,
                )
                local_runtime_save_result = gr.Textbox(
                    label=tr("保存结果"), value=tr("尚未保存本地运行配置。"), lines=3, interactive=False
                )
            with gr.Accordion(tr("日志面板"), open=False):
                log_box = gr.Textbox(label=tr("最近日志"), value=get_log_text, lines=14, interactive=False)
                with gr.Row():
                    refresh_logs_button = gr.Button(tr("刷新日志"))
                    clear_logs_button = gr.Button(tr("清空日志"))
                log_timer = gr.Timer(value=3.0)

    locale_status_timer = gr.Timer(value=0.75)

    interface_mode_input.change(
        fn=interface_mode_visibility,
        inputs=interface_mode_input,
        outputs=[
            advanced_chat_settings,
            memory_advanced_controls,
            stable_advanced_controls,
            knowledge_advanced_controls,
            style_advanced_controls,
            admin_recovery_group,
            advanced_tab,
            main_tabs,
        ],
        show_progress="hidden",
    )
    provider_input.change(
        fn=provider_changed,
        inputs=[provider_input, api_key_input, locale_probe_input],
        outputs=[model_input, base_url_input, status_box],
        js=SYNC_LOCALE_JS,
    )
    refresh_button.click(
        fn=refresh_status,
        inputs=[provider_input, model_input, base_url_input, api_key_input, locale_probe_input],
        outputs=status_box,
        js=SYNC_LOCALE_JS,
    )
    status_timer.tick(
        fn=refresh_status,
        inputs=[provider_input, model_input, base_url_input, api_key_input, locale_probe_input],
        outputs=status_box,
        js=SYNC_LOCALE_JS,
    )
    locale_status_timer.tick(
        fn=relocalize_status_values_and_locale,
        inputs=[
            status_box,
            connection_result,
            connection_detail_result,
            save_model_config_result,
            local_runtime_status,
            local_runtime_save_result,
            login_status_box,
            access_key_status,
            export_user_data_status,
            admin_recovery_status_box,
            locale_probe_input,
        ],
        outputs=[
            status_box,
            connection_result,
            connection_detail_result,
            save_model_config_result,
            local_runtime_status,
            local_runtime_save_result,
            login_status_box,
            access_key_status,
            export_user_data_status,
            admin_recovery_status_box,
            locale_probe_input,
        ],
        js=SYNC_LOCALE_JS,
        show_progress="hidden",
    )
    connection_button.click(
        fn=test_model_connection_and_refresh,
        inputs=[
            provider_input,
            model_input,
            base_url_input,
            api_key_input,
            temperature_input,
            top_p_input,
            max_new_tokens_input,
            locale_probe_input,
        ],
        outputs=[connection_result, connection_detail_result, status_box],
        js=SYNC_LOCALE_JS,
    )
    save_model_config_button.click(
        fn=save_model_config_and_refresh,
        inputs=[
            provider_input,
            model_input,
            base_url_input,
            api_key_input,
            temperature_input,
            top_p_input,
            max_new_tokens_input,
            locale_probe_input,
        ],
        outputs=[save_model_config_result, status_box],
        js=SYNC_LOCALE_JS,
    )
    save_local_runtime_button.click(
        fn=save_local_runtime_config_and_refresh,
        inputs=[
            local_dtype_input,
            local_attention_input,
            local_low_cpu_mem_input,
            local_compile_input,
            local_cpu_threads_input,
            locale_probe_input,
        ],
        outputs=[local_runtime_save_result, local_runtime_status],
        js=SYNC_LOCALE_JS,
    )
    refresh_local_runtime_button.click(
        fn=build_local_runtime_config_text_localized,
        inputs=[
            local_dtype_input,
            local_attention_input,
            local_low_cpu_mem_input,
            local_compile_input,
            local_cpu_threads_input,
            locale_probe_input,
        ],
        outputs=local_runtime_status,
        js=SYNC_LOCALE_JS,
    )
    save_access_key_button.click(
        fn=save_access_key_and_status,
        inputs=[user_id_input, access_key_input, locale_probe_input],
        outputs=[access_key_status, login_status_box],
        js=SYNC_LOCALE_JS,
    )
    change_access_key_button.click(
        fn=change_access_key_and_status,
        inputs=[user_id_input, access_key_input, new_access_key_input, locale_probe_input],
        outputs=[access_key_status, access_key_input, new_access_key_input, login_status_box],
        js=SYNC_LOCALE_JS,
    )
    admin_recovery_button.click(
        fn=admin_recover_access_key_and_status,
        inputs=[user_id_input, admin_recovery_key_input, admin_new_access_key_input, locale_probe_input],
        outputs=[
            admin_recovery_status_box,
            access_key_input,
            admin_recovery_key_input,
            login_status_box,
        ],
        js=SYNC_LOCALE_JS,
    )
    export_user_data_button.click(
        fn=export_user_data_from_gui,
        inputs=[user_id_input, access_key_input, locale_probe_input],
        outputs=[export_user_data_status, export_user_data_file],
        js=SYNC_LOCALE_JS,
    )
    refresh_logs_button.click(
        fn=get_log_text,
        outputs=log_box,
    )
    clear_logs_button.click(
        fn=clear_logs,
        outputs=log_box,
    )
    log_timer.tick(
        fn=get_log_text,
        outputs=log_box,
    )
    save_mood_button.click(
        fn=submit_mood_checkin_and_refresh_dashboard,
        inputs=[
            user_id_input,
            access_key_input,
            mood_date_input,
            mood_choice_input,
            mood_intensity_input,
            mood_note_input,
            weekly_mood_end_date_input,
            theme_mode_input,
            locale_probe_input,
        ],
        outputs=[
            mood_box,
            weekly_mood_end_date_input,
            weekly_mood_chart,
            weekly_mood_summary,
            weekly_mood_analysis,
        ],
    )
    refresh_mood_button.click(
        fn=refresh_mood_panel,
        inputs=[user_id_input, access_key_input, locale_probe_input],
        outputs=mood_box,
    )
    delete_mood_button.click(
        fn=delete_mood_checkin_and_refresh_dashboard,
        inputs=[
            user_id_input,
            access_key_input,
            mood_date_input,
            weekly_mood_end_date_input,
            theme_mode_input,
            locale_probe_input,
        ],
        outputs=[
            mood_box,
            weekly_mood_end_date_input,
            weekly_mood_chart,
            weekly_mood_summary,
            weekly_mood_analysis,
        ],
    )
    refresh_weekly_mood_button.click(
        fn=refresh_weekly_mood_dashboard,
        inputs=[
            user_id_input,
            access_key_input,
            weekly_mood_end_date_input,
            theme_mode_input,
            locale_probe_input,
        ],
        outputs=[weekly_mood_chart, weekly_mood_summary, weekly_mood_analysis],
    )
    theme_mode_input.change(
        fn=refresh_weekly_mood_dashboard,
        inputs=[
            user_id_input,
            access_key_input,
            weekly_mood_end_date_input,
            theme_mode_input,
            locale_probe_input,
        ],
        outputs=[weekly_mood_chart, weekly_mood_summary, weekly_mood_analysis],
        js=REFRESH_CHART_THEME_JS,
        show_progress="hidden",
    )
    demo.load(
        fn=sync_locale_value,
        inputs=locale_probe_input,
        outputs=locale_probe_input,
        js=SYNC_LOCALE_JS,
        show_progress="hidden",
    )
    demo.load(
        fn=load_theme_and_weekly_dashboard,
        inputs=[
            user_id_input,
            access_key_input,
            weekly_mood_end_date_input,
            theme_mode_input,
            locale_probe_input,
        ],
        outputs=[theme_mode_input, weekly_mood_chart, weekly_mood_summary, weekly_mood_analysis],
        js=LOAD_THEME_JS,
        show_progress="hidden",
    )
    load_memory_button.click(
        fn=load_memory_editor,
        inputs=[user_id_input, access_key_input, memory_section, locale_probe_input],
        outputs=[memory_editor, memory_box],
    )
    save_memory_button.click(
        fn=save_memory_editor,
        inputs=[user_id_input, access_key_input, memory_section, memory_editor, locale_probe_input],
        outputs=[memory_editor, memory_box],
    )
    clear_memory_button.click(
        fn=clear_memory_section_and_reload,
        inputs=[user_id_input, access_key_input, memory_section, locale_probe_input],
        outputs=[memory_editor, memory_box],
    )
    refresh_memory_events_button.click(
        fn=load_memory_event_log,
        inputs=[user_id_input, access_key_input, locale_probe_input],
        outputs=memory_event_box,
    )
    backup_memory_button.click(
        fn=backup_memory_from_gui,
        inputs=[user_id_input, access_key_input, locale_probe_input],
        outputs=[memory_backup_status, memory_backup_file],
    )
    restore_memory_button.click(
        fn=restore_memory_from_gui,
        inputs=[
            user_id_input,
            access_key_input,
            restore_memory_file,
            restore_memory_mode,
            locale_probe_input,
        ],
        outputs=[memory_backup_status, memory_safety_backup_file, memory_box, memory_event_box],
    )
    add_stable_profile_button.click(
        fn=add_stable_profile,
        inputs=[user_id_input, access_key_input, stable_profile_input, locale_probe_input],
        outputs=[stable_profile_input, stable_profile_editor, stable_profile_box],
    )
    load_stable_profile_button.click(
        fn=load_stable_profile_editor,
        inputs=[user_id_input, access_key_input, locale_probe_input],
        outputs=[stable_profile_editor, stable_profile_box],
    )
    save_stable_profile_button.click(
        fn=save_stable_profile_editor,
        inputs=[user_id_input, access_key_input, stable_profile_editor, locale_probe_input],
        outputs=[stable_profile_editor, stable_profile_box],
    )
    clear_stable_profile_button.click(
        fn=clear_stable_profile,
        inputs=[user_id_input, access_key_input, locale_probe_input],
        outputs=[stable_profile_editor, stable_profile_box],
    )
    import_knowledge_button.click(
        fn=import_knowledge_files_and_refresh,
        inputs=knowledge_files,
        outputs=[knowledge_box, knowledge_document_selector, knowledge_document_box],
    )
    rebuild_knowledge_button.click(
        fn=rebuild_knowledge_panel,
        outputs=knowledge_box,
    )
    refresh_knowledge_button.click(
        fn=knowledge_status,
        outputs=knowledge_box,
    )
    preview_knowledge_button.click(
        fn=preview_knowledge_search,
        inputs=[knowledge_query, knowledge_top_k, knowledge_threshold, knowledge_candidate_multiplier],
        outputs=knowledge_box,
    )
    inspect_knowledge_quality_button.click(
        fn=format_knowledge_quality_report,
        outputs=knowledge_box,
    )
    refresh_knowledge_documents_button.click(
        fn=refresh_knowledge_documents_from_gui,
        outputs=[knowledge_document_selector, knowledge_document_box],
    )
    delete_knowledge_document_button.click(
        fn=delete_knowledge_document_from_gui,
        inputs=knowledge_document_selector,
        outputs=[knowledge_document_selector, knowledge_document_box, knowledge_box],
    )
    clear_knowledge_documents_button.click(
        fn=clear_knowledge_documents_from_gui,
        outputs=[knowledge_document_selector, knowledge_document_box, knowledge_box],
    )
    import_style_button.click(
        fn=import_style_files,
        inputs=style_files,
        outputs=style_box,
    )
    rebuild_style_button.click(
        fn=rebuild_style_panel,
        outputs=style_box,
    )
    refresh_style_button.click(
        fn=style_status,
        outputs=style_box,
    )
    preview_style_button.click(
        fn=preview_style_search,
        inputs=style_query,
        outputs=style_box,
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
