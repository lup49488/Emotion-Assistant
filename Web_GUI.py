from __future__ import annotations

import logging
import os
import threading
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
    make_model_config,
)
from config import (
    BASE_DIR,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    KNOWLEDGE_ENABLED,
    LOADED_ENV_FILES,
    STYLE_ENABLED,
)
from gui_auth import (
    AUTH_REQUIRED_MESSAGE,
    authorize_or_message as _authorize_or_message,
    change_saved_access_key,
    save_or_verify_access_key,
)
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
    format_memory_snapshot,
    load_memory_editor,
    load_memory_panel,
    save_memory_editor,
)
from gui_mood import (
    _render_weekly_mood_chart,
    delete_mood_checkin_from_gui,
    load_theme_and_weekly_chart,
    refresh_mood_panel,
    refresh_weekly_mood_chart,
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
    provider: str,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
):
    user_id, auth_error = _authorize_or_message(user_id, access_key)
    if auth_error:
        yield auth_error
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
        user_id or "local",
        config.normalized_provider(),
        config.resolved_model(),
    )
    set_chat_status(f"正在请求 {config.normalized_provider()} / {config.resolved_model()}")
    try:
        for chunk in handle_user_message_stream(
            user_id or "local",
            message,
            model_config=config,
            use_knowledge=use_knowledge,
            use_style=use_style,
        ):
            partial += chunk
            yield partial
        set_chat_status("回复完成")
    except Exception as exc:
        set_chat_status(f"请求失败：{exc}")
        yield f"请求失败：{exc}"


def provider_changed(provider: str, api_key: str = ""):
    choices = MODEL_CHOICES.get(provider, MODEL_CHOICES["custom"])
    default_model = DEFAULT_MODELS.get(provider, choices[0])
    default_base_url = DEFAULT_BASE_URLS.get(provider, "")
    return (
        gr.update(choices=choices, value=default_model),
        gr.update(value=default_base_url),
        build_status_text(provider, default_model, default_base_url, api_key),
    )


def refresh_status(provider: str, model: str, base_url: str, api_key: str) -> str:
    return build_status_text(provider, model, base_url, api_key)


def save_access_key_from_gui(user_id: str, access_key: str) -> str:
    return save_or_verify_access_key(user_id, access_key)


def login_status_text(user_id: str, access_key: str) -> str:
    user_id, auth_error = _authorize_or_message(user_id, access_key)
    if auth_error:
        return f"未验证：{auth_error}"
    return f"已验证：{user_id}"


def save_access_key_and_status(user_id: str, access_key: str) -> tuple[str, str]:
    message = save_access_key_from_gui(user_id, access_key)
    return message, login_status_text(user_id, access_key)


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
) -> tuple[str, Any, Any, str]:
    message, access_key_update, new_key_update = change_access_key_from_gui(
        user_id,
        current_access_key,
        new_access_key,
    )
    if message.startswith("访问密码已修改"):
        return message, access_key_update, new_key_update, login_status_text(user_id, new_access_key)
    return message, access_key_update, new_key_update, login_status_text(user_id, current_access_key)


def export_user_data_from_gui(user_id: str, access_key: str) -> tuple[str, str | None]:
    user_id, auth_error = _authorize_or_message(user_id, access_key)
    if auth_error:
        return auth_error, None
    try:
        path = export_user_data(user_id)
        return f"已导出 {user_id} 的用户数据：{path}", str(path)
    except Exception as exc:
        return f"导出用户数据失败：{exc}", None


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
    return result, build_status_text(provider, model, base_url, api_key)


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


def test_model_connection_and_refresh(
    provider: str,
    model: str,
    base_url: str,
    api_key: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
) -> tuple[str, str]:
    result = test_model_connection(
        provider,
        model,
        base_url,
        api_key,
        temperature,
        top_p,
        max_new_tokens,
    )
    return result, build_status_text(provider, model, base_url, api_key)


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


def preview_knowledge_search(query: str) -> str:
    query = (query or "").strip()
    if not query:
        return "请输入检索问题。"
    context = build_knowledge_context(query)
    return context or "没有检索到高相关资料。"


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

with gr.Blocks(title="情绪感知对话助手") as demo:
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
        placeholder="本地模型可留空；OpenAI 兼容接口可填写服务地址",
        render=False,
    )
    api_key_input = gr.Textbox(
        label="API Key",
        value="",
        type="password",
        placeholder="可留空并改用 .env，例如 DEEPSEEK_API_KEY",
        render=False,
    )
    user_id_input = gr.Textbox(
        label="User ID",
        value=os.getenv("CHATBOT_USER_ID", "local"),
        placeholder="输入用户名",
        render=False,
    )
    access_key_input = gr.Textbox(
        label="访问密码",
        value="",
        type="password",
        placeholder="首次使用该用户名请设置密码；之后需输入同样的密码才能访问其数据",
        render=False,
    )
    use_knowledge_input = gr.Checkbox(
        label="启用知识库",
        value=KNOWLEDGE_ENABLED,
        render=False,
    )
    use_style_input = gr.Checkbox(
        label="启用风格参考",
        value=STYLE_ENABLED,
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

    theme_mode_input = gr.Radio(
        label="页面主题",
        choices=[("浅色", "light"), ("深色", "dark")],
        value="light",
        elem_id="theme-mode",
    )

    gr.ChatInterface(
        fn=respond,
        title="情绪感知对话助手",
        additional_inputs=[
            user_id_input,
            access_key_input,
            use_knowledge_input,
            use_style_input,
            provider_input,
            model_input,
            base_url_input,
            api_key_input,
            temperature_input,
            top_p_input,
            max_new_tokens_input,
        ],
        additional_inputs_accordion="模型设置",
    )

    with gr.Accordion("用户访问", open=False):
        new_access_key_input = gr.Textbox(
            label="新访问密码",
            value="",
            type="password",
            placeholder="修改密码时填写，至少 8 位，可包含特殊符号",
        )
        with gr.Row():
            save_access_key_button = gr.Button("保存/验证密码")
            change_access_key_button = gr.Button("修改密码")
            export_user_data_button = gr.Button("导出用户数据")
        login_status_box = gr.Textbox(
            label="登录状态",
            value="未验证",
            lines=2,
            interactive=False,
        )
        access_key_status = gr.Textbox(
            label="访问密码状态",
            value="尚未验证访问密码。",
            lines=3,
            interactive=False,
        )
        export_user_data_status = gr.Textbox(
            label="数据导出状态",
            value="尚未导出。",
            lines=3,
            interactive=False,
        )
        export_user_data_file = gr.File(
            label="导出的用户数据",
            interactive=False,
        )

    with gr.Accordion("运行状态", open=False):
        status_box = gr.Textbox(
            label="状态",
            value=build_status_text(initial_provider, initial_model, initial_base_url),
            lines=9,
            interactive=False,
        )
        with gr.Row():
            refresh_button = gr.Button("刷新状态")
            connection_button = gr.Button("测试连接")
            save_model_config_button = gr.Button("保存模型配置")
        connection_result = gr.Textbox(label="连接测试结果", interactive=False)
        save_model_config_result = gr.Textbox(label="模型配置保存结果", interactive=False)
        status_timer = gr.Timer(value=2.0)

    with gr.Accordion("日志面板", open=False):
        log_box = gr.Textbox(
            label="最近日志",
            value=get_log_text,
            lines=14,
            interactive=False,
        )
        with gr.Row():
            refresh_logs_button = gr.Button("刷新日志")
            clear_logs_button = gr.Button("清空日志")
        log_timer = gr.Timer(value=3.0)

    with gr.Accordion("Mood Check-in", open=False):
        with gr.Row():
            mood_date_input = gr.Textbox(
                label="日期",
                value=date.today().isoformat(),
                placeholder="YYYY-MM-DD",
            )
            mood_choice_input = gr.Dropdown(
                label="今天的心情",
                choices=MOOD_CHOICES,
                value="一般",
                allow_custom_value=True,
            )
            mood_intensity_input = gr.Slider(
                label="强度",
                minimum=1,
                maximum=5,
                value=3,
                step=1,
            )
        mood_note_input = gr.Textbox(
            label="备注",
            placeholder="可以写下触发原因、身体状态、需要被记住的背景。",
            lines=3,
        )
        with gr.Row():
            save_mood_button = gr.Button("保存 Mood")
            refresh_mood_button = gr.Button("刷新 Mood")
            delete_mood_button = gr.Button("删除当天记录")
        mood_box = gr.Textbox(
            label="最近 Mood Check-ins",
            value="点击“刷新 Mood”查看当前用户的记录。",
            lines=10,
            interactive=False,
        )

    with gr.Accordion("Weekly Mood Chart", open=False):
        weekly_mood_end_date_input = gr.Textbox(
            label="结束日期",
            value=date.today().isoformat(),
            placeholder="YYYY-MM-DD",
        )
        refresh_weekly_mood_button = gr.Button("刷新周情绪图")
        weekly_mood_chart = gr.HTML(
            value=_render_weekly_mood_chart([], "light")
        )
        weekly_mood_summary = gr.Textbox(
            label="一周摘要",
            value=AUTH_REQUIRED_MESSAGE,
            lines=6,
            interactive=False,
        )


    with gr.Accordion("记忆管理", open=False):
        memory_section = gr.Radio(
            label="清理范围",
            choices=[
                ("短期对话", "history"),
                ("情绪记忆", "emotion"),
                ("长期记忆", "long"),
                ("兴趣记忆", "interest"),
                ("全部记忆", "all"),
            ],
            value="history",
        )
        with gr.Row():
            load_memory_button = gr.Button("查看记忆")
            save_memory_button = gr.Button("保存修改")
            clear_memory_button = gr.Button("清理所选记忆")
        memory_editor = gr.Textbox(
            label="记忆 JSON 编辑区",
            value="[]",
            lines=14,
            interactive=True,
        )
        memory_box = gr.Textbox(
            label="当前用户记忆",
            value="点击“查看记忆”加载当前用户的记忆。",
            lines=18,
            interactive=False,
        )

    with gr.Accordion("知识库 / RAG", open=False):
        knowledge_files = gr.File(
            label="导入资料",
            file_count="multiple",
            file_types=[".txt", ".md", ".markdown", ".csv", ".json", ".pdf", ".docx"],
        )
        with gr.Row():
            import_knowledge_button = gr.Button("导入并重建索引")
            rebuild_knowledge_button = gr.Button("重建索引")
            refresh_knowledge_button = gr.Button("查看知识库状态")
        knowledge_query = gr.Textbox(
            label="检索预览问题",
            placeholder="输入一个问题，查看会被检索进 Prompt 的资料片段",
        )
        preview_knowledge_button = gr.Button("检索预览")
        knowledge_box = gr.Textbox(
            label="知识库状态 / 检索结果",
            value=knowledge_status,
            lines=16,
            interactive=False,
        )

    with gr.Accordion("风格库 / Style RAG", open=False):
        style_files = gr.File(
            label="导入风格样例",
            file_count="multiple",
            file_types=[".txt", ".md", ".markdown", ".csv", ".json", ".jsonl"],
        )
        with gr.Row():
            import_style_button = gr.Button("导入并重建风格索引")
            rebuild_style_button = gr.Button("重建风格索引")
            refresh_style_button = gr.Button("查看风格库状态")
        style_query = gr.Textbox(
            label="风格检索预览",
            placeholder="输入当前问题或场景，查看会被参考的回复风格样例",
        )
        preview_style_button = gr.Button("风格检索预览")
        style_box = gr.Textbox(
            label="风格库状态 / 检索结果",
            value=style_status,
            lines=16,
            interactive=False,
        )

    provider_input.change(
        fn=provider_changed,
        inputs=[provider_input, api_key_input],
        outputs=[model_input, base_url_input, status_box],
    )
    refresh_button.click(
        fn=refresh_status,
        inputs=[provider_input, model_input, base_url_input, api_key_input],
        outputs=status_box,
    )
    status_timer.tick(
        fn=refresh_status,
        inputs=[provider_input, model_input, base_url_input, api_key_input],
        outputs=status_box,
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
        ],
        outputs=[connection_result, status_box],
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
        ],
        outputs=[save_model_config_result, status_box],
    )
    save_access_key_button.click(
        fn=save_access_key_and_status,
        inputs=[user_id_input, access_key_input],
        outputs=[access_key_status, login_status_box],
    )
    change_access_key_button.click(
        fn=change_access_key_and_status,
        inputs=[user_id_input, access_key_input, new_access_key_input],
        outputs=[access_key_status, access_key_input, new_access_key_input, login_status_box],
    )
    export_user_data_button.click(
        fn=export_user_data_from_gui,
        inputs=[user_id_input, access_key_input],
        outputs=[export_user_data_status, export_user_data_file],
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
        fn=submit_mood_checkin,
        inputs=[
            user_id_input,
            access_key_input,
            mood_date_input,
            mood_choice_input,
            mood_intensity_input,
            mood_note_input,
        ],
        outputs=mood_box,
    )
    refresh_mood_button.click(
        fn=refresh_mood_panel,
        inputs=[user_id_input, access_key_input],
        outputs=mood_box,
    )
    delete_mood_button.click(
        fn=delete_mood_checkin_from_gui,
        inputs=[user_id_input, access_key_input, mood_date_input],
        outputs=mood_box,
    )
    refresh_weekly_mood_button.click(
        fn=refresh_weekly_mood_chart,
        inputs=[user_id_input, access_key_input, weekly_mood_end_date_input, theme_mode_input],
        outputs=[weekly_mood_chart, weekly_mood_summary],
    )
    theme_mode_input.change(
        fn=refresh_weekly_mood_chart,
        inputs=[user_id_input, access_key_input, weekly_mood_end_date_input, theme_mode_input],
        outputs=[weekly_mood_chart, weekly_mood_summary],
        js=REFRESH_CHART_THEME_JS,
        show_progress="hidden",
    )
    demo.load(
        fn=load_theme_and_weekly_chart,
        inputs=[user_id_input, access_key_input, weekly_mood_end_date_input, theme_mode_input],
        outputs=[theme_mode_input, weekly_mood_chart, weekly_mood_summary],
        js=LOAD_THEME_JS,
        show_progress="hidden",
    )
    load_memory_button.click(
        fn=load_memory_editor,
        inputs=[user_id_input, access_key_input, memory_section],
        outputs=[memory_editor, memory_box],
    )
    save_memory_button.click(
        fn=save_memory_editor,
        inputs=[user_id_input, access_key_input, memory_section, memory_editor],
        outputs=[memory_editor, memory_box],
    )
    clear_memory_button.click(
        fn=clear_memory_section_and_reload,
        inputs=[user_id_input, access_key_input, memory_section],
        outputs=[memory_editor, memory_box],
    )
    import_knowledge_button.click(
        fn=import_knowledge_files,
        inputs=knowledge_files,
        outputs=knowledge_box,
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
        inputs=knowledge_query,
        outputs=knowledge_box,
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
    )
