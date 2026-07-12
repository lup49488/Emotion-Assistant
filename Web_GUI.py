from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

import gradio as gr

import goemotions_local as goemotions
from chatbot import (
    DEFAULT_LLM_PROVIDER,
    get_embedding_model,
    get_llm,
    handle_user_message_stream,
    make_model_config,
    session_store,
)
from config import (
    DEFAULT_API_BASE_URL,
    DEFAULT_API_MODEL,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    KNOWLEDGE_ENABLED,
    LOADED_ENV_FILES,
    STYLE_ENABLED,
)
from knowledge_store import (
    build_knowledge_context,
    import_documents,
    knowledge_status,
    rebuild_knowledge_index,
)
from llm_providers import ModelRuntimeConfig, require_openai_client
from style_store import (
    build_style_context,
    import_style_documents,
    rebuild_style_index,
    style_status,
)


PROVIDER_CHOICES = [
    "local_hf",
    "deepseek",
    "openai",
    "openrouter",
    "openai_compatible",
    "custom",
]

PROVIDER_API_KEYS = {
    "deepseek": ["DEEPSEEK_API_KEY", "LLM_API_KEY"],
    "openai": ["OPENAI_API_KEY", "LLM_API_KEY"],
    "openrouter": ["OPENROUTER_API_KEY", "LLM_API_KEY"],
    "openai_compatible": ["LLM_API_KEY"],
    "custom": ["LLM_API_KEY"],
}

MEMORY_SECTION_LABELS = {
    "history": "短期对话",
    "emotion": "情绪记忆",
    "long": "长期记忆",
    "interest": "兴趣记忆",
    "all": "全部记忆",
}


def unique_choices(values: list[str]) -> list[str]:
    choices: list[str] = []
    for value in values:
        if value and value not in choices:
            choices.append(value)
    return choices


MODEL_CHOICES = {
    "local_hf": unique_choices([
        os.getenv("CHAT_MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct"),
        "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct",
    ]),
    "deepseek": unique_choices([
        os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        "deepseek-chat",
        "deepseek-reasoner",
    ]),
    "openai": unique_choices([
        os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        "gpt-4.1-mini",
        "gpt-4.1",
        "gpt-4o-mini",
        "gpt-4o",
    ]),
    "openrouter": unique_choices([
        os.getenv("OPENROUTER_MODEL", "openai/gpt-4.1-mini"),
        "openai/gpt-4.1-mini",
        "openai/gpt-4o-mini",
        "anthropic/claude-3.5-sonnet",
        "google/gemini-2.0-flash-001",
        "deepseek/deepseek-chat",
    ]),
    "openai_compatible": unique_choices([
        DEFAULT_API_MODEL,
        "deepseek-chat",
        "qwen-plus",
        "moonshot-v1-8k",
    ]),
    "custom": unique_choices([
        DEFAULT_API_MODEL,
    ]),
}

DEFAULT_MODELS = {
    provider: choices[0] for provider, choices in MODEL_CHOICES.items()
}

DEFAULT_BASE_URLS = {
    "local_hf": "",
    "deepseek": "https://api.deepseek.com",
    "openai": "",
    "openrouter": "https://openrouter.ai/api/v1",
    "openai_compatible": DEFAULT_API_BASE_URL,
    "custom": DEFAULT_API_BASE_URL,
}

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


def _test_local_connection(config: ModelRuntimeConfig) -> str:
    tokenizer, model = get_llm()
    model_name = getattr(getattr(model, "config", None), "name_or_path", None)
    tokenizer_name = getattr(tokenizer, "name_or_path", None)
    loaded_name = model_name or tokenizer_name or config.resolved_model()
    return f"本地模型加载成功：{loaded_name}"


def _test_api_connection(config: ModelRuntimeConfig) -> str:
    api_key = config.resolved_api_key()
    if not api_key:
        raise RuntimeError("当前 Provider 缺少 API Key，请在 GUI 输入或写入 .env")

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
    return f"API 连接成功：{config.normalized_provider()} / {config.resolved_model()}"


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
        result = f"连接失败：{exc}"
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


def _format_item(item: dict[str, Any], index: int) -> str:
    time_text = str(item.get("time", ""))[:19]
    role = item.get("role")
    content = item.get("content")
    text = item.get("text")
    label = item.get("label")
    score = item.get("score")
    emotion = item.get("emotion")

    if role is not None and content is not None:
        return f"{index}. [{role}] {content}"
    if label is not None:
        score_text = f", score={score}" if score is not None else ""
        return f"{index}. {label}{score_text} ({time_text})"
    if text is not None:
        suffix = f" | emotion={emotion}" if emotion else ""
        return f"{index}. {text}{suffix} ({time_text})"
    return f"{index}. {item}"


def format_memory_snapshot(user_id: str, state: Any) -> str:
    history = list(state.history)
    emotion_memory = list(state.emotion_memory)
    long_memory = list(state.long_memory)
    interest_items = list(state.interest_store.items)

    sections = [
        ("短期对话", history),
        ("情绪记忆", emotion_memory),
        ("长期记忆", long_memory),
        ("兴趣记忆", interest_items),
    ]
    lines = [
        f"用户：{user_id}",
        (
            "统计："
            f"短期对话 {len(history)} 条，"
            f"情绪记忆 {len(emotion_memory)} 条，"
            f"长期记忆 {len(long_memory)} 条，"
            f"兴趣记忆 {len(interest_items)} 条"
        ),
    ]
    for title, items in sections:
        lines.append("")
        lines.append(f"## {title}")
        if not items:
            lines.append("暂无记录")
            continue
        for index, item in enumerate(items[-20:], start=1):
            lines.append(_format_item(item, index))
    return "\n".join(lines)


def load_memory_panel(user_id: str) -> str:
    user_id = user_id or "local"
    try:
        with session_store.session(user_id) as state:
            return format_memory_snapshot(user_id, state)
    except Exception as exc:
        return f"读取记忆失败：{exc}"


def _memory_section_items(state: Any, section: str) -> list[dict[str, Any]]:
    if section == "history":
        return list(state.history)
    if section == "emotion":
        return list(state.emotion_memory)
    if section == "long":
        return list(state.long_memory)
    if section == "interest":
        return list(state.interest_store.items)
    if section == "all":
        return [
            {"section": "history", "items": list(state.history)},
            {"section": "emotion", "items": list(state.emotion_memory)},
            {"section": "long", "items": list(state.long_memory)},
            {"section": "interest", "items": list(state.interest_store.items)},
        ]
    raise ValueError(f"未知记忆区块：{section}")


def _replace_memory_section(state: Any, section: str, items: list[dict[str, Any]]) -> None:
    if section == "history":
        state.history = items
        return
    if section == "emotion":
        state.emotion_memory = items
        return
    if section == "long":
        state.long_memory = items
        return
    if section == "interest":
        state.interest_store.replace_all(items)
        if state.vector_index is not None:
            state.vector_index.mark_dirty_for_rebuild()
        return
    raise ValueError(f"未知记忆区块：{section}")


def load_memory_editor(user_id: str, section: str) -> tuple[str, str]:
    user_id = user_id or "local"
    section = section or "history"
    try:
        with session_store.session(user_id) as state:
            items = _memory_section_items(state, section)
            editor_text = json.dumps(items, ensure_ascii=False, indent=2)
            snapshot = format_memory_snapshot(user_id, state)
        return editor_text, snapshot
    except Exception as exc:
        return "", f"读取记忆失败：{exc}"


def save_memory_editor(user_id: str, section: str, editor_text: str) -> tuple[str, str]:
    user_id = user_id or "local"
    section = section or "history"
    if section == "all":
        return editor_text, "暂不支持直接保存“全部记忆”。请分别选择短期、情绪、长期或兴趣记忆保存。"

    try:
        data = json.loads(editor_text or "[]")
        if not isinstance(data, list):
            raise ValueError("记忆内容必须是 JSON 数组。")
        if not all(isinstance(item, dict) for item in data):
            raise ValueError("JSON 数组中的每一项都必须是对象。")

        with session_store.session(user_id) as state:
            _replace_memory_section(state, section, data)
            snapshot = format_memory_snapshot(user_id, state)
        normalized = json.dumps(data, ensure_ascii=False, indent=2)
        label = MEMORY_SECTION_LABELS.get(section, section)
        return normalized, f"已保存 {user_id} 的{label}。\n\n{snapshot}"
    except Exception as exc:
        return editor_text, f"保存记忆失败：{exc}"


def clear_memory_section(user_id: str, section: str) -> str:
    user_id = user_id or "local"
    section = section or "history"
    try:
        with session_store.session(user_id) as state:
            if section in {"history", "all"}:
                state.history.clear()
            if section in {"emotion", "all"}:
                state.emotion_memory.clear()
            if section in {"long", "all"}:
                state.long_memory.clear()
            if section in {"interest", "all"}:
                state.interest_store.replace_all([])
                state.vector_index.mark_dirty_for_rebuild()
        label = MEMORY_SECTION_LABELS.get(section, section)
        return f"已清理 {user_id} 的{label}。\n\n{load_memory_panel(user_id)}"
    except Exception as exc:
        return f"清理记忆失败：{exc}"


def clear_memory_section_and_reload(user_id: str, section: str) -> tuple[str, str]:
    message = clear_memory_section(user_id, section)
    editor_text, snapshot = load_memory_editor(user_id, section)
    return editor_text, f"{message}\n\n{snapshot}"


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

    gr.ChatInterface(
        fn=respond,
        title="情绪感知对话助手",
        additional_inputs=[
            user_id_input,
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
        connection_result = gr.Textbox(label="连接测试结果", interactive=False)
        status_timer = gr.Timer(value=2.0)

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
    load_memory_button.click(
        fn=load_memory_editor,
        inputs=[user_id_input, memory_section],
        outputs=[memory_editor, memory_box],
    )
    save_memory_button.click(
        fn=save_memory_editor,
        inputs=[user_id_input, memory_section, memory_editor],
        outputs=[memory_editor, memory_box],
    )
    clear_memory_button.click(
        fn=clear_memory_section_and_reload,
        inputs=[user_id_input, memory_section],
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
    )
