from __future__ import annotations

from typing import Any

import gradio as gr

from chatbot import session_store
from config import SHORT_TERM_LIMIT
from conversation_store import create_conversation, delete_conversation, get_conversation, list_conversations
from gui_auth import authorize_or_message


def _message(locale: str, zh: str, en: str) -> str:
    return en if str(locale or "").lower().startswith("en") else zh


def _set_short_term_context(user_id: str, messages: list[dict[str, str]]) -> None:
    with session_store.session(user_id) as state:
        state.history = list(messages[-SHORT_TERM_LIMIT:])


def conversation_choices(user_id: str) -> list[tuple[str, str]]:
    choices: list[tuple[str, str]] = []
    for item in list_conversations(user_id):
        title = str(item["title"] or "New conversation")
        updated = str(item["updated_at"] or "")[:16].replace("T", " ")
        choices.append((f"{title} ({updated})", str(item["id"])))
    return choices


def refresh_conversation_panel(
    user_id: str, access_key: str, selected_id: str | None, locale: str = "zh-CN"
) -> tuple[Any, str]:
    user_id, error = authorize_or_message(user_id, access_key, locale)
    if error:
        return gr.update(choices=[], value=None), error
    choices = conversation_choices(user_id)
    valid_ids = {value for _, value in choices}
    value = selected_id if selected_id in valid_ids else None
    return gr.update(choices=choices, value=value), _message(
        locale, f"已加载 {len(choices)} 个历史会话。", f"Loaded {len(choices)} archived conversations."
    )


def new_conversation_from_gui(
    user_id: str, access_key: str, locale: str = "zh-CN"
) -> tuple[Any, list[dict[str, str]], str]:
    user_id, error = authorize_or_message(user_id, access_key, locale)
    if error:
        return gr.update(), [], error
    record = create_conversation(user_id)
    _set_short_term_context(user_id, [])
    return (
        gr.update(choices=conversation_choices(user_id), value=record["id"]),
        [],
        _message(locale, "已新建会话。", "Created a new conversation."),
    )


def load_conversation_from_gui(
    user_id: str, access_key: str, conversation_id: str | None, locale: str = "zh-CN"
) -> tuple[list[dict[str, str]], str]:
    user_id, error = authorize_or_message(user_id, access_key, locale)
    if error:
        return [], error
    record = get_conversation(user_id, conversation_id or "")
    if record is None:
        return [], _message(locale, "请选择一个有效的历史会话。", "Select a valid archived conversation.")
    messages = [
        {"role": str(item["role"]), "content": str(item["content"])}
        for item in record.get("messages", [])
        if item.get("role") in {"user", "assistant", "system"} and str(item.get("content", "")).strip()
    ]
    _set_short_term_context(user_id, messages)
    return messages, _message(locale, f"已加载会话：{record['title']}。", f"Loaded conversation: {record['title']}.")


def delete_conversation_from_gui(
    user_id: str, access_key: str, conversation_id: str | None, locale: str = "zh-CN"
) -> tuple[Any, list[dict[str, str]], str]:
    user_id, error = authorize_or_message(user_id, access_key, locale)
    if error:
        return gr.update(), [], error
    if not conversation_id:
        return gr.update(), [], _message(locale, "请先选择要删除的历史会话。", "Select a conversation to delete first.")
    if not delete_conversation(user_id, conversation_id):
        return gr.update(choices=conversation_choices(user_id)), [], _message(locale, "未找到要删除的历史会话。", "The selected conversation was not found.")
    _set_short_term_context(user_id, [])
    return gr.update(choices=conversation_choices(user_id), value=None), [], _message(
        locale, "历史会话已删除。", "Conversation deleted."
    )
