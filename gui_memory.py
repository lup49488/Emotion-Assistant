from __future__ import annotations

import json
from typing import Any

from chatbot import session_store
from gui_auth import authorize_or_message


MEMORY_SECTION_LABELS = {
    "history": "短期对话",
    "emotion": "情绪记忆",
    "long": "长期记忆",
    "interest": "兴趣记忆",
    "all": "全部记忆",
}


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


def load_memory_panel(user_id: str, access_key: str) -> str:
    user_id, auth_error = authorize_or_message(user_id, access_key)
    if auth_error:
        return auth_error
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


def load_memory_editor(user_id: str, access_key: str, section: str) -> tuple[str, str]:
    user_id, auth_error = authorize_or_message(user_id, access_key)
    if auth_error:
        return "", auth_error
    section = section or "history"
    try:
        with session_store.session(user_id) as state:
            items = _memory_section_items(state, section)
            editor_text = json.dumps(items, ensure_ascii=False, indent=2)
            snapshot = format_memory_snapshot(user_id, state)
        return editor_text, snapshot
    except Exception as exc:
        return "", f"读取记忆失败：{exc}"


def save_memory_editor(
    user_id: str, access_key: str, section: str, editor_text: str
) -> tuple[str, str]:
    user_id, auth_error = authorize_or_message(user_id, access_key)
    if auth_error:
        return editor_text, auth_error
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


def clear_memory_section(user_id: str, access_key: str, section: str) -> str:
    user_id, auth_error = authorize_or_message(user_id, access_key)
    if auth_error:
        return auth_error
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
        return f"已清理 {user_id} 的{label}。\n\n{load_memory_panel(user_id, access_key)}"
    except Exception as exc:
        return f"清理记忆失败：{exc}"


def clear_memory_section_and_reload(
    user_id: str, access_key: str, section: str
) -> tuple[str, str]:
    user_id, auth_error = authorize_or_message(user_id, access_key)
    if auth_error:
        return "", auth_error
    message = clear_memory_section(user_id, access_key, section)
    editor_text, snapshot = load_memory_editor(user_id, access_key, section)
    return editor_text, f"{message}\n\n{snapshot}"
