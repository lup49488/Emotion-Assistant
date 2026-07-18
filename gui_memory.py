from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

from chatbot import session_store
from config import LONG_TERM_EXPIRY_DAYS
from gui_auth import authorize_or_message
from memory_backup import create_memory_backup, load_memory_backup, restore_memory_payload
from memory_store import record_memory_event, update_stable_profile


MEMORY_SECTION_LABELS = {
    "history": "短期对话",
    "emotion": "情绪记忆",
    "long": "长期记忆",
    "interest": "兴趣记忆",
    "stable": "稳定资料",
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
    stable_profile = list(state.stable_profile)
    interest_items = list(state.interest_store.items)

    sections = [
        ("短期对话", history),
        ("情绪记忆", emotion_memory),
        ("长期记忆", long_memory),
        ("稳定资料", stable_profile),
        ("兴趣记忆", interest_items),
    ]
    lines = [
        f"用户：{user_id}",
        (
            "统计："
            f"短期对话 {len(history)} 条，"
            f"情绪记忆 {len(emotion_memory)} 条，"
            f"长期记忆 {len(long_memory)} 条，"
            f"稳定资料 {len(stable_profile)} 条，"
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


def load_memory_panel(user_id: str, access_key: str, locale: str = "zh-CN") -> str:
    user_id, auth_error = authorize_or_message(user_id, access_key, locale)
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
    if section == "stable":
        return list(state.stable_profile)
    if section == "all":
        return [
            {"section": "history", "items": list(state.history)},
            {"section": "emotion", "items": list(state.emotion_memory)},
            {"section": "long", "items": list(state.long_memory)},
            {"section": "interest", "items": list(state.interest_store.items)},
            {"section": "stable", "items": list(state.stable_profile)},
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
    if section == "stable":
        normalized: list[dict[str, Any]] = []
        for item in items:
            text = str(item.get("text", "")).strip()
            if not text:
                raise ValueError("稳定资料的每一项都需要 text 字段。")
            normalized.append({**item, "text": text, "kind": "profile"})
        state.stable_profile = normalized
        return
    raise ValueError(f"未知记忆区块：{section}")


def load_memory_editor(
    user_id: str, access_key: str, section: str, locale: str = "zh-CN"
) -> tuple[str, str]:
    user_id, auth_error = authorize_or_message(user_id, access_key, locale)
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
    user_id: str, access_key: str, section: str, editor_text: str, locale: str = "zh-CN"
) -> tuple[str, str]:
    user_id, auth_error = authorize_or_message(user_id, access_key, locale)
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
            record_memory_event(
                state,
                section=section,
                action="updated",
                text=f"手动保存 {len(data)} 条{MEMORY_SECTION_LABELS.get(section, section)}",
                reason="用户在记忆管理面板中保存了编辑内容",
            )
            snapshot = format_memory_snapshot(user_id, state)
        normalized = json.dumps(data, ensure_ascii=False, indent=2)
        label = MEMORY_SECTION_LABELS.get(section, section)
        return normalized, f"已保存 {user_id} 的{label}。\n\n{snapshot}"
    except Exception as exc:
        return editor_text, f"保存记忆失败：{exc}"


def clear_memory_section(
    user_id: str, access_key: str, section: str, locale: str = "zh-CN"
) -> str:
    user_id, auth_error = authorize_or_message(user_id, access_key, locale)
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
            if section in {"stable", "all"}:
                state.stable_profile.clear()
        label = MEMORY_SECTION_LABELS.get(section, section)
        return f"已清理 {user_id} 的{label}。\n\n{load_memory_panel(user_id, access_key, locale)}"
    except Exception as exc:
        return f"清理记忆失败：{exc}"


def clear_memory_section_and_reload(
    user_id: str, access_key: str, section: str, locale: str = "zh-CN"
) -> tuple[str, str]:
    user_id, auth_error = authorize_or_message(user_id, access_key, locale)
    if auth_error:
        return "", auth_error
    message = clear_memory_section(user_id, access_key, section, locale)
    editor_text, snapshot = load_memory_editor(user_id, access_key, section, locale)
    return editor_text, f"{message}\n\n{snapshot}"


def load_stable_profile_editor(
    user_id: str, access_key: str, locale: str = "zh-CN"
) -> tuple[str, str]:
    return load_memory_editor(user_id, access_key, "stable", locale)


def save_stable_profile_editor(
    user_id: str, access_key: str, editor_text: str, locale: str = "zh-CN"
) -> tuple[str, str]:
    return save_memory_editor(user_id, access_key, "stable", editor_text, locale)


def clear_stable_profile(
    user_id: str, access_key: str, locale: str = "zh-CN"
) -> tuple[str, str]:
    return clear_memory_section_and_reload(user_id, access_key, "stable", locale)


def add_stable_profile(
    user_id: str, access_key: str, text: str, locale: str = "zh-CN"
) -> tuple[str, str, str]:
    user_id, auth_error = authorize_or_message(user_id, access_key, locale)
    if auth_error:
        return text, "", auth_error
    try:
        profile_text = (text or "").strip()
        if not profile_text:
            raise ValueError("请先输入要保存的稳定资料。")
        with session_store.session(user_id) as state:
            action = update_stable_profile(state, {"text": profile_text, "source": "manual"})
            record_memory_event(
                state,
                section="stable",
                action=action,
                text=profile_text,
                reason="用户在稳定资料面板中手动保存",
            )
            editor_text = json.dumps(state.stable_profile, ensure_ascii=False, indent=2)
            snapshot = format_memory_snapshot(user_id, state)
        return "", editor_text, f"已保存稳定资料。\n\n{snapshot}"
    except Exception as exc:
        return text, "", f"保存稳定资料失败：{exc}"


def format_memory_event_log(user_id: str, state: Any, limit: int = 20) -> str:
    events = list(getattr(state, "memory_events", []))
    if not events:
        return f"用户：{user_id}\n\n尚无记忆写入记录。"
    labels = {
        "stable": "稳定资料",
        "interest": "兴趣记忆",
        "long": "长期记忆",
        "emotion": "情绪记忆",
        "none": "未写入",
    }
    actions = {
        "added": "新增",
        "updated": "更新",
        "merged": "合并",
        "unchanged": "保持不变",
        "skipped": "跳过",
    }
    recent = events[-max(1, int(limit)):]
    lines = [f"用户：{user_id}", f"最近 {len(recent)} 条记忆判断：", ""]
    for event in reversed(recent):
        time_text = str(event.get("time", ""))[:19]
        section = labels.get(str(event.get("section")), str(event.get("section", "记忆")))
        action = actions.get(str(event.get("action")), str(event.get("action", "处理")))
        text = str(event.get("text", "")).strip() or "（无文本）"
        reason = str(event.get("reason", "")).strip() or "未记录原因"
        score = event.get("score")
        score_text = f" | 评分 {score}" if score is not None else ""
        lines.append(f"{time_text} | {section} | {action}{score_text}")
        lines.append(f"内容：{text}")
        lines.append(f"原因：{reason}")
        lines.append("")
    return "\n".join(lines).rstrip()


def load_memory_event_log(user_id: str, access_key: str, locale: str = "zh-CN") -> str:
    user_id, auth_error = authorize_or_message(user_id, access_key, locale)
    if auth_error:
        return auth_error
    try:
        with session_store.session(user_id) as state:
            return format_memory_event_log(user_id, state)
    except Exception as exc:
        return f"读取记忆写入记录失败：{exc}"


def _normalized_memory_text(item: Any, field: str) -> str:
    if not isinstance(item, dict):
        return ""
    return " ".join(str(item.get(field, "")).split()).casefold()


def _is_stale_timestamp(value: Any) -> tuple[bool, bool]:
    """Return (is_stale, is_invalid_or_missing) for a memory timestamp."""
    text = str(value or "").strip()
    if not text:
        return False, True
    try:
        timestamp = datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return False, True
    return timestamp < datetime.now() - timedelta(days=LONG_TERM_EXPIRY_DAYS), False


def build_memory_quality_report(user_id: str, state: Any, locale: str = "zh-CN") -> str:
    """Summarize memory hygiene without changing the stored user data."""
    sections = {
        "短期对话": list(state.history),
        "情绪记忆": list(state.emotion_memory),
        "长期记忆": list(state.long_memory),
        "兴趣记忆": list(state.interest_store.items),
        "稳定资料": list(state.stable_profile),
    }
    is_english = (locale or "").lower().startswith("en")
    issues: list[str] = []
    empty_items = 0
    invalid_time_items = 0
    stale_long_items = 0

    text_fields = {"情绪记忆": "label", "长期记忆": "text", "兴趣记忆": "text", "稳定资料": "text"}
    timed_sections = {"情绪记忆", "长期记忆", "兴趣记忆"}
    seen_texts: dict[str, str] = {}
    duplicate_items = 0

    for section, items in sections.items():
        for item in items:
            field = text_fields.get(section)
            if field:
                normalized = _normalized_memory_text(item, field)
                if not normalized:
                    empty_items += 1
                    continue
                if section != "情绪记忆":
                    previous_section = seen_texts.get(normalized)
                    if previous_section:
                        duplicate_items += 1
                    else:
                        seen_texts[normalized] = section
            if section in timed_sections:
                stale, invalid = _is_stale_timestamp(item.get("time") if isinstance(item, dict) else None)
                stale_long_items += int(section == "长期记忆" and stale)
                invalid_time_items += int(invalid)

    if empty_items:
        issues.append(
            f"{empty_items} 条持久记忆缺少有效内容" if not is_english
            else f"{empty_items} persistent memory item(s) have no usable content"
        )
    if duplicate_items:
        issues.append(
            f"发现 {duplicate_items} 条重复的长期、兴趣或稳定资料" if not is_english
            else f"Found {duplicate_items} duplicate long-term, interest, or profile item(s)"
        )
    if stale_long_items:
        issues.append(
            f"{stale_long_items} 条长期记忆超过 {LONG_TERM_EXPIRY_DAYS} 天" if not is_english
            else f"{stale_long_items} long-term memory item(s) exceed {LONG_TERM_EXPIRY_DAYS} days"
        )
    if invalid_time_items:
        issues.append(
            f"{invalid_time_items} 条记忆缺少或使用了无法解析的时间" if not is_english
            else f"{invalid_time_items} memory item(s) have missing or invalid timestamps"
        )

    score = max(0, 100 - min(40, empty_items * 10) - min(25, duplicate_items * 5)
                - min(20, stale_long_items * 5) - min(15, invalid_time_items * 3))
    if score >= 90:
        level = "良好" if not is_english else "Good"
    elif score >= 70:
        level = "需要关注" if not is_english else "Needs attention"
    else:
        level = "建议整理" if not is_english else "Cleanup recommended"

    counts = "，".join(f"{name} {len(items)} 条" for name, items in sections.items())
    if is_english:
        counts = ", ".join(f"{name}: {len(items)}" for name, items in sections.items())
        lines = [f"Memory quality report for {user_id}", f"Score: {score}/100 ({level})", f"Counts: {counts}"]
        lines.append("No issues found. Current memory structure looks healthy." if not issues else "Findings:")
    else:
        lines = [f"用户：{user_id}", f"记忆质量评分：{score}/100（{level}）", f"数量：{counts}"]
        lines.append("未发现需要处理的问题，当前记忆结构良好。" if not issues else "发现的问题：")
    if issues:
        lines.extend(f"- {issue}" for issue in issues)
    return "\n".join(lines)


def assess_memory_quality(user_id: str, access_key: str, locale: str = "zh-CN") -> str:
    user_id, auth_error = authorize_or_message(user_id, access_key, locale)
    if auth_error:
        return auth_error
    try:
        with session_store.session(user_id) as state:
            return build_memory_quality_report(user_id, state, locale)
    except Exception as exc:
        return f"评估记忆质量失败：{exc}" if not (locale or "").lower().startswith("en") else f"Memory quality assessment failed: {exc}"


def backup_memory_from_gui(
    user_id: str, access_key: str, locale: str = "zh-CN"
) -> tuple[str, str | None]:
    user_id, auth_error = authorize_or_message(user_id, access_key, locale)
    if auth_error:
        return auth_error, None
    try:
        with session_store.session(user_id) as state:
            path = create_memory_backup(user_id, state)
        return f"已备份 {user_id} 的全部记忆。", str(path)
    except Exception as exc:
        return f"备份记忆失败：{exc}", None


def _uploaded_backup_path(file_obj: Any) -> str:
    if isinstance(file_obj, (str, bytes)):
        return str(file_obj)
    path = getattr(file_obj, "name", None) or getattr(file_obj, "path", None)
    if path:
        return str(path)
    raise ValueError("请先选择记忆备份 JSON 文件。")


def restore_memory_from_gui(
    user_id: str,
    access_key: str,
    file_obj: Any,
    mode: str,
    locale: str = "zh-CN",
) -> tuple[str, str | None, str, str]:
    user_id, auth_error = authorize_or_message(user_id, access_key, locale)
    if auth_error:
        return auth_error, None, auth_error, auth_error
    try:
        payload = load_memory_backup(_uploaded_backup_path(file_obj))
        with session_store.session(user_id) as state:
            safety_backup = create_memory_backup(user_id, state, reason="pre_restore")
            result = restore_memory_payload(state, payload, mode=mode or "merge")
            record_memory_event(
                state,
                section="all",
                action="updated",
                text=f"从 {result['source_user_id']} 恢复全部记忆",
                reason=f"用户使用 {result['mode']} 模式恢复记忆备份",
            )
            snapshot = format_memory_snapshot(user_id, state)
            event_log = format_memory_event_log(user_id, state)
        mode_label = "合并" if result["mode"] == "merge" else "覆盖"
        counts = result["counts"]
        status = (
            f"记忆恢复完成：{mode_label}模式，来源用户 {result['source_user_id']}。\n"
            f"短期 {counts['history']}，情绪 {counts['emotion_memory']}，长期 {counts['long_memory']}，"
            f"稳定资料 {counts['stable_profile']}，兴趣 {counts['interest_memory']}。\n"
            "恢复前的记忆已自动备份。"
        )
        return status, str(safety_backup), snapshot, event_log
    except Exception as exc:
        message = f"恢复记忆失败：{exc}"
        return message, None, message, message
