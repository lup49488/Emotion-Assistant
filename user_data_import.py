"""Validation and restoration for user-owned Serenova export files."""
from __future__ import annotations

import json
import re
from typing import Any

from conversation_store import restore_conversations
from memory_store import reconcile_memory_ownership
from mood_store import restore_mood_checkins
from session_store import validate_user_id


MAX_IMPORT_BYTES = 10 * 1024 * 1024
_CONVERSATION_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_MESSAGE_ROLES = {"user", "assistant", "system"}
_MEMORY_SECTIONS = {"stable", "interest", "long", "emotion"}


def _list_of_objects(payload: dict[str, Any], key: str, *, limit: int = 2_000) -> list[dict[str, Any]]:
    value = payload.get(key, [])
    if not isinstance(value, list) or len(value) > limit or any(not isinstance(item, dict) for item in value):
        raise ValueError(f"Invalid exported {key} data.")
    return [dict(item) for item in value]


def _text_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for item in items:
        text = str(item.get("text", "")).strip()
        if not text or len(text) > 2_000:
            raise ValueError("Exported memory text is invalid.")
        result.append({**item, "text": text})
    return result


def _validate_conversations(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    validated = []
    seen_ids: set[str] = set()
    for item in items:
        conversation_id = str(item.get("id", "")).strip()
        title = str(item.get("title", "")).strip()
        messages = item.get("messages", [])
        if not _CONVERSATION_ID.match(conversation_id) or conversation_id in seen_ids:
            raise ValueError("Exported conversation ID is invalid or duplicated.")
        if not title or len(title) > 200 or not isinstance(messages, list) or len(messages) > 2_000:
            raise ValueError("Exported conversation data is invalid.")
        normalized_messages = []
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("Exported conversation message is invalid.")
            role = str(message.get("role", ""))
            content = str(message.get("content", ""))
            if role not in _MESSAGE_ROLES or not content or len(content) > 20_000:
                raise ValueError("Exported conversation message is invalid.")
            normalized_messages.append({
                "role": role,
                "content": content,
                "created_at": str(message.get("created_at") or item.get("updated_at") or ""),
            })
        seen_ids.add(conversation_id)
        validated.append({
            "id": conversation_id,
            "title": title,
            "created_at": str(item.get("created_at") or ""),
            "updated_at": str(item.get("updated_at") or ""),
            "messages": normalized_messages,
        })
    return validated


def _merge_by_text(existing: list[dict[str, Any]], imported: list[dict[str, Any]]) -> list[dict[str, Any]]:
    known = {str(item.get("text", "")).strip() for item in existing}
    return [*existing, *(item for item in imported if item["text"] not in known)]


def parse_export_bytes(raw: bytes, user_id: str) -> dict[str, Any]:
    if not raw or len(raw) > MAX_IMPORT_BYTES:
        raise ValueError("Import file must be a non-empty JSON export under 10 MB.")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Import file must be valid UTF-8 JSON.") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("schema_version"), int):
        raise ValueError("This file is not a recognized Serenova export.")
    if int(payload["schema_version"]) < 1 or int(payload["schema_version"]) > 4:
        raise ValueError("This export version is not supported.")
    if str(payload.get("user_id", "")) != validate_user_id(user_id):
        raise ValueError("The export belongs to a different user ID.")

    history = _list_of_objects(payload, "history", limit=1_000)
    for item in history:
        if str(item.get("role", "")) not in _MESSAGE_ROLES or not str(item.get("content", "")).strip():
            raise ValueError("Exported chat history is invalid.")
    emotion = _list_of_objects(payload, "emotion_memory")
    if any(not str(item.get("label", "")).strip() for item in emotion):
        raise ValueError("Exported emotion memory is invalid.")
    pending = _list_of_objects(payload, "pending_memory")
    for item in pending:
        candidate = item.get("candidate")
        if str(item.get("section", "")) not in _MEMORY_SECTIONS or not isinstance(candidate, dict) or not str(candidate.get("text", "")).strip():
            raise ValueError("Exported pending memory is invalid.")
    events = _list_of_objects(payload, "memory_events")
    moods = _list_of_objects(payload, "mood_checkins", limit=5_000)
    for item in moods:
        date = str(item.get("date", ""))
        mood = str(item.get("mood", "")).strip()
        intensity = item.get("intensity")
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date) or not mood or len(mood) > 100:
            raise ValueError("Exported mood check-in is invalid.")
        try:
            if int(intensity) not in range(1, 6):
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise ValueError("Exported mood check-in is invalid.") from exc
    return {
        "history": history,
        "conversations": _validate_conversations(_list_of_objects(payload, "conversations")),
        "emotion_memory": emotion,
        "long_memory": _text_items(_list_of_objects(payload, "long_memory")),
        "stable_profile": _text_items(_list_of_objects(payload, "stable_profile")),
        "interest_memory": _text_items(_list_of_objects(payload, "interest_memory")),
        "memory_events": events,
        "pending_memory": pending,
        "mood_checkins": moods,
    }


def import_user_export(user_id: str, raw: bytes, *, mode: str) -> dict[str, int | str]:
    if mode not in {"merge", "replace"}:
        raise ValueError("Import mode must be merge or replace.")
    user_id = validate_user_id(user_id)
    imported = parse_export_bytes(raw, user_id)

    # Import only after the full file has passed validation.
    from chatbot import session_store

    with session_store.session(user_id) as state:
        if mode == "replace":
            state.history = imported["history"]
            state.emotion_memory = imported["emotion_memory"]
            state.long_memory = imported["long_memory"]
            state.stable_profile = imported["stable_profile"]
            state.interest_store.replace_all(imported["interest_memory"])
            state.memory_events = imported["memory_events"][-100:]
            state.pending_memory = imported["pending_memory"]
        else:
            history_keys = {(item.get("role"), item.get("content")) for item in state.history}
            state.history.extend(item for item in imported["history"] if (item.get("role"), item.get("content")) not in history_keys)
            state.history = state.history[-1_000:]
            state.emotion_memory.extend(
                item for item in imported["emotion_memory"]
                if (item.get("label"), item.get("time")) not in {(value.get("label"), value.get("time")) for value in state.emotion_memory}
            )
            state.long_memory = _merge_by_text(state.long_memory, imported["long_memory"])
            state.stable_profile = _merge_by_text(state.stable_profile, imported["stable_profile"])
            state.interest_store.replace_all(_merge_by_text(state.interest_store.items, imported["interest_memory"]))
            event_ids = {str(item.get("id", "")) for item in state.memory_events}
            state.memory_events = [*state.memory_events, *(item for item in imported["memory_events"] if str(item.get("id", "")) not in event_ids)][-100:]
            pending_ids = {str(item.get("id", "")) for item in state.pending_memory}
            state.pending_memory.extend(item for item in imported["pending_memory"] if str(item.get("id", "")) not in pending_ids)
        if state.vector_index is not None:
            state.vector_index.mark_dirty_for_rebuild()
        reconcile_memory_ownership(state)

    conversations = restore_conversations(user_id, imported["conversations"], mode=mode)
    moods = restore_mood_checkins(user_id, imported["mood_checkins"], mode=mode)
    return {
        "mode": mode,
        "conversations": conversations,
        "mood_checkins": moods,
        "memories": len(imported["long_memory"]) + len(imported["stable_profile"]) + len(imported["interest_memory"]),
    }
