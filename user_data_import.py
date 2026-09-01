"""Validation and restoration for user-owned Serenova export files."""
from __future__ import annotations

import json
import re
import hashlib
import io
import zipfile
from typing import Any

import logging

from config import SHORT_TERM_LIMIT
from conversation_store import restore_conversations
from export_store import build_user_export_payload
from memory_backup import create_memory_backup
from memory_store import MEMORY_SECTIONS, expire_overflowing_pending_memory, reconcile_memory_ownership
from mood_store import restore_mood_checkins
from session_store import validate_user_id


logger = logging.getLogger(__name__)

MAX_IMPORT_BYTES = 10 * 1024 * 1024
MAX_EXTERNAL_UNCOMPRESSED_BYTES = 30 * 1024 * 1024
MAX_EXTERNAL_USER_BYTES = 512 * 1024
_CONVERSATION_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_MESSAGE_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_MESSAGE_ROLES = {"user", "assistant", "system"}


def _external_id(prefix: str, value: Any) -> str:
    digest = hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:40]}"


def _read_zip_member(archive: zipfile.ZipFile, info: zipfile.ZipInfo, limit: int) -> bytes:
    """Read one member, bounded by what it actually decompresses to.

    ZipInfo.file_size comes from the archive's own header, so an uploader can
    understate it and slip an oversized member past a pre-read check. Reading
    one byte past the limit and rejecting on that keeps the ceiling real.
    """
    with archive.open(info) as member:
        data = member.read(limit + 1)
    if len(data) > limit:
        raise ValueError("The archive contains too much uncompressed data.")
    return data


def _read_external_payload(raw: bytes) -> Any:
    if not raw or len(raw) > MAX_IMPORT_BYTES:
        raise ValueError("Import file must be non-empty and under 10 MB.")
    if raw[:2] == b"PK":
        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as archive:
                names = [name for name in archive.namelist() if not name.endswith("/") and name.lower().endswith(".json")]
                if len(names) > 30 or not names:
                    raise ValueError("The archive does not contain a supported JSON export.")
                info = next((archive.getinfo(name) for name in names if name.lower().endswith("conversations.json")), archive.getinfo(names[0]))
                payload = json.loads(_read_zip_member(archive, info, MAX_EXTERNAL_UNCOMPRESSED_BYTES).decode("utf-8"))
                user_info = next((archive.getinfo(name) for name in names if name.lower().endswith("user.json")), None)
                if user_info is not None:
                    user_payload = json.loads(_read_zip_member(archive, user_info, MAX_EXTERNAL_USER_BYTES).decode("utf-8"))
                    if isinstance(payload, list) and isinstance(user_payload, dict):
                        return {"conversations": payload, "user": user_payload}
                return payload
        except (zipfile.BadZipFile, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("The archive does not contain a readable JSON conversation export.") from exc
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Import file must be valid UTF-8 JSON or a supported ZIP export.") from exc


def _external_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(filter(None, (_external_text(item) for item in value))).strip()
    if isinstance(value, dict):
        return str(value.get("text") or value.get("content") or "").strip()
    return ""


def _normalize_external_conversation(source: str, item: dict[str, Any], messages: list[dict[str, Any]], index: int) -> dict[str, Any] | None:
    normalized = []
    for position, message in enumerate(messages):
        role = str(message.get("role") or message.get("sender") or message.get("author", {}).get("role") or "").lower()
        role = "assistant" if role in {"assistant", "claude", "chatgpt"} else "user" if role in {"user", "human"} else "system" if role == "system" else ""
        content = _external_text(message.get("content") if "content" in message else message.get("text") or message.get("parts"))
        if role and content:
            normalized.append({"id": _external_id("msg", [source, index, position, role, content]), "role": role, "content": content[:20_000], "created_at": str(message.get("created_at") or message.get("create_time") or "")})
    if not normalized:
        return None
    title = str(item.get("title") or item.get("name") or f"Imported {source.title()} conversation {index + 1}").strip()[:200]
    return {"id": _external_id("conv", [source, index, title, normalized]), "title": title or f"Imported conversation {index + 1}", "created_at": str(item.get("created_at") or item.get("create_time") or ""), "updated_at": str(item.get("updated_at") or item.get("update_time") or ""), "messages": normalized}


def _chatgpt_conversations(payload: Any) -> list[dict[str, Any]]:
    items = payload if isinstance(payload, list) else payload.get("conversations", []) if isinstance(payload, dict) else []
    conversations = []
    for index, item in enumerate(items[:2_000]):
        if not isinstance(item, dict) or not isinstance(item.get("mapping"), dict):
            continue
        messages = []
        for node in item["mapping"].values():
            message = node.get("message") if isinstance(node, dict) else None
            if not isinstance(message, dict):
                continue
            author = message.get("author") if isinstance(message.get("author"), dict) else {}
            parts = message.get("content", {}).get("parts", []) if isinstance(message.get("content"), dict) else []
            messages.append({"role": author.get("role"), "parts": parts, "create_time": message.get("create_time")})
        normalized = _normalize_external_conversation("chatgpt", item, messages, index)
        if normalized:
            conversations.append(normalized)
    return conversations


def _claude_or_generic_conversations(payload: Any) -> tuple[str, list[dict[str, Any]]]:
    if isinstance(payload, dict) and isinstance(payload.get("chat_messages"), list):
        groups: dict[str, list[dict[str, Any]]] = {}
        for message in payload["chat_messages"][:10_000]:
            if isinstance(message, dict): groups.setdefault(str(message.get("conversation_uuid") or message.get("conversation_id") or "0"), []).append(message)
        source, items = "claude", [{"name": f"Claude conversation {index + 1}", "chat_messages": messages} for index, messages in enumerate(groups.values())]
    else:
        source = "claude" if isinstance(payload, list) and any(isinstance(item, dict) and isinstance(item.get("chat_messages"), list) for item in payload) else "generic"
        items = payload.get("conversations", []) if isinstance(payload, dict) else payload if isinstance(payload, list) else []
    conversations = []
    for index, item in enumerate(items[:2_000]):
        if not isinstance(item, dict): continue
        messages = item.get("chat_messages") or item.get("messages") or []
        if isinstance(messages, list):
            normalized = _normalize_external_conversation(source, item, messages, index)
            if normalized: conversations.append(normalized)
    return source, conversations


def _external_profile(payload: Any) -> list[dict[str, str]]:
    if not isinstance(payload, dict): return []
    user = payload.get("user") or payload.get("account") or payload.get("profile")
    if not isinstance(user, dict): return []
    fields = []
    for key, label in (("name", "Name"), ("email", "Email"), ("location", "Location"), ("bio", "Bio")):
        value = _external_text(user.get(key))
        if value and len(value) <= 500: fields.append({"key": key, "label": label, "value": value})
    return fields


def parse_external_export_bytes(raw: bytes, selected_profile_fields: list[str] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    payload = _read_external_payload(raw)
    chatgpt = _chatgpt_conversations(payload)
    source, conversations = ("chatgpt", chatgpt) if chatgpt else _claude_or_generic_conversations(payload)
    if not conversations:
        raise ValueError("No supported conversations were found. Choose a ChatGPT or Claude JSON/ZIP export.")
    profile = _external_profile(payload)
    selected = set(selected_profile_fields or [])
    stable_profile = [{"text": f"Imported from {source.title()} — {field['label']}: {field['value']}", "kind": "imported_profile"} for field in profile if field["key"] in selected]
    imported = {"history": [], "conversations": conversations, "emotion_memory": [], "long_memory": [], "stable_profile": stable_profile, "interest_memory": [], "memory_events": [], "pending_memory": [], "mood_checkins": []}
    preview = {"source": source, "conversations": len(conversations), "messages": sum(len(item["messages"]) for item in conversations), "profile_fields": profile, "sample_titles": [item["title"] for item in conversations[:3]]}
    return imported, preview


def preview_external_import(raw: bytes) -> dict[str, Any]:
    return parse_external_export_bytes(raw)[1]


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
        message_ids: set[str] = set()
        for message in messages:
            if not isinstance(message, dict):
                raise ValueError("Exported conversation message is invalid.")
            role = str(message.get("role", ""))
            content = str(message.get("content", ""))
            if role not in _MESSAGE_ROLES or not content or len(content) > 20_000:
                raise ValueError("Exported conversation message is invalid.")
            message_id = str(message.get("id") or "").strip()
            if message_id and (not _MESSAGE_ID.match(message_id) or message_id in message_ids):
                raise ValueError("Exported conversation message ID is invalid or duplicated.")
            normalized = {
                "role": role,
                "content": content,
                "created_at": str(message.get("created_at") or item.get("updated_at") or ""),
            }
            if message_id:
                normalized["id"] = message_id
                message_ids.add(message_id)
            reply_to = str(message.get("reply_to_message_id") or "").strip()
            if reply_to:
                if not _MESSAGE_ID.match(reply_to):
                    raise ValueError("Exported conversation quote is invalid.")
                normalized["reply_to_message_id"] = reply_to
            normalized_messages.append(normalized)
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
    if int(payload["schema_version"]) < 1 or int(payload["schema_version"]) > 5:
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
        if str(item.get("section", "")) not in MEMORY_SECTIONS or not isinstance(candidate, dict) or not str(candidate.get("text", "")).strip():
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


def _durable_memory_count(state: Any) -> int:
    return len(state.long_memory) + len(state.stable_profile) + len(state.interest_store.items)


def _apply_memory_import(user_id: str, imported: dict[str, Any], *, mode: str) -> int:
    """Write the memory sections and report how many durable memories were added."""
    from chatbot import session_store

    with session_store.session(user_id) as state:
        before = _durable_memory_count(state)
        if mode == "replace":
            state.history = imported["history"]
            state.emotion_memory = imported["emotion_memory"]
            state.long_memory = imported["long_memory"]
            state.stable_profile = imported["stable_profile"]
            state.interest_store.replace_all(imported["interest_memory"])
            state.memory_events = imported["memory_events"][-100:]
            state.pending_memory = list(imported["pending_memory"])
        else:
            history_keys = {(item.get("role"), item.get("content")) for item in state.history}
            state.history.extend(item for item in imported["history"] if (item.get("role"), item.get("content")) not in history_keys)
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
        # An import must leave the same invariants a normal turn would. Without the
        # trim the next request would send the whole imported transcript to the
        # model, and the queue would be truncated at save time with no audit trail.
        state.history = state.history[-SHORT_TERM_LIMIT:]
        expire_overflowing_pending_memory(state)
        if state.vector_index is not None:
            state.vector_index.mark_dirty_for_rebuild()
        reconcile_memory_ownership(state)
        return max(0, _durable_memory_count(state) - before)


def _apply_import(user_id: str, imported: dict[str, Any], *, mode: str) -> dict[str, int | str]:
    # Restore independent stores before replacing the in-session memory state. If
    # a conversation or Mood Check-in write fails, the existing memory remains
    # intact and rollback has less work to do.
    conversations = restore_conversations(user_id, imported["conversations"], mode=mode)
    moods = restore_mood_checkins(user_id, imported["mood_checkins"], mode=mode)
    memories = _apply_memory_import(user_id, imported, mode=mode)
    return {
        "mode": mode,
        "conversations": conversations,
        "mood_checkins": moods,
        # Actual additions, so all three counts mean the same thing. Re-importing
        # the same file reports zeros instead of the file's own size.
        "memories": memories,
    }


def _snapshot_for_rollback(user_id: str) -> dict[str, Any]:
    """Capture what the replace is about to overwrite, from the same source.

    build_user_export_payload() reloads the account from storage, but the import
    writes through session_store's cached SessionState. Those are two different
    objects, so snapshotting the reloaded copy can restore something other than
    what was replaced. Reading the live session keeps the rollback exact.
    """
    from chatbot import session_store

    with session_store.session(user_id) as state:
        memory = {
            "history": list(state.history),
            "emotion_memory": list(state.emotion_memory),
            "long_memory": list(state.long_memory),
            "stable_profile": list(state.stable_profile),
            "interest_memory": list(state.interest_store.items),
            "memory_events": list(state.memory_events),
            "pending_memory": list(state.pending_memory),
        }
    exported = build_user_export_payload(user_id)
    return {
        **memory,
        # These two live in their own stores, which the import writes directly.
        "conversations": _validate_conversations(exported["conversations"]),
        "mood_checkins": list(exported["mood_checkins"]),
    }


def import_user_export(user_id: str, raw: bytes, *, mode: str) -> dict[str, int | str]:
    if mode not in {"merge", "replace"}:
        raise ValueError("Import mode must be merge or replace.")
    user_id = validate_user_id(user_id)
    imported = parse_export_bytes(raw, user_id)

    if mode != "replace":
        # Merge only ever adds, so a failure part-way leaves the account usable.
        return _apply_import(user_id, imported, mode=mode)

    # Replace overwrites memories, conversations and Mood Check-ins in three
    # separate stores. Without a rollback a failure between them would leave the
    # account half replaced, with the discarded half unrecoverable.
    snapshot = _snapshot_for_rollback(user_id)
    from chatbot import session_store

    with session_store.session(user_id) as state:
        backup_path = create_memory_backup(user_id, state, reason="pre_restore")
    try:
        return _apply_import(user_id, imported, mode=mode)
    except Exception:
        try:
            _apply_import(user_id, snapshot, mode="replace")
        except Exception:
            logger.exception(
                "导入失败后回滚也失败，导入前的记忆备份保存在 %s。user=%s", backup_path, user_id,
            )
        raise


def import_external_export(user_id: str, raw: bytes, *, mode: str, selected_profile_fields: list[str] | None = None) -> dict[str, int | str]:
    """Import a reviewed third-party export without accepting its credentials or settings."""
    if mode not in {"merge", "replace"}:
        raise ValueError("Import mode must be merge or replace.")
    user_id = validate_user_id(user_id)
    imported, preview = parse_external_export_bytes(raw, selected_profile_fields)
    if mode != "replace":
        return {**_apply_import(user_id, imported, mode=mode), "source": preview["source"]}
    snapshot = _snapshot_for_rollback(user_id)
    from chatbot import session_store
    with session_store.session(user_id) as state:
        backup_path = create_memory_backup(user_id, state, reason="pre_external_restore")
    try:
        return {**_apply_import(user_id, imported, mode=mode), "source": preview["source"]}
    except Exception:
        try:
            _apply_import(user_id, snapshot, mode="replace")
        except Exception:
            logger.exception("外部导入失败后回滚也失败，导入前的记忆备份保存在 %s。user=%s", backup_path, user_id)
        raise
