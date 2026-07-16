from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from config import BASE_DIR
from session_store import validate_user_id


BACKUPS_DIR = BASE_DIR / "exports" / "memory_backups"
MAX_BACKUP_BYTES = 10 * 1024 * 1024
MAX_BACKUP_ITEMS = 50_000
MEMORY_KEYS = (
    "history",
    "emotion_memory",
    "long_memory",
    "stable_profile",
    "interest_memory",
    "memory_events",
)


def _backup_name(user_id: str, reason: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    safe_reason = "pre_restore" if reason == "pre_restore" else "manual"
    return f"{user_id}_memory_{safe_reason}_{stamp}.json"


def build_memory_backup_payload(user_id: str, state: Any) -> dict[str, Any]:
    user_id = validate_user_id(user_id)
    return {
        "kind": "memory_backup",
        "schema_version": 1,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "source_user_id": user_id,
        "notes": [
            "此文件只包含记忆数据，不包含访问密码、API Key 或 Mood 数据。",
            "可以恢复到另一个已通过验证的 User ID。",
        ],
        "history": list(state.history),
        "emotion_memory": list(state.emotion_memory),
        "long_memory": list(state.long_memory),
        "stable_profile": list(state.stable_profile),
        "interest_memory": list(state.interest_store.items),
        "memory_events": list(getattr(state, "memory_events", [])),
    }


def create_memory_backup(user_id: str, state: Any, *, reason: str = "manual") -> Path:
    payload = build_memory_backup_payload(user_id, state)
    BACKUPS_DIR.mkdir(parents=True, exist_ok=True)
    target = BACKUPS_DIR / _backup_name(user_id, reason)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def load_memory_backup(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    if not source.exists() or not source.is_file():
        raise FileNotFoundError("备份文件不存在。")
    if source.suffix.lower() != ".json":
        raise ValueError("记忆备份必须是 JSON 文件。")
    if source.stat().st_size > MAX_BACKUP_BYTES:
        raise ValueError("记忆备份超过 10 MB 限制。")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("记忆备份不是有效的 UTF-8 JSON 文件。") from exc
    return validate_memory_backup_payload(payload)


def validate_memory_backup_payload(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("记忆备份的顶层必须是 JSON 对象。")
    kind = payload.get("kind")
    if kind not in {None, "memory_backup"}:
        raise ValueError("该文件不是受支持的记忆备份。")

    normalized = dict(payload)
    total_items = 0
    for key in MEMORY_KEYS:
        items = payload.get(key, [])
        if not isinstance(items, list) or not all(isinstance(item, dict) for item in items):
            raise ValueError(f"字段 {key} 必须是由 JSON 对象组成的数组。")
        total_items += len(items)
        normalized[key] = [dict(item) for item in items]
    if total_items > MAX_BACKUP_ITEMS:
        raise ValueError(f"记忆备份条目数超过 {MAX_BACKUP_ITEMS} 条限制。")

    for key in ("stable_profile", "interest_memory"):
        for item in normalized[key]:
            text = str(item.get("text", "")).strip()
            if not text:
                raise ValueError(f"字段 {key} 中的每一项都需要非空 text。")
            item["text"] = text
    for item in normalized["stable_profile"]:
        item["kind"] = "profile"
    return normalized


def _merge_unique(current: list[dict[str, Any]], incoming: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = [dict(item) for item in current]
    known = {json.dumps(item, ensure_ascii=False, sort_keys=True) for item in result}
    for item in incoming:
        key = json.dumps(item, ensure_ascii=False, sort_keys=True)
        if key not in known:
            result.append(dict(item))
            known.add(key)
    return result


def restore_memory_payload(state: Any, payload: dict[str, Any], *, mode: str = "merge") -> dict[str, Any]:
    normalized = validate_memory_backup_payload(payload)
    if mode not in {"merge", "replace"}:
        raise ValueError("恢复模式必须是 merge 或 replace。")

    current = {
        "history": list(state.history),
        "emotion_memory": list(state.emotion_memory),
        "long_memory": list(state.long_memory),
        "stable_profile": list(state.stable_profile),
        "interest_memory": list(state.interest_store.items),
        "memory_events": list(getattr(state, "memory_events", [])),
    }
    restored: dict[str, list[dict[str, Any]]] = {}
    for key in MEMORY_KEYS:
        incoming = normalized[key]
        restored[key] = _merge_unique(current[key], incoming) if mode == "merge" else incoming

    state.history = restored["history"]
    state.emotion_memory = restored["emotion_memory"]
    state.long_memory = restored["long_memory"]
    state.stable_profile = restored["stable_profile"]
    state.interest_store.replace_all(restored["interest_memory"])
    state.memory_events = restored["memory_events"][-100:]
    if state.vector_index is not None:
        state.vector_index.mark_dirty_for_rebuild()

    return {
        "mode": mode,
        "source_user_id": str(normalized.get("source_user_id") or normalized.get("user_id") or "未知"),
        "counts": {key: len(restored[key]) for key in MEMORY_KEYS},
    }
