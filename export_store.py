from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from config import BASE_DIR
from conversation_store import list_conversations, get_conversation
from mood_store import load_mood_checkins
from session_store import load_state, validate_user_id

EXPORTS_DIR = BASE_DIR / "exports"


def _safe_export_name(user_id: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{user_id}_export_{stamp}.json"


def build_user_export_payload(user_id: str) -> dict[str, Any]:
    user_id = validate_user_id(user_id)
    state = load_state(user_id)
    mood_checkins = load_mood_checkins(user_id)
    return {
        "schema_version": 5,
        "exported_at": datetime.now().isoformat(timespec="seconds"),
        "user_id": user_id,
        "notes": [
            "此导出包含当前用户的对话、记忆和 Mood 数据。",
            "不会导出访问密码、密码哈希、API Key 或本地 .env 配置。",
            "Mood Check-in 图片附件保留在服务器的私有目录中，未包含在 JSON 导出内。",
        ],
        "history": list(state.history),
        "conversations": [
            get_conversation(user_id, item["id"])
            for item in list_conversations(user_id)
        ],
        "emotion_memory": list(state.emotion_memory),
        "long_memory": list(state.long_memory),
        "stable_profile": list(state.stable_profile),
        "memory_events": list(state.memory_events),
        "pending_memory": list(state.pending_memory),
        "interest_memory": list(state.interest_store.items),
        "mood_checkins": mood_checkins,
    }


def export_user_data(user_id: str) -> Path:
    user_id = validate_user_id(user_id)
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    target = EXPORTS_DIR / _safe_export_name(user_id)
    payload = build_user_export_payload(user_id)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target
