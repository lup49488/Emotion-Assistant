from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from config import BASE_DIR
from json_utils import load_json
from session_store import user_file_lock, user_paths, validate_user_id

EXPORTS_DIR = BASE_DIR / "exports"


def _safe_export_name(user_id: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{user_id}_export_{stamp}.json"


def build_user_export_payload(user_id: str) -> dict[str, Any]:
    user_id = validate_user_id(user_id)
    paths = user_paths(user_id)
    with user_file_lock(user_id):
        return {
            "schema_version": 2,
            "exported_at": datetime.now().isoformat(timespec="seconds"),
            "user_id": user_id,
            "notes": [
                "此导出包含当前用户的对话、记忆和 Mood 数据。",
                "不会导出访问密码、密码哈希、API Key 或本地 .env 配置。",
            ],
            "history": load_json(paths["history"]),
            "emotion_memory": load_json(paths["emotion_memory"]),
            "long_memory": load_json(paths["long_memory"]),
            "stable_profile": load_json(paths["stable_profile"]),
            "memory_events": load_json(paths["memory_events"]),
            "interest_memory": load_json(paths["interest_memory"]),
            "mood_checkins": load_json(paths["mood_checkins"]),
        }


def export_user_data(user_id: str) -> Path:
    user_id = validate_user_id(user_id)
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    target = EXPORTS_DIR / _safe_export_name(user_id)
    payload = build_user_export_payload(user_id)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return target
