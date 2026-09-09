from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any

from api_usage_store import usage_summary
from conversation_store import list_conversations
from export_store import EXPORTS_DIR
from memory_backup import BACKUPS_DIR
from mood_store import load_mood_checkins
from session_store import load_state, user_dir, user_file_lock, validate_user_id
from sqlite_store import connection, sqlite_enabled, storage_backend


def privacy_summary(user_id: str) -> dict[str, Any]:
    user_id = validate_user_id(user_id)
    state = load_state(user_id)
    conversations = list_conversations(user_id)
    usage = usage_summary(user_id)
    return {
        "backend": storage_backend(),
        "conversation_count": len(conversations),
        "message_count": sum(int(item.get("message_count", 0)) for item in conversations),
        "history_count": len(state.history),
        "memory_count": sum([
            len(state.emotion_memory), len(state.long_memory), len(state.stable_profile),
            len(state.interest_store.items), len(state.memory_events), len(state.pending_memory),
        ]),
        "mood_count": len(load_mood_checkins(user_id)),
        "api_request_count": int(usage["month"]["requests"]),
    }


def _delete_user_exports(directory: Path, user_id: str, *, memory: bool = False) -> int:
    # User IDs may contain underscores, so an "alice_" prefix also matches
    # alice_bob's files. Match the complete generated suffix to establish owner.
    suffix = r"_memory_(?:manual|pre_restore)" if memory else r"_export"
    stamp = r"(?:_[0-9]{8}_[0-9]{6}(?:_[0-9]{6})?)?"
    filename = re.compile(re.escape(user_id) + suffix + stamp + r"\.json")
    if not directory.exists():
        return 0
    removed = 0
    for path in directory.iterdir():
        if path.is_file() and filename.fullmatch(path.name):
            path.unlink()
            removed += 1
    return removed


def _clear_cached_session(user_id: str) -> None:
    # chatbot 持有运行中的多用户会话缓存；延迟导入避免初始化时循环依赖。
    try:
        import chatbot

        store = chatbot.session_store
        with store._registry_lock:
            store._sessions.pop(user_id, None)
            store._active_counts.pop(user_id, None)
            store._locks.pop(user_id, None)
    except Exception:
        # 存储已删除，即使缓存清理失败也不应阻断隐私删除操作。
        pass


def delete_all_user_data(user_id: str) -> dict[str, int | str]:
    """Delete only data owned by one authenticated user; shared RAG data is untouched."""
    user_id = validate_user_id(user_id)
    backend = storage_backend()
    user_directory = user_dir(user_id)
    with user_file_lock(user_id):
        if sqlite_enabled():
            with connection() as conn:
                cursor = conn.execute("DELETE FROM users WHERE user_id = ?", (user_id,))
                database_rows = max(0, int(cursor.rowcount))
        else:
            database_rows = 0
        directory_removed = int(user_directory.exists())
        if user_directory.exists():
            # Windows 不能删除当前仍被 user_file_lock 打开的 .lock 文件。
            for path in user_directory.iterdir():
                if path.name == ".lock":
                    continue
                if path.is_dir():
                    shutil.rmtree(path)
                else:
                    path.unlink()
        exports_removed = _delete_user_exports(EXPORTS_DIR, user_id)
        backups_removed = _delete_user_exports(BACKUPS_DIR, user_id, memory=True)
    if user_directory.exists():
        shutil.rmtree(user_directory)
    _clear_cached_session(user_id)
    return {
        "backend": backend,
        "database_rows": database_rows,
        "user_directory": directory_removed,
        "exports": exports_removed,
        "backups": backups_removed,
    }
