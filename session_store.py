from __future__ import annotations

import logging
import os
import re
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from config import MAX_CACHED_SESSIONS, USERS_DIR, _USER_ID_PATTERN
from json_utils import load_json, save_json
from memory_store import (
    InterestMemoryStore,
    VectorIndexManager,
    clean_long_term,
    compact_long_memory,
)
from sqlite_store import (
    connection as sqlite_connection,
    has_session_items,
    legacy_import_completed,
    list_session_items,
    mark_legacy_import_completed,
    replace_session_items,
    sqlite_enabled,
)


logger = logging.getLogger(__name__)
_IS_WINDOWS = sys.platform.startswith("win")
_LOCK_RETRY_INTERVAL = 0.05
_LOCK_TIMEOUT = 30.0
SESSION_SECTIONS = (
    "history",
    "emotion_memory",
    "long_memory",
    "stable_profile",
    "interest_memory",
    "memory_events",
)


def validate_user_id(user_id: str) -> str:
    if not isinstance(user_id, str) or not _USER_ID_PATTERN.match(user_id):
        raise ValueError(
            f"非法 user_id: {user_id!r}。"
            "只允许字母、数字、下划线、短横线，长度 1-128。"
        )
    return user_id


def user_dir(user_id: str) -> Path:
    validate_user_id(user_id)
    path = USERS_DIR / user_id
    path.mkdir(parents=True, exist_ok=True)
    return path


def user_paths(user_id: str) -> dict[str, Path]:
    d = user_dir(user_id)
    return {
        "history": d / "history.json",
        "emotion_memory": d / "emotion_memory.json",
        "interest_memory": d / "interest_memory.json",
        "long_memory": d / "long_memory.json",
        "stable_profile": d / "stable_profile.json",
        "memory_events": d / "memory_events.json",
        "mood_checkins": d / "mood_checkins.json",
        "vector_index": d / "memory.index",
        "lock": d / ".lock",
    }


def _acquire_lock_unix(f) -> None:
    import fcntl
    fcntl.flock(f, fcntl.LOCK_EX)


def _release_lock_unix(f) -> None:
    import fcntl
    fcntl.flock(f, fcntl.LOCK_UN)


def _acquire_lock_windows(f) -> None:
    import msvcrt
    import time
    f.seek(0, 2)
    if f.tell() == 0:
        f.write(b"0")
        f.flush()
    f.seek(0)
    deadline = time.monotonic() + _LOCK_TIMEOUT
    while True:
        try:
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            return
        except OSError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"等待用户文件锁超时（>{_LOCK_TIMEOUT}秒）。")
            time.sleep(_LOCK_RETRY_INTERVAL)


def _release_lock_windows(f) -> None:
    import msvcrt
    f.seek(0)
    msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


@contextmanager
def user_file_lock(user_id: str):
    lock_path = user_paths(user_id)["lock"]
    lock_path.touch(exist_ok=True)
    mode = "r+b" if _IS_WINDOWS else "w"
    with open(lock_path, mode) as f:
        acquire = _acquire_lock_windows if _IS_WINDOWS else _acquire_lock_unix
        release = _release_lock_windows if _IS_WINDOWS else _release_lock_unix
        acquire(f)
        try:
            yield
        finally:
            release(f)


@dataclass
class SessionState:
    user_id: str = "default"
    history: list[dict[str, str]] = field(default_factory=list)
    emotion_memory: list[dict[str, Any]] = field(default_factory=list)
    long_memory: list[dict[str, Any]] = field(default_factory=list)
    stable_profile: list[dict[str, Any]] = field(default_factory=list)
    memory_events: list[dict[str, Any]] = field(default_factory=list)
    interest_store: InterestMemoryStore = field(default_factory=InterestMemoryStore)
    vector_index: VectorIndexManager | None = None
    preferences: dict[str, str | None] = field(
        default_factory=lambda: {"language": None, "tone": None}
    )
    last_accessed: datetime = field(default_factory=datetime.now)

    def __post_init__(self) -> None:
        if self.vector_index is None:
            self.vector_index = VectorIndexManager(user_paths(self.user_id)["vector_index"])

    def touch(self) -> None:
        self.last_accessed = datetime.now()


def _legacy_sections(user_id: str) -> dict[str, list[dict[str, Any]]]:
    paths = user_paths(user_id)
    return {section: load_json(paths[section]) for section in SESSION_SECTIONS}


def _hydrate_state(state: SessionState, sections: dict[str, list[dict[str, Any]]]) -> None:
    state.history = list(sections["history"])
    state.emotion_memory = list(sections["emotion_memory"])
    raw_long_memory = list(sections["long_memory"])
    state.stable_profile = [
        item for item in sections["stable_profile"]
        if isinstance(item, dict) and str(item.get("text", "")).strip()
    ]
    state.memory_events = [item for item in sections["memory_events"] if isinstance(item, dict)][-100:]

    # Move profiles created before the dedicated stable-profile store existed.
    legacy_profiles = [
        item for item in raw_long_memory
        if isinstance(item, dict) and item.get("kind") == "profile" and str(item.get("text", "")).strip()
    ]
    known_texts = {str(item["text"]).strip() for item in state.stable_profile}
    for item in legacy_profiles:
        text = str(item["text"]).strip()
        if text not in known_texts:
            state.stable_profile.append(dict(item))
            known_texts.add(text)
    raw_long_memory = [
        item for item in raw_long_memory
        if not (isinstance(item, dict) and item.get("kind") == "profile")
    ]
    state.long_memory = compact_long_memory(clean_long_term(raw_long_memory))
    state.interest_store.load(list(sections["interest_memory"]))


def _state_sections(state: SessionState) -> dict[str, list[dict[str, Any]]]:
    return {
        "history": list(state.history),
        "emotion_memory": list(state.emotion_memory),
        "long_memory": list(state.long_memory),
        "stable_profile": list(state.stable_profile),
        "interest_memory": list(state.interest_store.items),
        "memory_events": list(state.memory_events[-100:]),
    }


def _migrate_legacy_session_unlocked(conn: Any, user_id: str) -> int:
    if legacy_import_completed(conn, user_id, "session_json"):
        return 0
    if has_session_items(conn, user_id):
        mark_legacy_import_completed(conn, user_id, "session_json")
        return 0
    sections = _legacy_sections(user_id)
    for section, items in sections.items():
        replace_session_items(conn, user_id, section, items)
    mark_legacy_import_completed(conn, user_id, "session_json")
    return sum(len(items) for items in sections.values())


def migrate_legacy_session_state(user_id: str) -> int:
    """Import one user's JSON session and memory state exactly once into SQLite."""
    user_id = validate_user_id(user_id)
    if not sqlite_enabled():
        return 0
    with user_file_lock(user_id):
        with sqlite_connection() as conn:
            return _migrate_legacy_session_unlocked(conn, user_id)


def _load_json_state(user_id: str) -> SessionState:
    with user_file_lock(user_id):
        state = SessionState(user_id=user_id)
        _hydrate_state(state, _legacy_sections(user_id))
        paths = user_paths(user_id)
        save_json(paths["long_memory"], state.long_memory)
        save_json(paths["stable_profile"], state.stable_profile)
        save_json(paths["memory_events"], state.memory_events)
    return state


def _load_sqlite_state(user_id: str) -> SessionState:
    with user_file_lock(user_id):
        with sqlite_connection() as conn:
            _migrate_legacy_session_unlocked(conn, user_id)
            state = SessionState(user_id=user_id)
            sections = {
                section: list_session_items(conn, user_id, section)
                for section in SESSION_SECTIONS
            }
            _hydrate_state(state, sections)
            for section, items in _state_sections(state).items():
                replace_session_items(conn, user_id, section, items)
    return state


def load_state(user_id: str) -> SessionState:
    user_id = validate_user_id(user_id)
    return _load_sqlite_state(user_id) if sqlite_enabled() else _load_json_state(user_id)


def persist_state(state: SessionState) -> None:
    user_id = validate_user_id(state.user_id)
    if sqlite_enabled():
        with user_file_lock(user_id):
            with sqlite_connection() as conn:
                _migrate_legacy_session_unlocked(conn, user_id)
                for section, items in _state_sections(state).items():
                    replace_session_items(conn, user_id, section, items)
        return

    paths = user_paths(user_id)
    with user_file_lock(user_id):
        for section, items in _state_sections(state).items():
            save_json(paths[section], items)


class SessionStore:
    def __init__(self, max_sessions: int = MAX_CACHED_SESSIONS) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._active_counts: dict[str, int] = {}
        self._registry_lock = threading.Lock()
        self._max_sessions = max_sessions

    def _get_user_lock(self, user_id: str) -> threading.Lock:
        with self._registry_lock:
            if user_id not in self._locks:
                self._locks[user_id] = threading.Lock()
            return self._locks[user_id]

    def _evict_if_needed(self) -> None:
        if len(self._sessions) <= self._max_sessions:
            return
        inactive_users = [
            uid for uid in self._sessions
            if self._active_counts.get(uid, 0) == 0
        ]
        if not inactive_users:
            return
        oldest_user = min(inactive_users, key=lambda uid: self._sessions[uid].last_accessed)
        del self._sessions[oldest_user]
        self._active_counts.pop(oldest_user, None)

    @contextmanager
    def session(self, user_id: str):
        validate_user_id(user_id)
        user_lock = self._get_user_lock(user_id)
        with user_lock:
            with self._registry_lock:
                state = self._sessions.get(user_id)
            if state is None:
                state = load_state(user_id)
                with self._registry_lock:
                    self._sessions[user_id] = state
                    self._evict_if_needed()
            state.touch()
            with self._registry_lock:
                self._active_counts[user_id] = self._active_counts.get(user_id, 0) + 1
            try:
                yield state
            finally:
                try:
                    persist_state(state)
                finally:
                    with self._registry_lock:
                        current = self._active_counts.get(user_id, 0)
                        if current <= 1:
                            self._active_counts.pop(user_id, None)
                            self._evict_if_needed()
                        else:
                            self._active_counts[user_id] = current - 1
