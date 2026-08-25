from __future__ import annotations

import os
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

from config import BASE_DIR


DEFAULT_DATABASE_PATH = BASE_DIR / "data" / "chatbot.db"
SQLITE_BACKEND = "sqlite"
JSON_BACKEND = "json"


def storage_backend() -> str:
    backend = os.getenv("STORAGE_BACKEND", JSON_BACKEND).strip().lower()
    return SQLITE_BACKEND if backend == SQLITE_BACKEND else JSON_BACKEND


def sqlite_enabled() -> bool:
    return storage_backend() == SQLITE_BACKEND


def database_path() -> Path:
    configured = os.getenv("SQLITE_DATABASE_PATH", "").strip()
    return Path(configured).expanduser() if configured else DEFAULT_DATABASE_PATH


@contextmanager
def connection() -> Iterator[sqlite3.Connection]:
    path = database_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA journal_mode = WAL")
        ensure_schema(conn)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            applied_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS auth_credentials (
            user_id TEXT PRIMARY KEY REFERENCES users(user_id) ON DELETE CASCADE,
            salt TEXT NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            changed_at TEXT
        );

        CREATE TABLE IF NOT EXISTS auth_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            action TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_auth_events_user_created
            ON auth_events(user_id, created_at DESC);

        CREATE TABLE IF NOT EXISTS user_preferences (
            user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            preference_key TEXT NOT NULL,
            value TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (user_id, preference_key)
        );

        CREATE TABLE IF NOT EXISTS mood_checkins (
            user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            checkin_date TEXT NOT NULL,
            mood TEXT NOT NULL,
            intensity INTEGER NOT NULL CHECK (intensity BETWEEN 1 AND 5),
            note TEXT NOT NULL DEFAULT '',
            source TEXT NOT NULL DEFAULT 'checkin',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY (user_id, checkin_date)
        );
        CREATE INDEX IF NOT EXISTS idx_mood_checkins_user_date
            ON mood_checkins(user_id, checkin_date);

        CREATE TABLE IF NOT EXISTS legacy_imports (
            user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            source TEXT NOT NULL,
            imported_at TEXT NOT NULL,
            PRIMARY KEY (user_id, source)
        );

        CREATE TABLE IF NOT EXISTS session_items (
            user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            section TEXT NOT NULL CHECK (section IN (
                'history', 'emotion_memory', 'long_memory', 'stable_profile',
                'interest_memory', 'memory_events', 'pending_memory'
            )),
            position INTEGER NOT NULL CHECK (position >= 0),
            payload_json TEXT NOT NULL,
            PRIMARY KEY (user_id, section, position)
        );
        CREATE INDEX IF NOT EXISTS idx_session_items_user_section_position
            ON session_items(user_id, section, position);

        CREATE TABLE IF NOT EXISTS conversations (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_conversations_user_updated
            ON conversations(user_id, updated_at DESC);

        CREATE TABLE IF NOT EXISTS conversation_messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
            message_id TEXT,
            position INTEGER NOT NULL CHECK (position >= 0),
            role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            reply_to_message_id TEXT,
            UNIQUE(conversation_id, position)
        );
        CREATE INDEX IF NOT EXISTS idx_conversation_messages_conversation_position
            ON conversation_messages(conversation_id, position);

        CREATE TABLE IF NOT EXISTS rag_documents (
            collection TEXT NOT NULL CHECK (collection IN ('knowledge', 'style')),
            name TEXT NOT NULL,
            size_bytes INTEGER NOT NULL DEFAULT 0,
            modified_at TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (collection, name)
        );

        CREATE TABLE IF NOT EXISTS rag_chunks (
            collection TEXT NOT NULL CHECK (collection IN ('knowledge', 'style')),
            chunk_id TEXT NOT NULL,
            source TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            text TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (collection, chunk_id)
        );
        CREATE INDEX IF NOT EXISTS idx_rag_chunks_collection_source_position
            ON rag_chunks(collection, source, chunk_index);

        CREATE TABLE IF NOT EXISTS rag_migrations (
            collection TEXT PRIMARY KEY CHECK (collection IN ('knowledge', 'style')),
            imported_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS rag_citation_traces (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            conversation_id TEXT NOT NULL DEFAULT '',
            citations_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_rag_citation_traces_user_created
            ON rag_citation_traces(user_id, created_at DESC);

        CREATE TABLE IF NOT EXISTS rag_feedback (
            trace_id TEXT PRIMARY KEY REFERENCES rag_citation_traces(id) ON DELETE CASCADE,
            helpful INTEGER NOT NULL CHECK (helpful IN (0, 1)),
            comment TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS api_usage_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            estimated_cost_usd REAL NOT NULL DEFAULT 0,
            duration_ms INTEGER NOT NULL DEFAULT 0,
            success INTEGER NOT NULL CHECK (success IN (0, 1)),
            error_kind TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_api_usage_events_user_created
            ON api_usage_events(user_id, created_at DESC);

        CREATE TABLE IF NOT EXISTS observability_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT NOT NULL,
            method TEXT NOT NULL,
            path TEXT NOT NULL,
            status_code INTEGER NOT NULL,
            duration_ms INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_observability_events_created
            ON observability_events(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_observability_events_path_created
            ON observability_events(path, created_at DESC);

        CREATE TABLE IF NOT EXISTS operations_alert_events (
            fingerprint TEXT PRIMARY KEY,
            severity TEXT NOT NULL CHECK (severity IN ('warning', 'critical')),
            message TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('active', 'resolved')),
            metadata_json TEXT NOT NULL DEFAULT '{}',
            first_seen_at TEXT NOT NULL,
            last_seen_at TEXT NOT NULL,
            resolved_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_operations_alert_events_status_seen
            ON operations_alert_events(status, last_seen_at DESC);

        CREATE TABLE IF NOT EXISTS background_jobs (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            kind TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')),
            progress INTEGER NOT NULL DEFAULT 0 CHECK (progress BETWEEN 0 AND 100),
            message TEXT NOT NULL DEFAULT '',
            payload_json TEXT NOT NULL DEFAULT '{}',
            result_json TEXT,
            error TEXT,
            created_at TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_background_jobs_user_created
            ON background_jobs(user_id, created_at DESC);
        """
    )
    message_columns = {
        str(row["name"])
        for row in conn.execute("PRAGMA table_info(conversation_messages)").fetchall()
    }
    if "message_id" not in message_columns:
        conn.execute("ALTER TABLE conversation_messages ADD COLUMN message_id TEXT")
    if "reply_to_message_id" not in message_columns:
        conn.execute("ALTER TABLE conversation_messages ADD COLUMN reply_to_message_id TEXT")
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_conversation_messages_conversation_message_id "
        "ON conversation_messages(conversation_id, message_id)"
    )
    legacy_rows = conn.execute(
        """
        SELECT id, conversation_id, position, role, content, created_at
        FROM conversation_messages
        WHERE message_id IS NULL OR message_id = ''
        """
    ).fetchall()
    for row in legacy_rows:
        fingerprint = ":".join(str(row[key]) for key in ("conversation_id", "position", "role", "created_at", "content"))
        message_id = uuid.uuid5(uuid.NAMESPACE_URL, f"serenova-message:{fingerprint}").hex
        conn.execute("UPDATE conversation_messages SET message_id = ? WHERE id = ?", (message_id, row["id"]))
    _ensure_pending_memory_section(conn)
    conn.execute(
        "INSERT OR IGNORE INTO schema_migrations(version, applied_at) VALUES (?, ?)",
        (6, _now()),
    )


def _ensure_pending_memory_section(conn: sqlite3.Connection) -> None:
    """Upgrade older SQLite databases whose CHECK constraint omits the queue."""
    row = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'session_items'"
    ).fetchone()
    schema_sql = row["sql"] if isinstance(row, sqlite3.Row) else row[0] if row else ""
    if row is None or "pending_memory" in str(schema_sql or ""):
        return
    # Run the table swap as one transaction. executescript() would COMMIT first and
    # could leave the database with only session_items_legacy if a step failed.
    statements = (
        "ALTER TABLE session_items RENAME TO session_items_legacy",
        """
        CREATE TABLE session_items (
            user_id TEXT NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            section TEXT NOT NULL CHECK (section IN (
                'history', 'emotion_memory', 'long_memory', 'stable_profile',
                'interest_memory', 'memory_events', 'pending_memory'
            )),
            position INTEGER NOT NULL CHECK (position >= 0),
            payload_json TEXT NOT NULL,
            PRIMARY KEY (user_id, section, position)
        )
        """,
        """
        INSERT INTO session_items(user_id, section, position, payload_json)
            SELECT user_id, section, position, payload_json FROM session_items_legacy
        """,
        "DROP TABLE session_items_legacy",
        """
        CREATE INDEX IF NOT EXISTS idx_session_items_user_section_position
            ON session_items(user_id, section, position)
        """,
    )
    conn.execute("BEGIN IMMEDIATE")
    try:
        for statement in statements:
            conn.execute(statement)
    except Exception:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def ensure_user(conn: sqlite3.Connection, user_id: str, *, now: str | None = None) -> None:
    timestamp = now or _now()
    conn.execute(
        """
        INSERT INTO users(user_id, created_at, updated_at) VALUES (?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET updated_at = excluded.updated_at
        """,
        (user_id, timestamp, timestamp),
    )


def legacy_import_completed(conn: sqlite3.Connection, user_id: str, source: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM legacy_imports WHERE user_id = ? AND source = ?",
        (user_id, source),
    ).fetchone()
    return row is not None


def mark_legacy_import_completed(conn: sqlite3.Connection, user_id: str, source: str) -> None:
    ensure_user(conn, user_id)
    conn.execute(
        """
        INSERT OR IGNORE INTO legacy_imports(user_id, source, imported_at) VALUES (?, ?, ?)
        """,
        (user_id, source, _now()),
    )


def get_auth_credential(conn: sqlite3.Connection, user_id: str) -> dict[str, str] | None:
    row = conn.execute(
        """
        SELECT salt, password_hash, created_at, updated_at, changed_at
        FROM auth_credentials WHERE user_id = ?
        """,
        (user_id,),
    ).fetchone()
    return dict(row) if row else None


def save_auth_credential(conn: sqlite3.Connection, user_id: str, record: dict[str, Any]) -> None:
    ensure_user(conn, user_id, now=str(record["updated_at"]))
    conn.execute(
        """
        INSERT INTO auth_credentials(user_id, salt, password_hash, created_at, updated_at, changed_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            salt = excluded.salt,
            password_hash = excluded.password_hash,
            created_at = excluded.created_at,
            updated_at = excluded.updated_at,
            changed_at = excluded.changed_at
        """,
        (
            user_id,
            str(record["salt"]),
            str(record["hash"]),
            str(record["created_at"]),
            str(record["updated_at"]),
            record.get("changed_at"),
        ),
    )


def append_auth_event(conn: sqlite3.Connection, user_id: str, action: str, *, created_at: str | None = None) -> None:
    ensure_user(conn, user_id)
    conn.execute(
        "INSERT INTO auth_events(user_id, action, created_at) VALUES (?, ?, ?)",
        (user_id, action, created_at or _now()),
    )


def list_mood_checkins(conn: sqlite3.Connection, user_id: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT checkin_date AS date, mood, intensity, note, source, created_at, updated_at
        FROM mood_checkins WHERE user_id = ? ORDER BY checkin_date
        """,
        (user_id,),
    ).fetchall()
    return [dict(row) for row in rows]


def upsert_mood_checkin(conn: sqlite3.Connection, user_id: str, record: dict[str, Any]) -> dict[str, Any]:
    ensure_user(conn, user_id, now=str(record["updated_at"]))
    conn.execute(
        """
        INSERT INTO mood_checkins(
            user_id, checkin_date, mood, intensity, note, source, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(user_id, checkin_date) DO UPDATE SET
            mood = excluded.mood,
            intensity = excluded.intensity,
            note = excluded.note,
            source = excluded.source,
            updated_at = excluded.updated_at
        """,
        (
            user_id,
            record["date"],
            record["mood"],
            int(record["intensity"]),
            record["note"],
            record["source"],
            record["created_at"],
            record["updated_at"],
        ),
    )
    row = conn.execute(
        """
        SELECT checkin_date AS date, mood, intensity, note, source, created_at, updated_at
        FROM mood_checkins WHERE user_id = ? AND checkin_date = ?
        """,
        (user_id, record["date"]),
    ).fetchone()
    return dict(row)


def delete_mood_checkin(conn: sqlite3.Connection, user_id: str, checkin_date: str) -> bool:
    result = conn.execute(
        "DELETE FROM mood_checkins WHERE user_id = ? AND checkin_date = ?",
        (user_id, checkin_date),
    )
    return result.rowcount > 0


def list_session_items(conn: sqlite3.Connection, user_id: str, section: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT payload_json FROM session_items
        WHERE user_id = ? AND section = ?
        ORDER BY position
        """,
        (user_id, section),
    ).fetchall()
    items: list[dict[str, Any]] = []
    for row in rows:
        try:
            payload = json.loads(str(row["payload_json"]))
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            items.append(payload)
    return items


def has_session_items(conn: sqlite3.Connection, user_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM session_items WHERE user_id = ? LIMIT 1", (user_id,)
    ).fetchone()
    return row is not None


def replace_session_items(
    conn: sqlite3.Connection, user_id: str, section: str, items: list[dict[str, Any]]
) -> None:
    ensure_user(conn, user_id)
    conn.execute(
        "DELETE FROM session_items WHERE user_id = ? AND section = ?",
        (user_id, section),
    )
    conn.executemany(
        """
        INSERT INTO session_items(user_id, section, position, payload_json)
        VALUES (?, ?, ?, ?)
        """,
        [
            (user_id, section, position, json.dumps(item, ensure_ascii=False, sort_keys=True))
            for position, item in enumerate(items)
            if isinstance(item, dict)
        ],
    )


def rag_migration_completed(conn: sqlite3.Connection, collection: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM rag_migrations WHERE collection = ?", (collection,)
    ).fetchone()
    return row is not None


def mark_rag_migration_completed(conn: sqlite3.Connection, collection: str) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO rag_migrations(collection, imported_at) VALUES (?, ?)",
        (collection, _now()),
    )


def list_rag_chunks(conn: sqlite3.Connection, collection: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT chunk_id AS id, source, text, chunk_index, created_at
        FROM rag_chunks WHERE collection = ?
        ORDER BY rowid
        """,
        (collection,),
    ).fetchall()
    return [dict(row) for row in rows]


def replace_rag_chunks(
    conn: sqlite3.Connection, collection: str, chunks: list[dict[str, Any]]
) -> None:
    conn.execute("DELETE FROM rag_chunks WHERE collection = ?", (collection,))
    conn.executemany(
        """
        INSERT INTO rag_chunks(collection, chunk_id, source, chunk_index, text, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                collection,
                str(chunk.get("id") or f"{chunk.get('source', '')}#{chunk.get('chunk_index', position)}"),
                str(chunk.get("source") or ""),
                int(chunk.get("chunk_index", position)),
                str(chunk.get("text") or ""),
                str(chunk.get("created_at") or ""),
            )
            for position, chunk in enumerate(chunks)
            if isinstance(chunk, dict) and str(chunk.get("text") or "").strip()
        ],
    )


def list_rag_documents(conn: sqlite3.Connection, collection: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT name, size_bytes, modified_at, created_at
        FROM rag_documents WHERE collection = ? ORDER BY name
        """,
        (collection,),
    ).fetchall()
    return [dict(row) for row in rows]


def replace_rag_documents(
    conn: sqlite3.Connection, collection: str, documents: list[dict[str, Any]]
) -> None:
    conn.execute("DELETE FROM rag_documents WHERE collection = ?", (collection,))
    conn.executemany(
        """
        INSERT INTO rag_documents(collection, name, size_bytes, modified_at, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        [
            (
                collection,
                str(document["name"]),
                int(document.get("size_bytes") or 0),
                str(document.get("modified_at") or ""),
                str(document.get("created_at") or ""),
            )
            for document in documents
            if isinstance(document, dict) and str(document.get("name") or "").strip()
        ],
    )
