from pathlib import Path

import privacy_store
import session_store
from auth_store import verify_access
from sqlite_store import connection


def test_delete_all_user_data_removes_json_files_exports_and_backups(tmp_path, monkeypatch):
    users_dir = tmp_path / "users"
    exports_dir = tmp_path / "exports"
    backups_dir = exports_dir / "memory_backups"
    monkeypatch.setenv("STORAGE_BACKEND", "json")
    monkeypatch.setattr(session_store, "USERS_DIR", users_dir)
    monkeypatch.setattr(privacy_store, "EXPORTS_DIR", exports_dir)
    monkeypatch.setattr(privacy_store, "BACKUPS_DIR", backups_dir)

    assert verify_access("alice", "secret123")[0]
    user_directory = session_store.user_dir("alice")
    (user_directory / "history.json").write_text("[]", encoding="utf-8")
    exports_dir.mkdir(parents=True)
    backups_dir.mkdir(parents=True)
    (exports_dir / "alice_export.json").write_text("{}", encoding="utf-8")
    (backups_dir / "alice_memory_manual.json").write_text("{}", encoding="utf-8")

    result = privacy_store.delete_all_user_data("alice")

    assert result["backend"] == "json"
    assert not user_directory.exists()
    assert not (exports_dir / "alice_export.json").exists()
    assert not (backups_dir / "alice_memory_manual.json").exists()


def test_privacy_summary_only_returns_counts(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_BACKEND", "json")
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")
    assert verify_access("alice", "secret123")[0]

    summary = privacy_store.privacy_summary("alice")

    assert summary["backend"] == "json"
    assert "access_key" not in summary
    assert "api_key" not in summary


def test_delete_all_user_data_cascades_in_sqlite(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DATABASE_PATH", str(tmp_path / "data" / "chatbot.db"))
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")
    assert verify_access("alice", "secret123")[0]

    privacy_store.delete_all_user_data("alice")

    with connection() as conn:
        assert conn.execute("SELECT 1 FROM users WHERE user_id = ?", ("alice",)).fetchone() is None
        assert conn.execute("SELECT 1 FROM auth_credentials WHERE user_id = ?", ("alice",)).fetchone() is None
