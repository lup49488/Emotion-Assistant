from pathlib import Path

import pytest

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


@pytest.mark.parametrize("backend", ["json", "sqlite"])
def test_delete_preserves_exports_owned_by_users_with_shared_prefix(tmp_path, monkeypatch, backend):
    monkeypatch.setenv("STORAGE_BACKEND", backend)
    monkeypatch.setenv("SQLITE_DATABASE_PATH", str(tmp_path / "data" / "chatbot.db"))
    monkeypatch.setattr(privacy_store, "_clear_cached_session", lambda user_id: None)
    monkeypatch.setattr(session_store, "USERS_DIR", tmp_path / "users")
    monkeypatch.setattr(privacy_store, "EXPORTS_DIR", tmp_path / "exports")
    monkeypatch.setattr(privacy_store, "BACKUPS_DIR", tmp_path / "exports" / "memory_backups")
    exports_dir = privacy_store.EXPORTS_DIR
    backups_dir = privacy_store.BACKUPS_DIR
    backups_dir.mkdir(parents=True)
    own_files = [
        exports_dir / "alice_export_20260907_120000.json",
        backups_dir / "alice_memory_manual_20260907_120000_123456.json",
        backups_dir / "alice_memory_pre_restore_20260907_120000_123456.json",
    ]
    other_files = [
        exports_dir / "alice_bob_export_20260907_120000.json",
        exports_dir / "alice_export_export_20260907_120000.json",
        backups_dir / "alice_bob_memory_manual_20260907_120000_123456.json",
        backups_dir / "alice_memory_manual_memory_pre_restore_20260907_120000_123456.json",
    ]
    for path in own_files + other_files:
        path.write_text("{}", encoding="utf-8")

    result = privacy_store.delete_all_user_data("alice")

    assert all(path.exists() for path in other_files)
    assert all(not path.exists() for path in own_files)
    assert result["exports"] == 1
    assert result["backups"] == 2
