import sqlite3
from contextlib import closing

import pytest

from deployment_db import backup, current_version, database_path, restore, schema_version_at


def test_database_path_uses_the_api_bind_mount(tmp_path):
    compose = {"services": {"api": {"environment": {"STORAGE_BACKEND": "sqlite", "SQLITE_DATABASE_PATH": "/app/data/chatbot.db"}, "volumes": [{"source": str(tmp_path / "data"), "target": "/app/data"}]}}}
    assert database_path(compose) == tmp_path / "data" / "chatbot.db"


def test_backup_and_explicit_restore_preserve_data(tmp_path, monkeypatch):
    database = tmp_path / "data" / "chatbot.db"
    database.parent.mkdir()
    with closing(sqlite3.connect(database)) as conn:
        conn.execute("PRAGMA user_version = 8")
        conn.execute("CREATE TABLE items(value TEXT)")
        conn.execute("INSERT INTO items VALUES ('original')")
        conn.commit()
    saved = backup(database, tmp_path / "backups")
    with closing(sqlite3.connect(database)) as conn:
        conn.execute("INSERT INTO items VALUES ('later')")
        conn.commit()
    assert current_version(saved) == 8
    monkeypatch.setattr("deployment_db.subprocess.run", lambda *args, **kwargs: type("Result", (), {"stdout": ""})())
    with pytest.raises(RuntimeError, match="Move the current database"):
        restore(database, saved)
    database.rename(database.with_suffix(".failed.db"))
    restore(database, saved)
    with closing(sqlite3.connect(database)) as conn:
        assert conn.execute("SELECT value FROM items").fetchall() == [("original",)]


def test_schema_version_is_read_without_importing_old_code(tmp_path, monkeypatch):
    monkeypatch.setattr("deployment_db.subprocess.run", lambda *args, **kwargs: type("Result", (), {"stdout": "SCHEMA_VERSION = 6\nraise AssertionError('must not execute')"})())
    assert schema_version_at("previous") == 6
