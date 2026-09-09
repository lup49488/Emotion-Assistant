from __future__ import annotations

import file_security


def test_restrict_file_tolerates_a_sqlite_sidecar_disappearing(monkeypatch, tmp_path):
    """A concurrent SQLite checkpoint may remove a sidecar before chmod runs."""
    monkeypatch.setattr(file_security.os, "name", "posix")

    def missing_file(*_args, **_kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(file_security.os, "chmod", missing_file)

    file_security.restrict_file(tmp_path / "chatbot.db-shm")
