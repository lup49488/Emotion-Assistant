"""SQLite backup and rollback compatibility checks for the VPS deployment."""
from __future__ import annotations

import argparse
import ast
import json
import os
import sqlite3
import subprocess
import sys
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path


def database_path(compose: dict) -> Path | None:
    api = compose["services"]["api"]
    environment = api["environment"]
    if isinstance(environment, list):
        environment = dict(item.split("=", 1) for item in environment)
    if environment.get("STORAGE_BACKEND", "sqlite") != "sqlite":
        return None
    container_path = Path(environment.get("SQLITE_DATABASE_PATH", "/app/data/chatbot.db"))
    for volume in api["volumes"]:
        target = Path(volume["target"])
        if container_path == target or target in container_path.parents:
            return Path(volume["source"]) / container_path.relative_to(target)
    raise RuntimeError(f"SQLite path {container_path} is not on a bind-mounted volume")


def compose_database_path() -> Path | None:
    result = subprocess.run(["docker", "compose", "config", "--format", "json"], check=True, capture_output=True, text=True)
    return database_path(json.loads(result.stdout))


def schema_version_at(revision: str) -> int:
    source = subprocess.run(["git", "show", f"{revision}:sqlite_store.py"], check=True, capture_output=True, text=True).stdout
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "SCHEMA_VERSION" for target in node.targets):
            return int(ast.literal_eval(node.value))
    raise RuntimeError(f"Cannot determine schema version at {revision}")


def current_version(path: Path) -> int:
    if not path.exists():
        return 0
    with closing(sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)) as conn:
        return int(conn.execute("PRAGMA user_version").fetchone()[0])


def backup(path: Path, directory: Path) -> Path | None:
    if not path.exists():
        return None
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / f"chatbot-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}.db"
    with closing(sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True)) as source:
        with closing(sqlite3.connect(destination)) as target:
            source.backup(target)
            if target.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                raise RuntimeError("SQLite backup integrity check failed")
    os.chmod(destination, 0o600)
    return destination


def restore(path: Path, source: Path) -> None:
    if not source.is_file():
        raise RuntimeError(f"Backup not found: {source}")
    with closing(sqlite3.connect(f"file:{source.as_posix()}?mode=ro", uri=True)) as conn:
        if conn.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("Backup integrity check failed")
    running = subprocess.run(["docker", "compose", "ps", "--status", "running", "--services"], check=True, capture_output=True, text=True)
    if any(service in {"api", "web"} for service in running.stdout.splitlines()):
        raise RuntimeError("Stop the API and web services before restoring the database")
    if path.exists():
        raise RuntimeError("Move the current database and its -wal/-shm files aside before restoring")
    if any(path.with_name(path.name + suffix).exists() for suffix in ("-wal", "-shm")):
        raise RuntimeError("Move the SQLite sidecar files aside before restoring")
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(f"file:{source.as_posix()}?mode=ro", uri=True)) as original:
        with closing(sqlite3.connect(path)) as target:
            original.backup(target)
    os.chmod(path, 0o600)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("backup", "check", "restore"))
    parser.add_argument("argument", nargs="?")
    args = parser.parse_args()
    path = compose_database_path()
    if path is None:
        print("JSON storage: no SQLite database to check")
        return 0
    if args.action == "backup":
        saved = backup(path, Path(".deployment/backups"))
        print(f"SQLite backup: {saved or 'database not yet created'}")
    elif args.action == "check":
        if not args.argument:
            parser.error("check requires the previous commit SHA")
        actual, supported = current_version(path), schema_version_at(args.argument)
        print(f"SQLite schema: current={actual} previous_revision_supports={supported}")
        if actual > supported:
            return 2
    else:
        if not args.argument:
            parser.error("restore requires a backup path")
        restore(path, Path(args.argument))
        print(f"Restored SQLite backup to {path}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, sqlite3.Error, subprocess.CalledProcessError) as exc:
        print(f"Database recovery check failed: {exc}", file=sys.stderr)
        sys.exit(1)
