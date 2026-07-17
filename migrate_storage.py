from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path

from config import BASE_DIR, KNOWLEDGE_CHUNKS_PATH, STYLE_CHUNKS_PATH, USERS_DIR
from session_store import validate_user_id


def legacy_user_ids(users_dir: Path = USERS_DIR) -> list[str]:
    if not users_dir.exists():
        return []
    user_ids: list[str] = []
    for path in users_dir.iterdir():
        if not path.is_dir():
            continue
        try:
            user_ids.append(validate_user_id(path.name))
        except ValueError:
            continue
    return sorted(user_ids)


def backup_legacy_users(users_dir: Path = USERS_DIR) -> Path | None:
    if not users_dir.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = BASE_DIR / "exports" / "storage_migrations" / f"users_{stamp}"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(users_dir, target)
    return target


def backup_legacy_rag_metadata() -> Path | None:
    sources = [path for path in (KNOWLEDGE_CHUNKS_PATH, STYLE_CHUNKS_PATH) if path.exists()]
    if not sources:
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = BASE_DIR / "exports" / "storage_migrations" / f"rag_metadata_{stamp}"
    target.mkdir(parents=True, exist_ok=True)
    for source in sources:
        shutil.copy2(source, target / source.name.replace("chunks", source.parent.name + "_chunks"))
    return target


def migrate_users(user_ids: list[str]) -> dict[str, int]:
    # Import lazily to ensure STORAGE_BACKEND is set before the operation begins.
    from auth_store import migrate_legacy_auth
    from mood_store import migrate_legacy_mood_checkins
    from session_store import migrate_legacy_session_state
    from knowledge_store import migrate_legacy_knowledge_metadata
    from style_store import migrate_legacy_style_metadata

    result = {
        "users": 0,
        "auth_records": 0,
        "mood_records": 0,
        "session_items": 0,
        "knowledge_chunks": 0,
        "style_chunks": 0,
    }
    for user_id in user_ids:
        result["users"] += 1
        result["auth_records"] += int(migrate_legacy_auth(user_id))
        result["mood_records"] += migrate_legacy_mood_checkins(user_id)
        result["session_items"] += migrate_legacy_session_state(user_id)
    result["knowledge_chunks"] = migrate_legacy_knowledge_metadata()
    result["style_chunks"] = migrate_legacy_style_metadata()
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Migrate legacy user auth and Mood JSON into SQLite.")
    parser.add_argument("--dry-run", action="store_true", help="Only show the users that would be migrated.")
    parser.add_argument("--user", action="append", dest="users", help="Migrate one user ID; may be repeated.")
    parser.add_argument("--no-backup", action="store_true", help="Skip the automatic users directory backup.")
    args = parser.parse_args()

    try:
        user_ids = [validate_user_id(user_id) for user_id in args.users] if args.users else legacy_user_ids()
    except ValueError as exc:
        parser.error(str(exc))

    if args.dry_run:
        print(f"Found {len(user_ids)} legacy user directories: {', '.join(user_ids) or '(none)'}")
        return 0

    os.environ["STORAGE_BACKEND"] = "sqlite"
    backup = None if args.no_backup else backup_legacy_users()
    rag_backup = None if args.no_backup else backup_legacy_rag_metadata()
    result = migrate_users(user_ids)
    if backup:
        print(f"Legacy backup created: {backup}")
    if rag_backup:
        print(f"RAG metadata backup created: {rag_backup}")
    print(
        "Migration complete: "
        f"{result['users']} users, {result['auth_records']} auth records, "
        f"{result['mood_records']} Mood records, {result['session_items']} session items, "
        f"{result['knowledge_chunks']} knowledge chunks, {result['style_chunks']} style chunks."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
