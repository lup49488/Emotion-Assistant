"""Private raster-image attachments for user-owned Mood Check-ins."""
from __future__ import annotations

import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from session_store import user_dir, user_file_lock, validate_user_id


MAX_IMAGES_PER_CHECKIN = 3
MAX_IMAGE_BYTES = 5 * 1024 * 1024
_IMAGE_ID_PATTERN = re.compile(r"^[a-f0-9]{32}$")
_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_IMAGE_TYPES = {
    "png": ("image/png", ".png"),
    "jpeg": ("image/jpeg", ".jpg"),
    "gif": ("image/gif", ".gif"),
    "webp": ("image/webp", ".webp"),
}


def _validate_date(checkin_date: str) -> str:
    if not _DATE_PATTERN.fullmatch(checkin_date or ""):
        raise ValueError("Mood Check-in date must use YYYY-MM-DD.")
    try:
        datetime.strptime(checkin_date, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError("Mood Check-in date must use YYYY-MM-DD.") from exc
    return checkin_date


def _attachment_dir(user_id: str, checkin_date: str) -> Path:
    return user_dir(validate_user_id(user_id)) / "mood_images" / _validate_date(checkin_date)


def _optional_attachment_dir(user_id: str, checkin_date: str) -> Path | None:
    """Resolve the directory for read paths, where a bad date simply means 'no image'."""
    try:
        return _attachment_dir(user_id, checkin_date)
    except ValueError:
        return None


def _detect_image_type(contents: bytes) -> tuple[str, str]:
    if contents.startswith(b"\x89PNG\r\n\x1a\n"):
        return _IMAGE_TYPES["png"]
    if contents.startswith(b"\xff\xd8\xff"):
        return _IMAGE_TYPES["jpeg"]
    if contents.startswith((b"GIF87a", b"GIF89a")):
        return _IMAGE_TYPES["gif"]
    if len(contents) >= 12 and contents[:4] == b"RIFF" and contents[8:12] == b"WEBP":
        return _IMAGE_TYPES["webp"]
    raise ValueError("Only PNG, JPEG, WebP, and GIF images are supported.")


def _metadata(path: Path) -> dict[str, Any]:
    content_type, _ = next(value for value in _IMAGE_TYPES.values() if value[1] == path.suffix.lower())
    return {"id": path.stem, "filename": path.name, "content_type": content_type, "size_bytes": path.stat().st_size}


def list_mood_images(user_id: str, checkin_date: str) -> list[dict[str, Any]]:
    directory = _optional_attachment_dir(user_id, checkin_date)
    if directory is None or not directory.exists():
        return []
    return [_metadata(path) for path in sorted(directory.iterdir()) if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".gif", ".webp"}]


def save_mood_image(user_id: str, checkin_date: str, contents: bytes) -> dict[str, Any]:
    if not contents:
        raise ValueError("The uploaded image is empty.")
    if len(contents) > MAX_IMAGE_BYTES:
        raise ValueError("Each Mood Check-in image must be 5 MB or smaller.")
    content_type, suffix = _detect_image_type(contents)
    directory = _attachment_dir(user_id, checkin_date)
    # The browser uploads a check-in's images in parallel, so counting and writing
    # have to happen under one lock or the per-check-in cap lets extras through.
    with user_file_lock(user_id):
        directory.mkdir(parents=True, exist_ok=True)
        if len(list_mood_images(user_id, checkin_date)) >= MAX_IMAGES_PER_CHECKIN:
            raise ValueError(f"A Mood Check-in can contain at most {MAX_IMAGES_PER_CHECKIN} images.")
        image_id = uuid.uuid4().hex
        target = directory / f"{image_id}{suffix}"
        temporary = directory / f".{image_id}.upload"
        temporary.write_bytes(contents)
        temporary.replace(target)
    return {"id": image_id, "filename": target.name, "content_type": content_type, "size_bytes": len(contents)}


def get_mood_image_path(user_id: str, checkin_date: str, image_id: str) -> Path | None:
    if not _IMAGE_ID_PATTERN.fullmatch(image_id or ""):
        return None
    directory = _optional_attachment_dir(user_id, checkin_date)
    if directory is None:
        return None
    for _, suffix in _IMAGE_TYPES.values():
        candidate = directory / f"{image_id}{suffix}"
        if candidate.is_file():
            return candidate
    return None


def delete_mood_image(user_id: str, checkin_date: str, image_id: str) -> bool:
    path = get_mood_image_path(user_id, checkin_date, image_id)
    if path is None:
        return False
    path.unlink()
    return True


def delete_mood_checkin_images(user_id: str, checkin_date: str) -> None:
    directory = _optional_attachment_dir(user_id, checkin_date)
    if directory is not None and directory.exists():
        shutil.rmtree(directory)
