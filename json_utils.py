from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> list[Any]:
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return data if isinstance(data, list) else []


def save_json(path: Path, data: list[Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(path)


def safe_extract_json_object(text: str) -> dict[str, Any] | None:
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else None
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            data, _ = decoder.raw_decode(text, start)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
    return None


def safe_extract_json_array(text: str) -> list[Any] | None:
    try:
        data = json.loads(text)
        return data if isinstance(data, list) else None
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    start = text.find("[")
    while start != -1:
        try:
            data, _ = decoder.raw_decode(text, start)
            return data if isinstance(data, list) else None
        except json.JSONDecodeError:
            start = text.find("[", start + 1)
    return None
