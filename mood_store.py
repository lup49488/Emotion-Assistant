from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

from json_utils import load_json, save_json
from session_store import user_file_lock, user_paths, validate_user_id


MOOD_CHOICES = [
    "开心",
    "平静",
    "一般",
    "疲惫",
    "焦虑",
    "难过",
    "生气",
    "压力大",
    "有希望",
    "其他",
]


def _normalize_date(value: str | None) -> str:
    text = (value or date.today().isoformat()).strip()
    try:
        parsed = datetime.strptime(text, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError("日期需要使用 YYYY-MM-DD 格式。") from exc
    return parsed.isoformat()


def _parse_date(value: str | None) -> date:
    return datetime.strptime(_normalize_date(value), "%Y-%m-%d").date()


def _normalize_intensity(value: int | float | str) -> int:
    try:
        intensity = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("心情强度需要是 1 到 5 的整数。") from exc
    if intensity < 1 or intensity > 5:
        raise ValueError("心情强度需要在 1 到 5 之间。")
    return intensity


def _normalize_mood(value: str | None) -> str:
    mood = (value or "").strip()
    if not mood:
        raise ValueError("请先选择或填写今天的心情。")
    return mood


def _clean_note(value: str | None) -> str:
    return (value or "").strip()


def _load_records_unlocked(user_id: str) -> list[dict[str, Any]]:
    records = load_json(user_paths(user_id)["mood_checkins"])
    clean_records: list[dict[str, Any]] = []
    for item in records:
        if not isinstance(item, dict):
            continue
        try:
            item_date = _normalize_date(str(item.get("date", "")))
            mood = _normalize_mood(str(item.get("mood", "")))
            intensity = _normalize_intensity(item.get("intensity", 3))
        except ValueError:
            continue
        clean_records.append({
            "date": item_date,
            "mood": mood,
            "intensity": intensity,
            "note": _clean_note(item.get("note")),
            "source": str(item.get("source") or "checkin"),
            "created_at": str(item.get("created_at") or ""),
            "updated_at": str(item.get("updated_at") or ""),
        })
    return sorted(clean_records, key=lambda item: item["date"])


def load_mood_checkins(user_id: str) -> list[dict[str, Any]]:
    user_id = validate_user_id(user_id or "local")
    with user_file_lock(user_id):
        return _load_records_unlocked(user_id)


def add_mood_checkin(
    user_id: str,
    mood: str,
    intensity: int | float | str,
    note: str | None = "",
    checkin_date: str | None = None,
) -> dict[str, Any]:
    user_id = validate_user_id(user_id or "local")
    normalized_date = _normalize_date(checkin_date)
    normalized_mood = _normalize_mood(mood)
    normalized_intensity = _normalize_intensity(intensity)
    now = datetime.now().isoformat(timespec="seconds")

    with user_file_lock(user_id):
        records = _load_records_unlocked(user_id)
        existing = next((item for item in records if item["date"] == normalized_date), None)
        if existing is None:
            record = {
                "date": normalized_date,
                "mood": normalized_mood,
                "intensity": normalized_intensity,
                "note": _clean_note(note),
                "source": "checkin",
                "created_at": now,
                "updated_at": now,
            }
            records.append(record)
        else:
            existing.update({
                "mood": normalized_mood,
                "intensity": normalized_intensity,
                "note": _clean_note(note),
                "updated_at": now,
            })
            record = existing
        records.sort(key=lambda item: item["date"])
        save_json(user_paths(user_id)["mood_checkins"], records)
        return dict(record)


def delete_mood_checkin(user_id: str, checkin_date: str) -> bool:
    user_id = validate_user_id(user_id or "local")
    normalized_date = _normalize_date(checkin_date)
    with user_file_lock(user_id):
        records = _load_records_unlocked(user_id)
        kept = [item for item in records if item["date"] != normalized_date]
        if len(kept) == len(records):
            return False
        save_json(user_paths(user_id)["mood_checkins"], kept)
        return True


def format_mood_checkins(user_id: str, limit: int = 14) -> str:
    user_id = validate_user_id(user_id or "local")
    records = load_mood_checkins(user_id)
    if not records:
        return f"用户：{user_id}\n\n还没有 Mood Check-in 记录。"

    recent = records[-max(1, int(limit)) :]
    lines = [
        f"用户：{user_id}",
        f"共 {len(records)} 条记录，下面显示最近 {len(recent)} 条。",
        "",
    ]
    for item in reversed(recent):
        note = item.get("note") or "无备注"
        lines.append(
            f"{item['date']} | {item['mood']} | 强度 {item['intensity']}/5 | {note}"
        )
    return "\n".join(lines)


def get_weekly_mood_points(
    user_id: str,
    end_date: str | None = None,
    days: int = 7,
) -> list[dict[str, Any]]:
    user_id = validate_user_id(user_id or "local")
    day_count = max(1, int(days))
    end = _parse_date(end_date)
    start = end - timedelta(days=day_count - 1)
    records_by_date = {item["date"]: item for item in load_mood_checkins(user_id)}

    points: list[dict[str, Any]] = []
    for offset in range(day_count):
        current = start + timedelta(days=offset)
        date_text = current.isoformat()
        record = records_by_date.get(date_text)
        points.append({
            "date": date_text,
            "label": current.strftime("%m-%d"),
            "intensity": record.get("intensity") if record else None,
            "mood": record.get("mood", "") if record else "",
            "note": record.get("note", "") if record else "",
        })
    return points


def format_weekly_mood_summary(
    user_id: str,
    end_date: str | None = None,
    days: int = 7,
) -> str:
    points = get_weekly_mood_points(user_id, end_date=end_date, days=days)
    recorded = [point for point in points if point["intensity"] is not None]
    if not recorded:
        start = points[0]["date"]
        end = points[-1]["date"]
        return f"{start} 到 {end} 暂无 Mood Check-in 数据。"

    values = [int(point["intensity"]) for point in recorded]
    average = sum(values) / len(values)
    max_point = max(recorded, key=lambda point: int(point["intensity"]))
    min_point = min(recorded, key=lambda point: int(point["intensity"]))
    trend = "记录不足，暂不判断趋势"
    if len(recorded) >= 2:
        first = int(recorded[0]["intensity"])
        last = int(recorded[-1]["intensity"])
        if last > first:
            trend = "本周后段强度比前段更高"
        elif last < first:
            trend = "本周后段强度比前段更低"
        else:
            trend = "本周前后强度基本持平"

    return "\n".join([
        f"记录天数：{len(recorded)}/{len(points)}",
        f"平均强度：{average:.1f}/5",
        f"最高：{max_point['date']} {max_point['mood']} {max_point['intensity']}/5",
        f"最低：{min_point['date']} {min_point['mood']} {min_point['intensity']}/5",
        f"趋势：{trend}",
    ])
