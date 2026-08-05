from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

from json_utils import load_json, save_json
from session_store import user_file_lock, user_paths, validate_user_id
from sqlite_store import (
    connection as sqlite_connection,
    delete_mood_checkin as delete_sqlite_mood_checkin,
    legacy_import_completed,
    list_mood_checkins as list_sqlite_mood_checkins,
    mark_legacy_import_completed,
    sqlite_enabled,
    upsert_mood_checkin,
)


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
    """Keep the note as typed, including leading indentation.

    Only trailing whitespace is dropped: it is invisible either way, and keeping
    it would grow the note every time the record is edited and saved again. A
    whitespace-only note collapses to empty, which still counts as "no note".
    """
    return str(value or "").rstrip()


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


def _migrate_legacy_moods_unlocked(conn: Any, user_id: str) -> int:
    if legacy_import_completed(conn, user_id, "mood_json"):
        return 0
    if list_sqlite_mood_checkins(conn, user_id):
        mark_legacy_import_completed(conn, user_id, "mood_json")
        return 0
    records = _load_records_unlocked(user_id)
    for record in records:
        upsert_mood_checkin(conn, user_id, record)
    mark_legacy_import_completed(conn, user_id, "mood_json")
    return len(records)


def migrate_legacy_mood_checkins(user_id: str) -> int:
    """Import one user's legacy Mood JSON exactly once when SQLite is enabled."""
    user_id = validate_user_id(user_id or "local")
    if not sqlite_enabled():
        return 0
    with user_file_lock(user_id):
        with sqlite_connection() as conn:
            return _migrate_legacy_moods_unlocked(conn, user_id)


def _load_sqlite_records(user_id: str) -> list[dict[str, Any]]:
    with user_file_lock(user_id):
        with sqlite_connection() as conn:
            _migrate_legacy_moods_unlocked(conn, user_id)
            return list_sqlite_mood_checkins(conn, user_id)


def load_mood_checkins(user_id: str) -> list[dict[str, Any]]:
    user_id = validate_user_id(user_id or "local")
    if sqlite_enabled():
        return _load_sqlite_records(user_id)
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

    if sqlite_enabled():
        with user_file_lock(user_id):
            with sqlite_connection() as conn:
                _migrate_legacy_moods_unlocked(conn, user_id)
                existing = next(
                    (item for item in list_sqlite_mood_checkins(conn, user_id) if item["date"] == normalized_date),
                    None,
                )
                record = {
                    "date": normalized_date,
                    "mood": normalized_mood,
                    "intensity": normalized_intensity,
                    "note": _clean_note(note),
                    "source": str((existing or {}).get("source") or "checkin"),
                    "created_at": str(existing.get("created_at") if existing else now),
                    "updated_at": now,
                }
                return upsert_mood_checkin(conn, user_id, record)

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
    if sqlite_enabled():
        with user_file_lock(user_id):
            with sqlite_connection() as conn:
                _migrate_legacy_moods_unlocked(conn, user_id)
                return delete_sqlite_mood_checkin(conn, user_id, normalized_date)
    with user_file_lock(user_id):
        records = _load_records_unlocked(user_id)
        kept = [item for item in records if item["date"] != normalized_date]
        if len(kept) == len(records):
            return False
        save_json(user_paths(user_id)["mood_checkins"], kept)
        return True


def restore_mood_checkins(user_id: str, records: list[dict[str, Any]], *, mode: str) -> int:
    """Restore validated mood records; merge keeps an existing date unchanged."""
    user_id = validate_user_id(user_id or "local")
    if mode not in {"merge", "replace"}:
        raise ValueError("Import mode must be merge or replace.")
    normalized = [
        {
            "date": _normalize_date(str(item.get("date", ""))),
            "mood": _normalize_mood(str(item.get("mood", ""))),
            "intensity": _normalize_intensity(item.get("intensity")),
            "note": _clean_note(str(item.get("note", ""))),
            "source": str(item.get("source") or "checkin"),
            "created_at": str(item.get("created_at") or datetime.now().isoformat(timespec="seconds")),
            "updated_at": str(item.get("updated_at") or datetime.now().isoformat(timespec="seconds")),
        }
        for item in records
    ]
    if sqlite_enabled():
        with user_file_lock(user_id):
            with sqlite_connection() as conn:
                _migrate_legacy_moods_unlocked(conn, user_id)
                if mode == "replace":
                    conn.execute("DELETE FROM mood_checkins WHERE user_id = ?", (user_id,))
                existing_dates = {item["date"] for item in list_sqlite_mood_checkins(conn, user_id)}
                additions = [item for item in normalized if item["date"] not in existing_dates]
                for item in additions:
                    upsert_mood_checkin(conn, user_id, item)
        return len(additions)

    with user_file_lock(user_id):
        existing = [] if mode == "replace" else _load_records_unlocked(user_id)
        dates = {item["date"] for item in existing}
        additions = [item for item in normalized if item["date"] not in dates]
        save_json(user_paths(user_id)["mood_checkins"], sorted([*existing, *additions], key=lambda item: item["date"]))
    return len(additions)


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


def format_mood_fluctuation_analysis(points: list[dict[str, Any]]) -> str:
    """Summarize changes in self-reported intensity without inferring a diagnosis."""
    recorded = [point for point in points if point.get("intensity") is not None]
    if not recorded:
        return "暂无足够的 Mood Check-in 数据，记录几天后可查看情绪波动分析。"

    values = [int(point["intensity"]) for point in recorded]
    intensity_range = max(values) - min(values)
    lines = [
        "情绪波动分析（基于自评强度，不构成医学判断）",
        f"有效记录：{len(recorded)}/{len(points)} 天",
        f"强度范围：{min(values)} 到 {max(values)}/5，跨度 {intensity_range}",
    ]

    adjacent_changes: list[tuple[dict[str, Any], dict[str, Any], int]] = []
    for previous, current in zip(recorded, recorded[1:]):
        previous_date = datetime.strptime(previous["date"], "%Y-%m-%d").date()
        current_date = datetime.strptime(current["date"], "%Y-%m-%d").date()
        if (current_date - previous_date).days == 1:
            adjacent_changes.append((previous, current, int(current["intensity"]) - int(previous["intensity"])))

    if adjacent_changes:
        absolute_changes = [abs(change) for _, _, change in adjacent_changes]
        largest_previous, largest_current, largest_change = max(
            adjacent_changes, key=lambda item: abs(item[2])
        )
        direction = "上升" if largest_change > 0 else "下降" if largest_change < 0 else "持平"
        average_change = sum(absolute_changes) / len(absolute_changes)
        if intensity_range >= 3 or average_change >= 2:
            fluctuation = "较明显"
        elif intensity_range >= 2 or average_change >= 1:
            fluctuation = "中等"
        else:
            fluctuation = "较平稳"
        lines.extend([
            f"连续记录平均变化：{average_change:.1f}",
            f"最大单日变化：{largest_previous['date']} 到 {largest_current['date']} {direction} {abs(largest_change)} 点",
            f"波动程度：{fluctuation}",
        ])
    else:
        lines.append("连续记录不足，暂不计算单日波动。")

    if len(recorded) >= 2:
        recent_change = int(recorded[-1]["intensity"]) - int(recorded[0]["intensity"])
        if recent_change > 0:
            recent_text = f"近期强度较首条记录上升 {recent_change} 点"
        elif recent_change < 0:
            recent_text = f"近期强度较首条记录下降 {abs(recent_change)} 点"
        else:
            recent_text = "近期强度与首条记录基本持平"
        lines.append(f"近期变化：{recent_text}")

    latest = recorded[-1]
    negative_moods = {"疲惫", "焦虑", "难过", "生气", "压力大"}
    if latest["mood"] in negative_moods and int(latest["intensity"]) >= 4:
        lines.append("关怀提示：最近记录为高强度不适感受；若持续或影响生活，可以考虑和信任的人或专业人士聊聊。")
    return "\n".join(lines)


def format_weekly_mood_analysis(
    user_id: str,
    end_date: str | None = None,
    days: int = 7,
) -> str:
    return format_mood_fluctuation_analysis(
        get_weekly_mood_points(user_id, end_date=end_date, days=days)
    )
