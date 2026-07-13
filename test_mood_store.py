from __future__ import annotations

import pytest

import session_store
from mood_store import (
    add_mood_checkin,
    delete_mood_checkin,
    format_mood_checkins,
    format_weekly_mood_summary,
    get_weekly_mood_points,
    load_mood_checkins,
)


@pytest.fixture
def isolated_users_dir(tmp_path, monkeypatch):
    users_dir = tmp_path / "users"
    users_dir.mkdir()
    monkeypatch.setattr(session_store, "USERS_DIR", users_dir)
    return users_dir


def test_add_mood_checkin_creates_record(isolated_users_dir):
    record = add_mood_checkin(
        user_id="alice",
        mood="开心",
        intensity=4,
        note="完成了一个小目标",
        checkin_date="2026-07-12",
    )

    assert record["date"] == "2026-07-12"
    assert record["mood"] == "开心"
    assert record["intensity"] == 4
    assert record["note"] == "完成了一个小目标"
    assert load_mood_checkins("alice") == [record]


def test_same_day_checkin_updates_existing_record(isolated_users_dir):
    first = add_mood_checkin("alice", "一般", 3, "早上普通", "2026-07-12")
    updated = add_mood_checkin("alice", "平静", 2, "晚上好多了", "2026-07-12")

    records = load_mood_checkins("alice")
    assert len(records) == 1
    assert records[0]["created_at"] == first["created_at"]
    assert records[0]["updated_at"] == updated["updated_at"]
    assert records[0]["mood"] == "平静"
    assert records[0]["note"] == "晚上好多了"


def test_delete_mood_checkin_removes_record(isolated_users_dir):
    add_mood_checkin("alice", "焦虑", 5, "", "2026-07-12")

    assert delete_mood_checkin("alice", "2026-07-12") is True
    assert delete_mood_checkin("alice", "2026-07-12") is False
    assert load_mood_checkins("alice") == []


def test_format_mood_checkins_shows_recent_records(isolated_users_dir):
    add_mood_checkin("alice", "开心", 4, "A", "2026-07-10")
    add_mood_checkin("alice", "平静", 3, "B", "2026-07-11")

    text = format_mood_checkins("alice")

    assert "alice" in text
    assert "2026-07-11 | 平静 | 强度 3/5 | B" in text
    assert "2026-07-10 | 开心 | 强度 4/5 | A" in text


def test_invalid_mood_input_is_rejected(isolated_users_dir):
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        add_mood_checkin("alice", "开心", 3, "", "2026/07/12")

    with pytest.raises(ValueError, match="1 到 5"):
        add_mood_checkin("alice", "开心", 6, "", "2026-07-12")

    with pytest.raises(ValueError, match="心情"):
        add_mood_checkin("alice", "", 3, "", "2026-07-12")


def test_get_weekly_mood_points_keeps_missing_days(isolated_users_dir):
    add_mood_checkin("alice", "开心", 4, "A", "2026-07-10")
    add_mood_checkin("alice", "平静", 2, "B", "2026-07-12")

    points = get_weekly_mood_points("alice", end_date="2026-07-12")

    assert [point["date"] for point in points] == [
        "2026-07-06",
        "2026-07-07",
        "2026-07-08",
        "2026-07-09",
        "2026-07-10",
        "2026-07-11",
        "2026-07-12",
    ]
    assert points[4]["intensity"] == 4
    assert points[4]["mood"] == "开心"
    assert points[5]["intensity"] is None
    assert points[6]["intensity"] == 2


def test_format_weekly_mood_summary_describes_recorded_days(isolated_users_dir):
    add_mood_checkin("alice", "开心", 4, "A", "2026-07-10")
    add_mood_checkin("alice", "平静", 2, "B", "2026-07-12")

    text = format_weekly_mood_summary("alice", end_date="2026-07-12")

    assert "记录天数：2/7" in text
    assert "平均强度：3.0/5" in text
    assert "最高：2026-07-10 开心 4/5" in text
    assert "最低：2026-07-12 平静 2/5" in text
