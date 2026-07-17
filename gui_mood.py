from __future__ import annotations

import logging
from html import escape
from typing import Any

from gui_auth import AUTH_REQUIRED_MESSAGE, authorize_or_message
from gui_i18n import localize_status_text
from mood_store import (
    add_mood_checkin,
    delete_mood_checkin,
    format_mood_checkins,
    format_weekly_mood_analysis,
    format_weekly_mood_summary,
    get_weekly_mood_points,
)

logger = logging.getLogger(__name__)


def refresh_mood_panel(user_id: str, access_key: str, locale: str = "zh-CN") -> str:
    user_id, auth_error = authorize_or_message(user_id, access_key, locale)
    if auth_error:
        return auth_error
    try:
        return localize_status_text(format_mood_checkins(user_id), locale)
    except Exception as exc:
        return localize_status_text(f"读取 Mood Check-in 失败：{exc}", locale)


def submit_mood_checkin(
    user_id: str,
    access_key: str,
    checkin_date: str,
    mood: str,
    intensity: int | float,
    note: str,
    locale: str = "zh-CN",
) -> str:
    user_id, auth_error = authorize_or_message(user_id, access_key, locale)
    if auth_error:
        return auth_error
    try:
        record = add_mood_checkin(
            user_id=user_id,
            mood=mood,
            intensity=intensity,
            note=note,
            checkin_date=checkin_date,
        )
        logger.info("Mood check-in saved for user=%s date=%s", user_id, record["date"])
        return localize_status_text(
            f"已保存 {record['date']} 的 Mood Check-in。\n\n{format_mood_checkins(user_id)}",
            locale,
        )
    except Exception as exc:
        logger.warning("Failed to save mood check-in for user=%s: %s", user_id, exc)
        return localize_status_text(f"保存 Mood Check-in 失败：{exc}", locale)


def delete_mood_checkin_from_gui(
    user_id: str, access_key: str, checkin_date: str, locale: str = "zh-CN"
) -> str:
    user_id, auth_error = authorize_or_message(user_id, access_key, locale)
    if auth_error:
        return auth_error
    try:
        deleted = delete_mood_checkin(user_id, checkin_date)
        prefix = f"已删除 {checkin_date} 的 Mood Check-in。" if deleted else f"没有找到 {checkin_date} 的记录。"
        if deleted:
            logger.info("Mood check-in deleted for user=%s date=%s", user_id, checkin_date)
        return localize_status_text(f"{prefix}\n\n{format_mood_checkins(user_id)}", locale)
    except Exception as exc:
        logger.warning("Failed to delete mood check-in for user=%s: %s", user_id, exc)
        return localize_status_text(f"删除 Mood Check-in 失败：{exc}", locale)


def render_weekly_mood_chart(
    points: list[dict[str, Any]], theme_mode: str = "light"
) -> str:
    dark = theme_mode == "dark"
    colors = {
        "surface": "#111827" if dark else "#ffffff",
        "surface_alt": "#1f2937" if dark else "#f9fafb",
        "border": "#475569" if dark else "#e5e7eb",
        "grid": "#374151" if dark else "#e5e7eb",
        "axis": "#94a3b8" if dark else "#9ca3af",
        "text": "#f8fafc" if dark else "#111827",
        "muted": "#cbd5e1" if dark else "#6b7280",
        "date": "#e2e8f0" if dark else "#374151",
        "label_border": "#64748b" if dark else "#cbd5e1",
        "missing": "#64748b" if dark else "#d1d5db",
        "accent": "#60a5fa" if dark else "#2563eb",
    }
    width = 720
    height = 300
    left = 52
    right = 24
    top = 28
    bottom = 56
    chart_width = width - left - right
    chart_height = height - top - bottom
    step = chart_width / max(1, len(points) - 1)

    def x_at(index: int) -> float:
        return left + index * step

    def y_at(value: int | float) -> float:
        return top + (5 - float(value)) / 4 * chart_height

    y_lines = []
    for value in range(1, 6):
        y = y_at(value)
        y_lines.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}" '
            f'stroke="{colors["grid"]}" stroke-width="1" />'
            f'<text x="18" y="{y + 4:.1f}" font-size="12" fill="{colors["muted"]}">{value}</text>'
        )

    recorded_points = [
        (index, point)
        for index, point in enumerate(points)
        if point.get("intensity") is not None
    ]
    path = ""
    if recorded_points:
        commands = []
        for index, point in recorded_points:
            prefix = "M" if not commands else "L"
            commands.append(f"{prefix}{x_at(index):.1f},{y_at(int(point['intensity'])):.1f}")
        path = (
            f'<path d="{" ".join(commands)}" fill="none" '
            f'stroke="{colors["accent"]}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />'
        )

    marks = []
    for index, point in enumerate(points):
        x = x_at(index)
        label = escape(str(point["label"]))
        date_text = escape(str(point["date"]))
        intensity = point.get("intensity")
        if intensity is None:
            marks.append(
                f'<circle cx="{x:.1f}" cy="{y_at(1):.1f}" r="4" fill="{colors["missing"]}" />'
                f'<text x="{x:.1f}" y="{height - 24}" text-anchor="middle" font-size="12" fill="{colors["muted"]}">{label}</text>'
            )
            continue
        mood = escape(str(point.get("mood") or ""))
        note = escape(str(point.get("note") or ""))
        y = y_at(int(intensity))
        label_width = max(44, len(mood) * 14 + 16)
        label_x = min(max(x - label_width / 2, left), width - right - label_width)
        label_y = max(6, y - 30)
        tooltip = f"{date_text} {mood} 强度 {intensity}/5"
        if note:
            tooltip += f"：{note}"
        marks.append(
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="{colors["accent"]}">'
            f'<title>{escape(tooltip)}</title></circle>'
            f'<rect x="{label_x:.1f}" y="{label_y:.1f}" width="{label_width:.1f}" height="20" '
            f'rx="5" fill="{colors["surface_alt"]}" stroke="{colors["label_border"]}" stroke-width="1" />'
            f'<text x="{label_x + label_width / 2:.1f}" y="{label_y + 14:.1f}" '
            f'text-anchor="middle" font-size="12" font-weight="600" fill="{colors["text"]}">'
            f'{mood}</text>'
            f'<text x="{x:.1f}" y="{height - 24}" text-anchor="middle" font-size="12" fill="{colors["date"]}">{label}</text>'
        )

    empty_note = ""
    if not recorded_points:
        empty_note = (
            f'<text x="{width / 2:.1f}" y="{height / 2:.1f}" text-anchor="middle" '
            f'font-size="15" fill="{colors["muted"]}">暂无本周 Mood Check-in 数据</text>'
        )

    return (
        f'<div data-chart-theme="{theme_mode}" style="width:100%;overflow-x:auto;'
        f'background:{colors["surface"]};border:1px solid {colors["border"]};border-radius:8px;padding:8px;">'
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        'aria-label="Weekly Mood Chart" style="max-width:100%;height:auto;">'
        f'{"".join(y_lines)}'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="{colors["axis"]}" />'
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="{colors["axis"]}" />'
        f'{path}{"".join(marks)}{empty_note}'
        f'<text x="18" y="18" font-size="12" fill="{colors["muted"]}">强度</text>'
        '</svg></div>'
    )


def weekly_mood_view(user_id: str, end_date: str, theme_mode: str) -> tuple[str, str]:
    try:
        points = get_weekly_mood_points(user_id, end_date=end_date)
        return render_weekly_mood_chart(points, theme_mode), format_weekly_mood_summary(user_id, end_date=end_date)
    except Exception as exc:
        logger.warning("Failed to refresh weekly mood chart for user=%s: %s", user_id, exc)
        return "", f"刷新 Weekly Mood Chart 失败：{exc}"


def refresh_weekly_mood_chart(
    user_id: str, access_key: str, end_date: str, theme_mode: str = "light",
    locale: str = "zh-CN",
) -> tuple[str, str]:
    user_id, auth_error = authorize_or_message(user_id, access_key, locale)
    if auth_error:
        return "", auth_error
    chart, summary = weekly_mood_view(user_id, end_date, theme_mode)
    return localize_status_text(chart, locale), localize_status_text(summary, locale)


def refresh_weekly_mood_dashboard(
    user_id: str, access_key: str, end_date: str, theme_mode: str = "light",
    locale: str = "zh-CN",
) -> tuple[str, str, str]:
    user_id, auth_error = authorize_or_message(user_id, access_key, locale)
    if auth_error:
        return "", auth_error, auth_error
    chart, summary = weekly_mood_view(user_id, end_date, theme_mode)
    try:
        analysis = format_weekly_mood_analysis(user_id, end_date=end_date)
    except Exception as exc:
        logger.warning("Failed to analyze weekly mood for user=%s: %s", user_id, exc)
        analysis = f"情绪波动分析失败：{exc}"
    return (
        localize_status_text(chart, locale),
        localize_status_text(summary, locale),
        localize_status_text(analysis, locale),
    )


def submit_mood_checkin_and_refresh_dashboard(
    user_id: str,
    access_key: str,
    checkin_date: str,
    mood: str,
    intensity: int | float,
    note: str,
    current_end_date: str,
    theme_mode: str = "light",
    locale: str = "zh-CN",
) -> tuple[str, str, str, str, str]:
    _, auth_error = authorize_or_message(user_id, access_key, locale)
    if auth_error:
        return auth_error, current_end_date, "", "", auth_error
    raw_mood_panel = submit_mood_checkin(user_id, access_key, checkin_date, mood, intensity, note)
    mood_panel = localize_status_text(raw_mood_panel, locale)
    if not raw_mood_panel.startswith("已保存 "):
        return mood_panel, current_end_date, "", "", mood_panel
    chart, summary, analysis = refresh_weekly_mood_dashboard(
        user_id, access_key, checkin_date, theme_mode, locale
    )
    return mood_panel, checkin_date, chart, summary, analysis


def delete_mood_checkin_and_refresh_dashboard(
    user_id: str,
    access_key: str,
    checkin_date: str,
    current_end_date: str,
    theme_mode: str = "light",
    locale: str = "zh-CN",
) -> tuple[str, str, str, str, str]:
    _, auth_error = authorize_or_message(user_id, access_key, locale)
    if auth_error:
        return auth_error, current_end_date, "", "", auth_error
    raw_mood_panel = delete_mood_checkin_from_gui(user_id, access_key, checkin_date)
    mood_panel = localize_status_text(raw_mood_panel, locale)
    if raw_mood_panel.startswith(("删除 Mood Check-in 失败", "请先", "访问密码")):
        return mood_panel, current_end_date, "", "", mood_panel
    chart, summary, analysis = refresh_weekly_mood_dashboard(
        user_id, access_key, checkin_date, theme_mode, locale
    )
    return mood_panel, checkin_date, chart, summary, analysis


def load_theme_and_weekly_chart(
    user_id: str, access_key: str, end_date: str, theme_mode: str = "light",
    locale: str = "zh-CN",
) -> tuple[str, str, str]:
    user_id, auth_error = authorize_or_message(user_id, access_key, locale)
    if auth_error:
        empty_chart = localize_status_text(render_weekly_mood_chart([], theme_mode), locale)
        return theme_mode, empty_chart, auth_error
    chart, summary = weekly_mood_view(user_id, end_date, theme_mode)
    return theme_mode, localize_status_text(chart, locale), localize_status_text(summary, locale)


def load_theme_and_weekly_dashboard(
    user_id: str, access_key: str, end_date: str, theme_mode: str = "light",
    locale: str = "zh-CN",
) -> tuple[str, str, str, str]:
    user_id, auth_error = authorize_or_message(user_id, access_key, locale)
    if auth_error:
        empty_chart = localize_status_text(render_weekly_mood_chart([], theme_mode), locale)
        return theme_mode, empty_chart, auth_error, auth_error
    chart, summary = weekly_mood_view(user_id, end_date, theme_mode)
    return (
        theme_mode,
        localize_status_text(chart, locale),
        localize_status_text(summary, locale),
        localize_status_text(format_weekly_mood_analysis(user_id, end_date=end_date), locale),
    )


# Backward-compatible name used by existing tests and Web_GUI imports.
_render_weekly_mood_chart = render_weekly_mood_chart
