from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from config import (
    API_DAILY_BUDGET_USD,
    API_INPUT_COST_PER_1M_TOKENS,
    API_MAX_REQUESTS_PER_MINUTE,
    API_MONTHLY_BUDGET_USD,
    API_OUTPUT_COST_PER_1M_TOKENS,
)
from sqlite_store import connection, ensure_user, sqlite_enabled


_JSON_FILENAME = "api_usage.json"
_lock = threading.RLock()


def _now() -> datetime:
    return datetime.now()


def _timestamp(value: datetime | None = None) -> str:
    return (value or _now()).isoformat(timespec="seconds")


def _validate_user_id(user_id: str) -> str:
    # 延迟导入避免 session_store 与模型模块的循环依赖。
    from session_store import validate_user_id

    return validate_user_id(user_id)


def _json_path(user_id: str) -> Path:
    from session_store import user_dir

    return user_dir(user_id) / _JSON_FILENAME


def _read_json_events(user_id: str) -> list[dict[str, Any]]:
    path = _json_path(user_id)
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return payload if isinstance(payload, list) else []


def _write_json_events(user_id: str, events: list[dict[str, Any]]) -> None:
    path = _json_path(user_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(events[-5000:], ensure_ascii=False, indent=2), encoding="utf-8")


def estimate_tokens(text: str) -> int:
    """Conservative display estimate when a provider does not return token usage."""
    return max(1, (len(text or "") + 3) // 4)


def estimate_input_tokens(messages: list[dict[str, str]]) -> int:
    return sum(estimate_tokens(str(message.get("content", ""))) for message in messages)


def estimate_cost_usd(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens * API_INPUT_COST_PER_1M_TOKENS
        + output_tokens * API_OUTPUT_COST_PER_1M_TOKENS
    ) / 1_000_000


def record_usage(
    user_id: str | None,
    *,
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    duration_ms: int,
    success: bool,
    error_kind: str = "",
    created_at: datetime | None = None,
) -> None:
    if not user_id:
        return
    user_id = _validate_user_id(user_id)
    event = {
        "provider": provider,
        "model": model,
        "input_tokens": max(0, int(input_tokens)),
        "output_tokens": max(0, int(output_tokens)),
        "estimated_cost_usd": estimate_cost_usd(input_tokens, output_tokens),
        "duration_ms": max(0, int(duration_ms)),
        "success": bool(success),
        "error_kind": error_kind,
        "created_at": _timestamp(created_at),
    }
    with _lock:
        if sqlite_enabled():
            with connection() as conn:
                ensure_user(conn, user_id)
                conn.execute(
                    """
                    INSERT INTO api_usage_events(
                        user_id, provider, model, input_tokens, output_tokens,
                        estimated_cost_usd, duration_ms, success, error_kind, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (user_id, *event.values()),
                )
            return
        events = _read_json_events(user_id)
        events.append(event)
        _write_json_events(user_id, events)


def list_usage_events(user_id: str, *, limit: int = 5000) -> list[dict[str, Any]]:
    user_id = _validate_user_id(user_id)
    limit = max(1, min(int(limit), 5000))
    with _lock:
        if sqlite_enabled():
            with connection() as conn:
                rows = conn.execute(
                    """
                    SELECT provider, model, input_tokens, output_tokens, estimated_cost_usd,
                           duration_ms, success, error_kind, created_at
                    FROM api_usage_events WHERE user_id = ?
                    ORDER BY id DESC LIMIT ?
                    """,
                    (user_id, limit),
                ).fetchall()
            return [dict(row) | {"success": bool(row["success"])} for row in rows]
        return list(reversed(_read_json_events(user_id)[-limit:]))


def usage_summary(user_id: str, *, now: datetime | None = None) -> dict[str, Any]:
    user_id = _validate_user_id(user_id)
    now = now or _now()
    day = now.date().isoformat()
    month = now.strftime("%Y-%m")
    minute_start = now.timestamp() - 60
    events = list_usage_events(user_id)

    def totals(selected: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "requests": len(selected),
            "failures": sum(1 for item in selected if not item.get("success")),
            "input_tokens": sum(int(item.get("input_tokens", 0)) for item in selected),
            "output_tokens": sum(int(item.get("output_tokens", 0)) for item in selected),
            "estimated_cost_usd": sum(float(item.get("estimated_cost_usd", 0)) for item in selected),
        }

    today = [item for item in events if str(item.get("created_at", "")).startswith(day)]
    current_month = [item for item in events if str(item.get("created_at", "")).startswith(month)]
    recent = []
    for item in events:
        try:
            if datetime.fromisoformat(str(item["created_at"])).timestamp() >= minute_start:
                recent.append(item)
        except (KeyError, TypeError, ValueError):
            continue
    return {
        "today": totals(today),
        "month": totals(current_month),
        "requests_last_minute": len(recent),
        "limits": {
            "requests_per_minute": API_MAX_REQUESTS_PER_MINUTE,
            "daily_budget_usd": API_DAILY_BUDGET_USD,
            "monthly_budget_usd": API_MONTHLY_BUDGET_USD,
            "input_cost_per_1m": API_INPUT_COST_PER_1M_TOKENS,
            "output_cost_per_1m": API_OUTPUT_COST_PER_1M_TOKENS,
        },
    }


def check_request_allowed(user_id: str | None, *, projected_input_tokens: int, projected_output_tokens: int) -> None:
    if not user_id:
        return
    summary = usage_summary(user_id)
    limits = summary["limits"]
    rpm = int(limits["requests_per_minute"])
    if rpm and summary["requests_last_minute"] >= rpm:
        raise RuntimeError(f"API 请求频率已达到本机限制：{rpm} 次/分钟")
    projected_cost = estimate_cost_usd(projected_input_tokens, projected_output_tokens)
    daily_limit = float(limits["daily_budget_usd"])
    if daily_limit and summary["today"]["estimated_cost_usd"] + projected_cost > daily_limit:
        raise RuntimeError(f"API 今日预计费用将超过本机限额 ${daily_limit:.4f}")
    monthly_limit = float(limits["monthly_budget_usd"])
    if monthly_limit and summary["month"]["estimated_cost_usd"] + projected_cost > monthly_limit:
        raise RuntimeError(f"API 本月预计费用将超过本机限额 ${monthly_limit:.4f}")
