from __future__ import annotations

from typing import Any

from json_utils import safe_extract_json_object


def tool_translate(text: str, target_lang: str = "en") -> str:
    return f"[translate -> {target_lang}] {text}"


def tool_summarize(text: str) -> str:
    return f"[summary] {text[:120]}{'...' if len(text) > 120 else ''}"


def tool_explain_code(code: str) -> str:
    return f"[code explanation requested] {code[:120]}{'...' if len(code) > 120 else ''}"


def call_tool(tool_name: str, arguments: dict[str, Any]) -> str:
    tools = {"translate": tool_translate, "summarize": tool_summarize, "explain_code": tool_explain_code}
    tool = tools.get(tool_name)
    if tool is None:
        return "未知工具"
    return tool(**arguments)


def try_tool_call(reply: str) -> str | None:
    data = safe_extract_json_object(reply)
    if not data or "tool" not in data:
        return None
    return call_tool(str(data["tool"]), data.get("args", {}))
