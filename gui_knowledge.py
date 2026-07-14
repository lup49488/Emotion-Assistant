from __future__ import annotations

from knowledge_store import clear_documents, delete_document, list_document_details, knowledge_status


def format_knowledge_document_list() -> str:
    documents = list_document_details()
    if not documents:
        return "当前没有已入库的 RAG 文档。"

    lines = [f"已入库文档：{len(documents)}"]
    for item in documents:
        size_kb = item["size_bytes"] / 1024
        lines.append(
            f"- {item['name']} | {item['chunks']} 个片段 | {size_kb:.1f} KB | 更新于 {item['modified_at']}"
        )
    return "\n".join(lines)


def knowledge_document_names() -> list[str]:
    return [item["name"] for item in list_document_details()]


def refresh_knowledge_document_panel() -> tuple[list[str], str]:
    return knowledge_document_names(), format_knowledge_document_list()


def _format_management_result(action: str, result: dict[str, Any]) -> str:
    lines = [
        f"{action}完成。",
        f"当前文档数：{result.get('documents', 0)}",
        f"当前片段数：{result.get('chunks', 0)}",
    ]
    errors = result.get("errors") or []
    if errors:
        lines.append("重建警告：")
        lines.extend(f"- {error}" for error in errors)
    lines.extend(["", format_knowledge_document_list(), "", knowledge_status()])
    return "\n".join(lines)


def delete_knowledge_document(name: str) -> tuple[list[str], str, str]:
    try:
        result = delete_document(name)
        message = _format_management_result(f"已删除文档“{result['deleted']}”", result)
    except Exception as exc:
        message = f"删除失败：{exc}\n\n{format_knowledge_document_list()}"
    names, document_list = refresh_knowledge_document_panel()
    return names, document_list, message


def clear_knowledge_documents() -> tuple[list[str], str, str]:
    try:
        result = clear_documents()
        removed = result.get("removed") or []
        message = _format_management_result(f"已清空 {len(removed)} 个文档", result)
    except Exception as exc:
        message = f"清空失败：{exc}\n\n{format_knowledge_document_list()}"
    names, document_list = refresh_knowledge_document_panel()
    return names, document_list, message
