from __future__ import annotations

from typing import Any

from knowledge_store import (
    assess_knowledge_quality,
    clear_documents,
    delete_document,
    diagnose_knowledge_search,
    list_document_details,
    knowledge_status,
)


def format_knowledge_quality_report() -> str:
    report = assess_knowledge_quality()
    index_count = report["index_count"]
    lines = [
        f"质量等级：{report['level']}",
        f"文档 / 片段：{report['documents']} / {report['chunks']}",
        f"平均片段长度：{report['average_chunk_chars']} 字符",
        f"短片段 / 重复片段 / 孤立片段：{report['short_chunks']} / {report['duplicate_chunks']} / {report['orphan_chunks']}",
        f"索引向量数：{index_count if index_count is not None else '不可用'}",
    ]
    issues = report["issues"]
    lines.append("检查结果：")
    lines.extend(f"- {issue}" for issue in issues)
    if not issues:
        lines.append("- 未发现明显质量问题")
    return "\n".join(lines)


def format_knowledge_search_diagnostics(
    query: str,
    top_k: int,
    threshold: float,
    candidate_multiplier: int,
) -> str:
    query = (query or "").strip()
    if not query:
        return "请输入检索问题。"
    diagnostic = diagnose_knowledge_search(
        query,
        top_k=int(top_k),
        threshold=float(threshold),
        candidate_multiplier=int(candidate_multiplier),
    )
    candidates = diagnostic["candidates"]
    accepted = diagnostic["results"]
    lines = [
        f"候选 {len(candidates)} 条，采用 {len(accepted)} 条",
        f"Top K={diagnostic.get('top_k', top_k)}，阈值={diagnostic.get('threshold', threshold):.2f}",
    ]
    if not candidates:
        lines.append("没有可供检索的候选片段。")
        return "\n".join(lines)
    for index, item in enumerate(candidates, start=1):
        marker = "采用" if item.get("accepted") else "淘汰"
        text = " ".join(str(item.get("text", "")).split())
        lines.extend([
            "",
            f"[{index}] {marker} | {item.get('source', 'unknown')} | 相关度 {float(item.get('score', 0)):.3f}",
            f"原因：{item.get('decision', '未分类')}",
            text[:280] + ("..." if len(text) > 280 else ""),
        ])
    return "\n".join(lines)


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
