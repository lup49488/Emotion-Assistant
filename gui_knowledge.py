from __future__ import annotations

from typing import Any

from gui_i18n import localize_status_text
from knowledge_store import (
    assess_knowledge_quality,
    clear_documents,
    delete_document,
    diagnose_knowledge_search,
    list_document_details,
    knowledge_status,
)
from rag_evaluation_store import import_evaluation_cases, latest_evaluation_report, run_evaluation


def format_knowledge_quality_report(locale: str = "zh-CN") -> str:
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
    return localize_status_text("\n".join(lines), locale)


def format_knowledge_search_diagnostics(
    query: str,
    top_k: int,
    threshold: float,
    candidate_multiplier: int,
    locale: str = "zh-CN",
) -> str:
    query = (query or "").strip()
    if not query:
        return localize_status_text("请输入检索问题。", locale)
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
        return localize_status_text("\n".join(lines), locale)
    for index, item in enumerate(candidates, start=1):
        marker = "采用" if item.get("accepted") else "淘汰"
        text = " ".join(str(item.get("text", "")).split())
        lines.extend([
            "",
            f"[{index}] {marker} | {item.get('source', 'unknown')} | 相关度 {float(item.get('score', 0)):.3f}",
            f"原因：{item.get('decision', '未分类')}",
            text[:280] + ("..." if len(text) > 280 else ""),
        ])
    return localize_status_text("\n".join(lines), locale)


def format_knowledge_document_list(locale: str = "zh-CN") -> str:
    documents = list_document_details()
    if not documents:
        return localize_status_text("当前没有已入库的 RAG 文档。", locale)

    lines = [f"已入库文档：{len(documents)}"]
    for item in documents:
        size_kb = item["size_bytes"] / 1024
        lines.append(
            f"- {item['name']} | {item['chunks']} 个片段 | {size_kb:.1f} KB | 更新于 {item['modified_at']}"
        )
    return localize_status_text("\n".join(lines), locale)


def knowledge_document_names() -> list[str]:
    return [item["name"] for item in list_document_details()]


def refresh_knowledge_document_panel(locale: str = "zh-CN") -> tuple[list[str], str]:
    return knowledge_document_names(), format_knowledge_document_list(locale)


def _format_management_result(action: str, result: dict[str, Any], locale: str) -> str:
    lines = [
        f"{action}完成。",
        f"当前文档数：{result.get('documents', 0)}",
        f"当前片段数：{result.get('chunks', 0)}",
    ]
    errors = result.get("errors") or []
    if errors:
        lines.append("重建警告：")
        lines.extend(f"- {error}" for error in errors)
    localized = localize_status_text("\n".join(lines), locale)
    return "\n".join([
        localized,
        "",
        format_knowledge_document_list(locale),
        "",
        localize_status_text(knowledge_status(), locale),
    ])


def delete_knowledge_document(name: str, locale: str = "zh-CN") -> tuple[list[str], str, str]:
    try:
        result = delete_document(name)
        message = _format_management_result(f"已删除文档“{result['deleted']}”", result, locale)
    except Exception as exc:
        message = localize_status_text(f"删除失败：{exc}", locale) + "\n\n" + format_knowledge_document_list(locale)
    names, document_list = refresh_knowledge_document_panel(locale)
    return names, document_list, message


def clear_knowledge_documents(locale: str = "zh-CN") -> tuple[list[str], str, str]:
    try:
        result = clear_documents()
        removed = result.get("removed") or []
        message = _format_management_result(f"已清空 {len(removed)} 个文档", result, locale)
    except Exception as exc:
        message = localize_status_text(f"清空失败：{exc}", locale) + "\n\n" + format_knowledge_document_list(locale)
    names, document_list = refresh_knowledge_document_panel(locale)
    return names, document_list, message


def import_rag_evaluation_cases(file_obj: Any, locale: str = "zh-CN") -> str:
    if not file_obj:
        return localize_status_text("请先选择 RAG 评估集文件。", locale)
    try:
        path = getattr(file_obj, "name", None) or getattr(file_obj, "path", None) or str(file_obj)
        result = import_evaluation_cases(path)
    except Exception as exc:
        return localize_status_text(f"导入 RAG 评估集失败：{exc}", locale)
    return localize_status_text(f"RAG 评估集已导入：{result['cases']} 条样本。", locale)


def format_rag_evaluation_report(report: dict[str, Any] | None, locale: str = "zh-CN") -> str:
    english = str(locale or "").lower().startswith("en")
    if not report:
        return "No RAG evaluation has been run." if english else "尚未运行 RAG 评估。"
    settings = report["settings"]
    not_available = "n/a" if english else "不适用"
    mode = settings.get("retrieval_mode")
    if english:
        lines = [
            f"Latest run: {report['created_at']}",
            f"Settings: Top K={settings['top_k']}, threshold={settings['threshold']:.2f}, candidate multiplier={settings['candidate_multiplier']}" + (f", retrieval={mode}" if mode else ""),
            f"Overall pass rate: {report['passed_cases']}/{report['total_cases']} ({report['pass_rate']:.1f}%)",
            f"Source recall@K: {report['source_hits']}/{report['source_cases']} ({report['source_recall_at_k'] if report['source_recall_at_k'] is not None else not_available}%)",
            f"MRR: {report['mrr'] if report['mrr'] is not None else not_available}",
            f"Keyword coverage: {report['keyword_hits']}/{report['keyword_cases']} ({report['keyword_coverage'] if report['keyword_coverage'] is not None else not_available}%)",
            f"Insufficient-evidence refusal: {report.get('insufficient_passes', 0)}/{report.get('insufficient_cases', 0)} ({report.get('insufficient_refusal_rate') if report.get('insufficient_refusal_rate') is not None else not_available}%)",
        ]
    else:
        lines = [
            f"最近评估：{report['created_at']}",
            f"参数：Top K={settings['top_k']}，阈值={settings['threshold']:.2f}，候选池倍数={settings['candidate_multiplier']}" + (f"，检索模式={mode}" if mode else ""),
            f"总体通过率：{report['passed_cases']}/{report['total_cases']} ({report['pass_rate']:.1f}%)",
            f"来源召回率@K：{report['source_hits']}/{report['source_cases']} ({report['source_recall_at_k'] if report['source_recall_at_k'] is not None else not_available}%)",
            f"MRR：{report['mrr'] if report['mrr'] is not None else not_available}",
            f"关键词覆盖率：{report['keyword_hits']}/{report['keyword_cases']} ({report['keyword_coverage'] if report['keyword_coverage'] is not None else not_available}%)",
            f"资料不足拒答：{report.get('insufficient_passes', 0)}/{report.get('insufficient_cases', 0)} ({report.get('insufficient_refusal_rate') if report.get('insufficient_refusal_rate') is not None else not_available}%)",
        ]
    comparison = report.get("mode_comparison")
    if isinstance(comparison, dict):
        vector = comparison.get("vector") or {}
        hybrid = comparison.get("hybrid_rrf") or {}

        def metric(summary: dict[str, Any], key: str) -> str:
            # A metric with no denominator is stored as None, so a plain dict.get
            # default would print "None" instead of the not-available marker.
            value = summary.get(key)
            return not_available if value is None else str(value)

        if english:
            lines.append(
                "Mode comparison (vector -> hybrid RRF): "
                f"Recall@K {metric(vector, 'source_recall_at_k')}% -> {metric(hybrid, 'source_recall_at_k')}%; "
                f"MRR {metric(vector, 'mrr')} -> {metric(hybrid, 'mrr')}; "
                f"insufficient-evidence refusal {metric(vector, 'insufficient_refusal_rate')}% -> {metric(hybrid, 'insufficient_refusal_rate')}%."
            )
        else:
            lines.append(
                "模式对照（向量 -> Hybrid RRF）："
                f"召回率@K {metric(vector, 'source_recall_at_k')}% -> {metric(hybrid, 'source_recall_at_k')}%；"
                f"MRR {metric(vector, 'mrr')} -> {metric(hybrid, 'mrr')}；"
                f"资料不足拒答 {metric(vector, 'insufficient_refusal_rate')}% -> {metric(hybrid, 'insufficient_refusal_rate')}%。"
            )
    failures = report.get("failures") or []
    if failures:
        lines.append("Failed samples:" if english else "失败样本：")
        for item in failures[:5]:
            returned = ", ".join(item.get("returned_sources") or []) or ("none" if english else "无")
            lines.append(f"- {item['query'][:100]} | {'returned' if english else '返回'}: {returned}")
    return "\n".join(lines)


def run_rag_evaluation_from_gui(
    top_k: int, threshold: float, candidate_multiplier: int,
    compare_modes: bool = False, locale: str = "zh-CN",
) -> str:
    try:
        report = run_evaluation(
            top_k=int(top_k), threshold=float(threshold), candidate_multiplier=int(candidate_multiplier),
            compare_modes=bool(compare_modes),
        )
    except Exception as exc:
        return localize_status_text(f"运行 RAG 评估失败：{exc}", locale)
    return format_rag_evaluation_report(report, locale)


def latest_rag_evaluation_report(locale: str = "zh-CN") -> str:
    # 该函数用作报告框的启动初始值：报告文件损坏或缺键时绝不能让整个应用启动失败。
    try:
        return format_rag_evaluation_report(latest_evaluation_report(), locale)
    except Exception as exc:
        return localize_status_text(f"读取 RAG 评估报告失败：{exc}", locale)
