from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from config import KNOWLEDGE_DIR, KNOWLEDGE_RETRIEVAL_MODE
from json_utils import load_json, save_json
from knowledge_store import diagnose_knowledge_search, has_usable_evidence


RAG_EVALUATION_CASES_PATH = KNOWLEDGE_DIR / "rag_evaluation_cases.json"
RAG_EVALUATION_REPORTS_PATH = KNOWLEDGE_DIR / "rag_evaluation_reports.json"
MAX_SAVED_REPORTS = 20


def _as_string_list(value: Any) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def normalize_case(value: Any, *, position: int | None = None) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"第 {position or '?'} 条评估样本必须是对象。")
    query = str(value.get("query", value.get("question", ""))).strip()
    sources = _as_string_list(value.get("expected_sources", value.get("expected_source", [])))
    keywords = _as_string_list(value.get("expected_keywords", value.get("expected_keyword", [])))
    case_id = str(value.get("id") or "").strip() or f"case-{position or 0:03d}"
    category = str(value.get("category") or "general").strip()
    expected_outcome = str(value.get("expected_outcome") or "grounded").strip().lower()
    if not query:
        raise ValueError(f"第 {position or '?'} 条评估样本缺少 query。")
    if expected_outcome not in {"grounded", "insufficient"}:
        raise ValueError(f"第 {position or '?'} 条评估样本的 expected_outcome 必须是 grounded 或 insufficient。")
    if expected_outcome == "grounded" and not sources and not keywords:
        raise ValueError(f"第 {position or '?'} 条评估样本至少需要 expected_sources 或 expected_keywords。")
    if expected_outcome == "insufficient" and (sources or keywords):
        raise ValueError(f"第 {position or '?'} 条应拒答样本不能包含 expected_sources 或 expected_keywords。")
    return {
        "id": case_id,
        "category": category,
        "query": query,
        "expected_outcome": expected_outcome,
        "expected_sources": sources,
        "expected_keywords": keywords,
    }


def normalize_cases(rows: list[Any]) -> list[dict[str, Any]]:
    """Normalize a whole set, rejecting duplicate ids so reports stay traceable."""
    cases = [normalize_case(item, position=index) for index, item in enumerate(rows, start=1)]
    seen: set[str] = set()
    for case in cases:
        if case["id"] in seen:
            raise ValueError(f"评估集存在重复的样本 id：{case['id']}。")
        seen.add(case["id"])
    return cases


def load_evaluation_cases() -> list[dict[str, Any]]:
    return normalize_cases(load_json(RAG_EVALUATION_CASES_PATH))


def _load_uploaded_cases(path: str | Path) -> list[Any]:
    file_path = Path(path)
    raw = file_path.read_text(encoding="utf-8-sig")
    if file_path.suffix.lower() == ".jsonl":
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    else:
        rows = json.loads(raw)
    if not isinstance(rows, list):
        raise ValueError("评估集文件必须是 JSON 数组或 JSONL，每行一条对象。")
    return rows


def import_evaluation_cases(path: str | Path) -> dict[str, Any]:
    rows = _load_uploaded_cases(path)
    cases = normalize_cases(rows)
    if not cases:
        raise ValueError("评估集不能是空数组。")
    RAG_EVALUATION_CASES_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_json(RAG_EVALUATION_CASES_PATH, cases)
    return {"cases": len(cases), "path": RAG_EVALUATION_CASES_PATH.name}


def _normalized_sources(items: list[str]) -> set[str]:
    return {item.casefold() for item in items}


def _percent(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator * 100, 1) if denominator else None


def _evaluate_cases(
    cases: list[dict[str, Any]], *, top_k: int, threshold: float,
    candidate_multiplier: int, retrieval_mode: str,
) -> dict[str, Any]:

    source_cases = keyword_cases = source_hits = keyword_hits = passed_cases = 0
    insufficient_cases = insufficient_passes = 0
    reciprocal_rank_total = 0.0
    details: list[dict[str, Any]] = []
    for case in cases:
        diagnostic = diagnose_knowledge_search(
            case["query"], top_k=int(top_k), threshold=float(threshold),
            candidate_multiplier=int(candidate_multiplier),
            retrieval_mode=retrieval_mode,
        )
        results = diagnostic["results"]
        returned_sources = [str(item.get("source", "")) for item in results]
        returned_text = "\n".join(str(item.get("text", "")) for item in results).casefold()
        expected_sources = _normalized_sources(case["expected_sources"])
        expected_keywords = [item.casefold() for item in case["expected_keywords"]]

        source_pass: bool | None = None
        keyword_pass: bool | None = None
        hit_rank: int | None = None
        insufficient_pass: bool | None = None
        if case["expected_outcome"] == "insufficient":
            insufficient_cases += 1
            # Match the chat path: grounded means "yields citable context",
            # not merely "the index returned rows".
            insufficient_pass = not has_usable_evidence(results)
            if insufficient_pass:
                insufficient_passes += 1
            passed = insufficient_pass
        else:
            if expected_sources:
                source_cases += 1
                for index, source in enumerate(returned_sources, start=1):
                    if source.casefold() in expected_sources:
                        hit_rank = index
                        break
                source_pass = hit_rank is not None
                if source_pass:
                    source_hits += 1
                    reciprocal_rank_total += 1 / hit_rank
            if expected_keywords:
                keyword_cases += 1
                keyword_pass = all(keyword in returned_text for keyword in expected_keywords)
                if keyword_pass:
                    keyword_hits += 1
            passed = all(value is not False for value in (source_pass, keyword_pass))
        if passed:
            passed_cases += 1
        details.append({
            "id": case["id"], "category": case["category"], "query": case["query"],
            "expected_outcome": case["expected_outcome"], "expected_sources": case["expected_sources"],
            "expected_keywords": case["expected_keywords"], "returned_sources": returned_sources,
            "source_pass": source_pass, "keyword_pass": keyword_pass,
            "insufficient_pass": insufficient_pass, "hit_rank": hit_rank, "passed": passed,
        })

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "settings": {"top_k": int(top_k), "threshold": float(threshold), "candidate_multiplier": int(candidate_multiplier), "retrieval_mode": retrieval_mode},
        "total_cases": len(cases), "passed_cases": passed_cases,
        "pass_rate": _percent(passed_cases, len(cases)),
        "source_cases": source_cases, "source_hits": source_hits,
        "source_recall_at_k": _percent(source_hits, source_cases),
        "recall_at_k": _percent(source_hits, source_cases),
        "mrr": round(reciprocal_rank_total / source_cases, 3) if source_cases else None,
        "keyword_cases": keyword_cases, "keyword_hits": keyword_hits,
        "keyword_coverage": _percent(keyword_hits, keyword_cases),
        "insufficient_cases": insufficient_cases, "insufficient_passes": insufficient_passes,
        "insufficient_refusal_rate": _percent(insufficient_passes, insufficient_cases),
        "insufficient_refusal_accuracy": _percent(insufficient_passes, insufficient_cases),
        "failures": [item for item in details if not item["passed"]][:10],
    }
    return report


def _comparison_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "recall_at_k": report["recall_at_k"],
        "mrr": report["mrr"],
        "insufficient_refusal_accuracy": report["insufficient_refusal_accuracy"],
        "pass_rate": report["pass_rate"],
    }


def run_evaluation(
    *, top_k: int, threshold: float, candidate_multiplier: int,
    retrieval_mode: str | None = None, compare_modes: bool = False,
) -> dict[str, Any]:
    cases = load_evaluation_cases()
    if not cases:
        raise ValueError("还没有评估样本，请先导入 JSON 或 JSONL 评估集。")
    mode = str(retrieval_mode or KNOWLEDGE_RETRIEVAL_MODE).strip().lower()
    if mode not in {"vector", "hybrid_rrf"}:
        raise ValueError("检索模式仅支持 vector 或 hybrid_rrf。")
    report = _evaluate_cases(
        cases, top_k=top_k, threshold=threshold,
        candidate_multiplier=candidate_multiplier, retrieval_mode=mode,
    )
    if compare_modes:
        reports = {mode: report}
        alternate_mode = "vector" if mode == "hybrid_rrf" else "hybrid_rrf"
        reports[alternate_mode] = _evaluate_cases(
            cases, top_k=top_k, threshold=threshold,
            candidate_multiplier=candidate_multiplier, retrieval_mode=alternate_mode,
        )
        vector_summary = _comparison_summary(reports["vector"])
        hybrid_summary = _comparison_summary(reports["hybrid_rrf"])
        report["mode_comparison"] = {
            "vector": vector_summary,
            "hybrid_rrf": hybrid_summary,
            "delta_hybrid_minus_vector": {
                key: round(float(hybrid_summary[key] or 0) - float(vector_summary[key] or 0), 3)
                for key in vector_summary
            },
        }
    reports = load_json(RAG_EVALUATION_REPORTS_PATH)
    RAG_EVALUATION_REPORTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_json(RAG_EVALUATION_REPORTS_PATH, [report, *reports[: MAX_SAVED_REPORTS - 1]])
    return report


def latest_evaluation_report() -> dict[str, Any] | None:
    reports = load_json(RAG_EVALUATION_REPORTS_PATH)
    return reports[0] if reports and isinstance(reports[0], dict) else None
