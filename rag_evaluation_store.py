from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from config import KNOWLEDGE_DIR
from json_utils import load_json, save_json
from knowledge_store import diagnose_knowledge_search


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
    if not query:
        raise ValueError(f"第 {position or '?'} 条评估样本缺少 query。")
    if not sources and not keywords:
        raise ValueError(f"第 {position or '?'} 条评估样本至少需要 expected_sources 或 expected_keywords。")
    return {"query": query, "expected_sources": sources, "expected_keywords": keywords}


def load_evaluation_cases() -> list[dict[str, Any]]:
    return [normalize_case(item, position=index) for index, item in enumerate(load_json(RAG_EVALUATION_CASES_PATH), start=1)]


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
    cases = [normalize_case(item, position=index) for index, item in enumerate(rows, start=1)]
    if not cases:
        raise ValueError("评估集不能是空数组。")
    RAG_EVALUATION_CASES_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_json(RAG_EVALUATION_CASES_PATH, cases)
    return {"cases": len(cases), "path": RAG_EVALUATION_CASES_PATH.name}


def _normalized_sources(items: list[str]) -> set[str]:
    return {item.casefold() for item in items}


def _percent(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator * 100, 1) if denominator else None


def run_evaluation(
    *, top_k: int, threshold: float, candidate_multiplier: int
) -> dict[str, Any]:
    cases = load_evaluation_cases()
    if not cases:
        raise ValueError("还没有评估样本，请先导入 JSON 或 JSONL 评估集。")

    source_cases = keyword_cases = source_hits = keyword_hits = passed_cases = 0
    reciprocal_rank_total = 0.0
    details: list[dict[str, Any]] = []
    for case in cases:
        diagnostic = diagnose_knowledge_search(
            case["query"], top_k=int(top_k), threshold=float(threshold),
            candidate_multiplier=int(candidate_multiplier),
        )
        results = diagnostic["results"]
        returned_sources = [str(item.get("source", "")) for item in results]
        returned_text = "\n".join(str(item.get("text", "")) for item in results).casefold()
        expected_sources = _normalized_sources(case["expected_sources"])
        expected_keywords = [item.casefold() for item in case["expected_keywords"]]

        source_pass: bool | None = None
        keyword_pass: bool | None = None
        hit_rank: int | None = None
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
            "query": case["query"], "expected_sources": case["expected_sources"],
            "expected_keywords": case["expected_keywords"], "returned_sources": returned_sources,
            "source_pass": source_pass, "keyword_pass": keyword_pass,
            "hit_rank": hit_rank, "passed": passed,
        })

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "settings": {"top_k": int(top_k), "threshold": float(threshold), "candidate_multiplier": int(candidate_multiplier)},
        "total_cases": len(cases), "passed_cases": passed_cases,
        "pass_rate": _percent(passed_cases, len(cases)),
        "source_cases": source_cases, "source_hits": source_hits,
        "source_recall_at_k": _percent(source_hits, source_cases),
        "mrr": round(reciprocal_rank_total / source_cases, 3) if source_cases else None,
        "keyword_cases": keyword_cases, "keyword_hits": keyword_hits,
        "keyword_coverage": _percent(keyword_hits, keyword_cases),
        "failures": [item for item in details if not item["passed"]][:10],
    }
    reports = load_json(RAG_EVALUATION_REPORTS_PATH)
    RAG_EVALUATION_REPORTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_json(RAG_EVALUATION_REPORTS_PATH, [report, *reports[: MAX_SAVED_REPORTS - 1]])
    return report


def latest_evaluation_report() -> dict[str, Any] | None:
    reports = load_json(RAG_EVALUATION_REPORTS_PATH)
    return reports[0] if reports and isinstance(reports[0], dict) else None
