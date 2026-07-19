import json
from pathlib import Path
from unittest.mock import patch

import gui_knowledge
import rag_evaluation_store


def _configure_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(rag_evaluation_store, "RAG_EVALUATION_CASES_PATH", tmp_path / "cases.json")
    monkeypatch.setattr(rag_evaluation_store, "RAG_EVALUATION_REPORTS_PATH", tmp_path / "reports.json")


def test_import_and_run_evaluation_records_retrieval_metrics(tmp_path, monkeypatch):
    _configure_paths(tmp_path, monkeypatch)
    source = tmp_path / "evaluation.json"
    source.write_text(json.dumps([
        {"query": "奖学金申请", "expected_sources": ["student.md"], "expected_keywords": ["材料"]},
        {"query": "课程安排", "expected_sources": ["course.md"]},
    ], ensure_ascii=False), encoding="utf-8")
    rag_evaluation_store.import_evaluation_cases(source)

    def fake_diagnose(query, **kwargs):
        if query == "奖学金申请":
            return {"results": [{"source": "student.md", "text": "申请材料清单"}]}
        return {"results": [{"source": "other.md", "text": "无关资料"}]}

    with patch.object(rag_evaluation_store, "diagnose_knowledge_search", side_effect=fake_diagnose):
        report = rag_evaluation_store.run_evaluation(top_k=4, threshold=0.35, candidate_multiplier=3)

    assert report["passed_cases"] == 1
    assert report["source_recall_at_k"] == 50.0
    assert report["keyword_coverage"] == 100.0
    assert report["mrr"] == 0.5
    assert report["failures"][0]["query"] == "课程安排"
    assert rag_evaluation_store.latest_evaluation_report()["pass_rate"] == 50.0


def test_evaluation_case_requires_expected_ground_truth(tmp_path, monkeypatch):
    _configure_paths(tmp_path, monkeypatch)
    source = tmp_path / "invalid.json"
    source.write_text('[{"query": "only question"}]', encoding="utf-8")

    try:
        rag_evaluation_store.import_evaluation_cases(source)
    except ValueError as exc:
        assert "expected_sources" in str(exc)
    else:
        raise AssertionError("Expected missing ground truth to be rejected")


def test_evaluation_error_messages_are_fully_english_without_mixed_fragments(tmp_path, monkeypatch):
    _configure_paths(tmp_path, monkeypatch)
    source = tmp_path / "invalid.json"
    source.write_text('[{"query": "only question"}]', encoding="utf-8")

    message = gui_knowledge.import_rag_evaluation_cases(str(source), "en")

    # 通用短语 " 条"→" entries" 曾把这类消息错位替换成中英混杂的乱码。
    assert message.startswith("Failed to import the RAG evaluation set: Sample 1")
    assert "entries" not in message
    assert "条" not in message


def test_latest_report_with_corrupt_file_does_not_crash_startup(tmp_path, monkeypatch):
    _configure_paths(tmp_path, monkeypatch)
    rag_evaluation_store.RAG_EVALUATION_REPORTS_PATH.write_text(
        '[{"broken": true}]', encoding="utf-8"
    )

    # 该函数是报告框的启动初始值，损坏的报告文件绝不能抛异常。
    zh = gui_knowledge.latest_rag_evaluation_report("zh-CN")
    en = gui_knowledge.latest_rag_evaluation_report("en")

    assert zh.startswith("读取 RAG 评估报告失败：")
    assert en.startswith("Failed to read the RAG evaluation report:")


def test_evaluation_report_is_localized_to_english():
    report = {
        "created_at": "2026-07-19T10:00:00",
        "settings": {"top_k": 4, "threshold": 0.35, "candidate_multiplier": 3},
        "passed_cases": 1, "total_cases": 2, "pass_rate": 50.0,
        "source_hits": 1, "source_cases": 2, "source_recall_at_k": 50.0,
        "mrr": 0.5, "keyword_hits": 1, "keyword_cases": 1,
        "keyword_coverage": 100.0, "failures": [],
    }

    text = gui_knowledge.format_rag_evaluation_report(report, "en")

    assert "Overall pass rate: 1/2 (50.0%)" in text
    assert "Source recall@K" in text
