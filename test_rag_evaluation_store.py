import json
from pathlib import Path
from unittest.mock import patch

import gui_knowledge
import rag_evaluation_store


def test_bundled_regression_set_includes_hard_multilingual_and_near_domain_cases():
    cases_path = Path(__file__).parent / "knowledge_base" / "rag_evaluation_cases.json"
    cases = rag_evaluation_store.normalize_cases(json.loads(cases_path.read_text(encoding="utf-8")))

    categories = {case["category"] for case in cases}
    ids = {case["id"] for case in cases}

    assert len(cases) >= 48
    assert "hard-multilingual" in categories
    assert "hard-near-domain-insufficient" in categories
    assert {"hard-ai-supervised-english", "hard-reject-therapy-diagnosis"} <= ids


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


def test_evaluation_supports_insufficient_evidence_cases(tmp_path, monkeypatch):
    _configure_paths(tmp_path, monkeypatch)
    source = tmp_path / "evaluation.json"
    source.write_text('[{"id":"out-of-domain","category":"insufficient","query":"火星天气怎么样？","expected_outcome":"insufficient"}]', encoding="utf-8")
    rag_evaluation_store.import_evaluation_cases(source)

    with patch.object(rag_evaluation_store, "diagnose_knowledge_search", return_value={"results": []}):
        report = rag_evaluation_store.run_evaluation(top_k=4, threshold=0.35, candidate_multiplier=3)

    assert report["passed_cases"] == 1
    assert report["insufficient_cases"] == 1
    assert report["insufficient_passes"] == 1
    assert report["insufficient_refusal_rate"] == 100.0


def test_duplicate_case_ids_are_rejected(tmp_path, monkeypatch):
    _configure_paths(tmp_path, monkeypatch)
    source = tmp_path / "duplicate.json"
    source.write_text(
        '[{"id":"same","query":"A","expected_keywords":["a"]},'
        '{"id":"same","query":"B","expected_keywords":["b"]}]',
        encoding="utf-8",
    )

    try:
        rag_evaluation_store.import_evaluation_cases(source)
    except ValueError as exc:
        assert "重复的样本 id" in str(exc)
    else:
        raise AssertionError("Expected duplicate case ids to be rejected")


def test_insufficient_case_fails_when_results_carry_usable_text(tmp_path, monkeypatch):
    _configure_paths(tmp_path, monkeypatch)
    source = tmp_path / "evaluation.json"
    source.write_text('[{"id":"out-of-domain","query":"火星天气怎么样？","expected_outcome":"insufficient"}]', encoding="utf-8")
    rag_evaluation_store.import_evaluation_cases(source)

    results = {"results": [{"source": "a.txt", "text": "grounded text", "score": 0.9}]}
    with patch.object(rag_evaluation_store, "diagnose_knowledge_search", return_value=results):
        report = rag_evaluation_store.run_evaluation(top_k=4, threshold=0.35, candidate_multiplier=3)

    assert report["insufficient_passes"] == 0
    assert report["insufficient_refusal_rate"] == 0.0


def test_evaluation_compares_vector_and_hybrid_metrics(tmp_path, monkeypatch):
    _configure_paths(tmp_path, monkeypatch)
    source = tmp_path / "evaluation.json"
    source.write_text(json.dumps([
        {"id": "grounded", "query": "课程安排", "expected_sources": ["course.md"]},
        {"id": "insufficient", "query": "火星天气", "expected_outcome": "insufficient"},
    ], ensure_ascii=False), encoding="utf-8")
    rag_evaluation_store.import_evaluation_cases(source)

    def fake_diagnose(query, *, retrieval_mode, **_kwargs):
        if query == "课程安排":
            source_name = "course.md" if retrieval_mode == "hybrid_rrf" else "other.md"
            return {"results": [{"source": source_name, "text": "课程材料"}]}
        return {"results": [] if retrieval_mode == "hybrid_rrf" else [{"source": "other.md", "text": "unrelated"}]}

    with patch.object(rag_evaluation_store, "diagnose_knowledge_search", side_effect=fake_diagnose):
        report = rag_evaluation_store.run_evaluation(
            top_k=4, threshold=0.35, candidate_multiplier=3,
            retrieval_mode="hybrid_rrf", compare_modes=True,
        )

    comparison = report["mode_comparison"]
    assert report["settings"]["retrieval_mode"] == "hybrid_rrf"
    assert report["source_recall_at_k"] == 100.0
    assert comparison["vector"]["source_recall_at_k"] == 0.0
    assert comparison["hybrid_rrf"]["mrr"] == 1.0
    assert comparison["hybrid_rrf"]["insufficient_refusal_rate"] == 100.0
    assert comparison["delta_hybrid_minus_vector"]["pass_rate"] == 100.0


def test_comparison_delta_is_undefined_when_a_metric_has_no_denominator(tmp_path, monkeypatch):
    """A metric with no cases has no delta; reporting 0 would read as "no change"."""
    _configure_paths(tmp_path, monkeypatch)
    source = tmp_path / "evaluation.json"
    source.write_text(json.dumps([
        {"id": "grounded", "query": "课程安排", "expected_keywords": ["课程"]},
    ], ensure_ascii=False), encoding="utf-8")
    rag_evaluation_store.import_evaluation_cases(source)

    results = {"results": [{"source": "course.md", "text": "课程材料"}]}
    with patch.object(rag_evaluation_store, "diagnose_knowledge_search", return_value=results):
        report = rag_evaluation_store.run_evaluation(
            top_k=4, threshold=0.35, candidate_multiplier=3, compare_modes=True,
        )

    # No expected_sources and no insufficient cases, so both metrics are undefined.
    assert report["mode_comparison"]["delta_hybrid_minus_vector"]["source_recall_at_k"] is None
    assert report["mode_comparison"]["delta_hybrid_minus_vector"]["insufficient_refusal_rate"] is None
    assert report["mode_comparison"]["delta_hybrid_minus_vector"]["pass_rate"] == 0.0


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


def test_evaluation_report_shows_hybrid_comparison():
    report = {
        "created_at": "2026-07-31T10:00:00",
        "settings": {"top_k": 4, "threshold": 0.35, "candidate_multiplier": 3, "retrieval_mode": "hybrid_rrf"},
        "passed_cases": 2, "total_cases": 2, "pass_rate": 100.0,
        "source_hits": 1, "source_cases": 1, "source_recall_at_k": 100.0,
        "mrr": 1.0, "keyword_hits": 0, "keyword_cases": 0,
        "keyword_coverage": None, "failures": [],
        "mode_comparison": {
            "vector": {"source_recall_at_k": 50.0, "mrr": 0.5, "insufficient_refusal_rate": 0.0},
            "hybrid_rrf": {"source_recall_at_k": 100.0, "mrr": 1.0, "insufficient_refusal_rate": 100.0},
        },
    }

    text = gui_knowledge.format_rag_evaluation_report(report, "en")

    assert "Mode comparison (vector -> hybrid RRF)" in text
    assert "Recall@K 50.0% -> 100.0%" in text


def test_comparison_line_shows_not_available_instead_of_none():
    report = {
        "created_at": "2026-07-31T10:00:00",
        "settings": {"top_k": 4, "threshold": 0.35, "candidate_multiplier": 3, "retrieval_mode": "hybrid_rrf"},
        "passed_cases": 1, "total_cases": 1, "pass_rate": 100.0,
        "source_hits": 0, "source_cases": 0, "source_recall_at_k": None,
        "mrr": None, "keyword_hits": 1, "keyword_cases": 1,
        "keyword_coverage": 100.0, "failures": [],
        "mode_comparison": {
            "vector": {"source_recall_at_k": None, "mrr": None, "insufficient_refusal_rate": None},
            "hybrid_rrf": {"source_recall_at_k": None, "mrr": None, "insufficient_refusal_rate": None},
        },
    }

    text = gui_knowledge.format_rag_evaluation_report(report, "en")

    assert "None" not in text
    assert "Recall@K n/a% -> n/a%" in text
