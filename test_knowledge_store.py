import tempfile
from pathlib import Path
from unittest.mock import patch

import knowledge_store
from prompt_builder import build_messages
from session_store import SessionState


def test_split_text_keeps_overlap_and_non_empty_chunks():
    text = "第一段内容。" * 80
    chunks = knowledge_store.split_text(text, chunk_size=120, overlap=20)

    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)
    assert all(len(chunk) <= 140 for chunk in chunks)


def test_import_documents_copies_and_builds_chunks():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        docs_dir = base / "documents"
        chunks_path = base / "chunks.json"
        index_path = base / "knowledge.index"
        source = base / "source.md"
        source.write_text("# 标题\n\n这是一份用于 RAG 的测试资料。" * 20, encoding="utf-8")

        with patch.object(knowledge_store, "KNOWLEDGE_DIR", base), \
             patch.object(knowledge_store, "KNOWLEDGE_DOCS_DIR", docs_dir), \
             patch.object(knowledge_store, "KNOWLEDGE_CHUNKS_PATH", chunks_path), \
             patch.object(knowledge_store, "KNOWLEDGE_INDEX_PATH", index_path), \
             patch.object(knowledge_store, "_write_index") as write_index:
            result = knowledge_store.import_documents([source])
            chunks = knowledge_store.load_chunks()

        assert result["imported"] == ["source.md"]
        assert result["documents"] == 1
        assert result["chunks"] == len(chunks)
        assert chunks[0]["source"] == "source.md"
        assert "测试资料" in chunks[0]["text"]
        write_index.assert_called_once()


def test_knowledge_status_reports_counts():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        docs_dir = base / "documents"
        chunks_path = base / "chunks.json"
        index_path = base / "knowledge.index"
        docs_dir.mkdir(parents=True)
        (docs_dir / "a.txt").write_text("hello", encoding="utf-8")
        chunks_path.write_text('[{"source": "a.txt", "text": "hello"}]', encoding="utf-8")
        index_path.write_bytes(b"index")

        with patch.object(knowledge_store, "KNOWLEDGE_DIR", base), \
             patch.object(knowledge_store, "KNOWLEDGE_DOCS_DIR", docs_dir), \
             patch.object(knowledge_store, "KNOWLEDGE_CHUNKS_PATH", chunks_path), \
             patch.object(knowledge_store, "KNOWLEDGE_INDEX_PATH", index_path):
            status = knowledge_store.knowledge_status()

    assert "文档数：1" in status
    assert "片段数：1" in status
    assert "索引状态：已建立" in status


def test_delete_document_removes_only_selected_source_and_rebuilds():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        docs_dir = base / "documents"
        chunks_path = base / "chunks.json"
        index_path = base / "knowledge.index"
        docs_dir.mkdir(parents=True)
        (docs_dir / "keep.txt").write_text("保留", encoding="utf-8")
        (docs_dir / "remove.txt").write_text("删除", encoding="utf-8")

        with patch.object(knowledge_store, "KNOWLEDGE_DIR", base), \
             patch.object(knowledge_store, "KNOWLEDGE_DOCS_DIR", docs_dir), \
             patch.object(knowledge_store, "KNOWLEDGE_CHUNKS_PATH", chunks_path), \
             patch.object(knowledge_store, "KNOWLEDGE_INDEX_PATH", index_path), \
             patch.object(knowledge_store, "_write_index"):
            result = knowledge_store.delete_document("remove.txt")

        assert result["deleted"] == "remove.txt"
        assert (docs_dir / "keep.txt").exists()
        assert not (docs_dir / "remove.txt").exists()


def test_delete_document_rejects_path_traversal():
    with tempfile.TemporaryDirectory() as tmp:
        docs_dir = Path(tmp) / "documents"
        docs_dir.mkdir(parents=True)
        with patch.object(knowledge_store, "KNOWLEDGE_DOCS_DIR", docs_dir):
            try:
                knowledge_store.delete_document("../outside.txt")
            except ValueError as exc:
                assert "有效文档" in str(exc)
            else:
                raise AssertionError("Expected invalid document name to be rejected")


def test_build_messages_includes_knowledge_context():
    state = SessionState(user_id="default")
    messages = build_messages(
        state,
        "这个项目支持什么？",
        "neutral",
        0.1,
        knowledge_context="[资料 1 | readme.md]\n项目支持 RAG 知识库。",
    )

    assert "知识库参考资料" in messages[0]["content"]
    assert "项目支持 RAG 知识库" in messages[0]["content"]


def test_diagnose_search_filters_threshold_duplicates_and_preserves_diversity():
    chunks = [
        {"source": "a.txt", "text": "重复内容", "chunk_index": 0},
        {"source": "a.txt", "text": "重复内容", "chunk_index": 1},
        {"source": "a.txt", "text": "来源 A 的补充", "chunk_index": 2},
        {"source": "b.txt", "text": "来源 B 的内容", "chunk_index": 0},
        {"source": "c.txt", "text": "低相关内容", "chunk_index": 0},
    ]
    candidates = []
    for score, index in [(0.95, 0), (0.94, 1), (0.90, 2), (0.89, 3), (0.10, 4)]:
        item = dict(chunks[index])
        item.update({"score": score, "_index": index})
        candidates.append(item)

    with patch.object(knowledge_store, "load_chunks", return_value=chunks), \
         patch.object(knowledge_store, "_search_candidates", return_value=candidates):
        diagnostic = knowledge_store.diagnose_knowledge_search(
            "测试",
            top_k=3,
            threshold=0.3,
            max_per_source=1,
        )

    assert [item["source"] for item in diagnostic["results"]] == ["a.txt", "b.txt", "a.txt"]
    assert [item["text"] for item in diagnostic["results"]].count("重复内容") == 1
    decisions = [item["decision"] for item in diagnostic["candidates"]]
    assert "内容重复" in decisions
    assert "低于阈值 0.30" in decisions


def test_hybrid_rrf_promotes_chunks_present_in_both_rankings():
    chunks = [
        {"source": "vector-only.md", "text": "vector result", "chunk_index": 0},
        {"source": "hybrid-hit.md", "text": "transformer self attention", "chunk_index": 0},
    ]
    vector_candidates = [
        {**chunks[0], "score": 0.91, "_index": 0},
        {**chunks[1], "score": 0.82, "_index": 1},
    ]
    bm25_candidates = [{**chunks[1], "bm25_score": 2.4, "_index": 1}]

    with patch.object(knowledge_store, "_search_candidates", return_value=vector_candidates), \
         patch.object(knowledge_store, "_bm25_search_candidates", return_value=bm25_candidates):
        candidates = knowledge_store._hybrid_search_candidates(
            "transformer attention", chunks, top_k=2, candidate_multiplier=2,
        )

    assert [item["source"] for item in candidates] == ["hybrid-hit.md", "vector-only.md"]
    assert candidates[0]["vector_score"] == 0.82
    assert candidates[0]["bm25_score"] == 2.4
    assert candidates[0]["rrf_score"] > candidates[1]["rrf_score"]


def test_hybrid_bm25_only_match_does_not_bypass_evidence_threshold():
    chunks = [{"source": "term-only.md", "text": "common term", "chunk_index": 0}]
    candidates = [{
        **chunks[0], "score": 0.0, "vector_score": None,
        "bm25_score": 3.0, "rrf_score": 1 / 61, "_index": 0,
    }]

    with patch.object(knowledge_store, "load_chunks", return_value=chunks), \
         patch.object(knowledge_store, "_hybrid_search_candidates", return_value=candidates):
        diagnostic = knowledge_store.diagnose_knowledge_search(
            "common term", top_k=1, threshold=0.35, retrieval_mode="hybrid_rrf",
        )

    assert diagnostic["results"] == []
    assert diagnostic["candidates"][0]["decision"] == "低于阈值 0.35"


def test_build_context_enforces_character_budget_and_shows_scores():
    results = [
        {"source": "a.txt", "text": "甲" * 300, "score": 0.8},
        {"source": "b.txt", "text": "乙" * 300, "score": 0.7},
    ]
    with patch.object(knowledge_store, "retrieve_knowledge", return_value=results):
        context = knowledge_store.build_knowledge_context("问题", max_context_chars=200)

    assert len(context) <= 200
    assert "相关度 0.800" in context


def test_knowledge_bundle_only_cites_chunks_that_fit_the_context_budget():
    results = [
        {"source": "a.txt", "chunk_index": 2, "text": "A" * 300, "score": 0.8},
        {"source": "b.txt", "chunk_index": 3, "text": "B" * 300, "score": 0.7},
    ]
    with patch.object(knowledge_store, "retrieve_knowledge", return_value=results):
        bundle = knowledge_store.build_knowledge_bundle("question", max_context_chars=200)

    assert bundle["citations"]
    assert bundle["citations"][0]["source"] == "a.txt"
    assert bundle["citations"][0]["chunk_index"] == 2
    assert len(bundle["context"]) <= 200


def test_results_without_usable_text_are_not_treated_as_evidence():
    # The evaluation set and the chat path must agree, so both ask this predicate.
    assert knowledge_store.has_usable_evidence([]) is False
    assert knowledge_store.has_usable_evidence([{"source": "a.txt", "text": "  ", "score": 0.9}]) is False
    assert knowledge_store.has_usable_evidence([{"source": "a.txt", "text": "grounded", "score": 0.9}]) is True


def test_bundle_reports_unusable_excerpts_separately_from_missing_matches():
    with patch.object(knowledge_store, "retrieve_knowledge", return_value=[{"source": "a.txt", "text": "", "score": 0.9}]):
        bundle = knowledge_store.build_knowledge_bundle("question")

    assert bundle["evidence"]["status"] == "insufficient"
    assert bundle["evidence"]["reason"] == "no_usable_excerpt"
    assert bundle["evidence"]["matched_chunks"] == 1


def test_build_knowledge_bundle_marks_missing_evidence_structurally():
    with patch.object(knowledge_store, "retrieve_knowledge", return_value=[]), \
         patch.object(knowledge_store, "load_chunks", return_value=[]):
        bundle = knowledge_store.build_knowledge_bundle("question")

    assert bundle["context"] == ""
    assert bundle["citations"] == []
    assert bundle["evidence"] == {
        "status": "insufficient",
        "code": "insufficient_evidence",
        "reason": "knowledge_base_empty",
        "matched_chunks": 0,
        "matched_sources": 0,
        "retrieval_threshold": knowledge_store.KNOWLEDGE_RETRIEVAL_THRESHOLD,
    }


def test_quality_report_detects_duplicate_short_and_stale_index():
    chunks = [
        {"source": "a.txt", "text": "短"},
        {"source": "a.txt", "text": "短"},
    ]
    fake_index = type("FakeIndex", (), {"ntotal": 1})()
    with tempfile.TemporaryDirectory() as tmp:
        index_path = Path(tmp) / "knowledge.index"
        index_path.write_bytes(b"index")
        with patch.object(knowledge_store, "load_chunks", return_value=chunks), \
             patch.object(knowledge_store, "list_documents", return_value=["a.txt"]), \
             patch.object(knowledge_store, "KNOWLEDGE_INDEX_PATH", index_path), \
             patch.object(knowledge_store, "require_faiss") as require_faiss:
            require_faiss.return_value.read_index.return_value = fake_index
            report = knowledge_store.assess_knowledge_quality()

    assert report["level"] == "需关注"
    assert report["duplicate_chunks"] == 1
    assert report["short_chunks"] == 2
    assert not report["index_consistent"]


def test_read_index_rebuilds_when_vector_count_is_stale():
    chunks = [{"source": "a.txt", "text": "有效内容"}]
    stale_index = type("FakeIndex", (), {"ntotal": 0, "d": 3})()
    healthy_index = type("FakeIndex", (), {"ntotal": 1, "d": 3})()

    with tempfile.TemporaryDirectory() as tmp:
        index_path = Path(tmp) / "knowledge.index"
        index_path.write_bytes(b"index")
        with patch.object(knowledge_store, "KNOWLEDGE_INDEX_PATH", index_path), \
             patch.object(knowledge_store, "require_faiss") as require_faiss, \
             patch.object(knowledge_store, "get_embedding_dimension", return_value=3), \
             patch.object(knowledge_store, "_write_index") as write_index:
            require_faiss.return_value.read_index.side_effect = [stale_index, healthy_index]
            result = knowledge_store._read_index(chunks)

    assert result is healthy_index
    write_index.assert_called_once_with(chunks)


def test_release_gate_restores_candidate_when_evaluation_misses_thresholds(tmp_path):
    report = {"total_cases": 2, "pass_rate": 50.0, "source_cases": 1, "source_recall_at_k": 0.0, "keyword_cases": 0}
    restored = []
    with patch.object(knowledge_store, "KNOWLEDGE_DIR", tmp_path), \
         patch.object(knowledge_store, "RAG_RELEASE_GATE_ENABLED", True), \
         patch.object(knowledge_store, "RAG_RELEASE_MIN_CASES", 3), \
         patch.object(knowledge_store, "_snapshot_release_state", return_value=[]), \
         patch.object(knowledge_store, "_restore_release_state", side_effect=lambda *_: restored.append(True)), \
         patch.object(knowledge_store, "rebuild_knowledge_index", return_value={"documents": 1, "chunks": 2, "errors": []}), \
         patch("rag_evaluation_store.run_evaluation", return_value=report):
        with patch.object(knowledge_store, "ensure_knowledge_dirs"):
            try:
                knowledge_store.rebuild_with_release_gate()
            except knowledge_store.ReleaseGateRejected as exc:
                assert "未发布" in str(exc)
            else:
                raise AssertionError("Expected the release gate to reject the candidate")

        status = knowledge_store.release_gate_status()
        assert status["state"] == "rejected"
        assert status["report"]["pass_rate"] == 50.0

    assert restored == [True]


def test_release_gate_rejects_low_insufficient_evidence_refusal_rate(monkeypatch):
    report = {
        "total_cases": 10, "pass_rate": 100.0, "source_cases": 0,
        "keyword_cases": 0, "insufficient_cases": 4,
        "insufficient_refusal_rate": 50.0,
    }
    monkeypatch.setattr(knowledge_store, "RAG_RELEASE_MIN_CASES", 5)
    monkeypatch.setattr(knowledge_store, "RAG_RELEASE_MIN_PASS_RATE", 80.0)
    monkeypatch.setattr(knowledge_store, "RAG_RELEASE_MIN_INSUFFICIENT_REFUSAL", 80.0)

    assert knowledge_store._evaluate_release_gate(report) == ["资料不足拒答率低于 80%。"]
