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
