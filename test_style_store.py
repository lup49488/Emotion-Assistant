import tempfile
from pathlib import Path
from unittest.mock import patch

import style_store
from prompt_builder import build_messages
from session_store import SessionState


def test_jsonl_style_file_is_converted_to_readable_samples():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "style.jsonl"
        path.write_text(
            '{"title":"温柔解释","user":"我很累","assistant":"听起来你真的很累，我们先把目标放小一点。"}\n',
            encoding="utf-8",
        )

        text = style_store.extract_text(path)

    assert "温柔解释" in text
    assert "用户：我很累" in text
    assert "助手：听起来你真的很累" in text


def test_split_style_text_prefers_markdown_sections():
    text = "# 温柔解释型\n用户：我很累\n助手：慢慢来。\n\n# 技术协作型\n用户：报错了\n助手：我们先定位报错行。"
    chunks = style_store.split_style_text(text, chunk_size=200)

    assert len(chunks) == 2
    assert chunks[0].startswith("# 温柔解释型")
    assert chunks[1].startswith("# 技术协作型")


def test_import_style_documents_copies_and_builds_chunks():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        docs_dir = base / "documents"
        chunks_path = base / "chunks.json"
        index_path = base / "style.index"
        source = base / "style.md"
        source.write_text("# 技术协作型\n用户：这个函数报错\n助手：我先帮你拆开看。", encoding="utf-8")

        with patch.object(style_store, "STYLE_DIR", base), \
             patch.object(style_store, "STYLE_DOCS_DIR", docs_dir), \
             patch.object(style_store, "STYLE_CHUNKS_PATH", chunks_path), \
             patch.object(style_store, "STYLE_INDEX_PATH", index_path), \
             patch.object(style_store, "_write_index") as write_index:
            result = style_store.import_style_documents([source])
            chunks = style_store.load_chunks()

    assert result["imported"] == ["style.md"]
    assert result["documents"] == 1
    assert result["chunks"] == len(chunks)
    assert "技术协作型" in chunks[0]["text"]
    write_index.assert_called_once()


def test_style_status_reports_counts():
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        docs_dir = base / "documents"
        chunks_path = base / "chunks.json"
        index_path = base / "style.index"
        docs_dir.mkdir(parents=True)
        (docs_dir / "style.txt").write_text("sample", encoding="utf-8")
        chunks_path.write_text('[{"source": "style.txt", "text": "sample"}]', encoding="utf-8")
        index_path.write_bytes(b"index")

        with patch.object(style_store, "STYLE_DIR", base), \
             patch.object(style_store, "STYLE_DOCS_DIR", docs_dir), \
             patch.object(style_store, "STYLE_CHUNKS_PATH", chunks_path), \
             patch.object(style_store, "STYLE_INDEX_PATH", index_path):
            status = style_store.style_status()

    assert "风格文档数：1" in status
    assert "风格片段数：1" in status
    assert "索引状态：已建立" in status


def test_build_messages_includes_style_context():
    state = SessionState(user_id="default")
    messages = build_messages(
        state,
        "帮我解释一下这个错误",
        "neutral",
        0.1,
        style_context="[风格 1 | style.md]\n助手：我先帮你拆开看。",
    )

    assert "回复风格参考" in messages[0]["content"]
    assert "我先帮你拆开看" in messages[0]["content"]
