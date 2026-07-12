from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from config import (
    KNOWLEDGE_CHUNK_OVERLAP,
    KNOWLEDGE_CHUNK_SIZE,
    KNOWLEDGE_CHUNKS_PATH,
    KNOWLEDGE_DIR,
    KNOWLEDGE_DOCS_DIR,
    KNOWLEDGE_INDEX_PATH,
    KNOWLEDGE_RETRIEVAL_THRESHOLD,
    KNOWLEDGE_TOP_K,
)
from json_utils import load_json, save_json
from llm_providers import encode_texts, get_embedding_dimension
from memory_store import require_faiss


SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".csv", ".json", ".pdf", ".docx"}


@dataclass
class KnowledgeChunk:
    id: str
    source: str
    text: str
    chunk_index: int
    created_at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "text": self.text,
            "chunk_index": self.chunk_index,
            "created_at": self.created_at,
        }


def ensure_knowledge_dirs() -> None:
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    KNOWLEDGE_DOCS_DIR.mkdir(parents=True, exist_ok=True)


def _safe_filename(path: Path) -> str:
    name = path.name.strip() or "document.txt"
    return "".join(char if char.isalnum() or char in "._- " else "_" for char in name)


def copy_document_to_store(path: str | Path) -> Path:
    ensure_knowledge_dirs()
    source_path = Path(path)
    if not source_path.exists() or not source_path.is_file():
        raise FileNotFoundError(f"文件不存在：{source_path}")
    if source_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"暂不支持该文件格式：{source_path.suffix}")

    target = KNOWLEDGE_DOCS_DIR / _safe_filename(source_path)
    if target.exists():
        stem = target.stem
        suffix = target.suffix
        target = KNOWLEDGE_DOCS_DIR / f"{stem}_{datetime.now().strftime('%Y%m%d%H%M%S')}{suffix}"
    shutil.copy2(source_path, target)
    return target


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ModuleNotFoundError:
        try:
            from PyPDF2 import PdfReader
        except ModuleNotFoundError as exc:
            raise RuntimeError("读取 PDF 需要安装 pypdf 或 PyPDF2。") from exc

    reader = PdfReader(str(path))
    parts: list[str] = []
    for page in reader.pages:
        text = page.extract_text() or ""
        if text.strip():
            parts.append(text)
    return "\n\n".join(parts)


def _read_docx(path: Path) -> str:
    try:
        from docx import Document
    except ModuleNotFoundError as exc:
        raise RuntimeError("读取 DOCX 需要安装 python-docx。") from exc
    doc = Document(str(path))
    return "\n".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip())


def extract_text(path: str | Path) -> str:
    source_path = Path(path)
    suffix = source_path.suffix.lower()
    if suffix in {".txt", ".md", ".markdown", ".csv"}:
        return source_path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".json":
        data = json.loads(source_path.read_text(encoding="utf-8"))
        return json.dumps(data, ensure_ascii=False, indent=2)
    if suffix == ".pdf":
        return _read_pdf(source_path)
    if suffix == ".docx":
        return _read_docx(source_path)
    raise ValueError(f"暂不支持该文件格式：{suffix}")


def split_text(text: str, *, chunk_size: int = KNOWLEDGE_CHUNK_SIZE, overlap: int = KNOWLEDGE_CHUNK_OVERLAP) -> list[str]:
    normalized = "\n".join(line.strip() for line in text.splitlines())
    normalized = "\n".join(line for line in normalized.splitlines() if line)
    if not normalized:
        return []

    chunk_size = max(100, int(chunk_size))
    overlap = max(0, min(int(overlap), chunk_size // 2))
    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        end = min(start + chunk_size, len(normalized))
        window = normalized[start:end]
        if end < len(normalized):
            split_at = max(window.rfind("\n"), window.rfind("。"), window.rfind("."), window.rfind("；"), window.rfind(";"))
            if split_at > chunk_size * 0.5:
                end = start + split_at + 1
                window = normalized[start:end]
        chunk = window.strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        start = max(0, end - overlap)
    return chunks


def load_chunks() -> list[dict[str, Any]]:
    chunks = load_json(KNOWLEDGE_CHUNKS_PATH)
    return chunks if isinstance(chunks, list) else []


def save_chunks(chunks: list[dict[str, Any]]) -> None:
    ensure_knowledge_dirs()
    save_json(KNOWLEDGE_CHUNKS_PATH, chunks)


def list_documents() -> list[str]:
    ensure_knowledge_dirs()
    return sorted(path.name for path in KNOWLEDGE_DOCS_DIR.iterdir() if path.is_file())


def ingest_document(path: str | Path, *, copy_to_store: bool = True) -> list[dict[str, Any]]:
    stored_path = copy_document_to_store(path) if copy_to_store else Path(path)
    text = extract_text(stored_path)
    parts = split_text(text)
    now = datetime.now().isoformat()
    chunks = [
        KnowledgeChunk(
            id=f"{stored_path.name}#{index}",
            source=stored_path.name,
            text=part,
            chunk_index=index,
            created_at=now,
        ).as_dict()
        for index, part in enumerate(parts)
    ]
    return chunks


def rebuild_knowledge_index() -> dict[str, Any]:
    ensure_knowledge_dirs()
    all_chunks: list[dict[str, Any]] = []
    errors: list[str] = []

    for path in sorted(KNOWLEDGE_DOCS_DIR.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        try:
            all_chunks.extend(ingest_document(path, copy_to_store=False))
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")

    save_chunks(all_chunks)
    _write_index(all_chunks)
    return {"chunks": len(all_chunks), "documents": len(list_documents()), "errors": errors}


def import_documents(paths: list[str | Path]) -> dict[str, Any]:
    ensure_knowledge_dirs()
    imported: list[str] = []
    errors: list[str] = []
    for path in paths:
        try:
            target = copy_document_to_store(path)
            imported.append(target.name)
        except Exception as exc:
            errors.append(f"{Path(path).name}: {exc}")
    result = rebuild_knowledge_index()
    result["imported"] = imported
    result["errors"] = errors + result["errors"]
    return result


def _write_index(chunks: list[dict[str, Any]]) -> None:
    faiss = require_faiss()
    index = faiss.IndexFlatIP(get_embedding_dimension())
    texts = [chunk["text"] for chunk in chunks if chunk.get("text")]
    if texts:
        embeddings = encode_texts(texts)
        index.add(np.asarray(embeddings, dtype=np.float32))
    faiss.write_index(index, str(KNOWLEDGE_INDEX_PATH))


def _read_index(chunks: list[dict[str, Any]]) -> Any:
    faiss = require_faiss()
    if not KNOWLEDGE_INDEX_PATH.exists():
        _write_index(chunks)
    return faiss.read_index(str(KNOWLEDGE_INDEX_PATH))


def retrieve_knowledge(
    query: str,
    *,
    top_k: int = KNOWLEDGE_TOP_K,
    threshold: float = KNOWLEDGE_RETRIEVAL_THRESHOLD,
) -> list[dict[str, Any]]:
    chunks = load_chunks()
    if not query.strip() or not chunks:
        return []
    index = _read_index(chunks)
    top_k = max(1, min(int(top_k), len(chunks)))
    similarities, indices = index.search(encode_texts([query]), top_k)
    results: list[dict[str, Any]] = []
    for score, i in zip(similarities[0], indices[0]):
        if i < 0 or i >= len(chunks) or float(score) < threshold:
            continue
        item = dict(chunks[int(i)])
        item["score"] = float(score)
        results.append(item)
    return results


def build_knowledge_context(query: str, *, top_k: int = KNOWLEDGE_TOP_K) -> str:
    results = retrieve_knowledge(query, top_k=top_k)
    if not results:
        return ""
    lines = []
    for index, item in enumerate(results, start=1):
        source = item.get("source", "unknown")
        text = str(item.get("text", "")).strip()
        lines.append(f"[资料 {index} | {source}]\n{text}")
    return "\n\n".join(lines)


def knowledge_status() -> str:
    chunks = load_chunks()
    docs = list_documents()
    index_status = "已建立" if KNOWLEDGE_INDEX_PATH.exists() else "未建立"
    return "\n".join([
        f"文档数：{len(docs)}",
        f"片段数：{len(chunks)}",
        f"索引状态：{index_status}",
        f"目录：{KNOWLEDGE_DOCS_DIR}",
        "文档：" + (", ".join(docs[:20]) if docs else "暂无"),
    ])
