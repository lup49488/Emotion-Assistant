from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from config import (
    KNOWLEDGE_CHUNK_OVERLAP,
    KNOWLEDGE_CHUNK_SIZE,
    KNOWLEDGE_CANDIDATE_MULTIPLIER,
    KNOWLEDGE_CHUNKS_PATH,
    KNOWLEDGE_DIR,
    KNOWLEDGE_DOCS_DIR,
    KNOWLEDGE_INDEX_PATH,
    KNOWLEDGE_MAX_CONTEXT_CHARS,
    KNOWLEDGE_MAX_PER_SOURCE,
    KNOWLEDGE_MIN_CHUNK_CHARS,
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


def list_document_details() -> list[dict[str, Any]]:
    """Return stored-document metadata for management views without reading file contents."""
    chunks_by_source: dict[str, int] = {}
    for chunk in load_chunks():
        source = str(chunk.get("source", "")).strip()
        if source:
            chunks_by_source[source] = chunks_by_source.get(source, 0) + 1

    details: list[dict[str, Any]] = []
    for name in list_documents():
        path = KNOWLEDGE_DOCS_DIR / name
        stat = path.stat()
        details.append({
            "name": name,
            "chunks": chunks_by_source.get(name, 0),
            "size_bytes": stat.st_size,
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        })
    return details


def _stored_document_path(name: str) -> Path:
    clean_name = (name or "").strip()
    if not clean_name or Path(clean_name).name != clean_name:
        raise ValueError("请选择知识库中的有效文档。")
    path = KNOWLEDGE_DOCS_DIR / clean_name
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"知识库中不存在该文档：{clean_name}")
    return path


def delete_document(name: str) -> dict[str, Any]:
    """Delete one stored source document and rebuild derived RAG data."""
    ensure_knowledge_dirs()
    path = _stored_document_path(name)
    path.unlink()
    result = rebuild_knowledge_index()
    result["deleted"] = path.name
    return result


def clear_documents() -> dict[str, Any]:
    """Remove all supported RAG source documents and rebuild an empty index."""
    ensure_knowledge_dirs()
    removed: list[str] = []
    for path in KNOWLEDGE_DOCS_DIR.iterdir():
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            path.unlink()
            removed.append(path.name)
    result = rebuild_knowledge_index()
    result["removed"] = sorted(removed)
    return result


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
    try:
        index = faiss.read_index(str(KNOWLEDGE_INDEX_PATH))
    except Exception:
        _write_index(chunks)
        index = faiss.read_index(str(KNOWLEDGE_INDEX_PATH))
    if int(index.ntotal) != len(chunks) or int(index.d) != get_embedding_dimension():
        _write_index(chunks)
        index = faiss.read_index(str(KNOWLEDGE_INDEX_PATH))
    return index


def _normalized_chunk_key(text: str) -> str:
    return re.sub(r"\s+", "", text).casefold()


def _search_candidates(
    query: str,
    chunks: list[dict[str, Any]],
    *,
    top_k: int,
    candidate_multiplier: int,
) -> list[dict[str, Any]]:
    index = _read_index(chunks)
    candidate_count = min(len(chunks), max(top_k, top_k * max(1, int(candidate_multiplier))))
    similarities, indices = index.search(encode_texts([query]), candidate_count)
    candidates: list[dict[str, Any]] = []
    for score, i in zip(similarities[0], indices[0]):
        if i < 0 or i >= len(chunks):
            continue
        item = dict(chunks[int(i)])
        item["score"] = float(score)
        item["_index"] = int(i)
        candidates.append(item)
    return candidates


def diagnose_knowledge_search(
    query: str,
    *,
    top_k: int = KNOWLEDGE_TOP_K,
    threshold: float = KNOWLEDGE_RETRIEVAL_THRESHOLD,
    candidate_multiplier: int = KNOWLEDGE_CANDIDATE_MULTIPLIER,
    max_per_source: int = KNOWLEDGE_MAX_PER_SOURCE,
) -> dict[str, Any]:
    chunks = load_chunks()
    query = (query or "").strip()
    if not query or not chunks:
        return {"query": query, "results": [], "candidates": [], "reason": "查询或知识库为空"}

    top_k = max(1, min(int(top_k), len(chunks)))
    threshold = max(-1.0, min(float(threshold), 1.0))
    max_per_source = max(1, int(max_per_source))
    candidates = _search_candidates(
        query,
        chunks,
        top_k=top_k,
        candidate_multiplier=candidate_multiplier,
    )

    eligible: list[dict[str, Any]] = []
    seen_texts: set[str] = set()
    for item in candidates:
        score = float(item["score"])
        key = _normalized_chunk_key(str(item.get("text", "")))
        if score < threshold:
            item["accepted"] = False
            item["decision"] = f"低于阈值 {threshold:.2f}"
        elif not key:
            item["accepted"] = False
            item["decision"] = "空片段"
        elif key in seen_texts:
            item["accepted"] = False
            item["decision"] = "内容重复"
        else:
            seen_texts.add(key)
            eligible.append(item)

    selected: list[dict[str, Any]] = []
    deferred: list[dict[str, Any]] = []
    source_counts: Counter[str] = Counter()
    for item in eligible:
        source = str(item.get("source", "unknown"))
        if len(selected) >= top_k:
            deferred.append(item)
        elif source_counts[source] >= max_per_source:
            deferred.append(item)
        else:
            selected.append(item)
            source_counts[source] += 1

    for item in deferred:
        if len(selected) >= top_k:
            break
        selected.append(item)

    selected_indices = {int(item["_index"]) for item in selected}
    for item in eligible:
        if int(item["_index"]) in selected_indices:
            item["accepted"] = True
            item["decision"] = "采用"
        else:
            item["accepted"] = False
            item["decision"] = "超出 Top K"

    public_results = [{key: value for key, value in item.items() if key != "_index"} for item in selected]
    public_candidates = [{key: value for key, value in item.items() if key != "_index"} for item in candidates]
    return {
        "query": query,
        "results": public_results,
        "candidates": public_candidates,
        "threshold": threshold,
        "top_k": top_k,
    }


def retrieve_knowledge(
    query: str,
    *,
    top_k: int = KNOWLEDGE_TOP_K,
    threshold: float = KNOWLEDGE_RETRIEVAL_THRESHOLD,
    candidate_multiplier: int = KNOWLEDGE_CANDIDATE_MULTIPLIER,
    max_per_source: int = KNOWLEDGE_MAX_PER_SOURCE,
) -> list[dict[str, Any]]:
    return diagnose_knowledge_search(
        query,
        top_k=top_k,
        threshold=threshold,
        candidate_multiplier=candidate_multiplier,
        max_per_source=max_per_source,
    )["results"]


def build_knowledge_context(
    query: str,
    *,
    top_k: int = KNOWLEDGE_TOP_K,
    threshold: float = KNOWLEDGE_RETRIEVAL_THRESHOLD,
    max_context_chars: int = KNOWLEDGE_MAX_CONTEXT_CHARS,
) -> str:
    results = retrieve_knowledge(query, top_k=top_k, threshold=threshold)
    if not results:
        return ""
    lines: list[str] = []
    used_chars = 0
    max_context_chars = max(200, int(max_context_chars))
    for index, item in enumerate(results, start=1):
        source = item.get("source", "unknown")
        text = str(item.get("text", "")).strip()
        score = float(item.get("score", 0.0))
        header = f"[资料 {index} | {source} | 相关度 {score:.3f}]\n"
        remaining = max_context_chars - used_chars - len(header)
        if remaining <= 0:
            break
        excerpt = text[:remaining]
        if not excerpt:
            break
        block = header + excerpt
        lines.append(block)
        used_chars += len(block) + 2
    return "\n\n".join(lines)


def assess_knowledge_quality() -> dict[str, Any]:
    chunks = load_chunks()
    documents = set(list_documents())
    lengths = [len(str(item.get("text", "")).strip()) for item in chunks]
    keys = [_normalized_chunk_key(str(item.get("text", ""))) for item in chunks]
    duplicate_chunks = sum(count - 1 for count in Counter(key for key in keys if key).values() if count > 1)
    short_chunks = sum(length < KNOWLEDGE_MIN_CHUNK_CHARS for length in lengths)
    orphan_chunks = sum(str(item.get("source", "")) not in documents for item in chunks)

    index_count: int | None = None
    index_error = ""
    if KNOWLEDGE_INDEX_PATH.exists():
        try:
            index_count = int(require_faiss().read_index(str(KNOWLEDGE_INDEX_PATH)).ntotal)
        except Exception as exc:
            index_error = str(exc)
    index_consistent = index_count == len(chunks) if index_count is not None else not chunks

    issues: list[str] = []
    if not documents:
        issues.append("知识库中还没有文档")
    elif not chunks:
        issues.append("文档未生成可检索片段，请检查文件内容或读取依赖")
    if chunks and not index_consistent:
        issues.append("向量索引与片段数量不一致，下一次检索会自动重建")
    if index_error:
        issues.append("向量索引无法读取，下一次检索会自动重建")
    if short_chunks:
        issues.append(f"有 {short_chunks} 个片段短于 {KNOWLEDGE_MIN_CHUNK_CHARS} 字符")
    if duplicate_chunks:
        issues.append(f"发现 {duplicate_chunks} 个完全重复片段，检索时会自动去重")
    if orphan_chunks:
        issues.append(f"有 {orphan_chunks} 个片段找不到来源文档")

    if not documents:
        level = "未就绪"
    elif issues:
        level = "需关注"
    else:
        level = "良好"
    return {
        "level": level,
        "documents": len(documents),
        "chunks": len(chunks),
        "average_chunk_chars": round(sum(lengths) / len(lengths), 1) if lengths else 0.0,
        "short_chunks": short_chunks,
        "duplicate_chunks": duplicate_chunks,
        "orphan_chunks": orphan_chunks,
        "index_count": index_count,
        "index_consistent": index_consistent,
        "issues": issues,
    }


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
