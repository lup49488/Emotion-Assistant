from __future__ import annotations

import json
import re
import shutil
import tempfile
import threading
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
    RAG_RELEASE_GATE_ENABLED,
    RAG_RELEASE_MIN_CASES,
    RAG_RELEASE_MIN_INSUFFICIENT_REFUSAL,
    RAG_RELEASE_MIN_KEYWORD_COVERAGE,
    RAG_RELEASE_MIN_PASS_RATE,
    RAG_RELEASE_MIN_SOURCE_RECALL,
)
from json_utils import load_json, save_json
from llm_providers import encode_texts, get_embedding_dimension
from memory_store import require_faiss
from sqlite_store import (
    connection as sqlite_connection,
    list_rag_chunks,
    list_rag_documents,
    mark_rag_migration_completed,
    rag_migration_completed,
    replace_rag_chunks,
    replace_rag_documents,
    sqlite_enabled,
)


SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".csv", ".json", ".pdf", ".docx"}
RAG_COLLECTION = "knowledge"
_RELEASE_LOCK = threading.RLock()


class ReleaseGateRejected(RuntimeError):
    """Raised after a candidate index is restored because it missed the release gate."""


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


def _filesystem_document_details() -> list[dict[str, Any]]:
    ensure_knowledge_dirs()
    details: list[dict[str, Any]] = []
    for path in sorted(KNOWLEDGE_DOCS_DIR.iterdir()):
        if not path.is_file():
            continue
        stat = path.stat()
        details.append({
            "name": path.name,
            "size_bytes": stat.st_size,
            "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
            "created_at": "",
        })
    return details


def _sync_sqlite_metadata(conn: Any, chunks: list[dict[str, Any]]) -> None:
    replace_rag_chunks(conn, RAG_COLLECTION, chunks)
    replace_rag_documents(conn, RAG_COLLECTION, _filesystem_document_details())
    mark_rag_migration_completed(conn, RAG_COLLECTION)


def _migrate_legacy_metadata_unlocked(conn: Any) -> int:
    if rag_migration_completed(conn, RAG_COLLECTION):
        return 0
    chunks = load_json(KNOWLEDGE_CHUNKS_PATH)
    clean_chunks = chunks if isinstance(chunks, list) else []
    _sync_sqlite_metadata(conn, clean_chunks)
    return len(clean_chunks)


def migrate_legacy_knowledge_metadata() -> int:
    """Import the legacy chunk JSON and document metadata into SQLite once."""
    if not sqlite_enabled():
        return 0
    with sqlite_connection() as conn:
        return _migrate_legacy_metadata_unlocked(conn)


def load_chunks() -> list[dict[str, Any]]:
    if sqlite_enabled():
        with sqlite_connection() as conn:
            _migrate_legacy_metadata_unlocked(conn)
            return list_rag_chunks(conn, RAG_COLLECTION)
    chunks = load_json(KNOWLEDGE_CHUNKS_PATH)
    return chunks if isinstance(chunks, list) else []


def save_chunks(chunks: list[dict[str, Any]]) -> None:
    ensure_knowledge_dirs()
    save_json(KNOWLEDGE_CHUNKS_PATH, chunks)
    if sqlite_enabled():
        with sqlite_connection() as conn:
            _sync_sqlite_metadata(conn, chunks)


def list_documents() -> list[str]:
    ensure_knowledge_dirs()
    return sorted(path.name for path in KNOWLEDGE_DOCS_DIR.iterdir() if path.is_file())


def list_document_details() -> list[dict[str, Any]]:
    """Return stored-document metadata for management views without reading file contents."""
    if sqlite_enabled():
        load_chunks()
        with sqlite_connection() as conn:
            documents = list_rag_documents(conn, RAG_COLLECTION)
            chunks_by_source = Counter(
                str(chunk.get("source", "")) for chunk in list_rag_chunks(conn, RAG_COLLECTION)
            )
        return [
            {**item, "chunks": chunks_by_source.get(str(item["name"]), 0)}
            for item in documents
        ]

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
    path = delete_document_from_store(name)
    result = rebuild_knowledge_index()
    result["deleted"] = path.name
    return result


def delete_document_from_store(name: str) -> Path:
    """Delete one source document without rebuilding; used by gated publishing."""
    ensure_knowledge_dirs()
    path = _stored_document_path(name)
    path.unlink()
    return path


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


def _release_status_path() -> Path:
    return KNOWLEDGE_DIR / "release_status.json"


def release_gate_status() -> dict[str, Any]:
    defaults = {
        "enabled": RAG_RELEASE_GATE_ENABLED,
        "state": "not_configured" if not RAG_RELEASE_GATE_ENABLED else "awaiting_evaluation",
        "published_at": None,
        "last_attempt_at": None,
        "reason": None,
        "report": None,
        "thresholds": _release_thresholds(),
    }
    path = _release_status_path()
    if not path.exists():
        return defaults
    try:
        saved = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return defaults
    return {**defaults, **saved, "enabled": RAG_RELEASE_GATE_ENABLED, "thresholds": _release_thresholds()}


def _save_release_status(status: dict[str, Any]) -> None:
    ensure_knowledge_dirs()
    path = _release_status_path()
    temp_path = path.with_suffix(".tmp")
    temp_path.write_text(json.dumps(status, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def _release_thresholds() -> dict[str, int | float]:
    return {
        "min_cases": RAG_RELEASE_MIN_CASES,
        "min_pass_rate": RAG_RELEASE_MIN_PASS_RATE,
        "min_source_recall": RAG_RELEASE_MIN_SOURCE_RECALL,
        "min_keyword_coverage": RAG_RELEASE_MIN_KEYWORD_COVERAGE,
        "min_insufficient_refusal": RAG_RELEASE_MIN_INSUFFICIENT_REFUSAL,
    }


def _evaluate_release_gate(report: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    if int(report.get("total_cases", 0)) < RAG_RELEASE_MIN_CASES:
        failures.append(f"评估样本不足：需要至少 {RAG_RELEASE_MIN_CASES} 条。")
    if float(report.get("pass_rate") or 0) < RAG_RELEASE_MIN_PASS_RATE:
        failures.append(f"通过率低于 {RAG_RELEASE_MIN_PASS_RATE:.0f}%。")
    if int(report.get("source_cases", 0)) and float(report.get("source_recall_at_k") or 0) < RAG_RELEASE_MIN_SOURCE_RECALL:
        failures.append(f"来源召回率低于 {RAG_RELEASE_MIN_SOURCE_RECALL:.0f}%。")
    if int(report.get("keyword_cases", 0)) and float(report.get("keyword_coverage") or 0) < RAG_RELEASE_MIN_KEYWORD_COVERAGE:
        failures.append(f"关键词覆盖率低于 {RAG_RELEASE_MIN_KEYWORD_COVERAGE:.0f}%。")
    if int(report.get("insufficient_cases", 0)) and float(report.get("insufficient_refusal_rate") or 0) < RAG_RELEASE_MIN_INSUFFICIENT_REFUSAL:
        failures.append(f"资料不足拒答率低于 {RAG_RELEASE_MIN_INSUFFICIENT_REFUSAL:.0f}%。")
    return failures


def _snapshot_release_state(backup_dir: Path) -> list[dict[str, Any]]:
    backup_docs = backup_dir / "documents"
    shutil.copytree(KNOWLEDGE_DOCS_DIR, backup_docs)
    old_chunks = load_chunks()
    if KNOWLEDGE_INDEX_PATH.exists():
        shutil.copy2(KNOWLEDGE_INDEX_PATH, backup_dir / "knowledge.index")
    return old_chunks


def _restore_release_state(backup_dir: Path, old_chunks: list[dict[str, Any]]) -> None:
    backup_docs = backup_dir / "documents"
    if KNOWLEDGE_DOCS_DIR.exists():
        shutil.rmtree(KNOWLEDGE_DOCS_DIR)
    shutil.copytree(backup_docs, KNOWLEDGE_DOCS_DIR)
    save_chunks(old_chunks)
    backup_index = backup_dir / "knowledge.index"
    if backup_index.exists():
        shutil.copy2(backup_index, KNOWLEDGE_INDEX_PATH)
    else:
        KNOWLEDGE_INDEX_PATH.unlink(missing_ok=True)


def rebuild_with_release_gate(
    *, mutate: Any | None = None, top_k: int = KNOWLEDGE_TOP_K,
    threshold: float = KNOWLEDGE_RETRIEVAL_THRESHOLD,
    candidate_multiplier: int = KNOWLEDGE_CANDIDATE_MULTIPLIER,
) -> dict[str, Any]:
    """Publish a candidate rebuild only after the configured retrieval evaluation passes."""
    if not RAG_RELEASE_GATE_ENABLED:
        mutation_result = mutate() if mutate else None
        result = rebuild_knowledge_index()
        if isinstance(mutation_result, Path):
            result["document"] = mutation_result.name
        return {**result, "published": True, "gate": {"enabled": False}}

    ensure_knowledge_dirs()
    with _RELEASE_LOCK, tempfile.TemporaryDirectory(prefix="rag-release-", dir=KNOWLEDGE_DIR) as temp_dir:
        backup_dir = Path(temp_dir)
        old_chunks = _snapshot_release_state(backup_dir)
        attempted_at = datetime.now().isoformat(timespec="seconds")
        try:
            mutation_result = mutate() if mutate else None
            result = rebuild_knowledge_index()
            from rag_evaluation_store import run_evaluation

            report = run_evaluation(
                top_k=int(top_k), threshold=float(threshold),
                candidate_multiplier=int(candidate_multiplier),
            )
            reasons = _evaluate_release_gate(report)
            if reasons:
                _restore_release_state(backup_dir, old_chunks)
                status = {
                    "state": "rejected", "published_at": release_gate_status().get("published_at"),
                    "last_attempt_at": attempted_at, "reason": " ".join(reasons),
                    "report": report,
                }
                _save_release_status(status)
                raise ReleaseGateRejected("知识库候选版本未发布：" + " ".join(reasons))
            status = {
                "state": "published", "published_at": datetime.now().isoformat(timespec="seconds"),
                "last_attempt_at": attempted_at, "reason": None, "report": report,
            }
            _save_release_status(status)
            if isinstance(mutation_result, Path):
                result["document"] = mutation_result.name
            return {**result, "published": True, "gate": status}
        except ReleaseGateRejected:
            raise
        except Exception:
            _restore_release_state(backup_dir, old_chunks)
            raise


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
    return build_knowledge_bundle(
        query, top_k=top_k, threshold=threshold, max_context_chars=max_context_chars,
    )["context"]


def _collect_context_blocks(
    results: list[dict[str, Any]], max_context_chars: int
) -> tuple[list[str], list[dict[str, Any]]]:
    """Fit retrieved chunks into the prompt budget, keeping citations in step."""
    lines: list[str] = []
    citations: list[dict[str, Any]] = []
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
        citations.append({
            "source": str(source),
            "chunk_index": int(item.get("chunk_index", index - 1)),
            "score": round(score, 3),
            "excerpt": excerpt[:280],
        })
        used_chars += len(block) + 2
    return lines, citations


def has_usable_evidence(
    results: list[dict[str, Any]], max_context_chars: int = KNOWLEDGE_MAX_CONTEXT_CHARS
) -> bool:
    """Whether retrieved chunks actually yield citable context.

    The chat path and the evaluation set must agree on what counts as grounded,
    so both decide through this predicate rather than testing ``results`` alone.
    """
    return bool(results) and bool(_collect_context_blocks(results, max_context_chars)[1])


def _insufficient_bundle(
    reason: str, results: list[dict[str, Any]], threshold: float
) -> dict[str, Any]:
    return {
        "context": "",
        "citations": [],
        "evidence": {
            "status": "insufficient",
            "code": "insufficient_evidence",
            "reason": reason,
            "matched_chunks": len(results),
            "matched_sources": len({str(item.get("source", "")) for item in results}),
            "retrieval_threshold": threshold,
        },
    }


def build_knowledge_bundle(
    query: str,
    *,
    top_k: int = KNOWLEDGE_TOP_K,
    threshold: float = KNOWLEDGE_RETRIEVAL_THRESHOLD,
    max_context_chars: int = KNOWLEDGE_MAX_CONTEXT_CHARS,
) -> dict[str, Any]:
    """Build prompt context and the matching, display-safe citation metadata."""
    results = retrieve_knowledge(query, top_k=top_k, threshold=threshold)
    if not results:
        reason = "knowledge_base_empty" if not load_chunks() else "no_relevant_sources"
        return _insufficient_bundle(reason, [], threshold)
    lines, citations = _collect_context_blocks(results, max_context_chars)
    if not citations:
        # Excerpts are truncated to fit, so reaching this point means the top
        # chunks carried no usable text (empty body, or a header alone already
        # larger than the whole budget) rather than the budget running out.
        return _insufficient_bundle("no_usable_excerpt", results, threshold)
    return {
        "context": "\n\n".join(lines),
        "citations": citations,
        "evidence": {
            "status": "sufficient",
            "code": None,
            "reason": None,
            "matched_chunks": len(citations),
            "matched_sources": len({citation["source"] for citation in citations}),
            "retrieval_threshold": threshold,
        },
    }


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
