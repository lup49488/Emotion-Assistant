from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from config import (
    STYLE_CHUNK_OVERLAP,
    STYLE_CHUNK_SIZE,
    STYLE_CHUNKS_PATH,
    STYLE_DIR,
    STYLE_DOCS_DIR,
    STYLE_INDEX_PATH,
    STYLE_RETRIEVAL_THRESHOLD,
    STYLE_PREFIX_CANDIDATE_MULTIPLIER,
    STYLE_TOP_K,
)
from json_utils import load_json, save_json
from llm_providers import encode_texts, get_embedding_dimension
from memory_store import require_faiss
from sqlite_store import (
    connection as sqlite_connection,
    list_rag_chunks,
    mark_rag_migration_completed,
    rag_migration_completed,
    replace_rag_chunks,
    replace_rag_documents,
    sqlite_enabled,
)


SUPPORTED_STYLE_EXTENSIONS = {".txt", ".md", ".markdown", ".csv", ".json", ".jsonl"}
RAG_COLLECTION = "style"


@dataclass
class StyleChunk:
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


def ensure_style_dirs() -> None:
    STYLE_DIR.mkdir(parents=True, exist_ok=True)
    STYLE_DOCS_DIR.mkdir(parents=True, exist_ok=True)


def _safe_filename(path: Path) -> str:
    name = path.name.strip() or "style.txt"
    return "".join(char if char.isalnum() or char in "._- " else "_" for char in name)


def copy_document_to_store(path: str | Path) -> Path:
    ensure_style_dirs()
    source_path = Path(path)
    if not source_path.exists() or not source_path.is_file():
        raise FileNotFoundError(f"文件不存在：{source_path}")
    if source_path.suffix.lower() not in SUPPORTED_STYLE_EXTENSIONS:
        raise ValueError(f"暂不支持该风格文件格式：{source_path.suffix}")

    target = STYLE_DOCS_DIR / _safe_filename(source_path)
    if target.exists():
        stem = target.stem
        suffix = target.suffix
        target = STYLE_DOCS_DIR / f"{stem}_{datetime.now().strftime('%Y%m%d%H%M%S')}{suffix}"
    shutil.copy2(source_path, target)
    return target


def _jsonl_to_text(path: Path) -> str:
    lines: list[str] = []
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        data = json.loads(raw_line)
        if isinstance(data, dict):
            title = data.get("title") or data.get("name") or "style sample"
            user = data.get("user") or data.get("input") or data.get("prompt") or ""
            assistant = data.get("assistant") or data.get("output") or data.get("response") or ""
            lines.append(f"# {title}\n用户：{user}\n助手：{assistant}")
        else:
            lines.append(json.dumps(data, ensure_ascii=False))
    return "\n\n".join(lines)


def extract_text(path: str | Path) -> str:
    source_path = Path(path)
    suffix = source_path.suffix.lower()
    if suffix in {".txt", ".md", ".markdown", ".csv"}:
        return source_path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".json":
        data = json.loads(source_path.read_text(encoding="utf-8"))
        return json.dumps(data, ensure_ascii=False, indent=2)
    if suffix == ".jsonl":
        return _jsonl_to_text(source_path)
    raise ValueError(f"暂不支持该风格文件格式：{suffix}")


def split_style_text(
    text: str,
    *,
    chunk_size: int = STYLE_CHUNK_SIZE,
    overlap: int = STYLE_CHUNK_OVERLAP,
) -> list[str]:
    normalized = "\n".join(line.rstrip() for line in text.splitlines())
    normalized = normalized.strip()
    if not normalized:
        return []

    sections = [section.strip() for section in normalized.split("\n# ") if section.strip()]
    if len(sections) > 1:
        sections = [sections[0], *[f"# {section}" for section in sections[1:]]]
    else:
        sections = [normalized]

    chunk_size = max(120, int(chunk_size))
    overlap = max(0, min(int(overlap), chunk_size // 2))
    chunks: list[str] = []
    for section in sections:
        if len(section) <= chunk_size:
            chunks.append(section)
            continue
        start = 0
        while start < len(section):
            end = min(start + chunk_size, len(section))
            chunk = section[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(section):
                break
            start = max(0, end - overlap)
    return chunks


def _filesystem_document_details() -> list[dict[str, Any]]:
    ensure_style_dirs()
    details: list[dict[str, Any]] = []
    for path in sorted(STYLE_DOCS_DIR.iterdir()):
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
    chunks = load_json(STYLE_CHUNKS_PATH)
    clean_chunks = chunks if isinstance(chunks, list) else []
    _sync_sqlite_metadata(conn, clean_chunks)
    return len(clean_chunks)


def migrate_legacy_style_metadata() -> int:
    """Import legacy style chunk JSON and document metadata into SQLite once."""
    if not sqlite_enabled():
        return 0
    with sqlite_connection() as conn:
        return _migrate_legacy_metadata_unlocked(conn)


def load_chunks() -> list[dict[str, Any]]:
    if sqlite_enabled():
        with sqlite_connection() as conn:
            _migrate_legacy_metadata_unlocked(conn)
            return list_rag_chunks(conn, RAG_COLLECTION)
    chunks = load_json(STYLE_CHUNKS_PATH)
    return chunks if isinstance(chunks, list) else []


def save_chunks(chunks: list[dict[str, Any]]) -> None:
    ensure_style_dirs()
    save_json(STYLE_CHUNKS_PATH, chunks)
    if sqlite_enabled():
        with sqlite_connection() as conn:
            _sync_sqlite_metadata(conn, chunks)


def list_documents() -> list[str]:
    ensure_style_dirs()
    return sorted(path.name for path in STYLE_DOCS_DIR.iterdir() if path.is_file())


def ingest_document(path: str | Path, *, copy_to_store: bool = True) -> list[dict[str, Any]]:
    stored_path = copy_document_to_store(path) if copy_to_store else Path(path)
    text = extract_text(stored_path)
    parts = split_style_text(text)
    now = datetime.now().isoformat()
    return [
        StyleChunk(
            id=f"{stored_path.name}#{index}",
            source=stored_path.name,
            text=part,
            chunk_index=index,
            created_at=now,
        ).as_dict()
        for index, part in enumerate(parts)
    ]


def rebuild_style_index() -> dict[str, Any]:
    ensure_style_dirs()
    all_chunks: list[dict[str, Any]] = []
    errors: list[str] = []
    for path in sorted(STYLE_DOCS_DIR.iterdir()):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_STYLE_EXTENSIONS:
            continue
        try:
            all_chunks.extend(ingest_document(path, copy_to_store=False))
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")
    save_chunks(all_chunks)
    _write_index(all_chunks)
    return {"chunks": len(all_chunks), "documents": len(list_documents()), "errors": errors}


def import_style_documents(paths: list[str | Path]) -> dict[str, Any]:
    ensure_style_dirs()
    imported: list[str] = []
    errors: list[str] = []
    for path in paths:
        try:
            target = copy_document_to_store(path)
            imported.append(target.name)
        except Exception as exc:
            errors.append(f"{Path(path).name}: {exc}")
    result = rebuild_style_index()
    result["imported"] = imported
    result["errors"] = errors + result["errors"]
    return result


def _write_index(chunks: list[dict[str, Any]]) -> None:
    faiss = require_faiss()
    index = faiss.IndexFlatIP(get_embedding_dimension())
    texts = [chunk["text"] for chunk in chunks if chunk.get("text")]
    if texts:
        index.add(np.asarray(encode_texts(texts), dtype=np.float32))
    faiss.write_index(index, str(STYLE_INDEX_PATH))


def _read_index(chunks: list[dict[str, Any]]) -> Any:
    faiss = require_faiss()
    if not STYLE_INDEX_PATH.exists():
        _write_index(chunks)
    return faiss.read_index(str(STYLE_INDEX_PATH))


def _normalized_prefix(source_prefix: str | None) -> str:
    return (source_prefix or "").strip().casefold()


def style_prefixes() -> list[str]:
    """List selectable style families derived from the document filename prefix.

    Documents are named ``<style>.md`` or ``<style>_<variant>.md``, so the text
    before the first underscore identifies the style family.
    """
    prefixes = {Path(name).stem.split("_", 1)[0].strip() for name in list_documents()}
    return sorted(prefix for prefix in prefixes if prefix)


def _matches_prefix(source: str, prefix: str) -> bool:
    return Path(str(source)).stem.casefold().startswith(prefix)


def retrieve_style(
    query: str,
    *,
    top_k: int = STYLE_TOP_K,
    threshold: float = STYLE_RETRIEVAL_THRESHOLD,
    source_prefix: str | None = None,
) -> list[dict[str, Any]]:
    chunks = load_chunks()
    if not query.strip() or not chunks:
        return []
    prefix = _normalized_prefix(source_prefix)
    index = _read_index(chunks)
    top_k = max(1, min(int(top_k), len(chunks)))
    # The FAISS index covers every style document, so a prefix filter must search
    # a wider candidate pool; otherwise the nearest neighbours may all belong to
    # other style families and the filtered result would be empty.
    search_k = min(len(chunks), top_k * STYLE_PREFIX_CANDIDATE_MULTIPLIER) if prefix else top_k
    similarities, indices = index.search(encode_texts([query]), search_k)
    results: list[dict[str, Any]] = []
    for score, i in zip(similarities[0], indices[0]):
        if i < 0 or i >= len(chunks) or float(score) < threshold:
            continue
        item = dict(chunks[int(i)])
        if prefix and not _matches_prefix(item.get("source", ""), prefix):
            continue
        item["score"] = float(score)
        results.append(item)
        if len(results) >= top_k:
            break
    return results


def build_style_context(
    query: str, *, top_k: int = STYLE_TOP_K, source_prefix: str | None = None
) -> str:
    results = retrieve_style(query, top_k=top_k, source_prefix=source_prefix)
    if not results:
        return ""
    lines = [
        "以下内容只用于参考表达方式、语气、结构和详略程度，不作为事实依据；不要逐字复制。"
    ]
    for index, item in enumerate(results, start=1):
        source = item.get("source", "unknown")
        text = str(item.get("text", "")).strip()
        lines.append(f"[风格 {index} | {source}]\n{text}")
    return "\n\n".join(lines)


def style_status() -> str:
    chunks = load_chunks()
    docs = list_documents()
    index_status = "已建立" if STYLE_INDEX_PATH.exists() else "未建立"
    return "\n".join([
        f"风格文档数：{len(docs)}",
        f"风格片段数：{len(chunks)}",
        f"索引状态：{index_status}",
        f"目录：{STYLE_DOCS_DIR}",
        "文档：" + (", ".join(docs[:20]) if docs else "暂无"),
    ])
