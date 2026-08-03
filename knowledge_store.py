from __future__ import annotations

import hashlib
import json
import math
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
    KNOWLEDGE_BM25_MIN_SCORE,
    KNOWLEDGE_RETRIEVAL_THRESHOLD,
    KNOWLEDGE_RETRIEVAL_MODE,
    KNOWLEDGE_RETRIEVAL_MODES,
    KNOWLEDGE_RRF_K,
    KNOWLEDGE_TOP_K,
    RAG_RELEASE_GATE_ENABLED,
    RAG_RELEASE_MIN_CASES,
    RAG_RELEASE_MIN_INSUFFICIENT_REFUSAL,
    RAG_RELEASE_MIN_KEYWORD_COVERAGE,
    RAG_RELEASE_MIN_PASS_RATE,
    RAG_RELEASE_MIN_SOURCE_RECALL,
    RAG_NEAR_DOMAIN_GUARD_ENABLED,
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


# Request shapes the bundled overview corpus cannot substantiate even when it
# shares vocabulary with them.
#
# Every rule must match an explicit *request for out-of-scope specifics*, never a
# bare subject word. This service is an emotional-support chatbot, so its users
# routinely mention diagnoses, medication, money and travel while describing how
# they feel — "我被诊断出抑郁症" is a disclosure to respond to, not a request for a
# diagnosis. Matching such a message would replace support with a "no sources"
# notice, which is worse than answering without citations. English alternatives
# carry \b so "stock" cannot fire inside "Stockholm", nor "current" inside
# "concurrent". Keep test_near_domain_guard_never_blocks_ordinary_support_talk
# green when adding rules.
_NEAR_DOMAIN_SCOPE_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("realtime_or_latest_information", re.compile(
        r"(?:最新|实时|今天|明天|当前|现在)[^。？?!！]{0,12}"
        r"(?:价格|股价|排行榜|播出表|更新时间|版本号|新番)"
        r"|20\d{2}\s*年?[^。？?!！]{0,10}(?:新番|播出表|更新时间)"
        r"|\b(?:latest|current|real[-\s]?time|today'?s)\b[^.?!]{0,24}"
        r"\b(?:price|ranking|release\s+date|schedule|version\s+number)\b",
        re.IGNORECASE,
    )),
    ("product_or_software_release", re.compile(
        r"(?:下一代|下一版|新一代)[^。？?!！]{0,20}(?:什么时候|何时|发布日期|上市日期)"
        r"|(?:iphone\s*\d*|python\s*\d+(?:\.\d+)?)[^。？?!！]{0,20}"
        r"(?:什么时候|何时|发布日期|新特性有哪些)"
        r"|\bnext[-\s]gen(?:eration)?\b[^.?!]{0,24}\brelease\s+date\b",
        re.IGNORECASE,
    )),
    ("precise_future_prediction", re.compile(
        # Only a demand for an exact date; "AGI 会实现吗" is a discussion, not a lookup.
        r"(?:agi|通用人工智能)[^。？?!！]{0,24}(?:哪一天|哪一年|具体日期|具体时间|准确时间)"
        r"|(?:哪一天|哪一年|具体日期)[^。？?!！]{0,24}(?:agi|通用人工智能)"
        r"|\b(?:agi)\b[^.?!]{0,24}\b(?:exact\s+date|which\s+year)\b",
        re.IGNORECASE,
    )),
    ("specific_media_detail", re.compile(
        r"(?:第\s*\d+\s*集|具体剧情|结局)[^。？?!！]{0,16}"
        r"(?:是什么|讲了什么|剧透|告诉我|列出|说一下)"
        r"|(?:剧透|告诉我|列出)[^。？?!！]{0,16}(?:第\s*\d+\s*集|具体剧情|结局)",
        re.IGNORECASE,
    )),
    ("medication_dosage_request", re.compile(
        # Dosage and prescription asks only. Mentioning a diagnosis one already has
        # is disclosure, so 确诊/诊断 alone must never match.
        r"(?:多少|几)\s*(?:毫克|mg|片|粒)"
        r"|(?:用药|服用)\s*剂量|剂量\s*(?:是多少|怎么定|多少)"
        r"|(?:该|应该|需要|要)[^。？?!！]{0,8}(?:吃|服)[^。？?!！]{0,8}(?:多少|几片|几粒)"
        r"|(?:帮我|给我|请你?|能否|可以)[^。？?!！]{0,8}(?:确诊|诊断)"
        r"|\b(?:dosage|how\s+many\s+mg|prescription\s+dose|diagnose\s+me)\b",
        re.IGNORECASE,
    )),
    ("production_implementation", re.compile(
        r"(?:生产环境|可直接上线|完整代码)[^。？?!！]{0,24}(?:代码|配置|部署|faiss)"
        r"|(?:代码|配置|部署|faiss)[^。？?!！]{0,24}(?:生产环境|可直接上线|完整代码)"
        r"|\bproduction[-\s]ready\b[^.?!]{0,24}\b(?:code|config|deployment)\b",
        re.IGNORECASE,
    )),
    ("financial_recommendation", re.compile(
        # An investment recommendation request. Losing money on stocks is a feeling
        # to talk about, so bare 股票/仓位 must never match.
        r"(?:该|应该|要不要|建议我|帮我)[^。？?!！]{0,10}(?:买入|卖出|加仓|减仓|抄底)"
        r"|(?:投资建议|荐股|买哪只|该买哪|值不值得投)"
        r"|\b(?:should\s+i\s+(?:buy|sell)|investment\s+advice|stock\s+(?:pick|tip))\b",
        re.IGNORECASE,
    )),
    ("weather_or_travel_plan", re.compile(
        r"(?:今天|明天|后天)[^。？?!！]{0,10}(?:天气|下雨|气温|温度)"
        r"|(?:推荐|规划|安排|列出)[^。？?!！]{0,10}(?:景点|路线|行程)"
        r"|(?:地铁|公交)[^。？?!！]{0,6}(?:路线|怎么走|怎么坐)"
        r"|\b(?:tomorrow|today)'?s?\b[^.?!]{0,16}\b(?:weather|forecast)\b"
        r"|\b(?:plan|recommend)\b[^.?!]{0,16}\bitinerary\b",
        re.IGNORECASE,
    )),
)


def assess_knowledge_query_scope(query: str) -> str | None:
    """Return a reason when a related-looking query exceeds this corpus's scope."""
    if not RAG_NEAR_DOMAIN_GUARD_ENABLED:
        return None
    normalized = " ".join(str(query or "").split())
    for reason, pattern in _NEAR_DOMAIN_SCOPE_RULES:
        if pattern.search(normalized):
            return reason
    return None


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


_CJK_RUN_RE = re.compile(r"[\u3400-\u9fff]+")
_WORD_RE = re.compile(r"[a-z0-9][a-z0-9_+.#-]*", re.IGNORECASE)


def _bm25_tokens(text: str) -> list[str]:
    """Tokenize mixed Chinese and Latin text without adding a tokenizer dependency."""
    normalized = str(text or "").casefold()
    tokens = _WORD_RE.findall(normalized)
    for run in _CJK_RUN_RE.findall(normalized):
        tokens.extend(run[index:index + 2] for index in range(max(1, len(run) - 1)))
    return tokens


def _chunks_fingerprint(chunks: list[dict[str, Any]]) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for chunk in chunks:
        digest.update(str(chunk.get("text", "")).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


# (fingerprint, statistics). Rebound as a whole so concurrent readers never see a
# half-built index; a race only costs duplicated work, never a stale result.
_BM25_INDEX: tuple[str, dict[str, Any]] | None = None


def _bm25_index(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Tokenized corpus statistics, cached because hashing costs far less than tokenizing.

    Rebuilding per query is O(corpus) regex work on the request thread, which grows
    with the knowledge base; digesting the same text to detect changes does not.
    """
    global _BM25_INDEX
    fingerprint = _chunks_fingerprint(chunks)
    cached = _BM25_INDEX
    if cached is not None and cached[0] == fingerprint:
        return cached[1]
    documents = [_bm25_tokens(str(chunk.get("text", ""))) for chunk in chunks]
    document_frequency: Counter[str] = Counter()
    for tokens in documents:
        document_frequency.update(set(tokens))
    index = {
        "documents": documents,
        "frequencies": [Counter(tokens) for tokens in documents],
        "document_frequency": document_frequency,
        "average_length": (sum(len(tokens) for tokens in documents) / len(documents)) or 1.0,
    }
    _BM25_INDEX = (fingerprint, index)
    return index


def _bm25_search_candidates(
    query: str, chunks: list[dict[str, Any]], *, candidate_count: int
) -> list[dict[str, Any]]:
    query_tokens = _bm25_tokens(query)
    if not query_tokens or not chunks:
        return []
    index = _bm25_index(chunks)
    documents = index["documents"]
    document_frequency = index["document_frequency"]
    average_length = index["average_length"]
    query_terms = Counter(query_tokens)
    ranked: list[tuple[float, int]] = []
    for position, tokens in enumerate(documents):
        if not tokens:
            continue
        frequencies = index["frequencies"][position]
        score = 0.0
        for term, query_frequency in query_terms.items():
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            inverse_frequency = math.log(1 + (len(documents) - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5))
            denominator = frequency + 1.5 * (1 - 0.75 + 0.75 * len(tokens) / average_length)
            score += query_frequency * inverse_frequency * frequency * 2.5 / denominator
        if score > 0:
            ranked.append((score, position))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [
        {**chunks[position], "_index": position, "bm25_score": float(score)}
        for score, position in ranked[:candidate_count]
    ]


def _hybrid_search_candidates(
    query: str,
    chunks: list[dict[str, Any]],
    *,
    top_k: int,
    candidate_multiplier: int,
) -> list[dict[str, Any]]:
    candidate_count = min(len(chunks), max(top_k, top_k * max(1, int(candidate_multiplier))))
    vector_candidates = _search_candidates(
        query, chunks, top_k=top_k, candidate_multiplier=candidate_multiplier,
    )
    bm25_candidates = _bm25_search_candidates(query, chunks, candidate_count=candidate_count)
    merged: dict[int, dict[str, Any]] = {}
    for rank, candidate in enumerate(vector_candidates, start=1):
        index = int(candidate["_index"])
        # Keep bm25_score present on every candidate so the diagnostics panel does
        # not show the field for only part of the pool.
        merged[index] = {
            **candidate, "vector_score": float(candidate["score"]),
            "bm25_score": None, "rrf_score": 1 / (KNOWLEDGE_RRF_K + rank),
        }
    for rank, candidate in enumerate(bm25_candidates, start=1):
        index = int(candidate["_index"])
        item = merged.setdefault(index, {**candidate, "score": 0.0, "vector_score": None, "rrf_score": 0.0})
        item["bm25_score"] = float(candidate["bm25_score"])
        item["rrf_score"] += 1 / (KNOWLEDGE_RRF_K + rank)
    return sorted(merged.values(), key=lambda item: (-float(item["rrf_score"]), int(item["_index"])))


def _passes_lexical_gate(item: dict[str, Any]) -> bool:
    """Whether a chunk the vector search scored too low qualifies on lexical evidence.

    Disabled by default: without this gate BM25 can only reorder the vector
    candidate pool, so an exact rare term the embedding missed stays unreachable.
    """
    if KNOWLEDGE_BM25_MIN_SCORE <= 0:
        return False
    bm25_score = item.get("bm25_score")
    return bm25_score is not None and float(bm25_score) >= KNOWLEDGE_BM25_MIN_SCORE


def _normalize_retrieval_mode(retrieval_mode: str | None) -> str:
    mode = str(retrieval_mode or KNOWLEDGE_RETRIEVAL_MODE).strip().lower()
    if mode not in KNOWLEDGE_RETRIEVAL_MODES:
        raise ValueError(f"检索模式仅支持 {'、'.join(KNOWLEDGE_RETRIEVAL_MODES)}。")
    return mode


def diagnose_knowledge_search(
    query: str,
    *,
    top_k: int = KNOWLEDGE_TOP_K,
    threshold: float = KNOWLEDGE_RETRIEVAL_THRESHOLD,
    candidate_multiplier: int = KNOWLEDGE_CANDIDATE_MULTIPLIER,
    max_per_source: int = KNOWLEDGE_MAX_PER_SOURCE,
    retrieval_mode: str | None = None,
) -> dict[str, Any]:
    chunks = load_chunks()
    query = (query or "").strip()
    if not query or not chunks:
        return {"query": query, "results": [], "candidates": [], "reason": "查询或知识库为空", "scope_reason": None}

    top_k = max(1, min(int(top_k), len(chunks)))
    threshold = max(-1.0, min(float(threshold), 1.0))
    max_per_source = max(1, int(max_per_source))
    mode = _normalize_retrieval_mode(retrieval_mode)
    candidates = _hybrid_search_candidates(query, chunks, top_k=top_k, candidate_multiplier=candidate_multiplier) if mode == "hybrid_rrf" else _search_candidates(query, chunks, top_k=top_k, candidate_multiplier=candidate_multiplier)

    eligible: list[dict[str, Any]] = []
    seen_texts: set[str] = set()
    for item in candidates:
        score = float(item["score"])
        key = _normalized_chunk_key(str(item.get("text", "")))
        # RRF only reorders; the vector score stays the primary evidence gate so a
        # common BM25 term alone cannot turn out-of-domain text into evidence. A
        # lexical-only chunk is admitted only when KNOWLEDGE_BM25_MIN_SCORE is
        # configured and its BM25 score clears that separate, stricter bar.
        if score < threshold and not _passes_lexical_gate(item):
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
    # The guard is applied here rather than in build_knowledge_bundle so chat, the
    # evaluation set and the diagnostics panel all see the same decision. Candidates
    # are still reported, so an operator can see what a blocked query would have hit.
    scope_reason = assess_knowledge_query_scope(query)
    return {
        "query": query,
        "results": [] if scope_reason else public_results,
        "candidates": public_candidates,
        "threshold": threshold,
        "top_k": top_k,
        "retrieval_mode": mode,
        "scope_reason": scope_reason,
    }


def retrieve_knowledge(
    query: str,
    *,
    top_k: int = KNOWLEDGE_TOP_K,
    threshold: float = KNOWLEDGE_RETRIEVAL_THRESHOLD,
    candidate_multiplier: int = KNOWLEDGE_CANDIDATE_MULTIPLIER,
    max_per_source: int = KNOWLEDGE_MAX_PER_SOURCE,
    retrieval_mode: str | None = None,
) -> list[dict[str, Any]]:
    return diagnose_knowledge_search(
        query,
        top_k=top_k,
        threshold=threshold,
        candidate_multiplier=candidate_multiplier,
        max_per_source=max_per_source,
        retrieval_mode=retrieval_mode,
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
    diagnostic = diagnose_knowledge_search(query, top_k=top_k, threshold=threshold)
    results = diagnostic["results"]
    if not results:
        reason = diagnostic.get("scope_reason") or (
            "knowledge_base_empty" if not load_chunks() else "no_relevant_sources"
        )
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
