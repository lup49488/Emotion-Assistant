from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import torch

from config import (
    INTEREST_PATTERNS,
    INTEREST_RETRIEVAL_THRESHOLD,
    INTEREST_SIMILARITY_THRESHOLD,
    LONG_TERM_EXPIRY_DAYS,
    MEMORY_KEYWORDS,
    PERSONAL_KEYWORDS,
    SCORE_EMOTION_MULTIPLIER,
    SCORE_LONG_TERM_THRESHOLD,
    SCORE_MEMORY_KEYWORD_BONUS,
    SCORE_MID_TERM_THRESHOLD,
    SCORE_NEGATIVE_BONUS,
    SCORE_PERSONAL_KEYWORD_BONUS,
)
from json_utils import safe_extract_json_array
from llm_providers import encode_texts, get_embedding_dimension, get_llm


logger = logging.getLogger(__name__)

LONG_MEMORY_SUMMARIZE_THRESHOLD = 30
LONG_MEMORY_KEEP_RECENT = 10
INTEREST_MERGE_THRESHOLD = 0.80

_faiss_module: Any | None = None


def require_faiss():
    global _faiss_module
    if _faiss_module is None:
        try:
            import faiss
            _faiss_module = faiss
        except ModuleNotFoundError as exc:
            raise RuntimeError("缺少依赖 faiss-cpu，请先安装后再使用向量记忆功能。") from exc
    return _faiss_module


class InterestMemoryStore:
    def __init__(self) -> None:
        self._items: list[dict[str, Any]] = []
        self._exact_texts: set[str] = set()
        self._dirty = False

    def load(self, items: list[dict[str, Any]]) -> None:
        self._items = items
        self._exact_texts = {m["text"] for m in items if m.get("text")}
        self._dirty = True

    @property
    def items(self) -> list[dict[str, Any]]:
        return self._items

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)

    def exact_exists(self, text: str) -> bool:
        return text in self._exact_texts

    def append(self, memory: dict[str, Any]) -> None:
        text = str(memory.get("text", "")).strip()
        if not text:
            return
        memory["text"] = text
        self._items.append(memory)
        self._exact_texts.add(text)
        self._dirty = True

    def remove_by_text(self, text: str) -> bool:
        for i, item in enumerate(self._items):
            if item.get("text") == text:
                del self._items[i]
                self._exact_texts.discard(text)
                self._dirty = True
                return True
        return False

    def replace_all(self, items: list[dict[str, Any]]) -> None:
        self._items = items
        self._exact_texts = {m["text"] for m in items if m.get("text")}
        self._dirty = True

    @property
    def dirty(self) -> bool:
        return self._dirty

    def mark_clean(self) -> None:
        self._dirty = False


class VectorIndexManager:
    def __init__(self, index_path: Path) -> None:
        self._index_path = index_path
        self._index: Any | None = None
        self._force_rebuild = False

    def get(self, store: InterestMemoryStore) -> Any:
        faiss = require_faiss()
        if self._index is None and self._index_path.exists() and not self._force_rebuild:
            try:
                self._index = faiss.read_index(str(self._index_path))
                logger.debug("FAISS 索引从磁盘加载 (%s)，ntotal=%d", self._index_path.name, self._index.ntotal)
            except RuntimeError:
                logger.warning("FAISS 索引文件损坏 (%s)，将重建。", self._index_path)
                self._index = None

        if store.dirty or self._index is None or self._force_rebuild:
            self._rebuild(store)
            store.mark_clean()
            self._force_rebuild = False
        return self._index

    def _rebuild(self, store: InterestMemoryStore) -> None:
        faiss = require_faiss()
        self._index = faiss.IndexFlatIP(get_embedding_dimension())
        texts = [m["text"] for m in store.items if m.get("text")]
        if texts:
            self._index.add(encode_texts(texts))
        faiss.write_index(self._index, str(self._index_path))

    def add_one(self, text: str, store: InterestMemoryStore) -> None:
        faiss = require_faiss()
        idx = self.get(store)
        idx.add(encode_texts([text]))
        faiss.write_index(idx, str(self._index_path))
        store.mark_clean()

    def mark_dirty_for_rebuild(self) -> None:
        self._index = None
        self._force_rebuild = True


def clean_long_term(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    now = datetime.now()
    expiry = timedelta(days=LONG_TERM_EXPIRY_DAYS)
    cleaned: list[dict[str, Any]] = []
    for item in items:
        raw_time = item.get("time")
        if raw_time is None:
            logger.warning("长期记忆条目缺少 'time' 字段，保留但跳过过期检查: %r", item)
            cleaned.append(item)
            continue
        try:
            item_time = datetime.fromisoformat(raw_time)
        except (TypeError, ValueError):
            logger.warning("长期记忆条目 'time' 格式非法 (%r)，保留但跳过过期检查: %r", raw_time, item)
            cleaned.append(item)
            continue
        if now - item_time < expiry:
            cleaned.append(item)
    return cleaned


def _format_memories_for_summary(items: list[dict[str, Any]]) -> str:
    lines = []
    for item in items:
        text = item.get("text", "")
        emotion = item.get("emotion", "")
        time_str = (item.get("time") or "")[:10]
        suffix = f"（情绪：{emotion}）" if emotion else ""
        lines.append(f"- [{time_str}] {text}{suffix}")
    return "\n".join(lines)


def summarize_long_memories(items_to_summarize: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not items_to_summarize:
        return None
    memory_text = _format_memories_for_summary(items_to_summarize)
    prompt = (
        "以下是用户的一批历史记忆条目，可能包含重复或相近的主题。"
        "请将它们压缩成 1-3 条简洁的摘要，保留关键事实、稳定偏好和重要趋势，"
        "去掉重复内容和无关细节。"
        '只返回 JSON 数组，例如 [{"text": "用户长期对工作压力感到困扰"}]。'
        f"\n\n历史记忆：\n{memory_text}"
    )
    try:
        tokenizer, model = get_llm()
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        encoded = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.inference_mode():
            outputs = model.generate(**encoded, max_new_tokens=200, do_sample=False)
        new_tokens = outputs[0][encoded["input_ids"].shape[1]:]
        reply = tokenizer.decode(new_tokens, skip_special_tokens=True)
    except Exception:
        logger.exception("长期记忆摘要压缩时 LLM 推理失败，保留原始记忆。")
        return None

    data = safe_extract_json_array(reply)
    if not data:
        logger.warning("长期记忆摘要压缩：模型输出无法解析为 JSON 数组，保留原始记忆。")
        return None

    now_iso = datetime.now().isoformat()
    summarized = [
        {"text": str(d.get("text", "")).strip(), "time": now_iso, "kind": "summary"}
        for d in data if isinstance(d, dict) and d.get("text")
    ]
    if not summarized:
        return None
    return {"summarized": summarized, "source_count": len(items_to_summarize)}


def compact_long_memory(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(items) <= LONG_MEMORY_SUMMARIZE_THRESHOLD:
        return items
    sorted_items = sorted(items, key=lambda item: str(item.get("time", "")))
    to_summarize = sorted_items[:-LONG_MEMORY_KEEP_RECENT]
    to_keep = sorted_items[-LONG_MEMORY_KEEP_RECENT:]
    result = summarize_long_memories(to_summarize)
    if result is None:
        return items
    return result["summarized"] + to_keep


def score_memory(user_text: str, emo_label: str, emo_score: float) -> float:
    score = float(emo_score) * SCORE_EMOTION_MULTIPLIER
    if emo_label in {"sadness", "fear", "anger", "anxiety"}:
        score += SCORE_NEGATIVE_BONUS
    if any(kw in user_text for kw in MEMORY_KEYWORDS):
        score += SCORE_MEMORY_KEYWORD_BONUS
    if any(kw in user_text for kw in PERSONAL_KEYWORDS):
        score += SCORE_PERSONAL_KEYWORD_BONUS
    return score


def extract_long_term_interest(text: str) -> dict[str, Any] | None:
    lowered = text.lower()
    if any(p.lower() in lowered for p in INTEREST_PATTERNS):
        return {"text": text.strip(), "time": datetime.now().isoformat()}
    return None


def memory_exists(text: str, state: Any) -> bool:
    if not state.interest_store:
        return False
    if state.interest_store.exact_exists(text):
        return True
    try:
        embedding = encode_texts([text])
        idx = state.vector_index.get(state.interest_store)
        similarities, _ = idx.search(embedding, 1)
        return bool(similarities.size and similarities[0][0] > INTEREST_SIMILARITY_THRESHOLD)
    except RuntimeError:
        logger.warning("FAISS 相似度检索失败，跳过语义查重。")
        return False


def find_similar_interest(text: str, state: Any) -> tuple[dict[str, Any], float] | None:
    if not state.interest_store:
        return None
    try:
        embedding = encode_texts([text])
        idx = state.vector_index.get(state.interest_store)
        similarities, indices = idx.search(embedding, 1)
    except RuntimeError:
        return None
    if not similarities.size or indices[0][0] < 0:
        return None
    score = float(similarities[0][0])
    items = state.interest_store.items
    i = int(indices[0][0])
    if i >= len(items):
        return None
    if INTEREST_MERGE_THRESHOLD <= score < INTEREST_SIMILARITY_THRESHOLD:
        return items[i], score
    return None


def merge_interest_texts(old_text: str, new_text: str) -> str:
    if new_text in old_text:
        return old_text
    if old_text in new_text:
        return new_text
    return f"{old_text}；{new_text}"


def save_interest(memory: dict[str, Any], state: Any) -> str:
    text = str(memory.get("text", "")).strip()
    if not text:
        return "skipped"
    similar = find_similar_interest(text, state)
    if similar is not None:
        old_item, _ = similar
        old_text = old_item.get("text", "")
        merged_text = merge_interest_texts(old_text, text)
        if merged_text != old_text:
            state.interest_store.remove_by_text(old_text)
            merged_item = {**old_item, "text": merged_text, "time": datetime.now().isoformat()}
            state.interest_store.append(merged_item)
            state.vector_index.mark_dirty_for_rebuild()
        return "merged"
    memory["text"] = text
    state.interest_store.append(memory)
    try:
        state.vector_index.add_one(text, state.interest_store)
    except RuntimeError:
        logger.warning("FAISS 增量追加失败，索引将在下次 get() 时重建。")
    return "added"


def retrieve_interests(query: str, state: Any, top_k: int = 5) -> list[dict[str, Any]]:
    if not state.interest_store:
        return []
    top_k = min(top_k, len(state.interest_store))
    try:
        idx = state.vector_index.get(state.interest_store)
        similarities, indices = idx.search(encode_texts([query]), top_k)
    except RuntimeError:
        return []
    results: list[dict[str, Any]] = []
    items = state.interest_store.items
    for score, i in zip(similarities[0], indices[0]):
        if i >= 0 and score > INTEREST_RETRIEVAL_THRESHOLD and i < len(items):
            results.append(items[i])
    return results


def update_short_term(state: Any, user_text: str, assistant_reply: str) -> None:
    state.history.extend([
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": assistant_reply},
    ])
    from config import SHORT_TERM_LIMIT
    state.history = state.history[-SHORT_TERM_LIMIT:]


def update_mid_term(state: Any, emo_label: str, emo_score: float) -> None:
    state.emotion_memory.append(
        {"label": emo_label, "score": float(emo_score), "time": datetime.now().isoformat()}
    )
    from config import MID_TERM_LIMIT
    state.emotion_memory = state.emotion_memory[-MID_TERM_LIMIT:]


def update_long_term(state: Any, info: dict[str, Any]) -> None:
    if not any(m.get("text") == info.get("text") for m in state.long_memory):
        state.long_memory.append(info)


def smart_memory_filter(state: Any, user_text: str, emo_label: str, emo_score: float) -> str:
    score = score_memory(user_text, emo_label, emo_score)
    interest = extract_long_term_interest(user_text)
    if interest and not memory_exists(interest["text"], state):
        outcome = save_interest(interest, state)
        if outcome == "added":
            update_long_term(state, {**interest, "kind": "interest"})
    if score >= SCORE_LONG_TERM_THRESHOLD:
        if interest is None:
            update_long_term(state, {
                "text": user_text, "emotion": emo_label,
                "score": float(emo_score), "time": datetime.now().isoformat(),
            })
        return "long"
    if score >= SCORE_MID_TERM_THRESHOLD:
        update_mid_term(state, emo_label, emo_score)
        return "mid"
    return "discard"
