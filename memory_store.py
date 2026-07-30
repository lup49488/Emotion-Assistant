from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import torch

from config import (
    EMOTION_CONFIDENCE_THRESHOLD,
    INTEREST_PATTERNS,
    INTEREST_RETRIEVAL_THRESHOLD,
    INTEREST_SIMILARITY_THRESHOLD,
    LONG_TERM_EXPIRY_DAYS,
    MEMORY_KEYWORDS,
    NEGATIVE_EMOTIONS,
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
MEMORY_EVENT_LIMIT = 100

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
    if emo_label in NEGATIVE_EMOTIONS:
        score += SCORE_NEGATIVE_BONUS
    if not is_memory_query(user_text):
        if any(kw in user_text for kw in MEMORY_KEYWORDS):
            score += SCORE_MEMORY_KEYWORD_BONUS
        if any(kw in user_text for kw in PERSONAL_KEYWORDS):
            score += SCORE_PERSONAL_KEYWORD_BONUS
    return score


def is_memory_query(text: str) -> bool:
    """Identify questions about stored facts so they are not saved as new facts."""
    normalized = (text or "").strip()
    if not normalized:
        return False
    lowered = normalized.lower()
    if normalized.endswith(("?", "？", "吗", "呢")):
        return True
    if any(marker in normalized for marker in ("什么", "哪些", "谁", "哪里", "怎么", "为何", "为什么", "多少")):
        return True
    return any(phrase in lowered for phrase in (
        "what do i like", "what are my interests", "do you remember", "who am i",
    ))


_CLAUSE_BOUNDARY_RE = re.compile(r"[，,。；;！？!?\n]")
_CHINESE_QUERY_MARKERS = ("什么", "哪些", "谁", "哪里", "怎么", "如何", "为何", "为什么", "多少", "是否")
_ENGLISH_QUERY_RE = re.compile(r"\b(what|who|where|when|why|how|whether)\b", re.IGNORECASE)


def _clause_containing(text: str, position: int) -> tuple[str, str]:
    """Return the clause around a matched fact and the punctuation that ends it."""
    start = 0
    for match in _CLAUSE_BOUNDARY_RE.finditer(text):
        if match.start() < position:
            start = match.end()
            continue
        return text[start:match.start()].strip(), match.group(0)
    return text[start:].strip(), ""


def _is_question_fact(clause: str, remainder: str, terminator: str) -> bool:
    remainder = remainder.strip(" ：:，,。.!！?？")
    if not remainder:
        return True
    if terminator in {"?", "？"} or remainder.endswith(("吗", "呢", "么")):
        return True
    if any(marker in remainder for marker in _CHINESE_QUERY_MARKERS):
        return True
    return bool(_ENGLISH_QUERY_RE.search(remainder) and clause.rstrip().endswith(("?", "？")))


def extract_long_term_interest(text: str) -> dict[str, Any] | None:
    normalized = (text or "").strip()
    lowered = normalized.lower()
    for pattern in INTEREST_PATTERNS:
        position = lowered.find(pattern.lower())
        if position < 0:
            continue
        clause, terminator = _clause_containing(normalized, position)
        clause_position = clause.lower().find(pattern.lower())
        if clause_position < 0:
            continue
        remainder = clause[clause_position + len(pattern):]
        if not _is_question_fact(clause, remainder, terminator):
            return {"text": clause, "time": datetime.now().isoformat()}
    return None


def extract_personal_profile(text: str) -> dict[str, Any] | None:
    """Extract explicit, declarative identity facts as durable profile memory."""
    normalized = (text or "").strip()
    clause, terminator = _clause_containing(normalized, 0)
    lowered = clause.lower()
    chinese_prefixes = {
        "我是": "identity", "我叫": "name", "我的名字是": "name", "我来自": "origin", "我今年": "age",
    }
    english_prefixes = {
        "i am ": "identity", "i'm ": "identity", "my name is ": "name",
        "i am from ": "origin", "i live in ": "location",
    }
    for prefix, key in sorted(chinese_prefixes.items(), key=lambda item: len(item[0]), reverse=True):
        if clause.startswith(prefix):
            remainder = clause[len(prefix):]
            if not _is_question_fact(clause, remainder, terminator):
                return {"text": clause, "time": datetime.now().isoformat(), "kind": "profile", "key": key}
    for prefix, key in sorted(english_prefixes.items(), key=lambda item: len(item[0]), reverse=True):
        if lowered.startswith(prefix):
            remainder = clause[len(prefix):]
            if not _is_question_fact(clause, remainder, terminator):
                return {"text": clause, "time": datetime.now().isoformat(), "kind": "profile", "key": key}
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


def update_long_term(state: Any, info: dict[str, Any]) -> bool:
    if not any(m.get("text") == info.get("text") for m in state.long_memory):
        state.long_memory.append(info)
        return True
    return False


def _normalized_memory_text(item: dict[str, Any]) -> str:
    """Normalize a memory item's text for exact cross-section ownership checks."""
    return " ".join(str(item.get("text", "")).split()).casefold()


def reconcile_memory_ownership(state: Any) -> dict[str, int]:
    """Keep each exact personal fact in one durable memory section.

    Stable profile values are authoritative for identity facts, followed by
    interests for preferences, then general long-term memory.  This also
    repairs historical data produced while interests were mirrored into the
    long-term store.
    """
    claimed: set[str] = set()
    removed = {"stable": 0, "interest": 0, "long": 0}

    def keep_unique(items: list[dict[str, Any]], section: str) -> list[dict[str, Any]]:
        kept: list[dict[str, Any]] = []
        for item in items:
            if not isinstance(item, dict):
                kept.append(item)
                continue
            normalized = _normalized_memory_text(item)
            if normalized and normalized in claimed:
                removed[section] += 1
                continue
            if normalized:
                claimed.add(normalized)
            kept.append(item)
        return kept

    stable_profile = keep_unique(list(state.stable_profile), "stable")
    interest_items = keep_unique(list(state.interest_store.items), "interest")
    long_memory = keep_unique(list(state.long_memory), "long")

    if stable_profile != state.stable_profile:
        state.stable_profile = stable_profile
    if interest_items != state.interest_store.items:
        state.interest_store.replace_all(interest_items)
        if state.vector_index is not None:
            state.vector_index.mark_dirty_for_rebuild()
    if long_memory != state.long_memory:
        state.long_memory = long_memory
    return removed


def update_stable_profile(state: Any, info: dict[str, Any]) -> str:
    """Save a durable personal fact, replacing an older value in the same profile field."""
    text = str(info.get("text", "")).strip()
    if not text:
        return "skipped"
    now = datetime.now().isoformat()
    item = {**info, "text": text, "kind": "profile", "updated_at": now}
    key = str(item.get("key", "")).strip()
    for index, existing in enumerate(state.stable_profile):
        if (key and existing.get("key") == key) or existing.get("text") == text:
            item["created_at"] = existing.get("created_at") or now
            if existing.get("text") == item.get("text"):
                return "unchanged"
            state.stable_profile[index] = item
            return "updated"
    item["created_at"] = now
    state.stable_profile.append(item)
    return "added"


def record_memory_event(
    state: Any,
    *,
    section: str,
    action: str,
    text: str,
    reason: str,
    score: float | None = None,
) -> dict[str, Any]:
    event = {
        "id": datetime.now().strftime("%Y%m%d%H%M%S%f"),
        "time": datetime.now().isoformat(timespec="seconds"),
        "section": section,
        "action": action,
        "text": (text or "").strip(),
        "reason": reason,
    }
    if score is not None:
        event["score"] = round(float(score), 3)
    state.memory_events.append(event)
    state.memory_events = state.memory_events[-MEMORY_EVENT_LIMIT:]
    return event


def latest_memory_receipt(state: Any) -> str:
    if not state.memory_events:
        return "记忆回执：本轮没有记忆判断记录。"
    event = state.memory_events[-1]
    action_labels = {
        "added": "已新增",
        "updated": "已更新",
        "merged": "已合并",
        "unchanged": "未重复写入",
        "skipped": "未写入",
    }
    section_labels = {
        "stable": "稳定资料",
        "interest": "兴趣记忆",
        "long": "长期记忆",
        "emotion": "情绪记忆",
        "none": "记忆",
    }
    action = action_labels.get(str(event.get("action")), str(event.get("action", "已处理")))
    section = section_labels.get(str(event.get("section")), str(event.get("section", "记忆")))
    text = str(event.get("text", "")).strip()
    reason = str(event.get("reason", "")).strip()
    content = f"“{text}”" if text else "本轮内容"
    return f"记忆回执：{section}{action} {content}。原因：{reason}"


def smart_memory_filter(state: Any, user_text: str, emo_label: str, emo_score: float) -> str:
    score = score_memory(user_text, emo_label, emo_score)
    emotion_recorded = (
        emo_label != "uncertain"
        and float(emo_score) >= EMOTION_CONFIDENCE_THRESHOLD
    )
    if emotion_recorded:
        # Emotion history has its own confidence gate. It should not depend on
        # personal-fact keywords or the unrelated long-term-memory score.
        update_mid_term(state, emo_label, emo_score)
        record_memory_event(
            state,
            section="emotion",
            action="added",
            text=user_text,
            reason=(
                f"情绪模型置信度 {float(emo_score):.2f} 达到情绪记录阈值 "
                f"{EMOTION_CONFIDENCE_THRESHOLD:.2f}"
            ),
            score=score,
        )
    profile = extract_personal_profile(user_text)
    if profile is not None:
        action = update_stable_profile(state, profile)
        record_memory_event(
            state,
            section="stable",
            action=action,
            text=profile["text"],
            reason=(
                "从混合陈述与提问中提取到明确的身份或个人背景"
                if profile["text"] != user_text.strip()
                else "识别到明确的身份或个人背景陈述"
            ),
            score=score,
        )
        return "stable"
    interest = extract_long_term_interest(user_text)
    if interest is not None:
        if memory_exists(interest["text"], state):
            record_memory_event(
                state,
                section="interest",
                action="unchanged",
                text=interest["text"],
                reason="相同或高度相似的兴趣已经存在",
                score=score,
            )
            return "discard"
        outcome = save_interest(interest, state)
        record_memory_event(
            state,
            section="interest",
            action=outcome,
            text=interest["text"],
            reason=(
                "从混合陈述与提问中提取到明确的长期偏好或兴趣"
                if interest["text"] != user_text.strip()
                else "识别到明确的长期偏好或兴趣陈述"
            ),
            score=score,
        )
        return "interest"
    if score >= SCORE_LONG_TERM_THRESHOLD:
        added = update_long_term(state, {
            # An unreliable reading must not be stored as an emotion label; the
            # empty string keeps it out of summaries and the memory panel.
            "text": user_text, "emotion": "" if not emotion_recorded else emo_label,
            "score": float(emo_score), "time": datetime.now().isoformat(),
        })
        record_memory_event(
            state,
            section="long",
            action="added" if added else "unchanged",
            text=user_text,
            reason=f"记忆评分 {score:.2f} 达到长期记忆阈值 {SCORE_LONG_TERM_THRESHOLD:.2f}",
            score=score,
        )
        return "long"
    if score >= SCORE_MID_TERM_THRESHOLD and not emotion_recorded:
        record_memory_event(
            state,
            section="emotion",
            action="skipped",
            text=user_text,
            reason="情绪识别置信度不足，不强行写入情绪标签",
            score=score,
        )
        return "discard"
    if emotion_recorded:
        return "mid"
    reason = "这是回忆查询，不作为新的个人事实保存" if is_memory_query(user_text) else (
        f"记忆评分 {score:.2f} 未达到写入阈值 {SCORE_MID_TERM_THRESHOLD:.2f}"
    )
    record_memory_event(
        state,
        section="none",
        action="skipped",
        text=user_text,
        reason=reason,
        score=score,
    )
    return "discard"
