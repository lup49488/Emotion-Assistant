from __future__ import annotations

import logging
import re
from typing import Any

from langdetect import detect

import goemotions_local as goemotions
from config import (
    EMOTION_ANXIETY_REFINEMENT,
    EMOTION_CONFIDENCE_THRESHOLD,
    NEGATIVE_EMOTIONS,
)


logger = logging.getLogger(__name__)


def map_emotion_label(label: str | None) -> str:
    mapping = {
        "sadness": "sadness", "grief": "sadness", "disappointment": "sadness",
        "remorse": "sadness", "embarrassment": "sadness", "pessimism": "sadness",
        "anger": "anger", "annoyance": "anger", "disgust": "anger",
        "contempt": "anger", "frustration": "anger",
        "fear": "fear", "nervousness": "fear",
        "anxiety": "anxiety", "confusion": "anxiety",
        "joy": "joy", "amusement": "joy", "approval": "joy",
        "gratitude": "joy", "love": "joy", "optimism": "joy",
        "pride": "joy", "relief": "joy",
        "neutral": "neutral", "curiosity": "neutral",
        "realization": "neutral", "surprise": "neutral",
    }
    return mapping.get((label or "").strip().lower(), "uncertain")


# The multilingual emotion model has no anxiety label, so anticipatory worry
# arrives as "fear". These cues separate future-directed, ruminative worry from
# an acute fright, and cover both Chinese and English input.
ANXIETY_CUES = (
    "焦虑", "紧张", "担心", "担忧", "忧虑", "不安", "忐忑", "心慌", "慌张",
    "压力", "睡不着", "失眠", "万一", "会不会", "胡思乱想", "坐立不安", "放不下",
    # "worri" covers worries / worried / worrying; "worry" is spelled separately.
    "anxious", "anxiety", "nervous", "worri", "worry", "uneasy", "restless",
    "stress", "overthink", "panic", "insomnia", "on edge", "what if",
)

# 含线索词但并不表达焦虑的固定说法：「时间紧张」这类客观描述，以及 "no worries" 这类口头语。
# 先整体剔除，避免它们触发细分。
_NON_EMOTIONAL_CUE_PATTERN = re.compile(
    r"(?:时间|气氛|关系|局势|资金|供应|比赛|日程|行程)[^，。！？,.!?]{0,3}紧张"
    r"|\bno worries\b|\bnot to worry\b"
)

# 出现在线索词前方的否定表达：「别担心」「我并不焦虑」「don't worry」等是宽慰或否认，
# 不应判为焦虑。「别」用否定型后顾排除“特别/分别”这类词内误命中。
_ZH_NEGATOR_PATTERN = re.compile(
    r"(?<![特差分个区派级告送识类辨])别|不用|不必|不要|不太|无需|并不|没什么[好可]|没有必要|不至于"
)
_EN_NEGATOR_PATTERN = re.compile(r"\b(?:not|never|no need|nothing)\b|\bdon'?t\b")
# 仅回看这么多字符，避免把远处无关的否定词算到当前线索头上。
# 英文否定词与线索之间常隔着 "need to" / "to" 之类的虚词，所以窗口比中文宽。
_ZH_NEGATION_WINDOW = 6
_EN_NEGATION_WINDOW = 16


def _is_negated(text: str, cue_start: int) -> bool:
    # 紧贴线索词的单个「不」才算否定（不担心 / 不焦虑），否则「不知道为什么很焦虑」会被误判。
    if cue_start > 0 and text[cue_start - 1] == "不":
        return True
    if _ZH_NEGATOR_PATTERN.search(text[max(0, cue_start - _ZH_NEGATION_WINDOW):cue_start]):
        return True
    return bool(
        _EN_NEGATOR_PATTERN.search(text[max(0, cue_start - _EN_NEGATION_WINDOW):cue_start])
    )


def has_anxiety_cue(text: str) -> bool:
    """Report whether the text carries at least one non-negated anxiety cue."""
    lowered = _NON_EMOTIONAL_CUE_PATTERN.sub("", (text or "").lower())
    for cue in ANXIETY_CUES:
        start = lowered.find(cue)
        while start != -1:
            if not _is_negated(lowered, start):
                return True
            start = lowered.find(cue, start + 1)
    return False


def refine_fear_as_anxiety(label: str, text: str) -> str:
    """Split anticipatory worry out of the model's single fear label.

    Only a fear reading is refined: the model has already recognised the
    fear family, so this never invents an emotion the model did not see.
    """
    if not EMOTION_ANXIETY_REFINEMENT or label != "fear":
        return label
    return "anxiety" if has_anxiety_cue(text) else label


def safe_analyze(text: str, lang: str) -> tuple[str, float]:
    try:
        result = goemotions.predict_emotion(text)
    except Exception:
        logger.exception("情绪模型处理失败，不写入情绪标签。text=%r lang=%r", text, lang)
        return "uncertain", 0.0

    if isinstance(result, (list, tuple)):
        if not result:
            logger.warning("情绪模型返回空结果，不写入情绪标签")
            return "uncertain", 0.0
        raw_label, emo_score, *_ = result
    elif isinstance(result, dict):
        raw_label = result.get("label")
        emo_score = result.get("score", 0.0)
    else:
        logger.warning("未知情绪模型输出格式 %r，不写入情绪标签", type(result))
        return "uncertain", 0.0

    score = float(emo_score or 0.0)
    label = refine_fear_as_anxiety(map_emotion_label(raw_label), text)
    if score < EMOTION_CONFIDENCE_THRESHOLD or label == "uncertain":
        logger.info(
            "情绪置信度不足或标签未映射，不写入情绪标签。label=%r score=%.3f threshold=%.3f",
            raw_label,
            score,
            EMOTION_CONFIDENCE_THRESHOLD,
        )
        return "uncertain", score
    return label, score


def intensity_to_level(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.40:
        return "medium"
    return "low"


def emotion_to_style(label: str, level: str) -> str:
    styles: dict[str, dict[str, str]] = {
        "sadness": {"high": "非常温柔、克制，先共情再给小步骤建议。", "medium": "温和理解，帮助用户把感受说清楚。", "low": "轻柔支持，保持积极但不夸张。"},
        "anger":   {"high": "冷静稳定，先承认情绪，再帮助用户降低冲突。", "medium": "理性、理解，帮助用户整理事实和诉求。", "low": "平和引导，避免火上浇油。"},
        "fear":    {"high": "安全感优先，简短、稳定、可执行。", "medium": "温和安抚，并给出清晰下一步。", "low": "轻松但可靠，帮助建立信心。"},
        "anxiety": {"high": "非常稳定，减少信息量，帮助用户慢慢拆解。", "medium": "温和、结构化，给出简短建议。", "low": "轻松自然，适度鼓励。"},
        "joy":     {"high": "积极回应，分享用户的开心，但保持真实。", "medium": "愉快自然。", "low": "轻松温暖。"},
        "neutral": {"high": "自然、清晰、平稳。", "medium": "自然、清晰、平稳。", "low": "自然、清晰、平稳。"},
    }
    return styles.get(label, styles["neutral"]).get(level, styles["neutral"]["medium"])


def detect_emotion_fluctuation(
    memories: list[dict[str, Any]], new_label: str, new_score: float
) -> tuple[str, str]:
    if not memories or new_label == "uncertain":
        return "stable", "unknown"
    last_score = float(memories[-1].get("score", 0.0))
    delta = abs(float(new_score) - last_score)
    if new_label in NEGATIVE_EMOTIONS:
        direction = "worse" if new_score > last_score else "better"
    else:
        direction = "better" if new_score > last_score else "worse"

    if delta < 0.15:
        return "stable", direction
    if delta < 0.35:
        return "mild", direction
    if delta < 0.60:
        return "moderate", direction
    return "severe", direction


def fluctuation_to_style(fluct_level: str) -> str:
    styles = {
        "stable":   "保持自然、温和、稳定的语气。",
        "mild":     "语气稍微更柔和，给一点支持和确认。",
        "moderate": "更明确地表达理解，并帮助用户整理情绪和问题。",
        "severe":   "非常温和、简洁、稳定，优先安抚和陪伴。",
    }
    return styles.get(fluct_level, styles["stable"])


def summarize_emotion_trend(emotion_memory: list[dict[str, Any]]) -> str:
    if not emotion_memory:
        return "暂无足够的历史情绪记录。"
    recent = emotion_memory[-5:]
    labels = [str(e.get("label", "neutral")) for e in recent]
    dominant = max(set(labels), key=labels.count)
    avg_score = sum(float(e.get("score", 0.0)) for e in recent) / len(recent)
    return f"最近主要情绪为 {dominant}，平均强度 {avg_score:.2f}，程度 {intensity_to_level(avg_score)}。"


def detect_lang(text: str) -> str:
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh-cn"
    try:
        return detect(text)
    except Exception:
        return "unknown"


def update_preferences(state: Any, user_text: str, lang: str) -> None:
    state.preferences["language"] = lang
    tone_patterns = {
        "gentle":   ["温柔", "柔和", "gentle"],
        "direct":   ["直接", "简洁", "direct"],
        "humorous": ["幽默", "humorous"],
        "formal":   ["正式", "formal"],
        "casual":   ["随意", "自然", "casual"],
    }
    for tone, patterns in tone_patterns.items():
        if any(p in user_text for p in patterns):
            state.preferences["tone"] = tone
            break
