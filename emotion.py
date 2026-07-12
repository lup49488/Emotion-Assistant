from __future__ import annotations

import logging
import re
from typing import Any

from langdetect import detect

import goemotions_local as goemotions


logger = logging.getLogger(__name__)


def map_emotion_label(label: str | None) -> str:
    mapping = {
        "sadness": "sadness", "grief": "sadness", "disappointment": "sadness",
        "remorse": "sadness", "embarrassment": "sadness", "pessimism": "sadness",
        "anger": "anger", "annoyance": "anger", "disgust": "anger",
        "fear": "fear", "nervousness": "fear",
        "anxiety": "anxiety", "confusion": "anxiety",
        "joy": "joy", "amusement": "joy", "approval": "joy",
        "gratitude": "joy", "love": "joy", "optimism": "joy",
        "pride": "joy", "relief": "joy",
        "neutral": "neutral", "curiosity": "neutral",
        "realization": "neutral", "surprise": "neutral",
    }
    return mapping.get(label or "", "neutral")


def safe_analyze(text: str, lang: str) -> tuple[str, float]:
    try:
        result = (
            goemotions.predict_emotion_zh(text)
            if lang.startswith("zh")
            else goemotions.predict_emotion_en(text)
        )
    except Exception:
        logger.exception("情绪模型处理失败，回退到 neutral。text=%r lang=%r", text, lang)
        return "neutral", 0.0

    if isinstance(result, (list, tuple)):
        if not result:
            logger.warning("情绪模型返回空结果，回退到 neutral")
            return "neutral", 0.0
        raw_label, emo_score, *_ = result
    elif isinstance(result, dict):
        raw_label = result.get("label")
        emo_score = result.get("score", 0.0)
    else:
        logger.warning("未知情绪模型输出格式 %r，回退到 neutral", type(result))
        return "neutral", 0.0

    return map_emotion_label(raw_label), float(emo_score or 0.0)


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
    if not memories:
        return "stable", "unknown"
    last_score = float(memories[-1].get("score", 0.0))
    delta = abs(float(new_score) - last_score)
    negative_emotions = {"sadness", "fear", "anger", "anxiety"}
    if new_label in negative_emotions:
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
