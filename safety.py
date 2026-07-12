from __future__ import annotations

import logging
from typing import Any

from emotion import detect_emotion_fluctuation


logger = logging.getLogger(__name__)

CRISIS_KEYWORDS = [
    "不想活了", "活不下去", "伤害自己", "自杀", "自残",
    "想死", "没有意义", "解脱",
    "suicide", "kill myself", "end my life", "self-harm", "self harm",
]
CRISIS_SEVERE_STREAK_THRESHOLD = 2


def detect_crisis_keywords(text: str) -> bool:
    lowered = text.lower()
    return any(kw.lower() in lowered for kw in CRISIS_KEYWORDS)


def detect_crisis_trend(emotion_memory: list[dict[str, Any]], current_fluct_level: str) -> bool:
    if current_fluct_level != "severe":
        return False
    recent = emotion_memory[-(CRISIS_SEVERE_STREAK_THRESHOLD - 1):]
    if len(recent) < CRISIS_SEVERE_STREAK_THRESHOLD - 1:
        return False
    negative_emotions = {"sadness", "fear", "anger", "anxiety"}
    return all(
        m.get("label") in negative_emotions and float(m.get("score", 0.0)) >= 0.75
        for m in recent
    )


def crisis_response(lang: str) -> str:
    if lang.startswith("zh"):
        return (
            "听到你现在这么难受，我很担心你。这些感受是真实的，你不需要一个人扛着。\n\n"
            "如果你现在有伤害自己的想法，请立即联系当地的心理危机干预热线，"
            "或前往最近的医院急诊。如果方便的话，也可以联系身边信任的人陪着你。\n\n"
            "我在这里，但更希望你能得到专业、及时的现实支持。"
        )
    return (
        "I'm really concerned about how much pain you're in right now. "
        "These feelings are real, and you don't have to carry them alone.\n\n"
        "If you're having thoughts of harming yourself, please reach out to a local "
        "crisis line right away, or go to your nearest emergency room. "
        "If possible, also reach out to someone you trust who can be with you.\n\n"
        "I'm here, but I really want you to get real, professional support right now."
    )


def check_crisis(state: Any, user_text: str, emo_label: str, emo_score: float, lang: str) -> str | None:
    if detect_crisis_keywords(user_text):
        logger.warning("检测到危机关键词，触发代码层兜底应对。user_id=%s", state.user_id)
        return crisis_response(lang)
    fluct_level, _ = detect_emotion_fluctuation(state.emotion_memory, emo_label, emo_score)
    if detect_crisis_trend(state.emotion_memory, fluct_level):
        logger.warning("检测到连续严重情绪波动趋势，触发代码层兜底应对。user_id=%s", state.user_id)
        return crisis_response(lang)
    return None
