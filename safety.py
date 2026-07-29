from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

from emotion import detect_emotion_fluctuation
from safety_semantic import SemanticSafetySignals, assess_semantic_safety, semantic_safety_enabled


logger = logging.getLogger(__name__)


class SafetyRiskLevel(StrEnum):
    NONE = "none"
    CONCERN = "concern"
    ELEVATED = "elevated"
    IMMINENT = "imminent"


@dataclass(frozen=True)
class SafetyAssessment:
    """Privacy-minimized, rule-based assessment for a single user message.

    ``confidence`` is confidence in the rule match, not a clinical probability
    or diagnosis. ``reason_codes`` deliberately excludes the original text and
    any method details so it is safe to use in operational telemetry.
    """

    risk_level: SafetyRiskLevel
    confidence: float
    self_harm_reference: bool
    first_person: bool
    intent: bool
    plan: bool
    means_access: bool
    urgency: bool
    in_progress: bool
    reason_codes: tuple[str, ...]

    @property
    def requires_direct_checkin(self) -> bool:
        return self.risk_level is not SafetyRiskLevel.NONE

    @property
    def requires_immediate_support(self) -> bool:
        return self.risk_level in {SafetyRiskLevel.ELEVATED, SafetyRiskLevel.IMMINENT}

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "risk_level": self.risk_level.value, "reason_codes": list(self.reason_codes)}


# Retained for compatibility with existing callers.  The structured evaluator
# below adds first-person, denial, plan, means, and urgency context before it
# selects an actual response level.
CRISIS_KEYWORDS = [
    "不想活了", "活不下去", "伤害自己", "自杀", "自残", "想死",
    "suicide", "kill myself", "end my life", "self-harm", "self harm",
]
CRISIS_SEVERE_STREAK_THRESHOLD = 2

_FIRST_PERSON_RE = re.compile(
    r"(?:\b(?:i|me|my|myself)\b|我|本人|自己|我的)", re.IGNORECASE
)
_THIRD_PERSON_CONTEXT_RE = re.compile(
    r"(?:\b(?:he|she|they|friend|someone|character|news|article|movie|book)\b|"
    r"他|她|他们|朋友|同学|新闻|报道|文章|电影|小说|角色)",
    re.IGNORECASE,
)
_GENERAL_DISCUSSION_RE = re.compile(
    r"(?:\b(?:news|article|movie|book|novel|prevention)\b|新闻|报道|文章|电影|小说|预防)",
    re.IGNORECASE,
)
_SELF_HARM_RE = re.compile(
    r"(?:不想活了|活不下去|伤害自己|自杀|自残|想死|结束(?:自己)?的?生命|"
    r"\b(?:suicide|kill(?:ing)? myself|end my life|harm(?:ing)? myself|self[- ]harm)\b)",
    re.IGNORECASE,
)
_IMPLICIT_FIRST_PERSON_RE = re.compile(
    r"(?:不想活了|活不下去|想死|kill(?:ing)? myself|end my life)", re.IGNORECASE
)
_DENIAL_RE = re.compile(
    r"(?:没有(?:自杀|自残|伤害自己的想法|想死)|不想(?:自杀|自残|伤害自己)|"
    r"\b(?:do not|don't|not)\s+(?:want|plan|intend)\s+to\s+(?:die|kill myself|self[- ]harm)\b)",
    re.IGNORECASE,
)
_INTENT_RE = re.compile(
    r"(?:我(?:已经)?(?:想|打算|决定|准备)(?:要)?(?:自杀|自残|伤害自己|结束(?:自己)?的?生命)|"
    r"\b(?:i\s+)?(?:want|plan|intend|decided|am going)\s+to\s+(?:die|kill myself|end my life|harm myself|self[- ]harm)\b)",
    re.IGNORECASE,
)
_PLAN_RE = re.compile(
    r"(?:已经(?:计划|安排|准备)好了|有(?:一个)?(?:计划|安排)|计划(?:已经)?(?:定|好了)|"
    r"\b(?:have|made)\s+(?:a\s+)?plan\b|\bplanned\s+it\b)",
    re.IGNORECASE,
)
_NEGATED_PLAN_RE = re.compile(
    r"(?:没有(?:计划|安排)|还没有(?:计划|安排)|\b(?:do not|don't|no)\s+(?:have|made)\s+(?:a\s+)?plan\b)",
    re.IGNORECASE,
)
_MEANS_RE = re.compile(
    r"(?:手边有(?:危险物品|药物|武器)|已经拿到(?:危险物品|药物|武器)|"
    r"\b(?:have access to|have)\s+(?:dangerous means|medication|a weapon)\b)",
    re.IGNORECASE,
)
_URGENCY_RE = re.compile(
    r"(?:现在|此刻|今晚|马上|立刻|很快|\b(?:right now|tonight|immediately|soon)\b)",
    re.IGNORECASE,
)
_IN_PROGRESS_RE = re.compile(
    r"(?:正在(?:伤害自己|自残)|已经开始(?:伤害自己|自残)|"
    r"\b(?:am|i'm)\s+(?:hurting myself|self[- ]harming)\s+(?:right now|now)\b)",
    re.IGNORECASE,
)
_SEMANTIC_CANDIDATE_RE = re.compile(
    r"(?:想(?:消失|结束一切)|不想(?:醒来|继续撑下去)|没有出路|撑不住了|"
    r"\b(?:want to disappear|do not want to wake up|don't want to wake up|can't go on|no way out)\b)",
    re.IGNORECASE,
)


def detect_crisis_keywords(text: str) -> bool:
    lowered = (text or "").lower()
    return any(keyword.lower() in lowered for keyword in CRISIS_KEYWORDS)


def _risk_level_for_signals(
    *, intent: bool, plan: bool, means_access: bool, urgency: bool, in_progress: bool,
) -> tuple[SafetyRiskLevel, float]:
    if in_progress or (intent and urgency and (plan or means_access)):
        return SafetyRiskLevel.IMMINENT, 0.98
    if (intent and (plan or means_access or urgency)) or (plan and means_access):
        return SafetyRiskLevel.ELEVATED, 0.90
    return SafetyRiskLevel.CONCERN, 0.72


def _supplement_with_semantics(text: str, base: SafetyAssessment) -> SafetyAssessment:
    """Fuse bounded NLI signals without letting a model create imminent risk."""
    blocked_context = {"explicit_denial", "third_person_context", "general_discussion"}
    if not semantic_safety_enabled() or blocked_context.intersection(base.reason_codes):
        return base
    if base.risk_level is SafetyRiskLevel.NONE and not (
        base.first_person and _SEMANTIC_CANDIDATE_RE.search(text)
    ):
        return base

    semantic: SemanticSafetySignals = assess_semantic_safety(text)
    if not semantic.available or not (semantic.intent or semantic.plan_or_means or semantic.immediate_danger):
        return base

    reason_codes = list(base.reason_codes)
    if semantic.intent:
        reason_codes.append("semantic_intent")
    if semantic.plan_or_means:
        reason_codes.append("semantic_plan_or_means")
    if semantic.immediate_danger:
        reason_codes.append("semantic_immediate_danger")

    # A semantic-only match can invite a direct check-in, but cannot jump to an
    # emergency level. Higher levels still require corroborating rule signals.
    if base.risk_level is SafetyRiskLevel.NONE:
        return SafetyAssessment(
            SafetyRiskLevel.CONCERN,
            max(0.72, semantic.intent_score, semantic.plan_or_means_score),
            True,
            base.first_person,
            semantic.intent,
            False,
            False,
            False,
            False,
            tuple(reason_codes),
        )

    intent = base.intent or semantic.intent
    plan = base.plan or semantic.plan_or_means
    level, confidence = _risk_level_for_signals(
        intent=intent,
        plan=plan,
        means_access=base.means_access,
        urgency=base.urgency,
        in_progress=base.in_progress,
    )
    if level is SafetyRiskLevel.IMMINENT and base.risk_level is not SafetyRiskLevel.IMMINENT:
        level = SafetyRiskLevel.ELEVATED
    return SafetyAssessment(
        level,
        max(confidence, semantic.intent_score, semantic.plan_or_means_score),
        base.self_harm_reference,
        base.first_person,
        intent,
        plan,
        base.means_access,
        base.urgency,
        base.in_progress,
        tuple(reason_codes),
    )


def assess_safety(text: str) -> SafetyAssessment:
    """Assess self-harm risk from one message without storing its content.

    This is a conservative routing aid, not a medical assessment. Explicit
    denials and clearly third-person/general discussion remain below concern
    unless the message also contains a first-person safety signal.
    """
    normalized = (text or "").strip()
    if not normalized:
        return SafetyAssessment(SafetyRiskLevel.NONE, 1.0, False, False, False, False, False, False, False, ())

    first_person = bool(_FIRST_PERSON_RE.search(normalized))
    self_harm_lexical = bool(_SELF_HARM_RE.search(normalized))
    implicit_first_person = bool(_IMPLICIT_FIRST_PERSON_RE.search(normalized))
    third_person_only = bool(_THIRD_PERSON_CONTEXT_RE.search(normalized)) and not first_person
    general_discussion = bool(_GENERAL_DISCUSSION_RE.search(normalized))
    intent = bool(_INTENT_RE.search(normalized))
    plan = bool(_PLAN_RE.search(normalized)) and not bool(_NEGATED_PLAN_RE.search(normalized))
    means_access = bool(_MEANS_RE.search(normalized))
    urgency = bool(_URGENCY_RE.search(normalized))
    in_progress = bool(_IN_PROGRESS_RE.search(normalized))
    denial = bool(_DENIAL_RE.search(normalized))
    self_harm_reference = (
        self_harm_lexical
        and (first_person or implicit_first_person)
        and not third_person_only
        and not (general_discussion and not intent and not means_access and not in_progress)
    )

    if denial and not (intent or means_access or in_progress):
        return SafetyAssessment(SafetyRiskLevel.NONE, 0.95, False, first_person, False, False, False, False, False, ("explicit_denial",))
    if not self_harm_reference and not in_progress:
        context_codes: list[str] = []
        if third_person_only:
            context_codes.append("third_person_context")
        if general_discussion:
            context_codes.append("general_discussion")
        base = SafetyAssessment(
            SafetyRiskLevel.NONE, 0.95, False, first_person, False, False, False, False, False, tuple(context_codes)
        )
        return _supplement_with_semantics(normalized, base)

    reason_codes = ["self_harm_reference"]
    if first_person or implicit_first_person:
        reason_codes.append("first_person")
    if intent:
        reason_codes.append("intent")
    if plan:
        reason_codes.append("plan")
    if means_access:
        reason_codes.append("means_access")
    if urgency:
        reason_codes.append("urgency")
    if in_progress:
        reason_codes.append("in_progress")

    level, confidence = _risk_level_for_signals(
        intent=intent, plan=plan, means_access=means_access, urgency=urgency, in_progress=in_progress,
    )
    base = SafetyAssessment(
        level, confidence, self_harm_reference, first_person, intent, plan,
        means_access, urgency, in_progress, tuple(reason_codes),
    )
    return _supplement_with_semantics(normalized, base)


def detect_crisis_trend(emotion_memory: list[dict[str, Any]], current_fluct_level: str) -> bool:
    if current_fluct_level != "severe":
        return False
    recent = emotion_memory[-(CRISIS_SEVERE_STREAK_THRESHOLD - 1):]
    if len(recent) < CRISIS_SEVERE_STREAK_THRESHOLD - 1:
        return False
    negative_emotions = {"sadness", "fear", "anger", "anxiety"}
    return all(
        memory.get("label") in negative_emotions and float(memory.get("score", 0.0)) >= 0.75
        for memory in recent
    )


def crisis_response(lang: str, risk_level: SafetyRiskLevel = SafetyRiskLevel.ELEVATED) -> str:
    if lang.startswith("zh"):
        if risk_level is SafetyRiskLevel.CONCERN:
            return (
                "听起来你现在承受了很多。为了确认你的安全：你现在是否有伤害自己或结束生命的想法？\n\n"
                "如果有，或你不确定自己能否保持安全，请马上联系身边信任的人陪着你，并联系当地危机支持服务或前往最近的急诊。"
            )
        if risk_level is SafetyRiskLevel.IMMINENT:
            return (
                "我非常担心你此刻的安全。请现在就联系当地紧急服务或危机支持热线，或请身边可信任的人立刻陪着你；不要独自待着。\n\n"
                "请尽量远离任何可能让你受伤的物品，前往有其他人在的安全地点。"
            )
        return (
            "听到你现在这么难受，我很担心你的安全。请立即联系身边信任的人，请他们陪着你，并联系当地危机支持服务或前往最近的医院急诊。\n\n"
            "在等待支持时，请尽量不要独自待着，并远离任何可能让你受伤的物品。"
        )
    if risk_level is SafetyRiskLevel.CONCERN:
        return (
            "It sounds like you are carrying a lot. To check on your safety: are you having thoughts of harming yourself or ending your life right now?\n\n"
            "If you are, or are not sure you can stay safe, please contact someone you trust to be with you and reach a local crisis service or emergency department."
        )
    if risk_level is SafetyRiskLevel.IMMINENT:
        return (
            "I'm very concerned about your immediate safety. Please contact local emergency services or a crisis line now, or ask someone you trust to come and stay with you right away.\n\n"
            "Try not to stay alone, and move away from anything that could be used to hurt you."
        )
    return (
        "I'm concerned about your safety. Please contact someone you trust to stay with you and reach a local crisis service or emergency department now.\n\n"
        "While you wait for support, try not to be alone and move away from anything that could be used to hurt you."
    )


def check_crisis(state: Any, user_text: str, emo_label: str, emo_score: float, lang: str) -> str | None:
    assessment = assess_safety(user_text)
    if assessment.risk_level is not SafetyRiskLevel.NONE:
        logger.warning(
            "event=safety_assessment user_id=%s risk_level=%s reason_codes=%s",
            state.user_id,
            assessment.risk_level.value,
            ",".join(assessment.reason_codes),
        )
        return crisis_response(lang, assessment.risk_level)

    fluct_level, _ = detect_emotion_fluctuation(state.emotion_memory, emo_label, emo_score)
    if detect_crisis_trend(state.emotion_memory, fluct_level):
        logger.warning("event=safety_trend user_id=%s", state.user_id)
        return crisis_response(lang, SafetyRiskLevel.ELEVATED)
    return None
