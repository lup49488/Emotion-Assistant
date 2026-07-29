"""Optional local multilingual NLI signals for the safety routing layer."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from config import (
    SAFETY_SEMANTIC_ENABLED,
    SAFETY_SEMANTIC_MODEL_NAME,
    SAFETY_SEMANTIC_PRELOAD,
    SAFETY_SEMANTIC_THRESHOLD,
)


logger = logging.getLogger(__name__)
_classifier = None
_load_failed = False


@dataclass(frozen=True)
class SemanticSafetySignals:
    """Model scores only; no user text or method details are retained."""

    available: bool
    intent_score: float = 0.0
    plan_or_means_score: float = 0.0
    immediate_danger_score: float = 0.0

    @property
    def intent(self) -> bool:
        return self.intent_score >= SAFETY_SEMANTIC_THRESHOLD

    @property
    def plan_or_means(self) -> bool:
        return self.plan_or_means_score >= SAFETY_SEMANTIC_THRESHOLD

    @property
    def immediate_danger(self) -> bool:
        return self.immediate_danger_score >= SAFETY_SEMANTIC_THRESHOLD


_CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
_HYPOTHESES = {
    "en": {
        "intent": "the user is expressing an intent to harm themself",
        "plan_or_means": "the user has a self-harm plan or access to dangerous means",
        "immediate_danger": "the user may be in immediate danger of self-harm",
    },
    "zh": {
        "intent": "用户表达了伤害自己的意图",
        "plan_or_means": "用户有伤害自己的计划或可及的危险手段",
        "immediate_danger": "用户可能正处于立即伤害自己的危险中",
    },
}


def semantic_safety_enabled() -> bool:
    return SAFETY_SEMANTIC_ENABLED


def semantic_safety_preload_enabled() -> bool:
    return semantic_safety_enabled() and SAFETY_SEMANTIC_PRELOAD


def get_semantic_classifier():
    """Load the local classifier lazily so ordinary deployments stay light."""
    global _classifier, _load_failed
    if not semantic_safety_enabled() or _load_failed:
        return None
    if _classifier is None:
        try:
            from transformers import pipeline

            _classifier = pipeline(
                "zero-shot-classification", model=SAFETY_SEMANTIC_MODEL_NAME, device=-1
            )
        except Exception as exc:
            _load_failed = True
            logger.warning(
                "event=safety_semantic_unavailable error_type=%s", type(exc).__name__
            )
            return None
    return _classifier


def assess_semantic_safety(text: str) -> SemanticSafetySignals:
    """Return constrained NLI scores, falling back safely when unavailable."""
    classifier = get_semantic_classifier()
    if classifier is None or not (text or "").strip():
        return SemanticSafetySignals(available=False)

    labels = _HYPOTHESES["zh" if _CHINESE_RE.search(text) else "en"]
    hypothesis_template = "这段消息表示{}。" if _CHINESE_RE.search(text) else "This message means that {}."
    try:
        result = classifier(
            text,
            candidate_labels=list(labels.values()),
            multi_label=True,
            hypothesis_template=hypothesis_template,
        )
    except Exception as exc:
        logger.warning("event=safety_semantic_inference_failed error_type=%s", type(exc).__name__)
        return SemanticSafetySignals(available=False)

    scores = dict(zip(result.get("labels", []), result.get("scores", []), strict=False))
    return SemanticSafetySignals(
        available=True,
        intent_score=float(scores.get(labels["intent"], 0.0)),
        plan_or_means_score=float(scores.get(labels["plan_or_means"], 0.0)),
        immediate_danger_score=float(scores.get(labels["immediate_danger"], 0.0)),
    )
