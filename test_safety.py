"""Regression tests for privacy-minimized self-harm safety routing."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import chatbot
import safety
from safety import SafetyRiskLevel, assess_safety, check_crisis
from safety_semantic import SemanticSafetySignals


CASES_PATH = Path(__file__).with_name("safety_evaluation_cases.json")


def test_safety_evaluation_set_matches_expected_levels():
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))

    for case in cases:
        assessment = assess_safety(case["text"])
        assert assessment.risk_level.value == case["expected_level"], case["id"]


def test_assessment_exposes_only_structured_reason_codes():
    assessment = assess_safety("I intend to kill myself and have a plan for tonight.")

    assert assessment.risk_level is SafetyRiskLevel.IMMINENT
    assert assessment.intent and assessment.plan and assessment.urgency
    assert assessment.requires_immediate_support
    payload = assessment.to_dict()
    assert "text" not in payload
    assert payload["reason_codes"] == ["self_harm_reference", "first_person", "intent", "plan", "urgency"]


def test_concern_response_asks_a_direct_safety_question_without_emergency_claims():
    response = check_crisis(
        chatbot.SessionState(), "我不想活了，感觉没有人能理解我。", "sadness", 0.9, "zh-cn"
    )

    assert response is not None
    assert "是否有伤害自己" in response
    assert "紧急服务" not in response


def test_imminent_response_interrupts_normal_chat_with_immediate_support_guidance():
    response = check_crisis(
        chatbot.SessionState(), "I intend to kill myself and have a plan for tonight.", "sadness", 0.9, "en"
    )

    assert response is not None
    assert "immediate safety" in response
    assert "emergency services" in response


def test_third_person_context_does_not_interrupt_normal_chat():
    response = check_crisis(
        chatbot.SessionState(), "My friend said he feels suicidal. How can I support him?", "sadness", 0.9, "en"
    )

    assert response is None


def test_crisis_message_is_not_promoted_into_long_or_emotion_memory():
    state = chatbot.SessionState()

    with patch.object(chatbot, "safe_analyze", return_value=("sadness", 0.95)):
        response = list(chatbot.chat(state, "I intend to kill myself and have a plan for tonight."))

    assert response
    assert state.long_memory == []
    assert state.emotion_memory == []
    assert list(state.interest_store.items) == []


def test_semantic_layer_promotes_implicit_first_person_signal_to_concern(monkeypatch):
    monkeypatch.setattr(safety, "semantic_safety_enabled", lambda: True)
    monkeypatch.setattr(
        safety,
        "assess_semantic_safety",
        lambda _text: SemanticSafetySignals(available=True, intent_score=0.91),
    )

    assessment = assess_safety("我想消失，也不想再醒来了。")

    assert assessment.risk_level is SafetyRiskLevel.CONCERN
    assert "semantic_intent" in assessment.reason_codes


def test_semantic_layer_can_escalate_rule_based_concern_but_not_to_imminent(monkeypatch):
    monkeypatch.setattr(safety, "semantic_safety_enabled", lambda: True)
    monkeypatch.setattr(
        safety,
        "assess_semantic_safety",
        lambda _text: SemanticSafetySignals(available=True, intent_score=0.93, plan_or_means_score=0.89),
    )

    assessment = assess_safety("I keep thinking about killing myself.")

    assert assessment.risk_level is SafetyRiskLevel.ELEVATED
    assert assessment.risk_level is not SafetyRiskLevel.IMMINENT


def test_semantic_layer_does_not_override_explicit_denial(monkeypatch):
    monkeypatch.setattr(safety, "semantic_safety_enabled", lambda: True)
    monkeypatch.setattr(
        safety,
        "assess_semantic_safety",
        lambda _text: SemanticSafetySignals(available=True, intent_score=0.99, plan_or_means_score=0.99),
    )

    assessment = assess_safety("我没有自杀的想法，只是最近很累。")

    assert assessment.risk_level is SafetyRiskLevel.NONE
