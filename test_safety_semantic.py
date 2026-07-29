"""Unit tests for the optional local multilingual NLI safety supplement."""
from __future__ import annotations

import safety_semantic


def test_semantic_classifier_maps_multilingual_candidate_scores(monkeypatch):
    calls = []

    def fake_classifier(text, **kwargs):
        calls.append((text, kwargs))
        labels = kwargs["candidate_labels"]
        return {"labels": [labels[2], labels[0], labels[1]], "scores": [0.81, 0.93, 0.86]}

    monkeypatch.setattr(safety_semantic, "SAFETY_SEMANTIC_ENABLED", True)
    monkeypatch.setattr(safety_semantic, "_classifier", fake_classifier)
    monkeypatch.setattr(safety_semantic, "_load_failed", False)

    signals = safety_semantic.assess_semantic_safety("我想消失，也不想再醒来了。")

    assert signals.available
    assert signals.intent_score == 0.93
    assert signals.plan_or_means_score == 0.86
    assert signals.immediate_danger_score == 0.81
    assert signals.intent and signals.plan_or_means and signals.immediate_danger
    assert calls[0][1]["hypothesis_template"] == "这段消息表示{}。"


def test_semantic_classifier_stays_unloaded_when_disabled(monkeypatch):
    monkeypatch.setattr(safety_semantic, "SAFETY_SEMANTIC_ENABLED", False)
    monkeypatch.setattr(safety_semantic, "_classifier", None)

    assert safety_semantic.assess_semantic_safety("I do not want to wake up.").available is False
