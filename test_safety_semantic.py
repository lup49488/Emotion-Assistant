"""Unit tests for the optional local multilingual NLI safety supplement."""
from __future__ import annotations

import sys
import threading
import types

import safety_semantic


def test_semantic_classifier_maps_multilingual_candidate_scores(monkeypatch):
    calls = []

    def fake_classifier(text, **kwargs):
        calls.append((text, kwargs))
        labels = kwargs["candidate_labels"]
        return {"labels": [labels[2], labels[0], labels[1]], "scores": [0.81, 0.93, 0.86]}

    monkeypatch.setattr(safety_semantic, "SAFETY_SEMANTIC_ENABLED", True)
    monkeypatch.setattr(safety_semantic, "SAFETY_SEMANTIC_IDLE_UNLOAD_SECONDS", 0)
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


def test_semantic_classifier_defers_load_when_memory_is_low(monkeypatch):
    monkeypatch.setattr(safety_semantic, "SAFETY_SEMANTIC_ENABLED", True)
    monkeypatch.setattr(safety_semantic, "SAFETY_SEMANTIC_MIN_AVAILABLE_MB", 1800)
    monkeypatch.setattr(safety_semantic, "_available_memory_bytes", lambda: 512 * 1024 * 1024)
    monkeypatch.setattr(safety_semantic, "_classifier", None)
    monkeypatch.setattr(safety_semantic, "_load_failed", False)

    assert safety_semantic.get_semantic_classifier() is None
    assert safety_semantic._load_failed is False


def test_concurrent_lazy_load_creates_only_one_classifier(monkeypatch):
    calls = []
    classifier = object()

    def fake_pipeline(*args, **kwargs):
        calls.append((args, kwargs))
        return classifier

    monkeypatch.setattr(safety_semantic, "SAFETY_SEMANTIC_ENABLED", True)
    monkeypatch.setattr(safety_semantic, "SAFETY_SEMANTIC_IDLE_UNLOAD_SECONDS", 0)
    monkeypatch.setattr(safety_semantic, "_memory_allows_load", lambda: True)
    monkeypatch.setattr(safety_semantic, "_classifier", None)
    monkeypatch.setattr(safety_semantic, "_load_failed", False)
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(pipeline=fake_pipeline))

    results = []
    workers = [threading.Thread(target=lambda: results.append(safety_semantic.get_semantic_classifier())) for _ in range(4)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    assert results == [classifier] * 4
    assert len(calls) == 1


def test_release_semantic_classifier_clears_loaded_model(monkeypatch):
    monkeypatch.setattr(safety_semantic, "_classifier", object())
    monkeypatch.setattr(safety_semantic, "_idle_timer", None)

    assert safety_semantic.release_semantic_classifier() is True
    assert safety_semantic._classifier is None
