from __future__ import annotations

from unittest.mock import patch

import chatbot
import dialogue_regression
import memory_store
import pytest
from prompt_builder import build_messages


def test_regression_suite_has_unique_case_ids_and_live_smoke_cases():
    cases = dialogue_regression.load_cases()
    identifiers = [case["id"] for case in cases]

    assert len(identifiers) == len(set(identifiers))
    assert {"identity_serenova", "english_short_greeting", "empty_model_stream"} <= set(identifiers)
    assert sum(bool(case.get("live")) for case in cases) >= 3


@pytest.mark.parametrize("case", [case for case in dialogue_regression.load_cases() if case.get("deterministic") == "profile_extraction"])
def test_profile_regression_cases(case):
    profile = memory_store.extract_personal_profile(case["input"])

    assert profile is not None
    assert case["checks"]["profile_contains"] in profile["text"]


def test_identity_and_language_cases_are_present_in_the_system_prompt():
    cases = {case["id"]: case for case in dialogue_regression.load_cases()}
    system_prompt = build_messages(chatbot.SessionState(), cases["identity_serenova"]["input"], "neutral", 0.0)[0]["content"]

    assert "你的名字是 Serenova" in system_prompt
    assert "普通回答不要主动反复强调自己的名字或身份" in system_prompt
    assert '"Hi"、"Hello" 或 "Hey"' in system_prompt


def test_empty_stream_regression_case_is_retryable_and_not_archived():
    state = chatbot.SessionState()
    with patch.object(chatbot, "safe_analyze", return_value=("neutral", 0.0)), \
        patch.object(chatbot, "smart_memory_filter", return_value="discard"), \
        patch.object(chatbot, "stream_model_response", return_value=iter(())):
        with pytest.raises(chatbot.ServiceError) as captured:
            list(chatbot.chat(state, "regression empty stream", use_style=False))

    checks = next(case["checks"] for case in dialogue_regression.load_cases() if case["id"] == "empty_model_stream")
    assert captured.value.code == checks["error_code"]
    assert captured.value.retryable is checks["retryable"]
    assert state.history == []


def test_reply_validator_checks_identity_and_language_contracts():
    cases = {case["id"]: case for case in dialogue_regression.load_cases()}

    assert dialogue_regression.validate_reply(cases["identity_serenova"], "My name is Serenova.") == []
    assert dialogue_regression.validate_reply(cases["english_short_greeting"], "Hallo, wie geht es dir?")
