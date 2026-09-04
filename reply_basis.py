from __future__ import annotations

from typing import Any

from emotion import emotion_to_style, intensity_to_level


_NEGATIVE_LABELS = {"sadness", "anger", "fear", "anxiety"}
_MODE_STYLE = {
    "supportive": "温和、稳定、结构化；先回应感受，再给出简短、可选择的下一步。",
    "steady": "自然、清晰、平稳；直接回答问题，避免过度解读。",
    "encouraging": "温暖、真诚地回应积极内容，并保持克制和具体。",
}


def build_reply_basis(
    preference: dict[str, Any], emotion_label: str, emotion_score: float, turn_id: str | None = None,
) -> dict[str, str | bool]:
    """Create the sole user-facing communication cue used by the prompt.

    Raw classifier labels and confidence stay internal to safety and memory code.
    The returned object intentionally contains neither.
    """
    enabled = bool(preference.get("enabled", False))
    correction = preference.get("correction")
    if not enabled:
        return {"enabled": False, "used": False, "mode": "steady", "source": "disabled", "turn_id": turn_id}
    if correction in _MODE_STYLE:
        return {"enabled": True, "used": True, "mode": str(correction), "source": "corrected", "turn_id": turn_id}
    if emotion_label in _NEGATIVE_LABELS:
        mode = "supportive"
    elif emotion_label == "joy":
        mode = "encouraging"
    else:
        mode = "steady"
    return {"enabled": True, "used": True, "mode": mode, "source": "message", "turn_id": turn_id}


def reply_style(basis: dict[str, Any]) -> str:
    mode = str(basis.get("mode", "steady"))
    if basis.get("used"):
        return _MODE_STYLE.get(mode, _MODE_STYLE["steady"])
    return _MODE_STYLE["steady"]


def legacy_emotion_style(label: str, score: float) -> str:
    """Keep the classifier mapping available for non-chat callers only."""
    return emotion_to_style(label, intensity_to_level(score))
