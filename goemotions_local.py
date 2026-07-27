from __future__ import annotations

from config import EMOTION_MODEL_MULTI_LABEL, EMOTION_MODEL_NAME

_emotion_tokenizer = None
_emotion_model = None


def require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("缺少依赖 torch，请先安装后再使用情绪识别功能。") from exc
    return torch


def require_transformers():
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise RuntimeError("缺少依赖 transformers，请先安装后再使用情绪识别功能。") from exc
    return AutoModelForSequenceClassification, AutoTokenizer


def get_emotion_model():
    global _emotion_tokenizer, _emotion_model
    AutoModelForSequenceClassification, AutoTokenizer = require_transformers()
    if _emotion_tokenizer is None:
        _emotion_tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL_NAME)
    if _emotion_model is None:
        _emotion_model = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL_NAME)
        _emotion_model.eval()
    return _emotion_tokenizer, _emotion_model


def predict_emotion(text: str) -> tuple[str, float]:
    """Classify the original text with a multilingual emotion model."""
    torch = require_torch()
    emotion_tokenizer, emotion_model = get_emotion_model()
    inputs = emotion_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.inference_mode():
        outputs = emotion_model(**inputs)
    is_multi_label = (
        getattr(emotion_model.config, "problem_type", "") == "multi_label_classification"
        or EMOTION_MODEL_MULTI_LABEL
    )
    probs = torch.sigmoid(outputs.logits)[0] if is_multi_label else torch.softmax(outputs.logits, dim=-1)[0]
    label_id = torch.argmax(probs).item()
    id2label = emotion_model.config.id2label
    label = id2label.get(label_id, id2label.get(str(label_id), str(label_id)))
    score = probs[label_id].item()
    return label, float(score)


def predict_emotion_en(text: str) -> tuple[str, float]:
    """Compatibility alias; input is no longer translated before inference."""
    return predict_emotion(text)


def predict_emotion_zh(text: str) -> tuple[str, float]:
    """Compatibility alias; input is no longer translated before inference."""
    return predict_emotion(text)
