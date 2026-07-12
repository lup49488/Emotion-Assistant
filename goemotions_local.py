from __future__ import annotations


EMOTION_MODEL_NAME = "SamLowe/roberta-base-go_emotions"
TRANSLATION_MODEL_NAME = "Helsinki-NLP/opus-mt-zh-en"

_emotion_tokenizer = None
_emotion_model = None
_translation_tokenizer = None
_translation_model = None


def require_torch():
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("缺少依赖 torch，请先安装后再使用情绪识别功能。") from exc
    return torch


def require_transformers():
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, MarianMTModel, MarianTokenizer
    except ModuleNotFoundError as exc:
        raise RuntimeError("缺少依赖 transformers，请先安装后再使用情绪识别功能。") from exc
    return AutoModelForSequenceClassification, AutoTokenizer, MarianMTModel, MarianTokenizer


def get_emotion_model():
    global _emotion_tokenizer, _emotion_model
    AutoModelForSequenceClassification, AutoTokenizer, _, _ = require_transformers()
    if _emotion_tokenizer is None:
        _emotion_tokenizer = AutoTokenizer.from_pretrained(EMOTION_MODEL_NAME)
    if _emotion_model is None:
        _emotion_model = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL_NAME)
        _emotion_model.eval()
    return _emotion_tokenizer, _emotion_model


def get_translation_model():
    global _translation_tokenizer, _translation_model
    _, _, MarianMTModel, MarianTokenizer = require_transformers()
    if _translation_tokenizer is None:
        _translation_tokenizer = MarianTokenizer.from_pretrained(TRANSLATION_MODEL_NAME)
    if _translation_model is None:
        _translation_model = MarianMTModel.from_pretrained(TRANSLATION_MODEL_NAME, use_safetensors=True)
        _translation_model.eval()
    return _translation_tokenizer, _translation_model


def translate_zh_to_en(text: str) -> str:
    torch = require_torch()
    tokenizer, model = get_translation_model()
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.inference_mode():
        outputs = model.generate(**inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def predict_emotion_en(text: str) -> tuple[str, float]:
    torch = require_torch()
    emotion_tokenizer, emotion_model = get_emotion_model()
    inputs = emotion_tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.inference_mode():
        outputs = emotion_model(**inputs)
    probs = torch.sigmoid(outputs.logits)[0]
    label_id = torch.argmax(probs).item()
    label = emotion_model.config.id2label[label_id]
    score = probs[label_id].item()
    return label, float(score)


def predict_emotion_zh(text: str) -> tuple[str, float]:
    return predict_emotion_en(translate_zh_to_en(text))
