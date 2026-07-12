from __future__ import annotations


MODEL_NAME = "nlptown/bert-base-multilingual-uncased-sentiment"
_star_model = None


def get_star_model():
    global _star_model
    if _star_model is None:
        try:
            from transformers import pipeline
        except ModuleNotFoundError as exc:
            raise RuntimeError("缺少依赖 transformers，请先安装后再使用星级情感分析功能。") from exc
        _star_model = pipeline("sentiment-analysis", model=MODEL_NAME)
    return _star_model


def predict_star(text: str) -> tuple[str, float]:
    result = get_star_model()(text)[0]
    return result["label"], float(result["score"])
