from __future__ import annotations

import re


_nlp_en = None
_jieba = None


def detect_lang(text: str) -> str:
    return "zh" if re.search(r"[\u4e00-\u9fff]", text) else "en"


def get_nlp_en():
    global _nlp_en
    import spacy

    if _nlp_en is None:
        try:
            _nlp_en = spacy.load("en_core_web_sm")
        except OSError:
            _nlp_en = spacy.blank("en")
    return _nlp_en


def get_jieba():
    global _jieba
    if _jieba is None:
        import jieba

        _jieba = jieba
    return _jieba


def preprocess_zh(text: str) -> str:
    return " ".join(token for token in get_jieba().lcut(text) if token.strip())


def preprocess_en(text: str) -> str:
    doc = get_nlp_en()(text)
    tokens = []
    for token in doc:
        if token.is_stop or token.is_punct or not token.text.strip():
            continue
        tokens.append((token.lemma_ or token.text).lower())
    return " ".join(tokens)


def preprocess(text: str) -> str:
    return preprocess_zh(text) if detect_lang(text) == "zh" else preprocess_en(text)
