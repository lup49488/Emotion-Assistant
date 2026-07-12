from __future__ import annotations

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "ChnSentiCorp"

_vectorizer: TfidfVectorizer | None = None
_clf: LogisticRegression | None = None
_target_names: list[str] = []


def train_model() -> tuple[TfidfVectorizer, LogisticRegression]:
    global _vectorizer, _clf, _target_names
    if _vectorizer is not None and _clf is not None:
        return _vectorizer, _clf

    import jieba
    from sklearn.datasets import load_files
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    data = load_files(str(DATA_DIR), categories=["neg", "pos"], encoding="utf-8")
    _target_names = list(data.target_names)
    _vectorizer = TfidfVectorizer(tokenizer=jieba.lcut, token_pattern=None, max_features=5000)
    x_train = _vectorizer.fit_transform(data.data)
    _clf = LogisticRegression(max_iter=300)
    _clf.fit(x_train, data.target)
    return _vectorizer, _clf


def predict_sentiment_zh(text: str) -> str:
    vectorizer, clf = train_model()
    pred = int(clf.predict(vectorizer.transform([text]))[0])
    label = _target_names[pred] if 0 <= pred < len(_target_names) else str(pred)
    return "正面" if label == "pos" else "负面"
