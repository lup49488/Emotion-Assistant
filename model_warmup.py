"""Non-blocking warmup helpers shared by server entry points.

The API server only warms lightweight supporting models.  Local chat-model
loading remains on demand so a web deployment cannot unexpectedly consume GPU
memory during startup.
"""
from __future__ import annotations

import logging
import os
import threading

import goemotions_local as goemotions
from chatbot import get_embedding_model
from safety_semantic import get_semantic_classifier, semantic_safety_preload_enabled


logger = logging.getLogger(__name__)
_warmup_lock = threading.Lock()
_warmup_started = False
_warmup_status: dict[str, str] = {
    "emotion": "pending", "embedding": "pending", "safety_semantic": "disabled",
}


def _api_preload_enabled() -> bool:
    return os.getenv("API_PRELOAD_MODELS", "true").strip().lower() == "true"


def warmup_api_models() -> None:
    """Load support models without loading the local LLM."""
    loaders = [
        ("emotion", lambda: goemotions.predict_emotion("hello")),
        ("embedding", get_embedding_model),
    ]
    if semantic_safety_preload_enabled():
        with _warmup_lock:
            _warmup_status["safety_semantic"] = "pending"
        loaders.append(("safety_semantic", get_semantic_classifier))
    for label, loader in loaders:
        with _warmup_lock:
            _warmup_status[label] = "running"
        try:
            loader()
            with _warmup_lock:
                _warmup_status[label] = "ready"
            logger.info("API background warmup completed for the %s model.", label)
        except Exception:
            with _warmup_lock:
                _warmup_status[label] = "degraded"
            # A failed optional warmup must never prevent the API from serving.
            logger.exception("API background warmup failed for the %s model.", label)


def start_api_background_warmup() -> bool:
    """Start the API support-model warmup once per process.

    Returns whether this call created a worker thread.  The thread is daemonized
    so application shutdown is never held up by optional model initialization.
    """
    global _warmup_started

    if not _api_preload_enabled():
        with _warmup_lock:
            for label in _warmup_status:
                _warmup_status[label] = "disabled"
        logger.info("API model warmup is disabled by API_PRELOAD_MODELS.")
        return False

    with _warmup_lock:
        if _warmup_started:
            return False
        _warmup_started = True

    threading.Thread(target=warmup_api_models, name="api-model-warmup", daemon=True).start()
    logger.info("API background model warmup thread started.")
    return True


def warmup_status() -> dict[str, str]:
    """Return model-preload state without triggering model initialization."""
    with _warmup_lock:
        return dict(_warmup_status)
