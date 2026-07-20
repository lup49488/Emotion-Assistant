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


logger = logging.getLogger(__name__)
_warmup_lock = threading.Lock()
_warmup_started = False


def _api_preload_enabled() -> bool:
    return os.getenv("API_PRELOAD_MODELS", "true").strip().lower() == "true"


def warmup_api_models() -> None:
    """Load support models without loading the local LLM."""
    for label, loader in (
        ("emotion", lambda: goemotions.predict_emotion_zh("hello")),
        ("embedding", get_embedding_model),
    ):
        try:
            loader()
            logger.info("API background warmup completed for the %s model.", label)
        except Exception:
            # A failed optional warmup must never prevent the API from serving.
            logger.exception("API background warmup failed for the %s model.", label)


def start_api_background_warmup() -> bool:
    """Start the API support-model warmup once per process.

    Returns whether this call created a worker thread.  The thread is daemonized
    so application shutdown is never held up by optional model initialization.
    """
    global _warmup_started

    if not _api_preload_enabled():
        logger.info("API model warmup is disabled by API_PRELOAD_MODELS.")
        return False

    with _warmup_lock:
        if _warmup_started:
            return False
        _warmup_started = True

    threading.Thread(target=warmup_api_models, name="api-model-warmup", daemon=True).start()
    logger.info("API background model warmup thread started.")
    return True
