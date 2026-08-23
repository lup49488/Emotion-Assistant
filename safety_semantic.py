"""Optional local multilingual NLI signals for the safety routing layer."""
from __future__ import annotations

import ctypes
import gc
import logging
import os
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from config import (
    SAFETY_SEMANTIC_ENABLED,
    SAFETY_SEMANTIC_IDLE_UNLOAD_SECONDS,
    SAFETY_SEMANTIC_LOAD_RETRY_SECONDS,
    SAFETY_SEMANTIC_MIN_AVAILABLE_MB,
    SAFETY_SEMANTIC_MODEL_NAME,
    SAFETY_SEMANTIC_PRELOAD,
    SAFETY_SEMANTIC_THRESHOLD,
)


logger = logging.getLogger(__name__)
_classifier = None
_load_failed = False
_load_failed_at = 0.0
_classifier_lock = threading.RLock()
_load_deferred_at = 0.0
_idle_timer: threading.Timer | None = None
_last_used_at = 0.0


@dataclass(frozen=True)
class SemanticSafetySignals:
    """Model scores only; no user text or method details are retained."""

    available: bool
    intent_score: float = 0.0
    plan_or_means_score: float = 0.0
    immediate_danger_score: float = 0.0

    @property
    def intent(self) -> bool:
        return self.intent_score >= SAFETY_SEMANTIC_THRESHOLD

    @property
    def plan_or_means(self) -> bool:
        return self.plan_or_means_score >= SAFETY_SEMANTIC_THRESHOLD

    @property
    def immediate_danger(self) -> bool:
        return self.immediate_danger_score >= SAFETY_SEMANTIC_THRESHOLD


_CHINESE_RE = re.compile(r"[\u4e00-\u9fff]")
_HYPOTHESES = {
    "en": {
        "intent": "the user is expressing an intent to harm themself",
        "plan_or_means": "the user has a self-harm plan or access to dangerous means",
        "immediate_danger": "the user may be in immediate danger of self-harm",
    },
    "zh": {
        "intent": "用户表达了伤害自己的意图",
        "plan_or_means": "用户有伤害自己的计划或可及的危险手段",
        "immediate_danger": "用户可能正处于立即伤害自己的危险中",
    },
}


def semantic_safety_enabled() -> bool:
    return SAFETY_SEMANTIC_ENABLED


def semantic_safety_preload_enabled() -> bool:
    return semantic_safety_enabled() and SAFETY_SEMANTIC_PRELOAD


def _read_int(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="ascii").strip()
        return None if value == "max" else int(value)
    except (OSError, ValueError):
        return None


def _available_memory_bytes() -> int | None:
    """Return the tighter of host RAM and Docker cgroup memory availability."""
    host_available = None
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]
        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            host_available = int(status.ullAvailPhys)
    else:
        try:
            for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
                if line.startswith("MemAvailable:"):
                    host_available = int(line.split()[1]) * 1024
                    break
        except OSError:
            pass
        if host_available is None:
            try:
                host_available = int(os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
            except (AttributeError, OSError, ValueError):
                pass

    cgroup_available = None
    for limit_path, usage_path in (
        (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory.current")),
        (Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"), Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")),
    ):
        limit, usage = _read_int(limit_path), _read_int(usage_path)
        if limit is not None and usage is not None and 0 < limit < (1 << 60):
            cgroup_available = max(0, limit - usage)
            break

    values = [value for value in (host_available, cgroup_available) if value is not None]
    return min(values) if values else None


def _memory_allows_load() -> bool:
    required = SAFETY_SEMANTIC_MIN_AVAILABLE_MB * 1024 * 1024
    available = _available_memory_bytes()
    if required <= 0 or available is None or available >= required:
        return True
    logger.warning(
        "event=safety_semantic_load_deferred available_mb=%d required_mb=%d",
        available // (1024 * 1024),
        SAFETY_SEMANTIC_MIN_AVAILABLE_MB,
    )
    return False


def _release_classifier_locked() -> bool:
    global _classifier, _idle_timer, _last_used_at
    if _idle_timer is not None:
        _idle_timer.cancel()
        _idle_timer = None
    if _classifier is None:
        return False
    _classifier = None
    _last_used_at = 0.0
    gc.collect()
    if sys.platform.startswith("linux"):
        try:
            ctypes.CDLL(None).malloc_trim(0)
        except (AttributeError, OSError):
            pass
    logger.info("event=safety_semantic_unloaded reason=idle")
    return True


def _mark_used() -> None:
    """Record the last use and re-arm the idle timer, briefly under the lock."""
    global _last_used_at
    with _classifier_lock:
        _last_used_at = time.monotonic()
        _schedule_idle_unload_locked()


def release_semantic_classifier() -> bool:
    """Release classifier weights immediately, primarily for shutdown and tests."""
    with _classifier_lock:
        return _release_classifier_locked()


def _idle_unload() -> None:
    global _idle_timer
    with _classifier_lock:
        idle_for = time.monotonic() - _last_used_at
        if _classifier is None:
            _idle_timer = None
        elif idle_for >= SAFETY_SEMANTIC_IDLE_UNLOAD_SECONDS:
            _release_classifier_locked()
        else:
            _schedule_idle_unload_locked(SAFETY_SEMANTIC_IDLE_UNLOAD_SECONDS - idle_for)


def _schedule_idle_unload_locked(delay: float | None = None) -> None:
    global _idle_timer
    if _idle_timer is not None:
        _idle_timer.cancel()
        _idle_timer = None
    if SAFETY_SEMANTIC_IDLE_UNLOAD_SECONDS <= 0 or _classifier is None:
        return
    _idle_timer = threading.Timer(
        max(0.1, delay if delay is not None else SAFETY_SEMANTIC_IDLE_UNLOAD_SECONDS),
        _idle_unload,
    )
    _idle_timer.daemon = True
    _idle_timer.start()


def get_semantic_classifier():
    """Load the local classifier once, subject to retry and RAM guards."""
    global _classifier, _load_failed, _load_failed_at, _load_deferred_at, _last_used_at
    if not semantic_safety_enabled():
        return None
    with _classifier_lock:
        now = time.monotonic()
        if _classifier is not None:
            _last_used_at = now
            _schedule_idle_unload_locked()
            return _classifier
        if _load_failed and now - _load_failed_at < SAFETY_SEMANTIC_LOAD_RETRY_SECONDS:
            return None
        if _load_deferred_at and now - _load_deferred_at < SAFETY_SEMANTIC_LOAD_RETRY_SECONDS:
            return None
        if not _memory_allows_load():
            # A RAM shortage is transient, so this stays separate from _load_failed,
            # but it still needs a cooldown: memory is tightest exactly when requests
            # pile up, and re-probing per message only floods the log.
            _load_deferred_at = now
            return None
        _load_deferred_at = 0.0
        try:
            from transformers import pipeline

            _classifier = pipeline(
                "zero-shot-classification", model=SAFETY_SEMANTIC_MODEL_NAME, device=-1
            )
        except Exception as exc:
            _load_failed = True
            _load_failed_at = now
            logger.warning(
                "event=safety_semantic_unavailable error_type=%s", type(exc).__name__
            )
            return None
        _load_failed = False
        _load_failed_at = 0.0
        _last_used_at = time.monotonic()
        _schedule_idle_unload_locked()
        logger.info("event=safety_semantic_loaded model=%s", SAFETY_SEMANTIC_MODEL_NAME)
        return _classifier


def assess_semantic_safety(text: str) -> SemanticSafetySignals:
    """Return constrained NLI scores, falling back safely when unavailable."""
    normalized = (text or "").strip()
    if not normalized:
        return SemanticSafetySignals(available=False)

    labels = _HYPOTHESES["zh" if _CHINESE_RE.search(normalized) else "en"]
    hypothesis_template = "这段消息表示{}。" if _CHINESE_RE.search(normalized) else "This message means that {}."
    # The lock covers the classifier's lifecycle, not the inference. Holding it
    # across a CPU-bound NLI pass would serialize every concurrent request —
    # exactly when several people are sending distress signals at once. Running
    # outside it is safe: `classifier` keeps the object alive even if the idle
    # timer clears the module reference mid-pass.
    classifier = get_semantic_classifier()
    if classifier is None:
        return SemanticSafetySignals(available=False)
    try:
        result = classifier(
            normalized,
            candidate_labels=list(labels.values()),
            multi_label=True,
            hypothesis_template=hypothesis_template,
        )
    except Exception as exc:
        logger.warning("event=safety_semantic_inference_failed error_type=%s", type(exc).__name__)
        return SemanticSafetySignals(available=False)
    finally:
        _mark_used()

    scores = dict(zip(result.get("labels", []), result.get("scores", []), strict=False))
    return SemanticSafetySignals(
        available=True,
        intent_score=float(scores.get(labels["intent"], 0.0)),
        plan_or_means_score=float(scores.get(labels["plan_or_means"], 0.0)),
        immediate_danger_score=float(scores.get(labels["immediate_danger"], 0.0)),
    )
