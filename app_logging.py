from __future__ import annotations

import logging
from collections import deque
from threading import Lock


class InMemoryLogHandler(logging.Handler):
    def __init__(self, capacity: int = 300) -> None:
        super().__init__()
        self._records: deque[str] = deque(maxlen=capacity)
        self._lock = Lock()
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
        except Exception:
            message = record.getMessage()
        with self._lock:
            self._records.append(message)

    def snapshot(self, limit: int = 120) -> str:
        with self._lock:
            records = list(self._records)[-limit:]
        return "\n".join(records) if records else "暂无日志"

    def clear(self) -> None:
        with self._lock:
            self._records.clear()


_memory_handler: InMemoryLogHandler | None = None


def setup_gui_logging(level: int = logging.INFO) -> InMemoryLogHandler:
    global _memory_handler
    root = logging.getLogger()
    root.setLevel(min(root.level or level, level))
    for noisy_logger in ["httpx", "httpcore", "urllib3", "gradio"]:
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
    if _memory_handler is None:
        _memory_handler = InMemoryLogHandler()
        _memory_handler.setLevel(level)
        root.addHandler(_memory_handler)
    return _memory_handler


def get_log_text(limit: int = 120) -> str:
    return setup_gui_logging().snapshot(limit)


def clear_logs() -> str:
    setup_gui_logging().clear()
    return "暂无日志"
