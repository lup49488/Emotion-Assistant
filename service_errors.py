"""Stable domain errors that can cross provider, API, and UI boundaries."""
from __future__ import annotations


class ServiceError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
