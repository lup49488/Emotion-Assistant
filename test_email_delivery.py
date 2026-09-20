from __future__ import annotations

import io
import logging
import urllib.error

import pytest

import config
import email_delivery


def test_resend_rejection_logs_only_status_and_error_category(monkeypatch, caplog):
    monkeypatch.setattr(config, "EMAIL_AUTH_PROVIDER", "resend")
    monkeypatch.setattr(config, "EMAIL_AUTH_FROM", "Serenova <no-reply@example.test>")
    monkeypatch.setattr(config, "EMAIL_AUTH_RESEND_API_KEY", "test-email-secret")

    def rejected(*_args, **_kwargs):
        raise urllib.error.HTTPError(
            "https://api.resend.com/emails",
            403,
            "Forbidden",
            {},
            io.BytesIO(b'{"name":"restricted_api_key","message":"student@example.test is blocked"}'),
        )

    monkeypatch.setattr(email_delivery.urllib.request, "urlopen", rejected)

    with caplog.at_level(logging.WARNING, logger="email_delivery"):
        with pytest.raises(email_delivery.EmailDeliveryError, match="rejected"):
            email_delivery.send_verification_code("student@example.test", "12345678")

    assert "status=403 category=restricted_api_key" in caplog.text
    assert "student@example.test" not in caplog.text
