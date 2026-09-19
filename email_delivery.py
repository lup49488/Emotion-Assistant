"""Outbound verification-email delivery with no secrets in browser code."""
from __future__ import annotations

import json
import urllib.error
import urllib.request

import config


class EmailDeliveryError(RuntimeError):
    pass


def configured() -> bool:
    return bool(
        config.EMAIL_AUTH_PROVIDER == "resend"
        and config.EMAIL_AUTH_FROM
        and config.EMAIL_AUTH_RESEND_API_KEY
    )


def send_verification_code(email: str, code: str) -> None:
    """Send one generic email challenge through Resend's HTTPS API."""
    if not configured():
        raise EmailDeliveryError("Email delivery is not configured.")
    payload = json.dumps(
        {
            "from": config.EMAIL_AUTH_FROM,
            "to": [email],
            "subject": "Your Serenova verification code",
            "text": (
                f"Your Serenova verification code is: {code}\n\n"
                f"It expires in {config.EMAIL_AUTH_CODE_TTL_SECONDS // 60} minutes. "
                "Do not share this code with anyone."
            ),
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        "https://api.resend.com/emails",
        data=payload,
        headers={
            "Authorization": f"Bearer {config.EMAIL_AUTH_RESEND_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            if not 200 <= response.status < 300:
                raise EmailDeliveryError("Email provider rejected the request.")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise EmailDeliveryError("Email provider is unavailable.") from exc
