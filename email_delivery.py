"""Outbound verification-email delivery with no secrets in browser code."""
from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.request

import config


logger = logging.getLogger(__name__)
_EMAIL_IN_TEXT = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")


def _safe_provider_detail(payload: object) -> tuple[str, str]:
    """Return a short diagnostic without retaining a recipient address."""
    if not isinstance(payload, dict):
        return "unknown", ""
    kind = payload.get("name") if isinstance(payload.get("name"), str) else "unknown"
    message = payload.get("message") if isinstance(payload.get("message"), str) else ""
    return kind[:80], _EMAIL_IN_TEXT.sub("<email>", message.strip())[:240]


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
            # Resend's Cloudflare edge rejects Python's default urllib user agent
            # before the request reaches the provider API.
            "User-Agent": "Serenova/1.0 (+https://chat.serenova.dev)",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            if not 200 <= response.status < 300:
                raise EmailDeliveryError("Email provider rejected the request.")
    except urllib.error.HTTPError as exc:
        # Resend may include the recipient address in its human-readable message.
        # Keep logs useful for operators without recording personal data.
        error_kind = "unknown"
        detail = ""
        try:
            payload = json.loads(exc.read().decode("utf-8"))
            error_kind, detail = _safe_provider_detail(payload)
        except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
            pass
        logger.warning("Resend verification delivery rejected: status=%s category=%s detail=%s", exc.code, error_kind, detail or "none")
        raise EmailDeliveryError("Email provider rejected the request.") from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        logger.warning("Resend verification delivery unavailable: error=%s", type(exc).__name__)
        raise EmailDeliveryError("Email provider is unavailable.") from exc
