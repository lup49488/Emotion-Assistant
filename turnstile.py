"""Server-side Cloudflare Turnstile verification."""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request

import config


def verify_email_auth_token(token: str, client_ip: str) -> bool:
    """Return true only for a fresh Siteverify result for this app and hostname."""
    if not config.EMAIL_AUTH_TURNSTILE_REQUIRED:
        return True
    if not token or not config.TURNSTILE_SECRET_KEY:
        return False
    body = urllib.parse.urlencode(
        {
            "secret": config.TURNSTILE_SECRET_KEY,
            "response": token,
            "remoteip": client_ip,
        }
    ).encode("ascii")
    request = urllib.request.Request(
        "https://challenges.cloudflare.com/turnstile/v0/siteverify",
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
        return False
    if not isinstance(payload, dict) or payload.get("success") is not True:
        return False
    if payload.get("action") != config.TURNSTILE_EMAIL_AUTH_ACTION:
        return False
    hostname = str(payload.get("hostname") or "").lower()
    return bool(not config.TURNSTILE_EXPECTED_HOSTNAME or hostname == config.TURNSTILE_EXPECTED_HOSTNAME)
