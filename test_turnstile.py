from __future__ import annotations

import json
import urllib.parse

import config
import turnstile


class _Response:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self) -> bytes:
        return json.dumps({"success": True, "action": "email-auth", "hostname": "chat.example.test"}).encode()


def test_turnstile_validation_does_not_send_a_reverse_proxy_address(monkeypatch):
    monkeypatch.setattr(config, "EMAIL_AUTH_TURNSTILE_REQUIRED", True)
    monkeypatch.setattr(config, "TURNSTILE_SECRET_KEY", "test-secret")
    monkeypatch.setattr(config, "TURNSTILE_EMAIL_AUTH_ACTION", "email-auth")
    monkeypatch.setattr(config, "TURNSTILE_EXPECTED_HOSTNAME", "chat.example.test")
    captured: dict[str, bytes] = {}

    def fake_urlopen(request, timeout):
        captured["body"] = request.data
        assert timeout == 10
        return _Response()

    monkeypatch.setattr(turnstile.urllib.request, "urlopen", fake_urlopen)

    assert turnstile.verify_email_auth_token("valid-token", "172.18.0.3") is True
    payload = urllib.parse.parse_qs(captured["body"].decode())
    assert payload["response"] == ["valid-token"]
    assert "remoteip" not in payload
