from __future__ import annotations

import subprocess

import deployment_check


def test_project_file_check_passes_for_repository():
    reporter = deployment_check.CheckReporter()

    deployment_check.check_project_files(reporter)

    assert reporter.failures == []


def test_docker_configuration_matches_same_origin_frontend():
    reporter = deployment_check.CheckReporter()

    deployment_check.check_docker_configuration(reporter)

    assert reporter.failures == []


def test_invalid_storage_backend_fails(monkeypatch):
    monkeypatch.setenv("STORAGE_BACKEND", "invalid")
    reporter = deployment_check.CheckReporter()

    deployment_check.check_runtime_configuration(reporter)

    assert reporter.failures == ["STORAGE_BACKEND must be either 'json' or 'sqlite'."]


def test_docker_runtime_accepts_running_services(monkeypatch):
    output = """
[
  {"Service": "api", "State": "running", "Health": "healthy"},
  {"Service": "web", "State": "running", "Health": ""}
]
"""

    monkeypatch.setattr(
        deployment_check,
        "_run_command",
        lambda command, timeout_seconds=20: subprocess.CompletedProcess(command, 0, output, ""),
    )
    reporter = deployment_check.CheckReporter()

    deployment_check.check_docker_runtime(reporter)

    assert reporter.failures == []


def test_local_docker_http_uses_trusted_host(monkeypatch):
    calls = []

    def fake_request(url, *, host=None, method="GET", timeout_seconds=8, follow_redirects=True):
        calls.append((url, host, method))
        return 200, ""

    monkeypatch.setenv("API_TRUSTED_HOSTS", "chat.serenova.dev")
    monkeypatch.setattr(deployment_check, "_request", fake_request)
    reporter = deployment_check.CheckReporter()

    deployment_check.check_local_docker_http(reporter)

    assert reporter.failures == []
    assert all(call[1] == "chat.serenova.dev" for call in calls)
    assert calls[0][2] == "HEAD"


def test_public_http_accepts_cloudflare_access_redirect(monkeypatch):
    monkeypatch.setattr(
        deployment_check,
        "_request",
        lambda *args, **kwargs: (302, "https://example.cloudflareaccess.com/login"),
    )
    reporter = deployment_check.CheckReporter()

    deployment_check.check_public_http(reporter, "https://chat.serenova.dev")

    assert reporter.failures == []


def test_public_http_fails_on_bad_gateway(monkeypatch):
    monkeypatch.setattr(deployment_check, "_request", lambda *args, **kwargs: (502, ""))
    reporter = deployment_check.CheckReporter()

    deployment_check.check_public_http(reporter, "https://chat.serenova.dev")

    assert reporter.failures == [
        "Public hostname returned HTTP 502; check cloudflared, Tunnel ingress, and local origin."
    ]


def test_cloudflared_warns_on_tunnel_transport_errors(monkeypatch):
    def fake_run(command, timeout_seconds=20):
        if command[:2] == ["systemctl", "is-active"]:
            return subprocess.CompletedProcess(command, 0, "active\n", "")
        return subprocess.CompletedProcess(
            command, 0, "failed to dial a quic connection: timeout: no recent network activity", ""
        )

    monkeypatch.setattr(deployment_check, "_run_command", fake_run)
    reporter = deployment_check.CheckReporter()

    deployment_check.check_cloudflared(reporter)

    assert reporter.failures == []
    assert reporter.warnings
