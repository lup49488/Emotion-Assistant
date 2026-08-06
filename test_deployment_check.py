from __future__ import annotations

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
