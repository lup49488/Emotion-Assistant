from __future__ import annotations

import operations_store


def test_operations_dashboard_generates_threshold_alerts(monkeypatch):
    monkeypatch.setattr(operations_store, "OPS_ALERT_MIN_REQUESTS", 10)
    monkeypatch.setattr(operations_store, "OPS_ALERT_HTTP_FAILURE_RATE", 10.0)
    monkeypatch.setattr(operations_store, "OPS_ALERT_AVERAGE_LATENCY_MS", 1000)
    monkeypatch.setattr(operations_store, "OPS_ALERT_PROVIDER_FAILURES", 3)
    monkeypatch.setattr(operations_store, "OPS_ALERT_JOB_FAILURES", 1)
    monkeypatch.setattr(operations_store, "observability_summary", lambda *, days: {
        "days": days, "requests": 20, "failures": 4, "failure_rate": 20.0,
        "average_duration_ms": 1400.0, "top_paths": [], "statuses": {"500": 4},
    })
    monkeypatch.setattr(operations_store, "_provider_failures", lambda days: [{"provider": "deepseek", "error_kind": "network", "failures": 3}])
    monkeypatch.setattr(operations_store, "_job_summary", lambda days: {"counts": {"failed": 2}, "recent_failures": []})
    monkeypatch.setattr(operations_store, "_persist_alerts", lambda candidates: candidates)

    result = operations_store.operations_dashboard(days=14)

    assert result["window_days"] == 14
    assert {alert["fingerprint"] for alert in result["alerts"]} == {
        "http_failure_rate", "http_latency", "provider_failures", "background_job_failures",
    }
