from datetime import datetime, timedelta
from unittest.mock import patch

import observability_store


def test_json_observability_events_are_persisted_summarized_and_retained(tmp_path):
    path = tmp_path / "observability_events.json"
    with patch.object(observability_store, "_JSON_PATH", path), \
         patch.object(observability_store, "sqlite_enabled", return_value=False), \
         patch.object(observability_store, "OBSERVABILITY_RETENTION_DAYS", 7):
        observability_store.record_http_event(
            request_id="request-1", method="get", path="/api/v1/chat", status_code=200, duration_ms=120,
        )
        observability_store.record_http_event(
            request_id="request-2", method="post", path="/api/v1/chat", status_code=503, duration_ms=240,
        )
        events = observability_store._read_json_events()
        events.append({"request_id": "old", "method": "GET", "path": "/old", "status_code": 200, "duration_ms": 1, "created_at": (datetime.now() - timedelta(days=8)).isoformat(timespec="seconds")})
        observability_store._write_json_events(events)
        observability_store.record_http_event(
            request_id="request-3", method="get", path="/health", status_code=200, duration_ms=30,
        )
        summary = observability_store.observability_summary(days=7)

    assert summary["requests"] == 3
    assert summary["failures"] == 1
    assert summary["failure_rate"] == 33.3
    assert summary["top_paths"][0] == {"path": "/api/v1/chat", "requests": 2}
    assert all(item["path"] != "/old" for item in observability_store._read_json_events())
