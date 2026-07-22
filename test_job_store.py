from __future__ import annotations

import threading
from pathlib import Path

import job_store


def test_background_job_runs_to_completion_and_hides_payload(monkeypatch, tmp_path):
    monkeypatch.setenv("STORAGE_BACKEND", "json")
    monkeypatch.setattr(job_store, "_JSON_PATH", Path(tmp_path) / "jobs.json")
    manager = job_store.BackgroundJobManager()
    completed = threading.Event()

    def worker(context):
        context.progress(45, "Indexing")
        completed.set()
        return {"chunks": 3}

    queued = manager.submit("job-user", "rag_rebuild", worker, payload={"internal": "hidden"})
    assert queued["status"] == "queued"
    assert "payload" not in queued
    assert completed.wait(timeout=3)

    job = job_store.get_job(queued["id"], "job-user")
    assert job is not None
    assert job["status"] == "succeeded"
    assert job["progress"] == 100
    assert job["result"] == {"chunks": 3}
    assert "payload" not in job


def test_background_jobs_are_scoped_to_the_requesting_user(monkeypatch, tmp_path):
    monkeypatch.setenv("STORAGE_BACKEND", "json")
    monkeypatch.setattr(job_store, "_JSON_PATH", Path(tmp_path) / "jobs.json")
    first = job_store.create_job("alice", "rag_rebuild", {"document": "private.md"})
    job_store.create_job("bob", "rag_evaluation")

    assert job_store.get_job(first["id"], "bob") is None
    assert [job["kind"] for job in job_store.list_jobs("alice")] == ["rag_rebuild"]


def test_background_job_is_persisted_with_sqlite(monkeypatch, tmp_path):
    monkeypatch.setenv("STORAGE_BACKEND", "sqlite")
    monkeypatch.setenv("SQLITE_DATABASE_PATH", str(Path(tmp_path) / "jobs.db"))

    created = job_store.create_job("sqlite-user", "rag_evaluation")
    loaded = job_store.get_job(created["id"], "sqlite-user")

    assert loaded is not None
    assert loaded["kind"] == "rag_evaluation"
    assert loaded["status"] == "queued"
