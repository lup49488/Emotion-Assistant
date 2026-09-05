"""Concurrency and input-validation guarantees for Mood Check-in image attachments."""
from __future__ import annotations

import threading

import pytest

import mood_image_store

PNG = b"\x89PNG\r\n\x1a\n" + b"payload"


@pytest.fixture
def store(monkeypatch, tmp_path):
    monkeypatch.setattr(mood_image_store, "user_dir", lambda user_id: tmp_path)
    monkeypatch.setattr(mood_image_store, "validate_user_id", lambda user_id: user_id)
    return mood_image_store


def test_parallel_uploads_cannot_exceed_the_per_checkin_cap(store):
    # The browser posts a check-in's images with Promise.all, so the cap has to
    # hold when every request checks the count at the same moment.
    attempts = store.MAX_IMAGES_PER_CHECKIN + 3
    gate = threading.Barrier(attempts)
    rejected = []

    def upload():
        gate.wait(timeout=5)
        try:
            store.save_mood_image("alice", "2026-07-18", PNG)
        except ValueError:
            rejected.append(1)

    workers = [threading.Thread(target=upload) for _ in range(attempts)]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join(timeout=10)

    assert len(store.list_mood_images("alice", "2026-07-18")) == store.MAX_IMAGES_PER_CHECKIN
    assert len(rejected) == attempts - store.MAX_IMAGES_PER_CHECKIN


def test_per_user_count_quota_spans_every_checkin(store, monkeypatch):
    # The per-check-in cap alone lets one user grow storage without bound by
    # spreading uploads across dates, so the quota has to count the whole tree.
    monkeypatch.setattr(store, "MAX_IMAGES_PER_USER", 2)
    store.save_mood_image("alice", "2026-07-18", PNG)
    store.save_mood_image("alice", "2026-07-19", PNG)

    with pytest.raises(ValueError, match="at most 2 Mood Check-in images"):
        store.save_mood_image("alice", "2026-07-20", PNG)


def test_per_user_byte_quota_rejects_the_upload_that_would_exceed_it(store, monkeypatch):
    monkeypatch.setattr(store, "MAX_IMAGES_PER_USER", 100)
    monkeypatch.setattr(store, "MAX_IMAGE_BYTES_PER_USER", len(PNG) + 1)
    store.save_mood_image("alice", "2026-07-18", PNG)

    with pytest.raises(ValueError, match="MB of Mood Check-in images"):
        store.save_mood_image("alice", "2026-07-19", PNG)
    assert store.list_mood_images("alice", "2026-07-19") == []


@pytest.mark.parametrize("checkin_date", ["not-a-date", "2026-13-45", "../../etc", "", "2026-7-8"])
def test_read_paths_treat_a_malformed_date_as_missing(store, checkin_date):
    # Read paths answer "no such image" instead of raising, which would surface as a 500.
    assert store.list_mood_images("alice", checkin_date) == []
    assert store.get_mood_image_path("alice", checkin_date, "0" * 32) is None
    assert store.delete_mood_image("alice", checkin_date, "0" * 32) is False
    store.delete_mood_checkin_images("alice", checkin_date)


def test_uploads_still_reject_a_malformed_date(store):
    # The write path stays strict: a bad date must never create a directory.
    with pytest.raises(ValueError):
        store.save_mood_image("alice", "not-a-date", PNG)


def test_only_recognised_image_signatures_are_accepted(store):
    with pytest.raises(ValueError):
        store.save_mood_image("alice", "2026-07-18", b"<html>not an image</html>")
    with pytest.raises(ValueError):
        store.save_mood_image("alice", "2026-07-18", b"")
