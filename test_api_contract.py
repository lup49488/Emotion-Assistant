"""Regression tests for the public v1 API contract.

Intentional API changes must update ``api_contract.openapi.json`` in the same
change set and bump ``API_CONTRACT_VERSION`` when they are breaking.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

import api_server
from api_contracts import ChatRequest
from config import DEFAULT_MAX_NEW_TOKENS


SNAPSHOT_PATH = Path(__file__).with_name("api_contract.openapi.json")


def test_openapi_contract_matches_versioned_snapshot():
    expected = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))

    assert api_server.app.openapi() == expected
    assert expected["info"]["x-api-version"] == "v1"
    assert expected["info"]["x-contract-version"] == api_server.API_CONTRACT_VERSION


def test_contract_endpoint_and_error_envelope_are_stable():
    with TestClient(api_server.app, base_url="https://testserver") as client:
        contract = client.get("/api/v1/contract")
        invalid_request = client.post("/api/v1/auth/login", json={"user_id": "only-id"})

    assert contract.status_code == 200
    assert contract.json() == {
        "api_version": "v1",
        "contract_version": api_server.API_CONTRACT_VERSION,
        "openapi_path": "/openapi.json",
    }
    assert invalid_request.status_code == 422
    assert invalid_request.json()["code"] == "request_validation_error"
    assert invalid_request.json()["message"] == "Request validation failed."
    assert isinstance(invalid_request.json()["details"], list)


def test_v1_request_models_reject_unknown_fields():
    with TestClient(api_server.app, base_url="https://testserver") as client:
        response = client.post(
            "/api/v1/auth/login",
            json={"user_id": "api-alice", "access_key": "api-secret", "unexpected": True},
        )

    assert response.status_code == 422
    assert response.json()["code"] == "request_validation_error"


def test_chat_request_uses_server_output_limit_when_client_omits_it():
    assert ChatRequest(message="hello").max_new_tokens == DEFAULT_MAX_NEW_TOKENS


def test_chat_request_accepts_one_bounded_mood_checkin_context():
    request = ChatRequest(
        message="I want to talk about this check-in.",
        mood_checkin={"date": "2026-08-22", "mood": "anxious", "intensity": 4, "note": "Interview tomorrow."},
    )

    assert request.mood_checkin is not None
    assert request.mood_checkin.intensity == 4
