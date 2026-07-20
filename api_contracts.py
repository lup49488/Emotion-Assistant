"""Versioned Pydantic models exposed by the public HTTP API.

Keep these models independent from GUI and storage implementation details.
Changing a required field here is a breaking API change and requires a new API
version (or an explicitly documented compatibility period).
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ContractModel(BaseModel):
    """Base class for stable request and response payloads."""

    model_config = ConfigDict(extra="forbid")


class ApiError(ContractModel):
    code: str
    message: str
    details: list[dict[str, Any]] | None = None


class ContractInfoResponse(ContractModel):
    api_version: Literal["v1"] = "v1"
    contract_version: str
    openapi_path: Literal["/openapi.json"] = "/openapi.json"


class HealthResponse(ContractModel):
    status: Literal["ok"]
    storage_backend: Literal["json", "sqlite"]


class LoginRequest(ContractModel):
    user_id: str = Field(min_length=1, max_length=128)
    access_key: str = Field(min_length=1, max_length=512)


class LoginResponse(ContractModel):
    token_type: Literal["cookie"]
    expires_in: int
    user_id: str


class SessionResponse(ContractModel):
    user_id: str
    authentication: Literal["signed_cookie"]


class ChatRequest(ContractModel):
    message: str = Field(min_length=1, max_length=20_000)
    # None 表示使用服务器 .env 中 LLM_PROVIDER 配置的默认提供者。
    provider: str | None = None
    model: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = Field(default=0.8, ge=0, le=2)
    top_p: float = Field(default=0.9, ge=0, le=1)
    max_new_tokens: int = Field(default=300, ge=1, le=8_192)
    use_knowledge: bool = False
    use_style: bool = False
    show_memory_receipt: bool = True
    conversation_id: str | None = Field(default=None, max_length=64)


class ChatResponse(ContractModel):
    reply: str
    memory_receipt: str | None = None


class MemoryQualityResponse(ContractModel):
    report: str


class MoodCheckinRequest(ContractModel):
    mood: str = Field(min_length=1, max_length=100)
    intensity: int = Field(ge=1, le=5)
    note: str = Field(default="", max_length=5_000)
    checkin_date: str | None = None


class MoodCheckin(ContractModel):
    date: str
    mood: str
    intensity: int
    note: str = ""
    source: str = "checkin"
    created_at: str | None = None
    updated_at: str | None = None


class MoodCheckinResponse(ContractModel):
    record: MoodCheckin


class MoodCheckinListResponse(ContractModel):
    records: list[MoodCheckin]


class WeeklyMoodResponse(ContractModel):
    points: list[dict[str, Any]]
    summary: str
    analysis: str


class ConversationCreateRequest(ContractModel):
    title: str = Field(default="New conversation", min_length=1, max_length=200)


class ConversationRenameRequest(ContractModel):
    title: str = Field(min_length=1, max_length=200)


class ConversationMessage(ContractModel):
    role: str
    content: str
    created_at: str | None = None


class ConversationData(ContractModel):
    id: str
    title: str
    created_at: str
    updated_at: str
    message_count: int | None = None
    messages: list[ConversationMessage] = Field(default_factory=list)


class ConversationListResponse(ContractModel):
    conversations: list[ConversationData]


class ConversationResponse(ContractModel):
    conversation: ConversationData


class MemorySnapshotResponse(ContractModel):
    history: list[dict[str, Any]]
    emotion_memory: list[dict[str, Any]]
    long_memory: list[dict[str, Any]]
    stable_profile: list[dict[str, Any]]
    interest_memory: list[dict[str, Any]]
    memory_events: list[dict[str, Any]]


class LongTermMemoryUpdateRequest(ContractModel):
    items: list[dict[str, Any]] = Field(default_factory=list, max_length=100)


class KnowledgeDocument(ContractModel):
    name: str
    size_bytes: int | None = None
    chunks: int | None = None
    modified_at: str | None = None
    created_at: str | None = None


class RagStatusResponse(ContractModel):
    status: str
    documents: list[KnowledgeDocument]


class RagDocumentMutationResponse(ContractModel):
    result: dict[str, Any]


class RagQualityResponse(BaseModel):
    """Quality output stays forward-compatible while its envelope is stable."""

    model_config = ConfigDict(extra="allow")


class RagSearchRequest(ContractModel):
    query: str = Field(min_length=1, max_length=20_000)
    top_k: int = Field(default=4, ge=1, le=20)
    threshold: float = Field(default=0.35, ge=-1, le=1)
    candidate_multiplier: int = Field(default=4, ge=1, le=16)


class RagSearchResponse(BaseModel):
    """Diagnostic results are intentionally extensible for retrieval tuning."""

    model_config = ConfigDict(extra="allow")


class RagEvaluationRequest(ContractModel):
    top_k: int = Field(default=4, ge=1, le=20)
    threshold: float = Field(default=0.35, ge=-1, le=1)
    candidate_multiplier: int = Field(default=4, ge=1, le=16)


class RagEvaluationResponse(ContractModel):
    report: dict[str, Any] | None


class UsageSummaryResponse(ContractModel):
    today: dict[str, int | float]
    month: dict[str, int | float]
    requests_last_minute: int
    limits: dict[str, int | float]


class UsageEventsResponse(ContractModel):
    events: list[dict[str, Any]]


class PrivacyResponse(ContractModel):
    backend: Literal["json", "sqlite"]
    conversation_count: int
    message_count: int
    history_count: int
    memory_count: int
    mood_count: int
    api_request_count: int


class ExportResponse(ContractModel):
    schema_version: int
    exported_at: str
    user_id: str
    notes: list[str]
    history: list[dict[str, Any]]
    conversations: list[dict[str, Any]]
    emotion_memory: list[dict[str, Any]]
    long_memory: list[dict[str, Any]]
    stable_profile: list[dict[str, Any]]
    memory_events: list[dict[str, Any]]
    interest_memory: list[dict[str, Any]]
    mood_checkins: list[dict[str, Any]]


class PrivacyDeletionResponse(ContractModel):
    deleted: dict[str, Any]


class PrivacyDeleteRequest(ContractModel):
    confirmation: str = Field(min_length=1, max_length=32)
