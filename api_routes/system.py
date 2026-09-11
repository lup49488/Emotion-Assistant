"""Health, contract, provider, and operator-facing route definitions."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from fastapi import APIRouter, Depends, Query, status
from fastapi.responses import JSONResponse

from api_contracts import (
    ContractInfoResponse,
    HealthResponse,
    LivenessResponse,
    ObservabilitySummaryResponse,
    OperationsDashboardResponse,
    ProviderCatalogResponse,
    StatusResponse,
)


def create_system_router(
    *,
    health_payload: Callable[[], dict[str, Any]],
    required_components_ok: Callable[[dict[str, Any]], bool],
    current_user: Callable[..., str],
    require_operations_user: Callable[[str], str],
    provider_catalog_loader: Callable[[], dict[str, object]],
    observability_summary_loader: Callable[[int], dict[str, Any]],
    operations_dashboard_loader: Callable[[int], dict[str, Any]],
    contract_version: str,
) -> APIRouter:
    """Create system routes without coupling them to the application module.

    Callbacks keep application configuration and test monkeypatches in the
    composition root while this module owns only HTTP-to-service adaptation.
    """
    router = APIRouter()

    @router.get("/health", response_model=HealthResponse, deprecated=True)
    def health() -> dict[str, Any]:
        """Deprecated compatibility report that always returns 200; use /health/live or /health/ready."""
        return health_payload()

    @router.get("/health/live", response_model=LivenessResponse)
    def health_live() -> dict[str, str]:
        """Report process liveness without treating optional startup work as failure."""
        return {"status": "ok"}

    @router.get("/health/ready", response_model=HealthResponse)
    def health_ready() -> dict[str, Any] | JSONResponse:
        """Report whether this instance can safely receive proxied application traffic.

        An optional component that failed to warm up leaves the overall status
        degraded but keeps this instance serving, so Docker does not restart a
        usable chat service over a preload that never had to succeed.
        """
        payload = health_payload()
        if not required_components_ok(payload):
            return JSONResponse(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, content=payload)
        return payload

    @router.get("/api/v1/status", response_model=StatusResponse)
    def api_status() -> dict[str, Any]:
        """Versioned operational status for reverse proxies and monitoring."""
        return {**health_payload(), "api_version": "v1"}

    @router.get("/api/v1/model/providers", response_model=ProviderCatalogResponse)
    def model_providers(_: str = Depends(current_user)) -> dict[str, object]:
        """Provider and model choices for the signed-in settings panel.

        No credentials are returned, but default_base_url comes from the deployment's
        own configuration and can name an internal gateway — so this stays behind the
        session like every other configuration view.
        """
        return provider_catalog_loader()

    @router.get("/api/v1/observability/summary", response_model=ObservabilitySummaryResponse)
    def observability_summary_endpoint(
        user_id: str = Depends(current_user),
        days: int = Query(default=7, ge=1, le=90),
    ) -> dict[str, Any]:
        require_operations_user(user_id)
        return observability_summary_loader(days)

    @router.get("/api/v1/operations/dashboard", response_model=OperationsDashboardResponse)
    def operations_dashboard_endpoint(
        user_id: str = Depends(current_user),
        days: int = Query(default=7, ge=1, le=90),
    ) -> dict[str, Any]:
        require_operations_user(user_id)
        return operations_dashboard_loader(days)

    @router.get("/api/v1/contract", response_model=ContractInfoResponse)
    def contract_info() -> dict[str, str]:
        return {"api_version": "v1", "contract_version": contract_version, "openapi_path": "/openapi.json"}

    return router
