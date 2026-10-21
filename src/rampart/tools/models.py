"""Pydantic contracts for every tool: validated inputs, one response envelope, typed payloads."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from ..intel.nvd import normalise_cve_id

Exposure = Literal["internet-facing", "internal", "isolated"]
Urgency = Literal["critical", "high", "medium", "low"]
Priority = Literal["P1", "P2", "P3", "P4"]


class ToolError(BaseModel):
    code: str
    message: str
    retryable: bool = False


class Envelope(BaseModel):
    """What every tool returns. `data` is tool-specific; `status` is the first thing to read."""

    status: Literal["ok", "error"]
    source: str
    data: dict[str, Any] | None = None
    error: ToolError | None = None
    fetched_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))
    cached: bool = False

    @classmethod
    def ok(cls, source: str, data: dict[str, Any], *, cached: bool = False, fetched_at: str | None = None) -> Envelope:
        return cls(
            status="ok", source=source, data=data, cached=cached, **({"fetched_at": fetched_at} if fetched_at else {})
        )

    @classmethod
    def fail(cls, source: str, code: str, message: str, retryable: bool = False) -> Envelope:
        return cls(status="error", source=source, error=ToolError(code=code, message=message, retryable=retryable))


class UrgencyEvidence(BaseModel):
    """Structured evidence for the urgency scorer. The agent fills it from the other tools."""

    cve_id: str
    cvss_score: float | None = Field(default=None, ge=0, le=10)
    cvss_severity: str | None = None
    epss: float | None = Field(default=None, ge=0, le=1)
    in_kev: bool = False
    kev_due_date: str | None = None
    known_ransomware_use: bool = False
    exposure: Exposure = "internal"
    affected_systems: list[str] = Field(default_factory=list)
    data_classification: str | None = None
    compensating_controls: list[str] = Field(default_factory=list)
    notes: str | None = Field(default=None, max_length=2000)

    @field_validator("cve_id")
    @classmethod
    def _norm(cls, v: str) -> str:
        return normalise_cve_id(v)


class UrgencyVerdict(BaseModel):
    urgency: Urgency
    score: float = Field(ge=0, le=1, description="0 = negligible, 1 = act now")
    confidence: float = Field(ge=0, le=1)
    reasoning: str = Field(max_length=1200)


class TicketRequest(BaseModel):
    cve_id: str
    system: str = Field(min_length=1, max_length=120)
    priority: Priority
    due_date: str = Field(description="ISO date YYYY-MM-DD, derived from the policy SLA")
    summary: str = Field(min_length=1, max_length=2000)
    owner_team: str | None = None

    @field_validator("cve_id")
    @classmethod
    def _norm(cls, v: str) -> str:
        return normalise_cve_id(v)
