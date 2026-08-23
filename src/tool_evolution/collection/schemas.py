from enum import Enum
from datetime import datetime, timezone
from pydantic import BaseModel, Field
from typing import Any


class TraceType(str, Enum):
    ATOMIC = "atomic"
    TASK_ROOT = "task_root"


class ErrorType(str, Enum):
    PARAM_ERROR = "param_error"
    PERMISSION_DENIED = "permission_denied"
    QUOTA_EXHAUSTED = "quota_exhausted"
    TIMEOUT = "timeout"
    SERVICE_UNAVAILABLE = "service_unavailable"


class TraceReport(BaseModel):
    trace_id: str
    parent_trace_id: str | None = None
    agent_id: str
    tool_name: str
    tool_version: str = "1.0.0"
    trace_type: TraceType = TraceType.ATOMIC
    params: dict[str, Any] = Field(default_factory=dict)
    success: bool
    result: dict[str, Any] | None = None
    error_type: ErrorType | None = None
    error_message: str | None = None
    latency_ms: int
    token_count: int = 0
    source: str = "synthetic"


class TraceSnapshot(TraceReport):
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
