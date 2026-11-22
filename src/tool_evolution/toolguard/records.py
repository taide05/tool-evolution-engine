"""Loading real tool-call records and agent-visible schemas."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ToolCall:
    trace_id: str
    method: str
    model: str
    strategy: str
    group: str
    query_id: str
    try_index: int
    step: int
    tool_name: str
    arguments: dict[str, Any]
    error: str | None
    recorded_ok: bool
    available_tools: list[str] = field(default_factory=list)
    valid_data: bool = False
    query: str = ""

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "ToolCall":
        return cls(
            trace_id=row["trace_id"],
            method=row["method"],
            model=row["model"],
            strategy=row["strategy"],
            group=row["group"],
            query_id=row["query_id"],
            try_index=row["try_index"],
            step=row["step"],
            tool_name=row["tool_name"],
            arguments=dict(row.get("arguments") or {}),
            error=row.get("observation_error"),
            recorded_ok=bool(row.get("observation_ok")),
            available_tools=list(row.get("available_tools") or []),
            valid_data=bool(row.get("valid_data")),
            query=str(row.get("query") or ""),
        )


@dataclass(frozen=True)
class FunctionSchema:
    name: str
    properties: dict[str, dict[str, Any]]
    required: tuple[str, ...]

    def has_param(self, name: str) -> bool:
        return name in self.properties

    def default_for(self, name: str) -> Any:
        prop = self.properties.get(name) or {}
        if "example_value" in prop and prop["example_value"] not in ("", None):
            return prop["example_value"]
        if "default" in prop and prop["default"] not in ("", None):
            return prop["default"]
        return None


def load_calls(path: Path) -> list[ToolCall]:
    return [ToolCall.from_dict(json.loads(line)) for line in Path(path).open(encoding="utf-8")]
