"""Runtime guard middleware: retries, schema coercion and tool fallbacks for live calls."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .live import sleep_s

_TRANSIENT = ("timeout", "rate_limit", "server_error")
# Tool-level fallbacks (source function -> fallback function).  Empty for the current
# live demo: recovery value comes from retries, schema coercion and argument repair.
_FALLBACKS: dict[str, str] = {}


@dataclass
class GuardEvent:
    rule: str
    function: str
    error: str
    repaired_function: str = ""
    repaired_arguments: dict[str, Any] = field(default_factory=dict)
    recovered: bool = False


def live_error_kind(message: str | None) -> str:
    text = (message or "").lower()
    for kind in ("timeout", "rate_limit", "invalid_argument", "not_found", "server_error", "request_error"):
        if text.startswith(kind) or f"{kind}:" in text[:64]:
            return kind
    if "no such function" in text:
        return "invalid_argument"
    return "other"


def _schemas_by_name(backend: Any) -> dict[str, dict[str, Any]]:
    try:
        schemas = backend.schemas()
    except Exception:  # noqa: BLE001 - backends without schemas still work
        return {}
    out: dict[str, dict[str, Any]] = {}
    for schema in schemas or []:
        fn = (schema.get("function") or {})
        name = fn.get("name")
        if name:
            out[name] = fn.get("parameters") or {}
    return out


def coerce_arguments(function: str, arguments: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
    """Coerce string arguments into the schema types (the most common live failure)."""
    props = (parameters or {}).get("properties") or {}
    repaired = dict(arguments)
    for name, value in list(repaired.items()):
        expected = str((props.get(name) or {}).get("type") or "").lower()
        if expected in ("number", "integer") and isinstance(value, str):
            try:
                number = float(value.strip().replace(",", ""))
                repaired[name] = int(number) if expected == "integer" and number.is_integer() else number
            except ValueError:
                continue
        elif expected == "string" and not isinstance(value, str):
            repaired[name] = json.dumps(value) if isinstance(value, (dict, list)) else str(value)
        elif expected == "boolean" and isinstance(value, str) and value.lower() in ("true", "false"):
            repaired[name] = value.lower() == "true"
    return repaired


class ToolGuard:
    """Middleware wrapper implementing retry / coercion / fallback over a tool backend."""

    def __init__(self, backend: Any, max_retries: int = 2, sleep_scale: float = 0.05):
        self.backend = backend
        self.max_retries = max_retries
        self.sleep_scale = sleep_scale
        self.events: list[GuardEvent] = []
        self._schemas = _schemas_by_name(backend)
        self.stats = {"calls": 0, "recovered": 0, "failed_after_repair": 0, "original_errors": 0}

    def call(self, function_name: str, arguments: dict[str, Any]):
        self.stats["calls"] += 1
        result = self.backend.call(function_name, arguments)
        if result.ok:
            return result
        self.stats["original_errors"] += 1
        kind = live_error_kind(result.error)
        event = GuardEvent(rule=kind or "unknown", function=function_name, error=result.error or "")
        if kind in _TRANSIENT:
            result = self._retry(function_name, arguments, result, event)
        elif kind == "invalid_argument":
            hinted = self._coerce_from_error_hint(arguments, result.error or "")
            if hinted is not None:
                event.rule = "error_hint_coercion"
                event.repaired_arguments = hinted
                result = self.backend.call(function_name, hinted)
                event.recovered = result.ok
                if result.ok:
                    self.stats["recovered"] += 1
                    self.events.append(event)
                    return result
            parameters = self._schemas.get(function_name) or {}
            coerced = coerce_arguments(function_name, arguments, parameters)
            if coerced != arguments:
                event.rule = "schema_coercion"
                event.repaired_arguments = coerced
                result = self.backend.call(function_name, coerced)
                event.recovered = result.ok
                if result.ok:
                    self.stats["recovered"] += 1
                    self.events.append(event)
                    return result
        elif kind == "not_found":
            result = self._fallback(function_name, arguments, result, event)
        if not result.ok:
            self.stats["failed_after_repair"] += 1
        self.events.append(event)
        return result

    # -- repair strategies --------------------------------------------------
    def _coerce_from_error_hint(self, arguments: dict[str, Any], error: str) -> dict[str, Any] | None:
        """Use the tool's own error text, e.g. ``story_id must be an integer``."""
        match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s+must be (?:an?\s+)?(integer|number)", error.lower())
        if not match:
            return None
        name, expected = match.group(1), match.group(2)
        for key, value in arguments.items():
            if key.lower() != name or not isinstance(value, str):
                continue
            try:
                number = float(value.strip().replace(",", ""))
            except ValueError:
                return None
            repaired = dict(arguments)
            repaired[key] = int(number) if expected == "integer" and number.is_integer() else number
            return repaired
        return None

    def _retry(self, function_name: str, arguments: dict[str, Any], result, event: GuardEvent):
        for attempt in range(1, self.max_retries + 1):
            sleep_s(self.sleep_scale * attempt)
            candidate = self.backend.call(function_name, arguments)
            if candidate.ok:
                event.rule = f"transient_retry:{attempt}"
                event.recovered = True
                self.stats["recovered"] += 1
                return candidate
            result = candidate
        return result

    def _fallback(self, function_name: str, arguments: dict[str, Any], result, event: GuardEvent):
        fallback_function = _FALLBACKS.get(function_name)
        if not fallback_function or not hasattr(self.backend, "backend"):
            return result
        # e.g. wikipedia summary miss -> search -> summary of the best hit
        query = arguments.get("title") or arguments.get("query") or ""
        search = self.backend.call(fallback_function, {"query": query})
        if not search.ok or not isinstance(search.response, dict):
            return result
        titles = search.response.get("results") or []
        for title in titles[:2]:
            candidate = self.backend.call(function_name, {"title": title})
            if candidate.ok:
                event.rule = "fallback_tool_chain"
                event.repaired_function = fallback_function
                event.repaired_arguments = {"query": query, "resolved_title": title}
                event.recovered = True
                self.stats["recovered"] += 1
                return candidate
        return result

    def dump_events(self) -> list[dict[str, Any]]:
        return [asdict(event) for event in self.events]
