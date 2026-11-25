"""Runtime guard middleware: retries, schema coercion and tool fallbacks for live calls."""

from __future__ import annotations

import difflib
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any

from .live import sleep_s
from .schema_check import validate_against_parameters

# Transport-level failures are retryable: a broken MCP session/pipe is recovered by
# reconnecting (MCPBridge) and retrying the call.  ``request_error`` is what the HTTP
# layer and the MCP adapter report for such transport failures.
_TRANSIENT = ("timeout", "rate_limit", "server_error", "request_error")
# Tool-level fallbacks (source function -> fallback function).  Empty for the current
# live demo: recovery value comes from retries, schema coercion and argument repair.
_FALLBACKS: dict[str, str] = {}
# Same resolution threshold the offline failure taxonomy uses for hallucinated names.
_NAME_RESOLUTION_THRESHOLD = 0.55
_REQUESTED_NAME_RE = re.compile(
    r"(?:unknown tool|no such function(?: name)?)\s*:?\s*([A-Za-z_][A-Za-z0-9_.\-]*)", re.IGNORECASE)


@dataclass
class GuardEvent:
    rule: str
    function: str
    error: str
    repaired_function: str = ""
    repaired_arguments: dict[str, Any] = field(default_factory=dict)
    recovered: bool = False
    # Structural schema validation of the *repaired* call before it is sent to the backend.
    # A rejected repair is recorded here and never silently swallowed.
    schema_validated: bool = False
    schema_unavailable: bool = False
    schema_errors: list[str] = field(default_factory=list)
    schema_overrides: dict[str, str] = field(default_factory=dict)


def live_error_kind(message: str | None) -> str:
    text = (message or "").lower()
    if "unknown tool" in text or "no such function" in text:
        return "no_such_function"
    for kind in ("timeout", "rate_limit", "invalid_argument", "not_found", "server_error", "request_error"):
        if text.startswith(kind) or f"{kind}:" in text[:64]:
            return kind
    return "other"


def requested_function_name(message: str | None) -> str | None:
    """Extract the requested tool name from an unknown-tool error message."""
    match = _REQUESTED_NAME_RE.search(message or "")
    return match.group(1) if match else None


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


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
        self.stats = {"calls": 0, "recovered": 0, "failed_after_repair": 0, "original_errors": 0,
                      "retry_sleep_s": 0.0, "schema_rejected": 0, "schema_validated": 0,
                      "schema_unchecked": 0}

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
        elif kind == "no_such_function":
            requested = requested_function_name(result.error)
            resolved = self._resolve_function_name(requested) if requested else None
            if resolved is not None:
                result = self._attempt_repaired(resolved, arguments, event,
                                                "function_name_resolution", original_result=result,
                                                repaired_function=resolved)
                if result.ok:
                    self.events.append(event)
                    return result
        elif kind == "invalid_argument":
            hint = self._coerce_from_error_hint(arguments, result.error or "")
            if hint is not None:
                hinted, overrides = hint
                result = self._attempt_repaired(function_name, hinted, event,
                                                "error_hint_coercion", original_result=result,
                                                type_overrides=overrides)
                if result.ok:
                    self.events.append(event)
                    return result
            parameters = self._schemas.get(function_name) or {}
            coerced = coerce_arguments(function_name, arguments, parameters)
            if coerced != arguments:
                result = self._attempt_repaired(function_name, coerced, event,
                                                "schema_coercion", original_result=result)
                if result.ok:
                    self.events.append(event)
                    return result
        elif kind == "not_found":
            result = self._fallback(function_name, arguments, result, event)
        if not result.ok:
            self.stats["failed_after_repair"] += 1
        self.events.append(event)
        return result

    # -- repair strategies --------------------------------------------------
    def _attempt_repaired(self, function_name: str, repaired_arguments: dict[str, Any],
                          event: GuardEvent, rule: str, original_result,
                          repaired_function: str = "",
                          type_overrides: dict[str, str] | None = None):
        """Validate a repaired call against the agent-visible schema, then send it.

        A repair that fails structural validation is *recorded* (``schema_errors``,
        rule suffix ``:schema_rejected``, ``stats["schema_rejected"]``) and the original
        error is returned to the agent instead of sending an invalid call.  If the target
        function is not in the agent-visible tool list (e.g. a hidden real tool recovered
        from catalog drift), there is no schema to validate against; that is recorded as
        ``schema_unavailable`` instead of being silently treated as valid.  Transient faults
        on the *repaired* call are still retried inside the middleware (see live traces).
        """
        if function_name not in self._schemas:
            event.schema_unavailable = True
            self.stats["schema_unchecked"] += 1
        else:
            parameters = self._schemas.get(function_name) or {}
            errors = validate_against_parameters(repaired_arguments, parameters,
                                                 type_overrides=type_overrides)
            event.schema_overrides = dict(type_overrides or {})
            if errors:
                event.rule = f"{rule}:schema_rejected"
                event.schema_errors = errors
                event.recovered = False
                self.stats["schema_rejected"] += 1
                return original_result
            event.schema_validated = True
            self.stats["schema_validated"] += 1
        event.rule = rule
        event.repaired_arguments = repaired_arguments
        if repaired_function:
            event.repaired_function = repaired_function
        result = self.backend.call(function_name, repaired_arguments)
        if not result.ok and live_error_kind(result.error) not in _TRANSIENT:
            return result
        if not result.ok:
            result = self._retry(function_name, repaired_arguments, result, event, count_recovery=False)
            if result.ok:
                event.rule = f"{rule}+{event.rule}"
        event.recovered = result.ok
        if result.ok:
            self.stats["recovered"] += 1
        return result

    def _available_function_names(self) -> list[str]:
        """Tool names the backend can actually execute (adapters may expose more)."""
        names = getattr(self.backend, "available_tool_names", None)
        if callable(names):
            try:
                resolved = list(names())
                if resolved:
                    return resolved
            except Exception:  # noqa: BLE001 - fall back to advertised schemas
                pass
        return list(self._schemas)

    def _resolve_function_name(self, requested: str) -> str | None:
        """Resolve a hallucinated/drifted tool name against the real tool catalogue.

        Same rule family the offline evaluation validated for hallucinated function
        names (sequence similarity >= 0.55), now applied at the runtime layer.
        """
        needle = _normalize_name(requested)
        best_name, best_ratio = None, 0.0
        for candidate in self._available_function_names():
            if candidate == requested:
                return candidate
            ratio = difflib.SequenceMatcher(None, needle, _normalize_name(candidate)).ratio()
            if ratio > best_ratio:
                best_name, best_ratio = candidate, ratio
        if best_name is None or best_ratio < _NAME_RESOLUTION_THRESHOLD:
            return None
        return best_name

    def _coerce_from_error_hint(self, arguments: dict[str, Any], error: str
                                ) -> tuple[dict[str, Any], dict[str, str]] | None:
        """Use the tool's own error text, e.g. ``story_id must be an integer``.

        Returns the repaired arguments plus the type override the backend asserted.  The
        advertised schema can itself be the drifting/buggy artefact (the HN ``story_id``
        schema deliberately advertises ``string`` while the backend requires an int), so the
        backend's own error is the authoritative type for that parameter; the override is
        recorded on the guard event rather than silently widening the advertised schema.
        """
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
            return repaired, {key: expected}
        return None

    def _retry(self, function_name: str, arguments: dict[str, Any], result, event: GuardEvent,
               count_recovery: bool = True):
        for attempt in range(1, self.max_retries + 1):
            sleep_s(self.sleep_scale * attempt)
            self.stats["retry_sleep_s"] += self.sleep_scale * attempt
            candidate = self.backend.call(function_name, arguments)
            if candidate.ok:
                event.rule = f"transient_retry:{attempt}"
                event.recovered = True
                if count_recovery:
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
        parameters = self._schemas.get(function_name) or {}
        for title in titles[:2]:
            candidate_arguments = {"title": title}
            errors = validate_against_parameters(candidate_arguments, parameters)
            if errors:
                event.rule = "fallback_tool_chain:schema_rejected"
                event.schema_errors = errors
                self.stats["schema_rejected"] += 1
                continue
            event.schema_validated = True
            candidate = self.backend.call(function_name, candidate_arguments)
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
