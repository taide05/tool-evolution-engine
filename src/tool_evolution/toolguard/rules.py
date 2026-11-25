"""Deterministic tool-call repair rules mined from real failure patterns.

The rules are intentionally small, auditable and *tool-layer only*: they never call
an LLM and never touch the tool implementation.  Each rule is derived from a real
error-message pattern observed in the public trajectory release (30k calls / 1.1k
real failures) and is applied only when the repaired call is valid against the
agent-visible function schema.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any

from .records import FunctionSchema
from .schema_check import validate_arguments

ERROR_ORDER = (
    "no_such_function",
    "unexpected_kwarg",
    "missing_required",
    "invalid_value",
    "server_error",
    "input_parse",
    "other",
)

_MISSING_PARAM_RE = re.compile(r"missing[^:]*param[^:]*:?\s*(.*)", re.I | re.S)
_UNEXPECTED_KW_RE = re.compile(r"unexpected keyword argument\s*'([^']+)'", re.I)
_QUOTED_RE = re.compile(r"'([^']+)'|\"([^\"]+)\"")
_BARE_AFTER_COLON_RE = re.compile(r":\s*([A-Za-z_][A-Za-z0-9_ ,]*)")
_CONTROL_FUNCTIONS = {"Finish", "finish"}


def classify_error(message: str | None) -> str:
    text = (message or "").lower()
    if not text:
        return "other"
    if "no such function" in text:
        return "no_such_function"
    if "unexpected keyword" in text or "got an unexpected" in text:
        return "unexpected_kwarg"
    if (
        "missing required parameter" in text
        or "missing parameter" in text
        or "missing input parameter" in text
        or "input parameters missing" in text
        or "no input parameters provided" in text
        or ("missing" in text and ("param" in text or "input" in text))
    ):
        return "missing_required"
    if "tool input parse error" in text:
        return "input_parse"
    if (
        "failed to generate fake response" in text
        or "api not working" in text
        or "status_code=5" in text
        or "status_code=429" in text
        or "gateway" in text
        or "temporarily" in text
        or "timed out" in text
        or "timeout" in text
    ):
        return "server_error"
    if (
        "invalid" in text
        or "not recognized" in text
        or "not supported" in text
        or "not a valid" in text
        or "incorrect" in text
        or "does not add up" in text
    ):
        return "invalid_value"
    return "other"


@dataclass
class RepairResult:
    kind: str
    repairable: bool
    rules: list[str] = field(default_factory=list)
    function_name: str | None = None
    arguments: dict[str, Any] | None = None
    schema_valid: bool = False
    reason: str = ""
    # A *proposed* repair is any candidate the strategy ladder produced, whether or not it
    # passed structural schema validation.  Counting only `repairable` hid the proposals
    # that schema validation rejected (the old constructional 100%).
    proposed: bool = False
    schema_errors: list[str] = field(default_factory=list)
    rejected_proposals: list[str] = field(default_factory=list)
    # every candidate the ladder produced (accepted or not), with its arguments; used for
    # the evidence report and by the evaluator's recorded-success verification tier
    candidates: list[dict] = field(default_factory=list)
    # accepted via an *exact recorded successful invocation* of the same task/function
    # (empirical evidence, deliberately kept separate from schema validity)
    value_verified: bool = False

    @property
    def changed_function(self) -> bool:
        return self.function_name is not None

    @property
    def changed_arguments(self) -> bool:
        return self.arguments is not None


def repair_call(
    tool_name: str,
    arguments: dict[str, Any],
    error: str | None,
    available_tools: list[str],
    schemas: dict[str, FunctionSchema],
) -> RepairResult:
    kind = classify_error(error)
    text = error or ""
    if kind == "no_such_function":
        return _repair_function_name(tool_name, arguments, available_tools, schemas)
    if kind == "unexpected_kwarg":
        return _repair_unexpected_kwarg(tool_name, arguments, text, schemas)
    if kind == "missing_required":
        return _repair_missing_required(tool_name, arguments, text, schemas)
    if kind == "invalid_value":
        return _repair_value_coercion(tool_name, arguments, schemas)
    if kind == "server_error":
        return RepairResult(kind=kind, repairable=False, reason="transient/server-side error; retry only")
    if kind == "input_parse":
        return RepairResult(kind=kind, repairable=False, reason="unparseable tool input")
    return RepairResult(kind=kind, repairable=False, reason="no deterministic rule for this error class")


# ---------------------------------------------------------------------------
# individual rules
# ---------------------------------------------------------------------------
def _repair_function_name(tool_name: str, arguments: dict[str, Any],
                          available_tools: list[str],
                          schemas: dict[str, FunctionSchema]) -> RepairResult:
    candidates = [t for t in available_tools if t not in _CONTROL_FUNCTIONS]
    best_name, best_ratio = None, 0.0
    needle = _normalize_identifier(tool_name)
    for candidate in candidates:
        ratio = difflib.SequenceMatcher(None, needle, _normalize_identifier(candidate)).ratio()
        if ratio > best_ratio:
            best_name, best_ratio = candidate, ratio
    if best_name is None or best_ratio < 0.55:
        return RepairResult(kind="no_such_function", repairable=False,
                            reason=f"no close available function (best ratio {best_ratio:.2f})")
    schema = schemas.get(best_name)
    errors = _validate(best_name, arguments, schema, available_tools)
    valid = not errors
    return RepairResult(
        kind="no_such_function",
        repairable=valid,
        proposed=True,
        rules=["function_name_resolution"],
        function_name=best_name,
        arguments=dict(arguments),
        schema_valid=valid,
        schema_errors=errors,
        reason=f"resolved to available function (similarity {best_ratio:.2f})",
    )


def _repair_unexpected_kwarg(tool_name: str, arguments: dict[str, Any], error: str,
                             schemas: dict[str, FunctionSchema]) -> RepairResult:
    schema = schemas.get(tool_name)
    if schema is None:
        return RepairResult(kind="unexpected_kwarg", repairable=False, reason="no schema for function")
    unknown = _UNEXPECTED_KW_RE.findall(error)
    if not unknown:
        unknown = [name for name in arguments if name not in schema.properties]
    repaired = dict(arguments)
    rules: list[str] = []
    for name in unknown:
        if name not in repaired:
            continue
        match = difflib.get_close_matches(name, list(schema.properties), n=1, cutoff=0.75)
        if match:
            repaired[match[0]] = repaired.pop(name)
            rules.append(f"arg_rename:{name}->{match[0]}")
        else:
            repaired.pop(name, None)
            rules.append(f"arg_drop:{name}")
    if not rules:
        return RepairResult(kind="unexpected_kwarg", repairable=False, reason="no unexpected argument found")
    errors = _validate(tool_name, repaired, schema, None)
    return RepairResult(kind="unexpected_kwarg", repairable=not errors, proposed=True, rules=rules,
                        function_name=tool_name, arguments=repaired,
                        schema_valid=not errors, schema_errors=errors,
                        reason="dropped/renamed arguments outside schema")


def _repair_missing_required(tool_name: str, arguments: dict[str, Any], error: str,
                             schemas: dict[str, FunctionSchema]) -> RepairResult:
    schema = schemas.get(tool_name)
    if schema is None:
        return RepairResult(kind="missing_required", repairable=False, reason="no schema for function")
    missing = _parse_missing_params(error)
    if not missing:
        missing = list(schema.required)
    repaired = dict(arguments)
    filled: list[str] = []
    unfilled: list[str] = []
    for raw_name in missing:
        name = _map_param_name(raw_name, schema)
        if name is None:
            unfilled.append(raw_name)
            continue
        if name in repaired and repaired[name] not in ("", None):
            continue
        default = schema.default_for(name)
        if default is not None:
            repaired[name] = default
            filled.append(name)
        else:
            unfilled.append(name)
    if filled and not unfilled:
        errors = _validate(tool_name, repaired, schema, None)
        return RepairResult(kind="missing_required", repairable=not errors, proposed=True,
                            rules=[f"fill_required_from_schema:{','.join(sorted(filled))}"],
                            function_name=tool_name, arguments=repaired,
                            schema_valid=not errors, schema_errors=errors,
                            reason="filled required params from schema defaults/examples")
    reason = f"filled={filled} unfilled={unfilled}" if (filled or unfilled) else "no recoverable parameter value"
    return RepairResult(kind="missing_required", repairable=False, reason=reason)


def _repair_value_coercion(tool_name: str, arguments: dict[str, Any],
                           schemas: dict[str, FunctionSchema]) -> RepairResult:
    schema = schemas.get(tool_name)
    if schema is None:
        return RepairResult(kind="invalid_value", repairable=False, reason="no schema for function")
    repaired = dict(arguments)
    rules: list[str] = []
    for name, value in list(repaired.items()):
        prop = schema.properties.get(name) or {}
        expected = str(prop.get("type") or "").lower()
        if not isinstance(value, str):
            continue
        if expected in ("integer", "number"):
            try:
                number = float(value)
                repaired[name] = int(number) if expected == "integer" and number.is_integer() else number
                rules.append(f"coerce_number:{name}")
            except ValueError:
                continue
        elif expected == "boolean" and value.lower() in ("true", "false"):
            repaired[name] = value.lower() == "true"
            rules.append(f"coerce_bool:{name}")
    if not rules:
        return RepairResult(kind="invalid_value", repairable=False, reason="no type coercion opportunity")
    errors = _validate(tool_name, repaired, schema, None)
    return RepairResult(kind="invalid_value", repairable=not errors, proposed=True, rules=rules,
                        function_name=tool_name, arguments=repaired,
                        schema_valid=not errors, schema_errors=errors,
                        reason="coerced argument types to schema")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _validate(tool_name: str, arguments: dict[str, Any], schema: FunctionSchema | None,
              available_tools: list[str] | None) -> list[str]:
    """Return structural schema violations for a repaired call (empty list = valid)."""
    if available_tools is not None and available_tools and tool_name not in available_tools:
        return [f"function {tool_name!r} is not in the agent-visible tool list"]
    if schema is None:
        return [f"no schema for function {tool_name!r}"]
    return validate_arguments(arguments, schema)


def _parse_missing_params(error: str) -> list[str]:
    return parse_missing_params(error)


def parse_missing_params(error: str) -> list[str]:
    """Extract parameter names mentioned in a 'missing parameters' tool error."""
    match = _MISSING_PARAM_RE.search(error)
    tail = match.group(1) if match else ""
    names = [a or b for a, b in _QUOTED_RE.findall(tail)]
    if not names and tail:
        # "Missing required parameters: start_date, end_date" (unquoted names)
        head = tail.split(".")[0]
        for token in re.split(r",| and | or ", head):
            token = token.strip().strip("'\"")
            if token and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_ ]*", token):
                names.append(token)
    return [n.strip() for n in names if n.strip()]


def _map_param_name(raw: str, schema: FunctionSchema) -> str | None:
    if raw in schema.properties:
        return raw
    norm = _normalize_identifier(raw)
    for name in schema.properties:
        if _normalize_identifier(name) == norm:
            return name
    matches = difflib.get_close_matches(norm, [_normalize_identifier(n) for n in schema.properties], n=1, cutoff=0.8)
    if matches:
        for name in schema.properties:
            if _normalize_identifier(name) == matches[0]:
                return name
    return None


def _normalize_identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
