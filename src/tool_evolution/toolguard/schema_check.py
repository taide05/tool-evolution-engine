"""Structural JSON-Schema validation for tool-call arguments.

The original ToolGuard ``schema_valid`` check only verified two things: every argument name
appears in ``properties`` and every required name is present.  That accepts
``{"year": "2023"}`` for an ``integer`` parameter and rejects nothing about enums, formats or
ranges, so ``schema_valid_rate = 100%`` was structural completeness rather than validity.
This module implements the JSON-Schema subset the schemas actually use, so both the offline
repair engine and the live guard validate the repaired call before it is counted/sent:

* object level: unknown parameters, required parameters, ``additionalProperties``;
* scalar types: string / integer / number / boolean / object / array / null (type unions too),
  with booleans never counting as integers/numbers;
* ``enum``, ``const``;
* string ``format`` (date, date-time, time, email, uri, uuid — unknown formats are ignored,
  as the JSON Schema spec says formats are annotations unless asserted);
* numeric ``minimum`` / ``maximum`` / ``exclusiveMinimum`` / ``exclusiveMaximum`` /
  ``multipleOf``;
* string ``minLength`` / ``maxLength`` / ``pattern``;
* array ``minItems`` / ``maxItems`` / ``items``; nested ``object`` properties/required.

Rules that are not declared by a schema are never invented: if a schema carries no enum,
format or bounds, none are checked.  The top-level tool-call object is stricter than raw
JSON Schema on purpose: a parameter the tool does not declare is an error, matching the
tool-call contract the failure taxonomy already assumes.
"""

from __future__ import annotations

import math
import re
from datetime import date, datetime, time
from typing import Any, Mapping

from .records import FunctionSchema

_TYPE_NAMES = {
    "string": "a string",
    "integer": "an integer",
    "number": "a number",
    "boolean": "a boolean",
    "object": "an object",
    "array": "an array",
    "null": "null",
}

_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_DATETIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:?\d{2})?$")
_TIME_RE = re.compile(r"^\d{2}:\d{2}:\d{2}(\.\d+)?$")
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_URI_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]*://[^\s]+$")
_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
                      r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


def _describe(value: Any) -> str:
    if isinstance(value, bool):
        return f"boolean ({value!r})"
    if isinstance(value, int):
        return f"integer ({value!r})"
    if isinstance(value, float):
        return f"number ({value!r})"
    if isinstance(value, str):
        preview = value if len(value) <= 40 else value[:37] + "..."
        return f"string ({preview!r})"
    if value is None:
        return "null"
    return type(value).__name__


def _matches_type(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "null":
        return value is None
    return True  # unknown annotation: do not invent a check


def _valid_date(value: str) -> bool:
    """Strict calendar date: regex shape plus a real calendar (2024-02-30 must fail)."""
    if not _DATE_RE.match(value):
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def _valid_datetime(value: str) -> bool:
    if not _DATETIME_RE.match(value):
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def _valid_time(value: str) -> bool:
    if not _TIME_RE.match(value):
        return False
    try:
        time.fromisoformat(value)
    except ValueError:
        return False
    return True


def _check_format(value: str, fmt: str) -> bool:
    if fmt == "date":
        return _valid_date(value)
    if fmt == "date-time":
        return _valid_datetime(value)
    if fmt == "time":
        return _valid_time(value)
    if fmt == "email":
        return bool(_EMAIL_RE.match(value))
    if fmt in ("uri", "url"):
        return bool(_URI_RE.match(value))
    if fmt == "uuid":
        return bool(_UUID_RE.match(value))
    return True  # unknown format: annotation only, never fail an unknown format


def _check_scalar(value: Any, prop: Mapping[str, Any], where: str) -> list[str]:
    problems: list[str] = []
    expected = prop.get("type")
    if isinstance(expected, str):
        types = [expected]
    elif isinstance(expected, (list, tuple)):
        types = [str(item) for item in expected]
    else:
        types = []
    if types and not any(_matches_type(value, item) for item in types):
        expected_human = " or ".join(_TYPE_NAMES.get(item, item) for item in types)
        return [f"{where}: expected {expected_human}, got {_describe(value)}"]

    enum = prop.get("enum")
    if isinstance(enum, list) and value not in enum:
        problems.append(f"{where}: value {value!r} not in enum {enum!r}")
    if "const" in prop and value != prop["const"]:
        problems.append(f"{where}: value {value!r} != const {prop['const']!r}")

    if isinstance(value, str):
        fmt = prop.get("format")
        if isinstance(fmt, str) and fmt and not _check_format(value, fmt):
            problems.append(f"{where}: value {value!r} is not a valid {fmt}")
        min_length = prop.get("minLength")
        if isinstance(min_length, int) and len(value) < min_length:
            problems.append(f"{where}: string shorter than minLength {min_length}")
        max_length = prop.get("maxLength")
        if isinstance(max_length, int) and len(value) > max_length:
            problems.append(f"{where}: string longer than maxLength {max_length}")
        pattern = prop.get("pattern")
        if isinstance(pattern, str):
            try:
                if re.search(pattern, value) is None:
                    problems.append(f"{where}: value {value!r} does not match pattern {pattern!r}")
            except re.error:  # malformed pattern in source schema: do not invent a failure
                pass

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        minimum = prop.get("minimum")
        if isinstance(minimum, (int, float)) and value < minimum:
            problems.append(f"{where}: value {value!r} below minimum {minimum!r}")
        maximum = prop.get("maximum")
        if isinstance(maximum, (int, float)) and value > maximum:
            problems.append(f"{where}: value {value!r} above maximum {maximum!r}")
        exclusive_minimum = prop.get("exclusiveMinimum")
        if isinstance(exclusive_minimum, (int, float)) and value <= exclusive_minimum:
            problems.append(f"{where}: value {value!r} not above exclusiveMinimum {exclusive_minimum!r}")
        exclusive_maximum = prop.get("exclusiveMaximum")
        if isinstance(exclusive_maximum, (int, float)) and value >= exclusive_maximum:
            problems.append(f"{where}: value {value!r} not below exclusiveMaximum {exclusive_maximum!r}")
        multiple_of = prop.get("multipleOf")
        if isinstance(multiple_of, (int, float)) and multiple_of:
            quotient = value / multiple_of
            if abs(quotient - round(quotient)) > 1e-9:
                problems.append(f"{where}: value {value!r} is not a multiple of {multiple_of!r}")
            elif not math.isfinite(quotient):  # pragma: no cover - defensive
                problems.append(f"{where}: value {value!r} is not finite")
    return problems


def _check_value(value: Any, prop: Mapping[str, Any], where: str,
                 type_overrides: Mapping[str, str] | None = None,
                 top_level_param: str | None = None) -> list[str]:
    if type_overrides and top_level_param and top_level_param in type_overrides:
        prop = dict(prop)
        prop["type"] = type_overrides[top_level_param]
    problems = _check_scalar(value, prop, where)
    if problems:
        return problems
    if isinstance(value, dict):
        nested_properties = prop.get("properties")
        nested_required = prop.get("required") or []
        if isinstance(nested_properties, Mapping):
            problems.extend(_validate_object(value, nested_properties, nested_required,
                                             where=where, type_overrides=None))
    if isinstance(value, list):
        min_items = prop.get("minItems")
        if isinstance(min_items, int) and len(value) < min_items:
            problems.append(f"{where}: array shorter than minItems {min_items}")
        max_items = prop.get("maxItems")
        if isinstance(max_items, int) and len(value) > max_items:
            problems.append(f"{where}: array longer than maxItems {max_items}")
        items = prop.get("items")
        if isinstance(items, Mapping):
            for index, item in enumerate(value):
                problems.extend(_check_value(item, items, f"{where}[{index}]"))
    return problems


def _validate_object(arguments: Mapping[str, Any], properties: Mapping[str, Any],
                     required: Any, where: str = "arguments",
                     type_overrides: Mapping[str, str] | None = None) -> list[str]:
    problems: list[str] = []
    declared = set(properties)
    for name in arguments:
        if name not in declared:
            problems.append(f"{where}: unexpected parameter {name!r} (not in schema)")
    for name in required or []:
        if name not in arguments:
            problems.append(f"{where}: missing required parameter {name!r}")
    for name, prop in properties.items():
        if name not in arguments:
            continue
        if not isinstance(prop, Mapping):
            prop = {}
        problems.extend(_check_value(arguments[name], prop, f"{where}.{name}"
                                     if where else str(name),
                                     type_overrides=type_overrides, top_level_param=name))
    return problems


def validate_arguments(arguments: Mapping[str, Any] | None,
                       schema: FunctionSchema | None,
                       type_overrides: Mapping[str, str] | None = None) -> list[str]:
    """Return a list of structural violations (empty list = valid)."""
    if schema is None:
        return ["no schema for function"]
    return _validate_object(arguments or {}, schema.properties, schema.required,
                            type_overrides=type_overrides)


def validate_against_parameters(arguments: Mapping[str, Any] | None,
                                parameters: Mapping[str, Any] | None,
                                type_overrides: Mapping[str, str] | None = None) -> list[str]:
    """Validate against a raw agent-visible JSON schema ``parameters`` block."""
    parameters = parameters or {}
    properties = parameters.get("properties") if isinstance(parameters, Mapping) else None
    required = parameters.get("required") if isinstance(parameters, Mapping) else None
    if not isinstance(properties, Mapping):
        properties = {}
    return _validate_object(arguments or {}, properties, required or [],
                            type_overrides=type_overrides)


def schema_valid(arguments: Mapping[str, Any] | None, schema: FunctionSchema | None) -> bool:
    """Backward-compatible boolean wrapper around :func:`validate_arguments`."""
    return not validate_arguments(arguments, schema)
