"""The ToolGuard repair engine.

Strategy ladder (cheap and general first):
  1. deterministic schema rules (rename/drop args, fill schema defaults, resolve names);
  2. task-text parameter extraction learned from training successes (cross-task general);
  3. experience memory: replay a previously successful invocation of the same task/function;
  4. learned per-parameter value modes from successful calls of the same function;
  5. schema default completion when the error names no parameters.
"""

from __future__ import annotations

import calendar
import difflib

from .extractors import ExtractorMap
from .memory import SuccessMemory, schema_valid
from .records import FunctionSchema, ToolCall
from .rules import RepairResult, classify_error, parse_missing_params, repair_call

_PARAM_LEVEL_KINDS = {"missing_required", "invalid_value", "unexpected_kwarg", "other"}


def repair_failure(
    call: ToolCall,
    schemas: dict[str, FunctionSchema],
    memory: SuccessMemory | None = None,
    extractor_map: ExtractorMap | None = None,
) -> RepairResult:
    result = repair_call(call.tool_name, call.arguments, call.error, call.available_tools, schemas)
    if result.repairable and result.schema_valid:
        return result
    if result.repairable and not result.schema_valid:
        result.repairable = False
        result.reason = f"rule produced schema-invalid call ({result.reason})"

    kind = classify_error(call.error)
    if kind not in _PARAM_LEVEL_KINDS:
        return result
    schema = schemas.get(call.tool_name)

    if extractor_map is not None and schema is not None:
        text_repair = _text_parameter_repair(call, schema, extractor_map)
        if text_repair is not None:
            return text_repair

    if memory is not None:
        replay = memory.best_replay(call, schema)
        if replay is not None and schema is not None and schema_valid(replay.arguments, schema):
            return RepairResult(
                kind=kind,
                repairable=True,
                rules=["memory_replay_successful_call"],
                function_name=call.tool_name,
                arguments=replay.arguments,
                schema_valid=True,
                reason=f"replayed a previously successful argument set (schema score {replay.schema_score})",
            )
        generic = _learned_parameter_repair(call, schema, memory, kind)
        if generic is not None:
            return generic

    if schema is not None:
        completion = _schema_default_completion(call, schema, kind)
        if completion is not None:
            return completion
    return result


# ---------------------------------------------------------------------------
# strategy 2: learned task-text extractors
# ---------------------------------------------------------------------------
def _text_parameter_repair(call: ToolCall, schema: FunctionSchema, extractor_map: ExtractorMap) -> RepairResult | None:
    if not call.query:
        return None
    missing_names = _missing_parameter_names(call, schema)
    if not missing_names:
        return None
    arguments = dict(call.arguments)
    rules: list[str] = []
    for raw_name in missing_names:
        name = _resolve_param_name(raw_name, set(schema.properties))
        if name is None:
            continue
        if arguments.get(name) not in (None, ""):
            continue
        value = _extract_for_param(name, call.query, extractor_map)
        if value is None:
            continue
        arguments[name] = value
        rules.append(f"text_extract:{name}")
    if not rules or not schema_valid(arguments, schema):
        return None
    return RepairResult(
        kind=classify_error(call.error),
        repairable=True,
        rules=rules,
        function_name=call.tool_name,
        arguments=arguments,
        schema_valid=True,
        reason="recovered missing parameters from the task text using learned extractors",
    )


def _extract_for_param(name: str, query: str, extractor_map: ExtractorMap) -> object | None:
    lowered = name.lower()
    extractor_name = extractor_map.extractor_for(name)
    # dates: pick first for start-ish params, last for end-ish params; expand month-year values
    if extractor_name == "dates":
        dates = extractor_map.candidates(name, query)
        if not dates:
            months = _extract_month_years(query)
            if months:
                is_end = _is_end_param(lowered)
                return _expand_month_year(months[-1] if is_end else months[0], is_end)
            return None
        return dates[-1] if _is_end_param(lowered) else dates[0]
    if extractor_name == "years":
        years = [int(y) for y in extractor_map.candidates(name, query)]
        if not years:
            return None
        if any(key in lowered for key in ("max", "end", "to")):
            return max(years)
        if any(key in lowered for key in ("min", "start", "from")):
            return min(years)
        return years[0]
    if extractor_name == "numbers":
        numbers = [int(n) for n in extractor_map.candidates(name, query)]
        return numbers[0] if numbers else None
    candidates = extractor_map.candidates(name, query)
    return candidates[0] if candidates else None


def _is_end_param(lowered: str) -> bool:
    return any(key in lowered for key in ("end", "to", "until", "max", "stop"))


def _expand_month_year(value: str, is_end: bool) -> str | None:
    parts = value.split("-")
    if len(parts) != 2:
        return value
    try:
        year, month = int(parts[0]), int(parts[1])
    except ValueError:
        return value
    if is_end:
        return f"{year:04d}-{month:02d}-{calendar.monthrange(year, month)[1]:02d}"
    return f"{year:04d}-{month:02d}-01"


def _extract_month_years(query: str) -> list[str]:
    from .extractors import extract_month_years

    return extract_month_years(query)


# ---------------------------------------------------------------------------
# strategy 3: learned parameter values (same function)
# ---------------------------------------------------------------------------
def _learned_parameter_repair(
    call: ToolCall,
    schema: FunctionSchema | None,
    memory: SuccessMemory,
    kind: str,
) -> RepairResult | None:
    function = call.tool_name
    observed = memory.observed_params(function)
    allowed = set(schema.properties) if schema and schema.properties else observed
    if not allowed:
        return None
    arguments = dict(call.arguments)
    rules: list[str] = []

    targets = _missing_parameter_names(call, schema) if schema is not None else list(parse_missing_params(call.error or ""))
    for raw in targets:
        name = _resolve_param_name(raw, allowed)
        if name is None or arguments.get(name) not in (None, ""):
            continue
        values = memory.param_values(function, name)
        if not values:
            continue
        arguments[name] = values.most_common(1)[0][0]
        rules.append(f"memory_param_fill:{name}")

    if kind == "invalid_value":
        for name, value in list(arguments.items()):
            values = memory.param_values(function, name)
            if len(values) < 2:
                continue
            known = {str(item) for item in values}
            if str(value) not in known:
                arguments[name] = values.most_common(1)[0][0]
                rules.append(f"memory_value_normalize:{name}")

    if not rules:
        return None
    valid = schema_valid(arguments, schema) if (schema is not None and schema.properties) else True
    if not valid:
        return None
    return RepairResult(
        kind=kind,
        repairable=True,
        rules=rules,
        function_name=function,
        arguments=arguments,
        schema_valid=valid,
        reason="learned parameter repair from successful calls of the same function",
    )


# ---------------------------------------------------------------------------
# strategy 4: schema default completion (unnamed missing parameters)
# ---------------------------------------------------------------------------
def _schema_default_completion(call: ToolCall, schema: FunctionSchema, kind: str) -> RepairResult | None:
    if kind != "missing_required" or schema.required:
        return None
    defaulted = {name: prop["example_value"] for name, prop in schema.properties.items()
                 if prop.get("example_value") not in (None, "")}
    if not defaulted:
        return None
    arguments = dict(call.arguments)
    rules: list[str] = []
    for name, value in defaulted.items():
        if arguments.get(name) not in (None, ""):
            continue
        arguments[name] = value
        rules.append(f"schema_default:{name}")
    if not rules or not schema_valid(arguments, schema):
        return None
    return RepairResult(
        kind=kind,
        repairable=True,
        rules=rules,
        function_name=call.tool_name,
        arguments=arguments,
        schema_valid=True,
        reason="completed the call with schema-declared default/example values",
    )


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _missing_parameter_names(call: ToolCall, schema: FunctionSchema) -> list[str]:
    names = parse_missing_params(call.error or "")
    if names:
        return names
    return [name for name in schema.required if call.arguments.get(name) in (None, "")]


def _resolve_param_name(raw: str, allowed: set[str]) -> str | None:
    if raw in allowed:
        return raw
    norm = raw.lower().replace(" ", "_").strip("_")
    for name in allowed:
        if name.lower().replace(" ", "_") == norm:
            return name
    matches = difflib.get_close_matches(norm, list(allowed), n=1, cutoff=0.8)
    return matches[0] if matches else None
