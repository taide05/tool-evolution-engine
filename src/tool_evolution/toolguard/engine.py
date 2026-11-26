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
from .memory import SuccessMemory
from .records import FunctionSchema, ToolCall
from .rules import RepairResult, classify_error, parse_missing_params, repair_call
from .schema_check import validate_arguments

_PARAM_LEVEL_KINDS = {"missing_required", "invalid_value", "unexpected_kwarg", "other"}


def coerce_values_to_schema(arguments: dict, schema: FunctionSchema) -> dict:
    """Deterministically convert proposed values to the *declared* scalar types.

    This is a repair, not a validation relaxation: a numeric string for a ``number``
    parameter, a float for an ``integer`` parameter or an int for a ``string`` parameter is
    converted so the call sends what the agent-visible schema declares.  Values that cannot
    be converted losslessly are left untouched (and will fail validation honestly).
    """
    coerced = dict(arguments)
    for name, value in list(coerced.items()):
        prop = schema.properties.get(name) or {}
        expected = str(prop.get("type") or "").lower()
        if isinstance(value, str) and expected in ("number", "integer"):
            text = value.strip().replace(",", "")
            try:
                number = float(text)
            except ValueError:
                continue
            if expected == "integer" and number.is_integer():
                coerced[name] = int(number)
            elif expected == "number":
                coerced[name] = number
        elif isinstance(value, float) and expected == "integer" and value.is_integer():
            # integral floats are losslessly representable as integers (e.g. 8863.0 -> 8863);
            # non-integral floats are left untouched and fail validation honestly.
            coerced[name] = int(value)
        elif isinstance(value, (int, float)) and not isinstance(value, bool) \
                and expected == "string":
            coerced[name] = str(value)
        elif isinstance(value, str) and expected == "boolean" \
                and value.lower() in ("true", "false"):
            coerced[name] = value.lower() == "true"
    return coerced


def _coerced_proposal(proposal: RepairResult, schema: FunctionSchema) -> RepairResult | None:
    """Return an accepted copy of ``proposal`` after schema-type coercion, if that suffices."""
    if proposal.arguments is None:
        return None
    coerced = coerce_values_to_schema(proposal.arguments, schema)
    if coerced == proposal.arguments:
        return None
    errors = validate_arguments(coerced, schema)
    if errors:
        return None
    return RepairResult(
        kind=proposal.kind,
        repairable=True,
        proposed=True,
        rules=list(proposal.rules) + ["schema_type_coercion"],
        function_name=proposal.function_name,
        arguments=coerced,
        schema_valid=True,
        reason=f"{proposal.reason} (values coerced to the declared schema types)",
    )


def _candidate(proposal: RepairResult) -> dict:
    return {
        "rules": list(proposal.rules),
        "function": proposal.function_name or "",
        "arguments": proposal.arguments,
        "schema_errors": list(proposal.schema_errors),
        "schema_valid": bool(proposal.repairable and proposal.schema_valid),
    }


def _rejection(p: RepairResult) -> str:
    rules = "+".join(p.rules) if p.rules else p.reason
    errors = "; ".join(p.schema_errors) if p.schema_errors else p.reason
    return f"{rules}: {errors}"


def _finish(result: RepairResult, proposals: list[RepairResult]) -> RepairResult:
    """Attach the full candidate list (used by the evaluator's value-verification tier)."""
    result.candidates = [_candidate(p) for p in proposals if p.proposed]
    return result


def _accept(accepted: RepairResult, proposals: list[RepairResult]) -> RepairResult:
    """Return an accepted proposal, recording the earlier proposals schema rejected."""
    accepted.rejected_proposals = [
        _rejection(p) for p in proposals
        if p is not accepted and p is not None and p.proposed
        and not (p.repairable and p.schema_valid)
    ]
    return _finish(accepted, proposals)


def _all_rejected(kind: str, proposals: list[RepairResult], base: RepairResult) -> RepairResult:
    rejected = [_rejection(p) for p in proposals
                if p.proposed and not (p.repairable and p.schema_valid)]
    result = base if base.proposed else RepairResult(
        kind=kind, repairable=False, proposed=bool(proposals),
        reason=f"all {len(rejected)} proposed repair(s) were rejected by schema validation")
    result.rejected_proposals = rejected
    return _finish(result, proposals)


def repair_failure(
    call: ToolCall,
    schemas: dict[str, FunctionSchema],
    memory: SuccessMemory | None = None,
    extractor_map: ExtractorMap | None = None,
    successful_arguments: list[dict] | None = None,
) -> RepairResult:
    """Apply the repair ladder; every *proposed* candidate is kept for the schema funnel.

    Candidates are tried in ladder order, each with up to three evidence outcomes:

    1. **schema-valid**: passes the declared structural schema as-is;
    2. **schema_type_coercion**: proposed values are deterministically converted to the
       declared scalar types and then pass the schema (reported as a repaired type
       violation, not hidden);
    3. **recorded_success_match**: the candidate is an *exact* recorded successful invocation
       of the same task/function.  This is empirical evidence, deliberately kept separate
       from schema validity; callers that do not have recorded successes (the live guard)
       never use this tier.

    Rejected candidates are counted and exposed on ``result.candidates`` instead of being
    silently dropped, so the schema funnel is honest.
    """
    base = repair_call(call.tool_name, call.arguments, call.error, call.available_tools, schemas)
    proposals: list[RepairResult] = []
    if base.proposed:
        proposals.append(base)

    kind = classify_error(call.error)
    schema = schemas.get(call.tool_name)
    if kind in _PARAM_LEVEL_KINDS:
        if extractor_map is not None and schema is not None:
            text_repair = _text_parameter_repair(call, schema, extractor_map)
            if text_repair is not None:
                proposals.append(text_repair)

        if memory is not None:
            replay = memory.best_replay(call, schema)
            if replay is not None and schema is not None:
                errors = validate_arguments(replay.arguments, schema)
                proposals.append(RepairResult(
                    kind=kind,
                    repairable=not errors,
                    proposed=True,
                    rules=["memory_replay_successful_call"],
                    function_name=call.tool_name,
                    arguments=replay.arguments,
                    schema_valid=not errors,
                    schema_errors=errors,
                    reason=f"replayed a previously successful argument set (schema score "
                           f"{replay.schema_score})",
                ))
            generic = _learned_parameter_repair(call, schema, memory, kind)
            if generic is not None:
                proposals.append(generic)

        if schema is not None:
            completion = _schema_default_completion(call, schema, kind)
            if completion is not None:
                proposals.append(completion)

    for index, proposal in enumerate(proposals):
        if proposal.repairable and proposal.schema_valid:
            return _accept(proposal, proposals)
        if schema is not None:
            coerced = _coerced_proposal(proposal, schema)
            if coerced is not None:
                proposals[index] = coerced
                return _accept(coerced, proposals)
        if successful_arguments and proposal.arguments is not None \
                and (proposal.function_name or call.tool_name) == call.tool_name:
            if any(proposal.arguments == recorded for recorded in successful_arguments):
                verified = RepairResult(
                    kind=kind,
                    repairable=True,
                    proposed=True,
                    rules=list(proposal.rules) + ["recorded_success_match"],
                    function_name=call.tool_name,
                    arguments=dict(proposal.arguments),
                    schema_valid=False,
                    schema_errors=list(proposal.schema_errors),
                    reason=f"{proposal.reason} (arguments are an exact recorded successful "
                           "invocation of this task/function)",
                    value_verified=True,
                )
                return _accept(verified, proposals)
    return _all_rejected(kind, proposals, base)


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
    if not rules:
        return None
    errors = validate_arguments(arguments, schema)
    return RepairResult(
        kind=classify_error(call.error),
        repairable=not errors,
        proposed=True,
        rules=rules,
        function_name=call.tool_name,
        arguments=arguments,
        schema_valid=not errors,
        schema_errors=errors,
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
    if schema is not None and schema.properties:
        errors = validate_arguments(arguments, schema)
    else:
        errors = []
    valid = not errors
    return RepairResult(
        kind=kind,
        repairable=valid,
        proposed=True,
        rules=rules,
        function_name=function,
        arguments=arguments,
        schema_valid=valid,
        schema_errors=errors,
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
    if not rules:
        return None
    errors = validate_arguments(arguments, schema)
    return RepairResult(
        kind=kind,
        repairable=not errors,
        proposed=True,
        rules=rules,
        function_name=call.tool_name,
        arguments=arguments,
        schema_valid=not errors,
        schema_errors=errors,
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
