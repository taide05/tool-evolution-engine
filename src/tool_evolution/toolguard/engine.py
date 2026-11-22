"""The ToolGuard repair engine: deterministic rules, experience memory, learned params."""

from __future__ import annotations

from .memory import SuccessMemory, schema_valid
from .records import FunctionSchema, ToolCall
from .rules import RepairResult, classify_error, parse_missing_params, repair_call

_PARAM_LEVEL_KINDS = {"missing_required", "invalid_value", "unexpected_kwarg", "other"}


def repair_failure(
    call: ToolCall,
    schemas: dict[str, FunctionSchema],
    memory: SuccessMemory | None = None,
) -> RepairResult:
    """Repair one real failed call using the cheapest applicable strategy.

    Order:
      1. deterministic schema rules (rename/drop args, fill schema defaults, resolve names);
      2. experience memory: replay a previously successful invocation of the same task/function;
      3. learned parameter repair: fill/normalize parameters with value modes learned from
         successful calls of the same function across the whole training split.
    """
    result = repair_call(call.tool_name, call.arguments, call.error, call.available_tools, schemas)
    if result.repairable and result.schema_valid:
        return result
    if result.repairable and not result.schema_valid:
        # A repair that does not produce a schema-valid call is not counted as a repair.
        result.repairable = False
        result.reason = f"rule produced schema-invalid call ({result.reason})"
    if memory is None:
        return result
    kind = classify_error(call.error)
    if kind not in _PARAM_LEVEL_KINDS:
        return result
    schema = schemas.get(call.tool_name)

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
    return result


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

    explicit_missing = parse_missing_params(call.error or "")
    targets: list[str] = []
    for raw in explicit_missing:
        name = _resolve_param_name(raw, allowed)
        if name is not None:
            targets.append(name)
    if not targets and schema is not None and schema.required:
        targets = [name for name in schema.required]
    if not targets:
        targets = sorted(observed - set(arguments))

    for name in targets:
        if arguments.get(name) not in (None, ""):
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


def _resolve_param_name(raw: str, allowed: set[str]) -> str | None:
    if raw in allowed:
        return raw
    import difflib  # noqa: PLC0415

    norm = raw.lower().replace(" ", "_").strip("_")
    for name in allowed:
        if name.lower().replace(" ", "_") == norm:
            return name
    matches = difflib.get_close_matches(norm, list(allowed), n=1, cutoff=0.8)
    return matches[0] if matches else None
