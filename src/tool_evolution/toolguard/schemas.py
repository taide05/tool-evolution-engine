"""Authoritative function schemas for the evaluation queries.

The per-query ``api_list`` in the released solvable-query files contains the full
agent-visible API specification (parameter names/types/descriptions/defaults).  We
rebuild the function schema index from that source because the answer files only
store a partial copy for some queries.
"""

from __future__ import annotations

import json
from pathlib import Path

from .naming import change_name, function_name, standardize
from .records import FunctionSchema

_TYPE_MAP = {"NUMBER": "number", "INTEGER": "integer", "STRING": "string", "BOOLEAN": "boolean"}


def build_schema_index(queries_dir: Path) -> tuple[dict[str, FunctionSchema], dict[str, int]]:
    """Return (function_name -> schema, stats) from solvable query files."""
    index: dict[str, FunctionSchema] = {}
    stats = {"queries": 0, "api_entries": 0, "conflicts": 0, "queries_seen": 0}
    for path in sorted(Path(queries_dir).glob("*.json")):
        queries = json.loads(path.read_text(encoding="utf-8"))
        for query in queries:
            stats["queries"] += 1
            for api in query.get("api_list") or []:
                if not isinstance(api, dict) or not api.get("api_name"):
                    continue
                stats["api_entries"] += 1
                fn = function_name(api["api_name"], api.get("tool_name") or "")
                properties: dict[str, dict[str, object]] = {}
                required: list[str] = []
                for param in api.get("required_parameters") or []:
                    name, prop = _param(param)
                    if name:
                        properties[name] = prop
                        required.append(name)
                for param in api.get("optional_parameters") or []:
                    name, prop = _param(param)
                    if name:
                        properties[name] = prop
                schema = FunctionSchema(name=fn, properties=properties, required=tuple(required))
                existing = index.get(fn)
                if existing is not None and existing != schema:
                    stats["conflicts"] += 1
                    continue
                index[fn] = schema
    return index, stats


def _param(param: dict) -> tuple[str | None, dict[str, object]]:
    if not isinstance(param, dict) or not param.get("name"):
        return None, {}
    name = change_name(standardize(str(param["name"])))
    raw_type = str(param.get("type") or "string").upper()
    prop: dict[str, object] = {
        "type": _TYPE_MAP.get(raw_type, "string"),
        "description": str(param.get("description") or ""),
    }
    default = param.get("default")
    if default not in (None, ""):
        prop["example_value"] = default
    return name, prop


def refine_schema_index(
    index: dict[str, FunctionSchema],
    calls,
    *,
    min_samples: int = 3,
    agreement: float = 0.9,
) -> tuple[dict[str, FunctionSchema], dict[str, object]]:
    """Correct schema-index declarations that contradict recorded successful calls.

    2026-11-25 schema-fidelity repair.  The released StableToolBench ``api_list`` is the
    agent-visible spec, but it is demonstrably wrong for some parameters: e.g. watchmode
    date parameters are declared ``NUMBER`` while every recorded *successful* call passes
    an ISO date string, and some functions declare empty parameter lists.  Repairing the
    index (instead of silently coercing/waiving at repair time) needs explicit rules and
    provenance:

    1. **Type correction** for scalar parameters: when at least ``min_samples`` recorded
       successful values exist and at least ``agreement`` of them share one concrete JSON
       type, and that type is *incompatible* with the declared scalar type
       (number/integer declared vs observed string, or string declared vs observed
       integer/number), the declared type is replaced by the observed type.  Numeric
       widening/narrowing (integer <-> number) is never performed.
    2. **Parameter back-fill**: when a function's declared parameter list is empty but its
       recorded successful calls consistently carry arguments, those argument names are
       added as *optional* properties with their observed dominant scalar type.  The same
       evidence bar as the type corrections applies: at least ``min_samples`` recorded
       successful values must carry the parameter and at least ``agreement`` of them must
       share one concrete scalar JSON type.  Required lists are never changed.

    Both kinds of corrections are counted; every correction is recorded (type corrections
    carry up to 10 examples, param back-fills are recorded in full) so the schema change is
    visible evidence rather than a quiet moving target.
    """
    # observed[(function, param)] = {concrete_type: count}
    observed: dict[tuple[str, str], dict[str, int]] = {}
    arg_presence: dict[str, dict[str, int]] = {}
    observed_functions: set[str] = set()
    for call in calls:
        if not call.recorded_ok or call.tool_name not in index:
            continue
        arguments = call.arguments or {}
        if "__raw__" in arguments:
            continue
        observed_functions.add(call.tool_name)
        for name, value in arguments.items():
            kind = _json_type(value)
            if kind is None:
                continue
            bucket = observed.setdefault((call.tool_name, name), {})
            bucket[kind] = bucket.get(kind, 0) + 1
            arg_presence.setdefault(call.tool_name, {})
            arg_presence[call.tool_name][name] = arg_presence[call.tool_name].get(name, 0) + 1

    refined: dict[str, FunctionSchema] = {}
    type_corrections: list[dict[str, object]] = []
    params_added: list[dict[str, object]] = []
    for function, schema in index.items():
        properties = {name: dict(prop) for name, prop in schema.properties.items()}
        required = schema.required
        if not properties and function in arg_presence:
            presence = arg_presence[function]
            for name, _presence_count in sorted(presence.items()):
                bucket = observed.get((function, name)) or {}
                total = sum(bucket.values())
                kind = _dominant_type(bucket, total, agreement, min_samples)
                if kind is None or kind not in ("string", "number", "integer", "boolean"):
                    continue
                agreement_value = round(max(bucket.values()) / total, 3)
                properties[name] = {
                    "type": kind,
                    "description": (
                        "inferred from recorded successful calls (api_list declared an empty "
                        f"parameter list); {total} samples, {agreement_value:.0%} scalar type "
                        "agreement"),
                }
                params_added.append({"function": function, "param": name, "observed_type": kind,
                                     "observations": total, "agreement": agreement_value})
        for name, prop in list(properties.items()):
            declared = prop.get("type")
            if declared not in ("number", "integer", "string", "boolean"):
                continue
            bucket = observed.get((function, name))
            if not bucket:
                continue
            total = sum(bucket.values())
            if total < min_samples:
                continue
            kind, count = max(bucket.items(), key=lambda item: item[1])
            if not count or count / total < agreement:
                continue
            if _incompatible(declared, kind):
                prop["type"] = kind
                prop["type_corrected_from"] = declared
                prop["type_correction_evidence"] = f"{count}/{total} recorded successful calls use {kind}"
                type_corrections.append({
                    "function": function,
                    "param": name,
                    "declared": declared,
                    "observed_type": kind,
                    "observations": total,
                    "agreement": round(count / total, 3),
                })
        refined[function] = FunctionSchema(
            name=function, properties=properties, required=required)
    stats: dict[str, object] = {
        "refined": True,
        "observed_functions": len(observed_functions),
        "type_corrections": len(type_corrections),
        "params_added": len(params_added),
        "type_correction_examples": type_corrections[:10],
        # Full back-fill list (not truncated): every added parameter is evidence that the
        # released api_list declared an empty parameter list for that function.
        "param_additions": params_added,
        "param_addition_examples": params_added[:10],  # deprecated alias, do not rely on it
        "min_samples": min_samples,
        "agreement_threshold": agreement,
        "explanation": (
            "The released api_list is the agent-visible spec; it is repaired only where "
            "recorded successful calls prove it wrong, using the same evidence bar for both "
            "kinds of change (>=3 recorded successful values, >=90% agreement on one concrete "
            "scalar type). (1) Type corrections: a declared scalar type is replaced only when "
            "the observed type is incompatible with it (number/integer declared vs observed "
            "string, or string declared vs observed integer/number); numeric widening/narrowing "
            "(integer<->number) is never applied. (2) Parameter back-fill: a function whose "
            "declared parameter list is empty gets the recorded argument names added as "
            "OPTIONAL properties with their observed dominant scalar type. Required lists are "
            "never changed; every correction is recorded in this block (param_additions is "
            "complete, not sampled)."),
    }
    return refined, stats


def _json_type(value) -> str | None:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    return None


def _dominant_type(bucket: dict[str, int], total: int, agreement: float,
                   min_samples: int = 3) -> str | None:
    if not bucket or total < min_samples:
        return None
    kind, count = max(bucket.items(), key=lambda item: item[1])
    if count / total < agreement:
        return None
    return kind


def _incompatible(declared: str, observed: str) -> bool:
    if declared == observed:
        return False
    if declared in ("number", "integer") and observed == "string":
        return True
    if declared == "string" and observed in ("integer", "number"):
        return True
    return False
