"""A1.8.1 experiment: evidence-based schema-index declarations for the 95 named
missing_required no-proposal failures (A1.5 subset).

Scope and guardrails
--------------------
* Read-only with respect to the committed evidence: no committed schema/artifact is
  changed.  The current refined schema index is copied, augmented **in memory only**,
  and the repair ladder is re-run on the 95 cases.
* A declaration for ``(function, parameter)`` is inferred **only from training-side
  successful calls** (fixed-three training set, excluding the evaluated fold), never
  from test-fold successes.  Evidence bar: >=3 successful samples carrying the
  parameter and >=90% agreement on one concrete scalar JSON type.
* No empty ``{}`` is ever treated as a declaration.  Parameters without evidence stay
  ``insufficient_samples`` (or ``conflict``) and are *not* added to the augmented
  schema, so the ladder cannot silently send an undeclared parameter.
* country/currency false positives ("US Dollars" -> country='US') are reported
  separately and are never counted as a fix.

Configurations measured on the 95 cases (per fold, same training-side memory):
  baseline            current ladder + current schema index (0 proposals by construction)
  a18_schema          in-memory augmented schema with evidence-based declarations,
                      default extractor map (headline configuration)
  a18_schema_extract  same schema + the A1.7 experimental extractors (sensitivity)

Usage:
    python scripts/experiment_a18_schema_completeness.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tool_evolution.toolguard.backend import RecordedBackend  # noqa: E402
from tool_evolution.toolguard.cache_index import CacheIndex  # noqa: E402
from tool_evolution.toolguard.engine import _resolve_param_name, repair_failure  # noqa: E402
from tool_evolution.toolguard.extractors import ExtractorMap  # noqa: E402
from tool_evolution.toolguard.memory import SuccessMemory  # noqa: E402
from tool_evolution.toolguard.records import FunctionSchema, load_calls  # noqa: E402
from tool_evolution.toolguard.rules import classify_error, parse_missing_params  # noqa: E402
from tool_evolution.toolguard.schemas import build_schema_index, refine_schema_index  # noqa: E402

TRAIN_MODELS = {"chatgpt", "toolllama", "gpt-3.5-turbo-1106"}
_SENTINEL_CALLS = {"invalid_hallucination_function_name"}
_SCALAR_TYPES = ("string", "integer", "number", "boolean")

# headline aggregate from benchmarks/toolguard/metrics.json results.lomo_aggregate
_EXPECTED_BASELINE = {
    "evaluable_failures": 969,
    "proposed_repairs": 583,
    "schema_passed": 519,
    "value_verified_repairs": 38,
    "coerced_repairs": 131,
    "accepted_repairs": 557,
}
_EXPECTED_SKIPPED_NO_SCHEMA = 64

_CURRENCY_ONLY_RE = re.compile(r"\bus\s+dollars?\b|\busd?\s+dollars?\b|\busd\b", re.I)


def _json_type(value) -> str | None:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return None


def _declared(prop: dict | None) -> bool:
    """True when the schema index carries an actual declaration, not empty {}."""
    if not isinstance(prop, dict) or not prop:
        return False
    return any(prop.get(key) not in (None, "", [], {}) for key in
               ("type", "enum", "const", "format", "pattern", "minimum", "maximum"))


def _param_evidence(function: str, param: str,
                    train_successes: dict[str, list[dict]]) -> dict:
    """Evidence for one (function, parameter) from *training-side* successful calls."""
    samples = [arguments[param] for arguments in train_successes.get(function, [])
               if param in arguments]
    types = Counter(_json_type(value) for value in samples)
    types.pop(None, None)
    total = len(samples)
    scalar_types = {kind: count for kind, count in types.items() if kind in _SCALAR_TYPES}
    dominant = None
    agreement = 0.0
    if scalar_types:
        dominant, count = max(scalar_types.items(), key=lambda item: item[1])
        agreement = round(count / total, 4) if total else 0.0
    enum_candidates: list[str] = []
    if dominant == "string" and samples:
        counts = Counter(str(value) for value in samples)
        if len(counts) <= 10 and counts.most_common(1)[0][1] / total >= 0.9:
            enum_candidates = [value for value, _ in counts.most_common()]
    if dominant is not None and total >= 3 and agreement >= 0.9:
        status = "inferred"
    elif total >= 3 and scalar_types:
        status = "conflict"
    else:
        status = "insufficient_samples"
    return {
        "function": function,
        "param": param,
        "success_samples": total,
        "types": dict(types),
        "dominant_type": dominant,
        "agreement": agreement,
        "enum_candidates": enum_candidates,
        "inferred_status": status,
    }


def _evidence_status(evidence: dict, prop: dict | None) -> str:
    if _declared(prop):
        return "declared"
    return evidence["inferred_status"]


def _fold_training_successes(train_calls) -> dict[str, list[dict]]:
    """function -> list of training-side successful argument dicts."""
    by_function: dict[str, list[dict]] = defaultdict(list)
    for call in train_calls:
        if call.recorded_ok and call.valid_data:
            by_function[call.tool_name].append(dict(call.arguments))
    return by_function


def _augment_schema(schema: FunctionSchema, params: list[str],
                    evidence: dict[tuple[str, str], dict],
                    declarations: dict[tuple[str, str], dict]) -> tuple[FunctionSchema, list[str]]:
    """Add only evidence-based declarations; never a typed-less {} placeholder."""
    properties = {name: dict(prop) for name, prop in schema.properties.items()}
    required = list(schema.required)
    added: list[str] = []
    for name in params:
        key = (schema.name, name)
        ev = evidence.get(key)
        if ev is None:
            continue
        status = declarations[key]["status"]
        if status in ("declared", "inferred"):
            if name not in properties:
                properties[name] = {
                    "type": declarations[key]["type"],
                    "description": (
                        "inferred from training-side recorded successful calls; "
                        f"{ev['success_samples']} samples, {ev['agreement']:.0%} scalar type "
                        "agreement"),
                }
                added.append(name)
            if name not in required:
                required.append(name)
        # insufficient / conflict: do not fabricate a declaration or a parameter
    return FunctionSchema(name=schema.name, properties=properties,
                          required=tuple(required)), added


def _classify(result) -> dict:
    if result.repairable:
        status = "accepted_schema_valid" if result.schema_valid else "accepted_value_verified"
    elif result.proposed:
        status = "proposed_schema_rejected"
    else:
        status = "no_proposal"
    return {
        "status": status,
        "rules": list(result.rules),
        "arguments": result.arguments,
        "schema_errors": list(result.schema_errors)[:5],
    }


def _md_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calls", default=".scratch/toolguard/tool_calls.jsonl")
    parser.add_argument("--queries", default=(
        ".scratch/stb/repo/intuit-ai-research-StableToolBench-74f76c1/"
        "solvable_queries/test_instruction"))
    parser.add_argument("--expected-metrics", default="benchmarks/toolguard/metrics.json")
    parser.add_argument("--cache", default=".scratch/stb/cache/tool_response_cache")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--output-dir", default=".scratch/toolguard/a18")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    calls = [call for call in load_calls(repo_root / args.calls)
             if call.tool_name not in _SENTINEL_CALLS]
    schemas, schema_stats = build_schema_index(repo_root / args.queries)
    schemas, schema_fidelity = refine_schema_index(schemas, calls)
    families = sorted({call.model for call in calls})
    print(f"calls: {len(calls)} | schemas: {len(schemas)} | families: {families}")

    backend = None
    if not args.no_cache and (repo_root / args.cache).is_dir():
        cache_root = repo_root / args.cache
        backend = RecordedBackend(cache_root, CacheIndex.from_root(cache_root))
        print(f"cache verification: {len(backend.index)} indexed responses")

    # ------------------------------------------------------------------
    # 1. baseline protocol + the 95 named missing_required no-proposal cases
    # ------------------------------------------------------------------
    protocol = Counter()
    cases: list[dict] = []
    case_calls = []
    for family in families:
        train = [call for call in calls
                 if call.model != family and call.model in TRAIN_MODELS]
        test = [call for call in calls if call.model == family]
        memory = SuccessMemory.build(train, schemas)
        extractor_map = ExtractorMap.learn(train, schemas)
        train_successes = _fold_training_successes(train)
        reference: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
        for call in test:
            if call.recorded_ok:
                reference[(call.group, call.query_id, call.tool_name)].append(call.arguments)

        for call in test:
            if call.recorded_ok:
                continue
            if "__raw__" in call.arguments or call.tool_name not in schemas:
                protocol["skipped_no_schema"] += 1
                continue
            kind = classify_error(call.error)
            protocol["evaluable_failures"] += 1
            result = repair_failure(call, schemas, memory, extractor_map,
                                    successful_arguments=reference.get(
                                        (call.group, call.query_id, call.tool_name)) or [])
            if result.proposed:
                protocol["proposed_repairs"] += 1
            if result.repairable:
                protocol["accepted_repairs"] += 1
                if result.schema_valid:
                    protocol["schema_passed"] += 1
                else:
                    protocol["value_verified_repairs"] += 1
                if "schema_type_coercion" in result.rules:
                    protocol["coerced_repairs"] += 1
            named = parse_missing_params(call.error or "") if kind == "missing_required" else []
            if not named or result.proposed:
                continue

            schema = schemas[call.tool_name]
            resolved: list[str] = []
            for raw in named:
                name = _resolve_param_name(raw, set(schema.properties))
                resolved.append(name if name is not None else raw)

            # evidence is training-side only
            evidence = {name: _param_evidence(call.tool_name, name, train_successes)
                        for name in resolved}
            declarations: dict[tuple[str, str], dict] = {}
            for name, ev in evidence.items():
                prop = schema.properties.get(name)
                status = _evidence_status(ev, prop)
                declarations[(call.tool_name, name)] = {
                    "status": status,
                    "type": ev["dominant_type"] if status == "inferred" else None,
                }

            statuses = [declarations[(call.tool_name, name)]["status"] for name in resolved]
            if all(status == "declared" for status in statuses):
                case_status = "declared"
            elif "conflict" in statuses:
                case_status = "conflict"
            elif "insufficient_samples" in statuses:
                case_status = "insufficient_samples"
            else:
                case_status = "inferred"

            cases.append({
                "fold": family,
                "tool": call.tool_name,
                "group": call.group,
                "query_id": call.query_id,
                "model": call.model,
                "error": call.error,
                "query": call.query,
                "named": named,
                "resolved": resolved,
                "case_status": case_status,
                "param_evidence": evidence,
                "declarations": {name: declarations[(call.tool_name, name)]
                                 for name in resolved},
                "failed_arguments": call.arguments,
                "reference_count": len(reference.get(
                    (call.group, call.query_id, call.tool_name)) or []),
            })
            case_calls.append(call)

    print(f"named missing_required no-proposal cases: {len(cases)}")
    computed = {key: int(protocol.get(key, 0)) for key in _EXPECTED_BASELINE}
    expected_metrics = json.loads(
        (repo_root / args.expected_metrics).read_text(encoding="utf-8"))
    headline = expected_metrics["results"]["lomo_aggregate"]
    expected = {
        "evaluable_failures": headline["evaluable_failures"],
        "proposed_repairs": headline["proposed_repairs"],
        "schema_passed": headline["schema_passed"],
        "value_verified_repairs": headline["value_verified_repairs"],
        "coerced_repairs": headline["coerced_repairs"],
        "accepted_repairs": headline["accepted_repairs"],
    }
    protocol_match = computed == expected
    print(f"protocol verification: {'MATCH' if protocol_match else 'MISMATCH'} {computed}")

    # ------------------------------------------------------------------
    # 2. per-(function, parameter) declaration table
    # ------------------------------------------------------------------
    param_rows: dict[tuple[str, str], dict] = {}
    for case in cases:
        for name in case["resolved"]:
            key = (case["tool"], name)
            row = param_rows.get(key)
            if row is None:
                ev = case["param_evidence"][name]
                prop = schemas[case["tool"]].properties.get(name)
                row = {
                    "function": case["tool"],
                    "param": name,
                    "success_samples": ev["success_samples"],
                    "types": ev["types"],
                    "dominant_type": ev["dominant_type"],
                    "agreement": ev["agreement"],
                    "enum_candidates": ev["enum_candidates"],
                    "status": case["declarations"][name]["status"],
                    "in_schema_index": name in schemas[case["tool"]].properties,
                    "schema_declaration": prop if isinstance(prop, dict) else None,
                    "cases": 0,
                }
                param_rows[key] = row
            row["cases"] += 1

    summary = Counter()
    for case in cases:
        summary["cases_total"] += 1
        summary[f"case_{case['case_status']}"] += 1
    for row in param_rows.values():
        summary["params_total"] += 1
        summary[f"param_{row['status']}"] += 1
        if row["status"] == "inferred":
            summary["param_inferred_scalar_type"] += 1
            if row["enum_candidates"]:
                summary["param_with_enum_candidates"] += 1

    # ------------------------------------------------------------------
    # 3. re-run the 95 cases against the in-memory augmented schema
    # ------------------------------------------------------------------
    schemas_augmented = dict(schemas)
    augmented_functions: dict[str, list[str]] = {}
    for function in sorted({case["tool"] for case in cases}):
        schema = schemas[function]
        function_params = sorted({name for case in cases if case["tool"] == function
                                  for name in case["resolved"]})
        evidence = {}
        declarations = {}
        for case in cases:
            if case["tool"] != function:
                continue
            for name in case["resolved"]:
                evidence[(function, name)] = case["param_evidence"][name]
                declarations[(function, name)] = case["declarations"][name]
        augmented, added = _augment_schema(schema, function_params, evidence, declarations)
        schemas_augmented[function] = augmented
        augmented_functions[function] = added
    augmented_delta = {
        function: {
            "params_added": added,
            "required_before": list(schemas[function].required),
            "required_after": list(schemas_augmented[function].required),
            "properties_before": sorted(schemas[function].properties),
            "properties_after": sorted(schemas_augmented[function].properties),
        }
        for function, added in augmented_functions.items()
    }

    configs: dict[str, dict] = {}
    accepted_rows: dict[str, list[dict]] = {}
    for config_name, experimental in (("a18_schema", False), ("a18_schema_extract", True)):
        # rebuild per-fold memory/extractors (training-only) and rerun each case
        fold_models = {}
        for family in families:
            train = [call for call in calls
                     if call.model != family and call.model in TRAIN_MODELS]
            fold_models[family] = (
                SuccessMemory.build(train, schemas_augmented),
                ExtractorMap.learn(train, schemas_augmented, include_experimental=experimental),
            )
        counts: Counter[str] = Counter()
        rows: list[dict] = []
        for case, call in zip(cases, case_calls):
            memory, extractor_map = fold_models[case["fold"]]
            result = repair_failure(call, schemas_augmented, memory, extractor_map,
                                    successful_arguments=[])
            after = _classify(result)
            counts[after["status"]] += 1
            row = {
                "fold": case["fold"],
                "tool": case["tool"],
                "params": case["resolved"],
                "case_status": case["case_status"],
                "after_status": after["status"],
                "after_rules": after["rules"],
                "after_arguments": after["arguments"],
                "cache_status": None,
            }
            if backend is not None and after["status"].startswith("accepted"):
                payload = backend.call(result.function_name or case["tool"],
                                       result.arguments or {})
                row["cache_status"] = payload.status
            rows.append(row)
        accepted_rows[config_name] = rows
        counts["accepted"] = counts.get("accepted_schema_valid", 0) + \
            counts.get("accepted_value_verified", 0)
        counts["proposed"] = counts["accepted"] + counts.get("proposed_schema_rejected", 0)
        configs[config_name] = dict(counts)

    # country/currency false-positive audit over accepted repairs
    false_positives: list[dict] = []
    for config_name, rows in accepted_rows.items():
        for row in rows:
            if not row["after_status"].startswith("accepted"):
                continue
            arguments = row["after_arguments"] or {}
            for name, value in arguments.items():
                if "country" not in name.lower() or not isinstance(value, str):
                    continue
                case = next(case for case in cases if case["fold"] == row["fold"]
                            and case["tool"] == row["tool"]
                            and case["resolved"] == row["params"])
                if _CURRENCY_ONLY_RE.search(case["query"] or ""):
                    false_positives.append({
                        "config": config_name,
                        "fold": row["fold"],
                        "tool": row["tool"],
                        "param": name,
                        "value": value,
                        "query_head": (case["query"] or "")[:160],
                        "reason": "country extracted from a currency phrase ('US Dollars'); "
                                  "schema-valid is structural only, not a verified fix",
                    })
    fp_distinct = {(fp["tool"], fp["param"], fp["value"]) for fp in false_positives}

    # context: why evidence is missing — recorded successes per named function (all folds)
    function_context: dict[str, dict] = {}
    for function in sorted({case["tool"] for case in cases}):
        function_calls = [call for call in calls if call.tool_name == function]
        successful = [call for call in function_calls if call.recorded_ok]
        valid_successful = [call for call in successful if call.valid_data]
        params = sorted({name for case in cases if case["tool"] == function
                         for name in case["resolved"]})
        function_context[function] = {
            "calls": len(function_calls),
            "successful_calls": len(successful),
            "valid_data_successful_calls": len(valid_successful),
            "named_params": params,
            "successful_calls_carrying_named_param": {
                name: sum(1 for call in successful if name in call.arguments)
                for name in params
            },
            "valid_data_successful_calls_carrying_named_param": {
                name: sum(1 for call in valid_successful if name in call.arguments)
                for name in params
            },
            "note": ("evidence is counted per (function, parameter) from training-side "
                     "successful calls only; a parameter that never appears in a successful "
                     "call of this function cannot be declared from record evidence"),
        }

    # ------------------------------------------------------------------
    # 4. output
    # ------------------------------------------------------------------
    result = {
        "experiment": "ToolGuard A1.8.1 evidence-based schema-index declarations",
        "protocol": {
            "protocol": "lomo fixed-three (training side = TRAIN_MODELS minus the fold)",
            "train_models": sorted(TRAIN_MODELS),
            "folds": families,
            "calls": len(calls),
            "schemas": schema_stats,
            "schema_fidelity_repair": {
                "type_corrections": schema_fidelity["type_corrections"],
                "params_added": schema_fidelity["params_added"],
            },
            "subset": ("95 named missing_required failures that get no proposal under the "
                       "baseline ladder (A1.5)"),
            "evidence_bar": {"min_samples": 3, "agreement": 0.9,
                             "enum_unique_max": 10, "enum_top1_agreement": 0.9},
            "declaration_source": "training-side successful calls only (test fold excluded)",
            "case_status_rule": (
                "declared: all named params already declared in the schema index; "
                "inferred: every non-declared named param gets an evidence-based scalar "
                "type; insufficient_samples: at least one param has <3 training samples "
                "and none conflicts; conflict: >=3 samples but <90% type agreement"),
        },
        "headline_default_unchanged": {
            "computed": computed,
            "expected": expected,
            "match": protocol_match,
        },
        "summary": dict(summary),
        "parameters": [row for row in sorted(param_rows.values(),
                                             key=lambda item: (item["status"], item["function"],
                                                               item["param"]))],
        "function_context": function_context,
        "cases": cases,
        "augmented_schema_delta": augmented_delta,
        "configurations": configs,
        "accepted_detail": accepted_rows,
        "false_positives_country_currency": {
            "count": len(false_positives),
            "distinct_tool_param_value": len(fp_distinct),
            "distinct_tasks": len({(fp["tool"], fp["query_head"]) for fp in false_positives}),
            "items": false_positives,
            "note": ("these are excluded from any recovered count: a country value taken "
                     "from a currency phrase is schema-valid but not a verified repair"),
        },
        "limitations": [
            "schema-valid is structural only: the augmented schema declares types from "
            "training-side evidence, but a repair still has to be sent to the backend to be "
            "proven; cache verification is the only independent check in this experiment",
            "the training-side evidence has <3 samples for many named parameters, so most "
            "cases stay insufficient_samples; that is the honest result, not a gap to fill "
            "by weakening the evidence bar",
            "the A1.7 experimental extractor configuration is a sensitivity row only; it does "
            "not change the default (headline) configuration",
        ],
    }
    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "schema_completeness.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

    report = _render_report(result)
    (output_dir / "schema_completeness.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nwrote {output_dir / 'schema_completeness.json'}")
    print(f"wrote {output_dir / 'schema_completeness.md'}")
    return 0 if protocol_match else 1


def _render_report(result: dict) -> str:
    lines = ["# A1.8.1 schema completeness (95 named missing_required no-proposal cases)\n"]
    headline = result["headline_default_unchanged"]
    lines.append(f"Protocol verification against metrics.json: "
                 f"**{'MATCH' if headline['match'] else 'MISMATCH'}** "
                 f"`{json.dumps(headline['computed'], ensure_ascii=False)}`\n")
    summary = result["summary"]
    lines.append("## Case status (95/95 have a status)\n")
    lines.append(_md_table(["status", "cases"], [
        ["declared (already in schema index)", summary.get("case_declared", 0)],
        ["inferred (evidence-based type added)", summary.get("case_inferred", 0)],
        ["insufficient_samples (<3 training samples)", summary.get("case_insufficient_samples", 0)],
        ["conflict (>=3 samples, <90% agreement)", summary.get("case_conflict", 0)],
        ["total", summary.get("cases_total", 0)],
    ]))
    lines.append("\n## Parameters (all named parameters, training-side evidence)\n")
    lines.append(_md_table(
        ["function", "param", "samples", "types", "dominant", "agreement", "status",
         "enum candidates"],
        [[row["function"], row["param"], row["success_samples"],
          json.dumps(row["types"], ensure_ascii=False), row["dominant_type"] or "-",
          row["agreement"], row["status"], ", ".join(row["enum_candidates"]) or "-"]
         for row in result["parameters"]]))
    lines.append("\n## In-memory augmented schema\n")
    lines.append(_md_table(
        ["function", "params added", "required before", "required after"],
        [[function, ", ".join(delta["params_added"]) or "-",
          ", ".join(delta["required_before"]) or "-",
          ", ".join(delta["required_after"]) or "-"]
         for function, delta in result["augmented_schema_delta"].items()]))
    lines.append("\n## Re-run of the 95 cases against the augmented schema\n")
    lines.append(_md_table(
        ["configuration", "proposed", "schema-valid", "value-verified", "schema rejected",
         "no proposal"],
        [[name, counts.get("proposed", 0), counts.get("accepted_schema_valid", 0),
          counts.get("accepted_value_verified", 0), counts.get("proposed_schema_rejected", 0),
          counts.get("no_proposal", 0)]
         for name, counts in result["configurations"].items()]))
    fp = result["false_positives_country_currency"]
    lines.append("\n## country/currency false positives\n")
    lines.append(f"count: **{fp['count']}** (distinct tool/param/value: {fp['distinct_tool_param_value']}, "
                 f"distinct tasks: {fp['distinct_tasks']}) — excluded from any recovered count.\n")
    rows = [[item["config"], item["tool"], item["param"], item["value"], item["query_head"]]
            for item in fp["items"]]
    if rows:
        lines.append(_md_table(["config", "tool", "param", "value", "query head"], rows))
    lines.append("\n## Why there is no declaration evidence (recorded successes per function)\n")
    lines.append(_md_table(
        ["function", "calls", "successful calls", "valid-data successes",
         "named params", "successful calls carrying the named param"],
        [[function, context["calls"], context["successful_calls"],
          context["valid_data_successful_calls"], ", ".join(context["named_params"]),
          json.dumps(context["successful_calls_carrying_named_param"], ensure_ascii=False)]
         for function, context in result["function_context"].items()]))
    lines.append("\n## Per-case status (95/95)\n")
    lines.append(_md_table(
        ["#", "fold", "tool", "named params", "resolved params", "case status"],
        [[index + 1, case["fold"], case["tool"], ", ".join(case["named"]),
          ", ".join(case["resolved"]), case["case_status"]]
         for index, case in enumerate(result["cases"])]))
    lines.append("\n## Limitations\n")
    for limitation in result["limitations"]:
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
