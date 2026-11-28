"""A1.7 experiment: deterministic value extractors + generic-error replay.

Read-only with respect to the committed evidence: rebuilds the LOMO fixed-three
evaluation in memory and compares configurations on the 228 ``missing_required``
failures that get **no repair proposal** under the current repair ladder.

Configurations
--------------
baseline      current ladder + default extractor map (must reproduce 228 no-proposal)
a16_schema    baseline ladder + in-memory error-driven schema completion (A1.6's
              experiment: the named missing parameters are added to properties/required,
              with no declared type, so "schema-valid" is structural only)
a17_extract   a16_schema + A1.7 deterministic extractors enabled (source/state/topic/ID)
a17_replay    a17_extract + generic-error same-task/function recorded-success replay
              (only for errors that name no parameter; schema gate applied as usual).
              A1.8.3: the replay candidate source is selected with ``--replay-source``:
                * ``training`` (default) -- training-side SuccessMemory successes only,
                  which is the honest replay gain (no test-fold oracle);
                * ``test-fold-oracle`` -- the A1.7 candidate source (recorded successes of
                  the *evaluated* fold).  This is an oracle, must be labelled as such and
                  must never be reported as independent recovery.
              Replay is also gated by the explicit read-only allowlist and the
              empty-schema/side-effect guards added in ``engine.repair_tool_allowed``.
a17_named_replay (sensitivity only, OUT of the documented scope): a17_extract + replay
              also for named errors, to quantify what the generic-only restriction costs.

Evidence tiers are reported separately and never mixed:
* schema-valid: passed the declared/augmented structural schema;
* value-verified: the proposal failed the schema but is an exact recorded successful
  invocation of the same task/function (the evaluator's existing tier);
* exact recorded success: the accepted arguments are an exact recorded success (for the
  replay rows these overlap with schema-valid; this column is value evidence, not a tier).

The output JSON also contains a protocol-verification block that recomputes the full-fold
baseline aggregate and compares it with ``benchmarks/toolguard/metrics.json`` so the
experiment is demonstrably on the same protocol as the headline numbers.

Usage:
    python scripts/experiment_a17_value_recovery.py
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
from tool_evolution.toolguard.engine import (  # noqa: E402
    _resolve_param_name, is_side_effect_tool, repair_failure,
)
from tool_evolution.toolguard.extractors import (  # noqa: E402
    EXPERIMENTAL_EXTRACTORS, ExtractorMap, extract_dates, extract_month_years,
)
from tool_evolution.toolguard.memory import SuccessMemory  # noqa: E402
from tool_evolution.toolguard.records import FunctionSchema, load_calls  # noqa: E402
from tool_evolution.toolguard.rules import classify_error, parse_missing_params  # noqa: E402
from tool_evolution.toolguard.schema_check import validate_arguments  # noqa: E402
from tool_evolution.toolguard.schemas import build_schema_index, refine_schema_index  # noqa: E402

TRAIN_MODELS = {"chatgpt", "toolllama", "gpt-3.5-turbo-1106"}
_SENTINEL_CALLS = {"invalid_hallucination_function_name"}

# expected full-fold baseline aggregate from benchmarks/toolguard/metrics.json
_EXPECTED_BASELINE = {
    "evaluable_failures": 969,
    "proposed_repairs": 583,
    "schema_passed": 519,
    "value_verified_repairs": 38,
    "coerced_repairs": 131,
    "accepted_repairs": 557,
    "skipped_no_schema": 64,
}

_RELATIVE_DATE_RE = re.compile(
    r"\b(today|yesterday|tomorrow|tonight|last\s+(?:week|month|year|night)|"
    r"this\s+(?:week|month|year)|next\s+(?:week|month|year)|"
    r"past\s+\d+\s+(?:days?|weeks?|months?|years?)|"
    r"(?:last|past|previous)\s+\d+\s+(?:days?|weeks?|months?|years?)|"
    r"recently|latest|current|nowadays)\b",
    re.I,
)


def _new_extractor(name: str | None) -> bool:
    return name in EXPERIMENTAL_EXTRACTORS


def _augment_schema(schema: FunctionSchema, named: list[str]) -> tuple[FunctionSchema, list[str]]:
    """A1.6 in-memory error-driven schema completion (properties added without a type)."""
    properties = dict(schema.properties)
    required = list(schema.required)
    resolved: list[str] = []
    for raw in named:
        name = _resolve_param_name(raw, set(properties))
        if name is None:
            properties[raw] = {}
            name = raw
        if name not in required:
            required.append(name)
        resolved.append(name)
    return FunctionSchema(name=schema.name, properties=properties, required=tuple(required)), resolved


def _classify(result) -> dict | None:
    if result is None:
        return None
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
        "rejected_proposals": list(result.rejected_proposals)[:5],
        "value_verified_tier": bool(result.value_verified),
    }


def _value_evidence(result, failed_arguments: dict, reference: list[dict]) -> dict:
    if result is None or not result.repairable or result.arguments is None:
        return {"value_verifiable": False, "value_agreement": None,
                "exact_recorded_success": False}
    changed = {name: value for name, value in result.arguments.items()
               if failed_arguments.get(name) != value}
    verifiable = bool(reference and changed)
    agreement = None
    if verifiable:
        agreement = any(all(ref.get(name) == value for name, value in changed.items())
                        for ref in reference)
    exact = result.arguments in reference if reference else False
    return {
        "value_verifiable": verifiable,
        "value_agreement": agreement,
        "exact_recorded_success": exact,
        "changed_arguments": changed,
    }


def _param_source(name: str, extractor_map: ExtractorMap, query: str) -> dict:
    extractor = extractor_map.extractor_for(name)
    values = extractor_map.candidates(name, query) if query else []
    return {
        "extractor": extractor,
        "extractor_new": _new_extractor(extractor),
        "candidate_count": len(values),
        "first_candidate": values[0] if values else None,
    }


def _simulate_named_replay(call, schema: FunctionSchema, reference: list[dict]) -> dict | None:
    """What the replay rule would accept if its scope were extended to *named* errors.

    Out of the documented generic-only scope, so the engine rule is deliberately not
    extended; this helper mirrors the engine outcome (schema gate first, then the existing
    exact-recorded-success tier) purely to quantify the scope restriction.
    """
    if not reference:
        return None
    for arguments in reference:
        if not isinstance(arguments, dict) or not arguments:
            continue
        errors = validate_arguments(arguments, schema)
        exact_evidence = {"value_verifiable": True, "value_agreement": True,
                          "exact_recorded_success": True, "changed_arguments": dict(arguments)}
        if not errors:
            return {"status": "accepted_schema_valid", "rules": ["generic_success_replay"],
                    "arguments": dict(arguments), "schema_errors": [],
                    "rejected_proposals": [], "value_verified_tier": False,
                    "value_evidence": exact_evidence}
        return {"status": "accepted_value_verified",
                "rules": ["generic_success_replay", "recorded_success_match"],
                "arguments": dict(arguments), "schema_errors": list(errors)[:5],
                "rejected_proposals": [], "value_verified_tier": True,
                "value_evidence": exact_evidence}
    return None


def _pick_samples(cases: list[dict], key, limit: int) -> list[dict]:
    """Deterministic stratified sample: one case per bucket per round, buckets sorted."""
    buckets: dict[tuple, list[dict]] = {}
    for case in cases:
        buckets.setdefault(key(case), []).append(case)
    picked: list[dict] = []
    round_index = 0
    while len(picked) < limit:
        added = False
        for bucket_key in sorted(buckets):
            if round_index < len(buckets[bucket_key]) and len(picked) < limit:
                picked.append(buckets[bucket_key][round_index])
                added = True
        if not added:
            break
        round_index += 1
    return picked


def _date_range_check(query: str) -> dict:
    explicit = extract_dates(query)
    months = extract_month_years(query)
    relative = sorted({match[0].lower() for match in _RELATIVE_DATE_RE.findall(query)})
    return {
        "explicit_dates": explicit,
        "month_years": months,
        "relative_date_phrases": relative,
        "has_value": bool(explicit or months),
    }


def _funnel_from_results(results: list[dict | None]) -> dict:
    counts: Counter[str] = Counter()
    for result in results:
        if result is None:
            continue
        status = result["status"]
        if status.startswith("accepted"):
            counts["accepted"] += 1
            if result["status"] == "accepted_schema_valid":
                counts["schema_valid"] += 1
            else:
                counts["value_verified"] += 1
            if result.get("value_evidence", {}).get("exact_recorded_success"):
                counts["exact_recorded_success"] += 1
        elif status == "proposed_schema_rejected":
            counts["proposed_rejected"] += 1
        else:
            counts["no_proposal"] += 1
    counts["proposed"] = counts["accepted"] + counts["proposed_rejected"]
    counts["value_verifiable"] = sum(1 for result in results
                                     if result and result.get("value_evidence", {}).get("value_verifiable"))
    counts["value_agreement"] = sum(1 for result in results
                                    if result and result.get("value_evidence", {}).get("value_agreement"))
    return dict(counts)


def _sample(case: dict, after: dict | None) -> dict:
    return {
        "fold": case["fold"],
        "tool": case["tool"],
        "error": (case["error"] or "")[:200],
        "query_head": case["query"][:220],
        "named_params": case["named"],
        "resolved_params": case["resolved"],
        "param_evidence": case["param_evidence"],
        "failed_arguments": case["failed_arguments"],
        "reference_successes": case["reference"][:3],
        "after_status": after["status"] if after else None,
        "after_rules": after["rules"] if after else None,
        "after_arguments": after["arguments"] if after else None,
        "schema_errors": after["schema_errors"] if after else None,
        "value_evidence": case.get("a17_value_evidence"),
        "cache_verification": case.get("cache_verification"),
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
    parser.add_argument("--cache", default=".scratch/stb/cache/tool_response_cache",
                        help="released real-response cache (for independent cache verification)")
    parser.add_argument("--no-cache", action="store_true",
                        help="skip cache verification (counts are then not cache-checked)")
    parser.add_argument("--output-dir", default=".scratch/toolguard/a17")
    parser.add_argument("--replay-source", choices=("training", "test-fold-oracle"),
                        default="training",
                        help="training: replay candidates are training-side SuccessMemory "
                             "successes (honest gain); test-fold-oracle: A1.7's recorded "
                             "successes of the evaluated fold (oracle, label separately)")
    parser.add_argument("--replay-allowlist", default="",
                        help="comma-separated replay allowlist (default: auto = every "
                             "observed tool whose name does not look side-effectful; the "
                             "engine still denies side-effect names and empty schemas)")
    parser.add_argument("--max-accepted-samples", type=int, default=20)
    parser.add_argument("--max-failed-samples", type=int, default=10)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    calls = [call for call in load_calls(repo_root / args.calls)
             if call.tool_name not in _SENTINEL_CALLS]
    schemas, schema_stats = build_schema_index(repo_root / args.queries)
    schemas, schema_fidelity = refine_schema_index(schemas, calls)
    families = sorted({call.model for call in calls})
    print(f"calls: {len(calls)} | schemas: {len(schemas)} | families: {families}")

    # A1.8.3 replay allowlist.  The default (auto) is an experiment upper bound: every tool
    # name observed in the dataset that does not look side-effectful.  The engine still
    # denies side-effect names and replay on an empty declared schema regardless.
    if args.replay_allowlist:
        replay_allowlist = frozenset(
            tool.strip() for tool in args.replay_allowlist.split(",") if tool.strip())
        replay_allowlist_source = "explicit"
    else:
        replay_allowlist = frozenset({call.tool_name for call in calls
                                      if not is_side_effect_tool(call.tool_name)})
        replay_allowlist_source = "auto: all observed non-side-effect tool names"
    print(f"replay source: {args.replay_source} | allowlist: {len(replay_allowlist)} "
          f"tool(s) [{replay_allowlist_source}]")

    backend = None
    if not args.no_cache and (repo_root / args.cache).is_dir():
        cache_root = repo_root / args.cache
        backend = RecordedBackend(cache_root, CacheIndex.from_root(cache_root))
        print(f"cache verification: {len(backend.index)} indexed responses")
    else:
        print("cache verification: disabled")

    # ------------------------------------------------------------------
    # 1. full-fold baseline: reproduce the headline aggregate
    # ------------------------------------------------------------------
    protocol_counts: Counter[str] = Counter()
    cases: list[dict] = []
    for family in families:
        train = [call for call in calls if call.model != family and call.model in TRAIN_MODELS]
        test = [call for call in calls if call.model == family]
        memory = SuccessMemory.build(train, schemas)
        extractor_map_baseline = ExtractorMap.learn(train, schemas)
        extractor_map_a17 = ExtractorMap.learn(train, schemas, include_experimental=True)
        successes: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
        for call in test:
            if call.recorded_ok:
                successes[(call.group, call.query_id, call.tool_name)].append(call.arguments)

        for call in test:
            if call.recorded_ok:
                continue
            if "__raw__" in call.arguments or call.tool_name not in schemas:
                protocol_counts["skipped_no_schema"] += 1
                continue
            kind = classify_error(call.error)
            protocol_counts["evaluable_failures"] += 1
            reference = successes.get((call.group, call.query_id, call.tool_name)) or []
            baseline = repair_failure(call, schemas, memory, extractor_map_baseline,
                                      successful_arguments=reference)
            if baseline.proposed:
                protocol_counts["proposed_repairs"] += 1
                protocol_counts["schema_rejected_proposals"] += len(baseline.rejected_proposals)
            if baseline.repairable:
                protocol_counts["accepted_repairs"] += 1
                if baseline.schema_valid:
                    protocol_counts["schema_passed"] += 1
                else:
                    protocol_counts["value_verified_repairs"] += 1
                if "schema_type_coercion" in baseline.rules:
                    protocol_counts["coerced_repairs"] += 1
            if baseline.proposed or kind != "missing_required":
                continue

            # candidate: missing_required failure with no proposal under the baseline
            named = parse_missing_params(call.error or "")
            schema = schemas[call.tool_name]
            if named:
                augmented_schema, resolved = _augment_schema(schema, named)
                schemas_augmented = dict(schemas)
                schemas_augmented[call.tool_name] = augmented_schema
                a16 = _classify(repair_failure(call, schemas_augmented, memory,
                                               extractor_map_baseline,
                                               successful_arguments=reference))
                a17_result = repair_failure(call, schemas_augmented, memory,
                                            extractor_map_a17,
                                            successful_arguments=reference)
                a17 = _classify(a17_result)
                sensitivity = _simulate_named_replay(call, augmented_schema, reference)
                param_schema = augmented_schema
                value_evidence = _value_evidence(a17_result, call.arguments, reference)
            else:
                resolved = []
                a16 = None
                training_reference = memory.by_task_function.get(
                    (call.group, call.query_id, call.tool_name)) or []
                replay_successes = (training_reference if args.replay_source == "training"
                                    else reference)
                a17_result = repair_failure(
                    call, schemas, memory, extractor_map_a17,
                    successful_arguments=reference, allow_recorded_success_replay=True,
                    replay_allowlist=replay_allowlist,
                    replay_successes=replay_successes)
                a17 = _classify(a17_result)
                a17["replay_source"] = args.replay_source
                a17["replay_candidate_count"] = len(replay_successes or [])
                sensitivity = None
                param_schema = schema
                value_evidence = _value_evidence(a17_result, call.arguments, reference)
            a17["value_evidence"] = value_evidence

            case = {
                "fold": family,
                "tool": call.tool_name,
                "group": call.group,
                "query_id": call.query_id,
                "trace_id": call.trace_id,
                "model": call.model,
                "error": call.error,
                "query": call.query,
                "named": named,
                "resolved": resolved,
                "failed_arguments": call.arguments,
                "schema_required_before": list(schema.required),
                "schema_props_before": sorted(schema.properties),
                "schema_props_augmented": (sorted(param_schema.properties) if named else None),
                "reference": reference,
                "reference_count": len(reference),
                "a16": a16,
                "a17": a17,
                "a17_named_replay_sensitivity": sensitivity if named else None,
                "param_evidence": {
                    name: _param_source(name, extractor_map_a17, call.query)
                    for name in (resolved or [])
                },
                "date_range_check": _date_range_check(call.query)
                if any("daterange" in name.lower() for name in resolved) else None,
                "a17_value_evidence": value_evidence,
            }
            if backend is not None and a17["status"].startswith("accepted"):
                payload = backend.call(call.tool_name, a17["arguments"])
                case["cache_verification"] = {
                    "status": payload.status,
                    "error": payload.error,
                }
            cases.append(case)

    named_cases = [case for case in cases if case["named"]]
    generic_cases = [case for case in cases if not case["named"]]
    print(f"subset candidates: {len(cases)} (named={len(named_cases)}, generic={len(generic_cases)})")

    # ------------------------------------------------------------------
    # 2. protocol verification against the committed headline aggregate
    # ------------------------------------------------------------------
    computed = {key: int(protocol_counts.get(key, 0)) for key in _EXPECTED_BASELINE}
    expected_metrics = json.loads((repo_root / args.expected_metrics).read_text(encoding="utf-8"))
    headline = expected_metrics["results"]["lomo_aggregate"]
    expected = {
        "evaluable_failures": headline["evaluable_failures"],
        "proposed_repairs": headline["proposed_repairs"],
        "schema_passed": headline["schema_passed"],
        "value_verified_repairs": headline["value_verified_repairs"],
        "coerced_repairs": headline["coerced_repairs"],
        "accepted_repairs": headline["accepted_repairs"],
        "skipped_no_schema": expected_metrics["results"]["skipped_no_schema"],
    }
    protocol_match = computed == expected
    print(f"protocol verification: {'MATCH' if protocol_match else 'MISMATCH'}")
    print(f"  computed: {json.dumps(computed, ensure_ascii=False)}")
    if not protocol_match:
        print(f"  expected: {json.dumps(expected, ensure_ascii=False)}")

    # ------------------------------------------------------------------
    # 3. before/after funnel tables
    # ------------------------------------------------------------------
    def case_funnel(key: str, subset: list[dict]) -> dict:
        return _funnel_from_results([case[key] if case[key] else None for case in subset])

    named_a16 = case_funnel("a16", named_cases)
    named_a17 = case_funnel("a17", named_cases)
    generic_a17 = case_funnel("a17", generic_cases)
    total_a17 = _funnel_from_results([case["a17"] for case in cases])
    # the simulated named-replay sensitivity has no result for the 90 named cases without
    # a same-task/function recorded success; those cases stay "no proposal" in that column
    named_sensitivity = _funnel_from_results([
        case["a17_named_replay_sensitivity"]
        if case["a17_named_replay_sensitivity"]
        else {"status": "no_proposal", "rules": [], "value_evidence": {}}
        for case in named_cases
    ])

    subset_tables = {
        "named": {
            "total": len(named_cases),
            "baseline_no_proposal": len(named_cases),
            "a16_schema_only": named_a16,
            "a17_extract": named_a17,
            "a17_named_replay_sensitivity": named_sensitivity,
        },
        "generic": {
            "total": len(generic_cases),
            "baseline_no_proposal": len(generic_cases),
            "a17_replay": generic_a17,
        },
        "total": {
            "total": len(cases),
            "baseline_no_proposal": len(cases),
            "a17_after": total_a17,
        },
    }

    # by parameter (named cases; multi-parameter cases contribute to each parameter row)
    by_param: dict[str, dict] = {}
    status_column = {
        "accepted_schema_valid": "accepted_schema_valid",
        "accepted_value_verified": "accepted_value_verified",
        "proposed_schema_rejected": "proposed_rejected",
        "no_proposal": "no_proposal",
    }
    for case in named_cases:
        after = case["a17"]
        for name in case["resolved"]:
            row = by_param.setdefault(name, {
                "total": 0, "accepted_schema_valid": 0, "accepted_value_verified": 0,
                "proposed_rejected": 0, "no_proposal": 0,
                "extractor": case["param_evidence"][name]["extractor"],
                "extractor_new": case["param_evidence"][name]["extractor_new"],
                "candidate_found": 0,
            })
            row["total"] += 1
            row[status_column[after["status"]]] += 1
            if case["param_evidence"][name]["candidate_count"]:
                row["candidate_found"] += 1
    generic_by_param = {
        "unnamed (generic error)": {
            "total": len(generic_cases),
            "accepted_schema_valid": generic_a17.get("schema_valid", 0),
            "accepted_value_verified": generic_a17.get("value_verified", 0),
            "proposed_rejected": generic_a17.get("proposed_rejected", 0),
            "no_proposal": generic_a17.get("no_proposal", 0),
            "exact_recorded_success": generic_a17.get("exact_recorded_success", 0),
        }
    }

    # by rule / attribution
    by_rule: Counter[str] = Counter()
    for case in cases:
        after = case["a17"]
        rules = list(after["rules"] or [])
        if after["status"].startswith("accepted"):
            if "generic_success_replay" in rules:
                by_rule["generic_success_replay (accepted)"] += 1
            elif any(rule.startswith("text_extract:") for rule in rules):
                style = "new_extractor" if any(
                    case["param_evidence"].get(name, {}).get("extractor_new")
                    for name in case["resolved"]) else "core_extractor"
                by_rule[f"text_extract[{style}] (accepted)"] += 1
            elif any(rule.startswith("schema_default") for rule in rules):
                by_rule["schema_default (accepted)"] += 1
            else:
                by_rule[f"other:{'+'.join(rules)} (accepted)"] += 1
        elif after["status"] == "proposed_schema_rejected":
            rejected = " ".join(after.get("rejected_proposals") or [])
            if "generic_success_replay" in rejected:
                by_rule["generic_success_replay (schema rejected)"] += 1
            elif "text_extract" in rejected:
                style = "new_extractor" if any(
                    case["param_evidence"].get(name, {}).get("extractor_new")
                    for name in case["resolved"]) else "core_extractor"
                by_rule[f"text_extract[{style}] (schema rejected; companion parameter has no value)"] += 1
            else:
                by_rule["other (schema rejected)"] += 1
        else:
            evidence = case["param_evidence"]
            if not evidence:
                by_rule["no_proposal: no named parameter"] += 1
            elif any(item["extractor"] is None for item in evidence.values()):
                by_rule["no_proposal: no deterministic extractor for parameter"] += 1
            elif all(item["candidate_count"] == 0 for item in evidence.values()):
                by_rule["no_proposal: extractor found no value in the task text"] += 1
            else:
                by_rule["no_proposal: other"] += 1

    by_tool: dict[str, dict] = {}
    for case in cases:
        row = by_tool.setdefault(case["tool"], {
            "total": 0, "accepted": 0, "proposed_rejected": 0, "no_proposal": 0,
            "named": bool(case["named"]),
        })
        row["total"] += 1
        status = case["a17"]["status"]
        if status.startswith("accepted"):
            row["accepted"] += 1
        elif status == "proposed_schema_rejected":
            row["proposed_rejected"] += 1
        else:
            row["no_proposal"] += 1

    # dateRange detail
    date_range_rows = [case for case in named_cases
                       if any("daterange" in name.lower() for name in case["resolved"])]
    date_range_summary = {
        "cases": len(date_range_rows),
        "explicit_dates_found": sum(1 for case in date_range_rows
                                    if case["date_range_check"]["explicit_dates"]),
        "month_years_found": sum(1 for case in date_range_rows
                                 if case["date_range_check"]["month_years"]),
        "relative_phrases_found": sum(1 for case in date_range_rows
                                      if case["date_range_check"]["relative_date_phrases"]),
        "cases_with_any_value": sum(1 for case in date_range_rows
                                    if case["date_range_check"]["has_value"]),
        "conclusion": ("no explicit date and no usable relative-date anchor in the task "
                       "text -> recorded as 'no value'; nothing fabricated"),
        "examples": [
            {"tool": case["tool"], "query_head": case["query"][:200],
             "check": case["date_range_check"], "after": case["a17"]["status"]}
            for case in date_range_rows[:5]
        ],
    }

    # named cases that do have a same-task/function recorded success (replay would be
    # possible if the generic-only scope were extended; out of the documented scope)
    named_with_reference = [case for case in named_cases if case["reference_count"]]
    named_replay_scope_note = {
        "named_cases_with_recorded_success": len(named_with_reference),
        "tools": sorted({case["tool"] for case in named_with_reference}),
        "sample_reference_values": (named_with_reference[0]["reference"][:2]
                                    if named_with_reference else []),
        "note": ("replay for named errors is OUT of the documented generic-only scope; the "
                 "simulated sensitivity row quantifies it. These recorded successes are "
                 "StableToolBench fake-response placeholders (e.g. ID 'abc123'), so even the "
                 "simulated gain is exact-recorded-success evidence, not semantic validation."),
    }

    # independent cache verification against the released real-response cache
    cache_counts: Counter[str] = Counter()
    cache_by_rule: dict[str, Counter] = defaultdict(Counter)
    for case in cases:
        cache = case.get("cache_verification")
        if cache is None:
            continue
        cache_counts["accepted_checked"] += 1
        cache_counts[cache["status"]] += 1
        rules = case["a17"]["rules"] or []
        if "generic_success_replay" in rules:
            rule_label = "generic_success_replay"
        elif any(rule.startswith("text_extract:") for rule in rules):
            rule_label = ("text_extract[new_extractor]" if any(
                case["param_evidence"].get(name, {}).get("extractor_new")
                for name in case["resolved"]) else "text_extract[core_extractor]")
        else:
            rule_label = "other"
        cache_by_rule[rule_label][cache["status"]] += 1
    cache_verification = {
        "enabled": backend is not None,
        "cache_entries": len(backend.index) if backend is not None else 0,
        "accepted_checked": cache_counts.get("accepted_checked", 0),
        "verified_recovered": cache_counts.get("ok", 0),
        "verified_still_error": cache_counts.get("error", 0),
        "cache_miss": cache_counts.get("miss", 0),
        "by_rule": {label: dict(counts) for label, counts in cache_by_rule.items()},
        "note": ("RecordedBackend against the released ToolBench response cache; a hit whose "
                 "payload has no error means the reference solution's exact call is known to "
                 "have succeeded. A miss is not a failure (the cache only covers "
                 "reference-solution inputs)."),
    }

    # samples for human review (deterministic, stratified so every rule/tool appears)
    accepted_cases = [case for case in cases if case["a17"]["status"].startswith("accepted")]
    failed_cases = [case for case in cases if not case["a17"]["status"].startswith("accepted")]

    # distinct accepted values (collapses the model/fold duplicates for human review)
    distinct_values: dict[tuple, dict] = {}
    for case in accepted_cases:
        arguments = case["a17"]["arguments"] or {}
        key = (case["tool"], tuple(case["resolved"]),
               json.dumps(arguments, sort_keys=True, ensure_ascii=False, default=str))
        row = distinct_values.setdefault(key, {
            "tool": case["tool"],
            "params": case["resolved"] or ["(unnamed)"],
            "arguments": arguments,
            "rules": case["a17"]["rules"],
            "cases": 0,
            "exact_recorded_success": bool(
                case.get("a17_value_evidence", {}).get("exact_recorded_success")),
            "cache_status": Counter(),
        })
        row["cases"] += 1
        if case.get("cache_verification"):
            row["cache_status"][case["cache_verification"]["status"]] += 1
    distinct_accepted_values = [
        {**row, "cache_status": dict(row["cache_status"])}
        for row in sorted(distinct_values.values(), key=lambda item: (-item["cases"], item["tool"]))
    ]

    accepted_samples = [_sample(case, case["a17"]) for case in _pick_samples(
        accepted_cases, lambda case: (case["tool"], case["a17"]["status"]),
        args.max_accepted_samples)]
    failed_samples = [_sample(case, case["a17"]) for case in _pick_samples(
        failed_cases, lambda case: case["a17"]["status"], args.max_failed_samples)]

    # ------------------------------------------------------------------
    # 4. artifacts + markdown report
    # ------------------------------------------------------------------
    result = {
        "experiment": "ToolGuard A1.7 deterministic extractors + generic-error replay",
        "protocol": {
            "protocol": "lomo fixed-three (same as metrics.json results.lomo_aggregate)",
            "train_models": sorted(TRAIN_MODELS),
            "folds": families,
            "calls": len(calls),
            "schemas": schema_stats,
            "schema_fidelity_repair": {
                "type_corrections": schema_fidelity["type_corrections"],
                "params_added": schema_fidelity["params_added"],
            },
            "subset": ("228 missing_required failures with no repair proposal under the "
                       "baseline ladder: 95 named + 133 generic (A1.5 definition)"),
            "configurations": {
                "baseline": "current ladder, default extractor map",
                "a16_schema_only": "in-memory error-driven schema completion (A1.6)",
                "a17_extract": "a16_schema_only + A1.7 source/state/topic/ID extractors",
                "a17_replay": ("a17_extract + same-task/function recorded-success replay for "
                               "generic errors only, schema-gated; A1.8.3 gated by the "
                               "read-only allowlist + side-effect/empty-schema guards"),
                "a17_named_replay_sensitivity": ("sensitivity only, out of the documented "
                                                 "scope: replay also for named errors"),
            },
            "replay": {
                "source": args.replay_source,
                "source_meaning": (
                    "training: training-side SuccessMemory successes only (honest gain)"
                    if args.replay_source == "training" else
                    "test-fold-oracle: recorded successes of the evaluated fold (A1.7 "
                    "oracle; must not be reported as independent recovery)"),
                "allowlist_source": replay_allowlist_source,
                "allowlist_size": len(replay_allowlist),
                "gates": ("allowlist membership + non-side-effect tool name + non-empty "
                          "declared schema (engine.replay_tool_allowed)"),
            },
        },
        "protocol_verification": {
            "computed": computed,
            "expected": expected,
            "match": protocol_match,
            "note": ("computed by re-running the same repair ladder on each held-out fold; "
                     "cache verification is reported separately and never changes these counts"),
        },
        "subset_tables": subset_tables,
        "by_parameter": {**by_param, **generic_by_param},
        "by_rule": dict(sorted(by_rule.items())),
        "by_tool": dict(sorted(by_tool.items(), key=lambda item: (-item[1]["total"], item[0]))),
        "cache_verification": cache_verification,
        "named_replay_scope_note": named_replay_scope_note,
        "date_range_analysis": date_range_summary,
        "distinct_accepted_values": distinct_accepted_values,
        "samples": {"accepted": accepted_samples, "still_failed": failed_samples},
        "cases": cases,
        "limitations": [
            "schema-valid for the named cases is structural only: the error-named parameters "
            "are added to the schema without a declared type (same convention as A1.6); "
            "schema-valid is NOT value-correctness",
            "no proposal in the extractor rows reached the value-verified tier, and the "
            "existing value-agreement check is unverifiable for them because no recorded "
            "success of the same task/function exists (reference_count=0)",
            "generic replay accepted proposals are exact recorded successes, so their value "
            "evidence is empirical for this dataset; some StableToolBench recorded calls use "
            "synthetic placeholder values (e.g. user_id '123456789', ID 'abc123'), so an "
            "accepted replay is not proof of a semantically correct argument",
            "a17_named_replay_sensitivity is out of the documented generic-only scope and "
            "must not be counted in the main result",
            "the extractor rows are policy upper bounds, not recovered call counts: a "
            "schema-valid extraction still has to be sent to the backend to be proven; the "
            "released-cache verification block is the independent check for those calls",
        ],
    }
    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "a17_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")

    report = _render_report(result)
    (output_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nwrote {output_dir / 'a17_results.json'}")
    print(f"wrote {output_dir / 'report.md'}")
    return 0 if protocol_match else 1


def _render_report(result: dict) -> str:
    lines: list[str] = []
    tables = result["subset_tables"]
    replay = result["protocol"]["replay"]
    lines.append("# A1.7/A1.8.3 before/after (228 missing_required no-proposal failures)\n")
    verification = result["protocol_verification"]
    lines.append(f"Protocol verification against metrics.json: "
                 f"**{'MATCH' if verification['match'] else 'MISMATCH'}** "
                 f"`{json.dumps(verification['computed'])}`\n")
    lines.append(f"Replay source: **{replay['source']}** — {replay['source_meaning']} "
                 f"(allowlist: {replay['allowlist_size']} tools, "
                 f"{replay['allowlist_source']}; gates: {replay['gates']}).\n")

    lines.append("## Funnel\n")
    named = tables["named"]
    a16 = named["a16_schema_only"]
    a17 = named["a17_extract"]
    sens = named["a17_named_replay_sensitivity"]
    gen = tables["generic"]["a17_replay"]
    total = tables["total"]["a17_after"]
    replay_label = f"A1.8.3 replay ({replay['source']})"
    lines.append(_md_table(
        ["subset", "config", "proposed", "schema-valid", "value-verified", "exact recorded success",
         "schema rejected", "no proposal"],
        [
            ["named 95", "baseline", 0, 0, 0, 0, 0, 95],
            ["named 95", "A1.6 schema only", a16.get("proposed", 0), a16.get("schema_valid", 0),
             a16.get("value_verified", 0), a16.get("exact_recorded_success", 0),
             a16.get("proposed_rejected", 0), a16.get("no_proposal", 0)],
            ["named 95", "A1.7 extract", a17.get("proposed", 0), a17.get("schema_valid", 0),
             a17.get("value_verified", 0), a17.get("exact_recorded_success", 0),
             a17.get("proposed_rejected", 0), a17.get("no_proposal", 0)],
            ["named 95", "A1.7 + named replay (sensitivity, simulated, out of scope)",
             sens.get("proposed", 0), sens.get("schema_valid", 0), sens.get("value_verified", 0),
             sens.get("exact_recorded_success", 0), sens.get("proposed_rejected", 0),
             sens.get("no_proposal", 0)],
            ["generic 133", "baseline", 0, 0, 0, 0, 0, 133],
            ["generic 133", replay_label, gen.get("proposed", 0), gen.get("schema_valid", 0),
             gen.get("value_verified", 0), gen.get("exact_recorded_success", 0),
             gen.get("proposed_rejected", 0), gen.get("no_proposal", 0)],
            ["total 228", "baseline", 0, 0, 0, 0, 0, 228],
            ["total 228", "A1.7/A1.8.3 after", total.get("proposed", 0), total.get("schema_valid", 0),
             total.get("value_verified", 0), total.get("exact_recorded_success", 0),
             total.get("proposed_rejected", 0), total.get("no_proposal", 0)],
        ]))
    lines.append("\n- `schema-valid` = accepted through the structural schema gate; it is "
                 "**not** value correctness.\n"
                 "- `value-verified` = the existing evaluator tier: a schema-failing proposal "
                 "that is an exact recorded successful invocation of the same task/function.\n")
    if replay["source"] == "training":
        lines.append(
            f"- **training-only replay gain on the 133 generic cases: "
            f"{gen.get('accepted', 0)} accepted** "
            f"({gen.get('schema_valid', 0)} schema-valid + {gen.get('value_verified', 0)} "
            f"value-verified; {gen.get('no_proposal', 0)} still no proposal). This is the "
            "honest gain: the baseline ladder already proposes every training-side success, so "
            "there is no training-side recorded success left for the generic cases. The A1.7 "
            "13 accepted replay cases came from the test-fold oracle and are not counted "
            "here.\n")
    else:
        lines.append(
            f"- **test-fold oracle replay (NOT independent): {gen.get('accepted', 0)} accepted** "
            f"({gen.get('schema_valid', 0)} schema-valid + {gen.get('value_verified', 0)} "
            f"value-verified; {gen.get('no_proposal', 0)} still no proposal). The A1.8.3 safety "
            "gates refuse replay on an empty declared schema, so the historical 13 unsafe "
            "admissions (7 schema-valid + 6 empty-schema `user_id='123456789'` placeholders) "
            "shrink to the schema-valid subset. This row is an oracle upper bound, never "
            "independent recovery.\n")
    scope_note = result["named_replay_scope_note"]
    lines.append(
        f"- the simulated named-replay sensitivity row is **out of the documented generic-only "
        f"scope**: only {scope_note['named_cases_with_recorded_success']} named cases even have a "
        f"same-task/function recorded success ({', '.join(scope_note['tools']) or '-'}); their "
        f"recorded values are StableToolBench fake-response placeholders "
        f"`{json.dumps(scope_note['sample_reference_values'], ensure_ascii=False)}`, so the "
        f"simulated gain is exact-recorded-success evidence, not semantic validation.\n")

    lines.append("## By parameter\n")
    rows = []
    for name, row in result["by_parameter"].items():
        rows.append([name, row.get("total", 0), row.get("extractor") or "-",
                     row.get("extractor_new", "-"),
                     row.get("candidate_found", "-"),
                     row.get("accepted_schema_valid", 0), row.get("accepted_value_verified", 0),
                     row.get("proposed_rejected", 0), row.get("no_proposal", 0),
                     row.get("exact_recorded_success", "-")])
    lines.append(_md_table(
        ["parameter", "cases", "extractor", "new extractor?", "value found",
         "schema-valid", "value-verified", "schema rejected", "no proposal",
         "exact recorded success"], rows))

    cache = result["cache_verification"]
    lines.append("\n## Independent cache verification (released real-response cache)\n")
    if cache["enabled"]:
        lines.append(_md_table(
            ["rule", "checked", "cache ok (recovered)", "cache error", "cache miss"],
            [[rule, counts.get("ok", 0) + counts.get("error", 0) + counts.get("miss", 0),
              counts.get("ok", 0), counts.get("error", 0), counts.get("miss", 0)]
             for rule, counts in cache["by_rule"].items()]))
        lines.append(f"\nAccepted proposals checked: {cache['accepted_checked']} / "
                     f"cache ok {cache['verified_recovered']} / "
                     f"cache error {cache['verified_still_error']} / "
                     f"cache miss {cache['cache_miss']} (cache entries: {cache['cache_entries']}).\n")
        lines.append("A cache hit whose payload has no error would be independent evidence that "
                     "this exact call worked; the released cache only covers reference-solution "
                     "inputs, so a miss is not a failure, it just leaves the repair unverified.\n")
    else:
        lines.append("disabled (`--no-cache`)\n")

    lines.append("\n## By rule / attribution\n")
    lines.append(_md_table(["rule", "count"],
                           [[rule, count] for rule, count in result["by_rule"].items()]))

    lines.append("\n## By tool\n")
    lines.append(_md_table(
        ["tool", "named?", "cases", "accepted", "schema rejected", "no proposal"],
        [[tool, row["named"], row["total"], row["accepted"], row["proposed_rejected"],
          row["no_proposal"]] for tool, row in result["by_tool"].items()]))

    dr = result["date_range_analysis"]
    lines.append("\n## dateRange check\n")
    lines.append(json.dumps(dr, ensure_ascii=False, indent=1))

    lines.append("\n## Distinct accepted values (human review: duplicates collapsed)\n")
    lines.append(_md_table(
        ["tool", "parameter", "value", "cases", "rule", "exact recorded success?",
         "cache status"],
        [[row["tool"], ", ".join(row["params"]),
          json.dumps(row["arguments"], ensure_ascii=False),
          row["cases"], ", ".join(row["rules"] or []),
          row["exact_recorded_success"], json.dumps(row["cache_status"], ensure_ascii=False)]
         for row in result["distinct_accepted_values"]]))

    lines.append(f"\n## Still-failed sample ({len(result['samples']['still_failed'])} of "
                 f"{total.get('no_proposal', 0)})\n")
    lines.append(_md_table(
        ["tool", "parameter(s)", "status", "reason"],
        [[s["tool"], ", ".join(s["named_params"]) or "(unnamed)", s["after_status"],
          (s["error"] or "")[:90]] for s in result["samples"]["still_failed"]]))
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
