"""B5 experiment: schema-fidelity extension audit (training-side evidence only).

Audits every ``(function, parameter)`` pair for schema-index gaps that recorded successful
calls can prove, using the same conservative evidence bar as the committed schema-fidelity
repair (>=3 recorded successful values, >=90% agreement on one concrete scalar type):

1. **missing parameters**: observed in training-side successful calls but not declared in
   the schema index (the committed back-fill only handles functions with an *empty*
   parameter list, so partially declared functions can still be missing parameters);
2. **type conflicts**: the declared scalar type is incompatible with the dominant observed
   type (number/integer declared vs observed string, or string declared vs observed
   integer/number; integer<->number widening is never performed);
3. **enum candidates** (report only): string parameter with <=10 unique values and top-1
   frequency >=90%;
4. **required candidates** (report only): parameter present in >=90% of the function's
   successful calls but declared optional.

Evidence comes from the *training side only*: for every model family, train successes are
the fixed-three training successes excluding that family; the held-out family's successes
are never used to declare anything.  The augmented schema is built in memory only, then
the repair ladder is re-run on the 969 evaluable failures to measure proposed /
schema-valid / value-verified / no-proposal and the suspected false positives
(country/currency phrases).  No committed schema/artifact is changed.

Usage:
    python scripts/experiment_b5_schema_fidelity.py
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
from tool_evolution.toolguard.engine import repair_failure  # noqa: E402
from tool_evolution.toolguard.extractors import ExtractorMap  # noqa: E402
from tool_evolution.toolguard.memory import SuccessMemory  # noqa: E402
from tool_evolution.toolguard.records import FunctionSchema, load_calls  # noqa: E402
from tool_evolution.toolguard.rules import classify_error  # noqa: E402
from tool_evolution.toolguard.schemas import build_schema_index, refine_schema_index  # noqa: E402

TRAIN_MODELS = {"chatgpt", "toolllama", "gpt-3.5-turbo-1106"}
_SENTINEL_CALLS = {"invalid_hallucination_function_name"}
_SCALAR_TYPES = ("string", "integer", "number", "boolean")
_MIN_SAMPLES = 3
_AGREEMENT = 0.9
_ENUM_UNIQUE_MAX = 10
_REQUIRED_PRESENCE = 0.9
_CURRENCY_RE = re.compile(r"\bus\s+dollars?\b|\busd?\s+dollars?\b|\busd\b", re.I)

_EXPECTED_BASELINE = {
    "evaluable_failures": 969,
    "proposed_repairs": 583,
    "schema_passed": 519,
    "value_verified_repairs": 38,
    "coerced_repairs": 131,
    "accepted_repairs": 557,
}


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


def _incompatible(declared: str, observed: str) -> bool:
    if declared == observed:
        return False
    if declared in ("number", "integer") and observed == "string":
        return True
    if declared == "string" and observed in ("integer", "number"):
        return True
    return False


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
    parser.add_argument("--output-dir", default=".scratch/toolguard/b5")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    calls = [call for call in load_calls(repo_root / args.calls)
             if call.tool_name not in _SENTINEL_CALLS]
    base_schemas, schema_stats = build_schema_index(repo_root / args.queries)
    base_schemas, schema_fidelity = refine_schema_index(base_schemas, calls)
    cache_root = repo_root / args.cache
    backend = RecordedBackend(cache_root, CacheIndex.from_root(cache_root))
    families = sorted({call.model for call in calls})
    print(f"calls: {len(calls)} | schemas: {len(base_schemas)} | "
          f"cache entries: {len(backend.index)} | families: {families}")

    candidates: dict[tuple, dict] = {}
    baseline_protocol: Counter[str] = Counter()
    augmented_protocol: Counter[str] = Counter()
    accepted_rows: list[dict] = []
    augmented_deltas: dict[str, dict] = {}
    baseline_accepted_keys: set[tuple] = set()
    augmented_accepted_keys: set[tuple] = set()

    for family in families:
        train = [call for call in calls
                 if call.model != family and call.model in TRAIN_MODELS]
        test = [call for call in calls if call.model == family]
        memory = SuccessMemory.build(train, base_schemas)
        extractor_map = ExtractorMap.learn(train, base_schemas)
        test_successes: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
        for call in test:
            if call.recorded_ok:
                test_successes[(call.group, call.query_id, call.tool_name)].append(
                    dict(call.arguments))

        # ---- training-side evidence -------------------------------------
        successful_by_function: dict[str, list[dict]] = defaultdict(list)
        for call in train:
            if call.recorded_ok and call.valid_data:
                successful_by_function[call.tool_name].append(call.arguments)
        for function in sorted(successful_by_function):
            schema = base_schemas.get(function)
            if schema is None:
                continue
            successes = successful_by_function[function]
            if not successes:
                continue
            presence: Counter[str] = Counter()
            types: dict[str, Counter] = defaultdict(Counter)
            for arguments in successes:
                for name, value in arguments.items():
                    kind = _json_type(value)
                    if kind is None:
                        continue
                    presence[name] += 1
                    types[name][kind] += 1
            for name, count in sorted(presence.items()):
                total = len(successes)
                declared_prop = schema.properties.get(name)
                if declared_prop is None:
                    bucket = types[name]
                    dominant, dominant_count = bucket.most_common(1)[0]
                    agreement = dominant_count / count
                    if count >= _MIN_SAMPLES and agreement >= _AGREEMENT:
                        key = ("missing_param", function, name, family)
                        candidates[key] = {
                            "kind": "missing_param",
                            "function": function,
                            "param": name,
                            "fold": family,
                            "samples": count,
                            "agreement": round(agreement, 4),
                            "dominant_type": dominant,
                            "applied": True,
                        }
                    else:
                        key = ("missing_param_insufficient", function, name, family)
                        candidates[key] = {
                            "kind": "missing_param_insufficient",
                            "function": function,
                            "param": name,
                            "fold": family,
                            "samples": count,
                            "agreement": round(agreement, 4),
                            "dominant_type": dominant,
                            "applied": False,
                        }
                    continue
                declared = str(declared_prop.get("type") or "").lower()
                if declared in _SCALAR_TYPES:
                    bucket = types[name]
                    dominant, dominant_count = bucket.most_common(1)[0]
                    agreement = dominant_count / count
                    if count >= _MIN_SAMPLES and agreement >= _AGREEMENT \
                            and _incompatible(declared, dominant):
                        key = ("type_conflict", function, name, family)
                        candidates[key] = {
                            "kind": "type_conflict",
                            "function": function,
                            "param": name,
                            "fold": family,
                            "samples": count,
                            "agreement": round(agreement, 4),
                            "declared_type": declared,
                            "dominant_type": dominant,
                            "applied": True,
                        }
                # enum candidate (report only)
                if declared == "string" and count >= _MIN_SAMPLES:
                    value_counter = Counter(str(arguments[name])
                                            for arguments in successes
                                            if name in arguments
                                            and _json_type(arguments[name]) == "string")
                    if 0 < len(value_counter) <= _ENUM_UNIQUE_MAX:
                        top_share = value_counter.most_common(1)[0][1] / count
                        if top_share >= _AGREEMENT:
                            key = ("enum_candidate", function, name, family)
                            candidates[key] = {
                                "kind": "enum_candidate",
                                "function": function,
                                "param": name,
                                "fold": family,
                                "samples": count,
                                "agreement": round(top_share, 4),
                                "values": [value for value, _ in value_counter.most_common()],
                                "applied": False,
                            }
                # required candidate (report only)
                if count / total >= _REQUIRED_PRESENCE and name not in schema.required:
                    key = ("required_candidate", function, name, family)
                    candidates[key] = {
                        "kind": "required_candidate",
                        "function": function,
                        "param": name,
                        "fold": family,
                        "samples": count,
                        "presence_ratio": round(count / total, 4),
                        "applied": False,
                    }

        # ---- build the in-memory augmented schema for this fold ----------
        augmented = dict(base_schemas)
        additions: dict[str, Counter] = defaultdict(Counter)
        corrections: dict[str, dict] = defaultdict(dict)
        for candidate in candidates.values():
            if candidate["fold"] != family or not candidate["applied"]:
                continue
            function, param = candidate["function"], candidate["param"]
            schema = augmented.get(function)
            if schema is None:
                continue
            properties = {name: dict(prop) for name, prop in schema.properties.items()}
            if candidate["kind"] == "missing_param":
                properties[param] = {
                    "type": candidate["dominant_type"],
                    "description": (
                        "inferred from training-side recorded successful calls; "
                        f"{candidate['samples']} samples, "
                        f"{candidate['agreement']:.0%} scalar-type agreement"),
                }
                additions[function][param + ":" + candidate["dominant_type"]] += 1
            elif candidate["kind"] == "type_conflict":
                properties[param]["type"] = candidate["dominant_type"]
                properties[param]["type_corrected_from"] = candidate["declared_type"]
                corrections[function][param] = {
                    "declared": candidate["declared_type"],
                    "observed": candidate["dominant_type"],
                    "evidence": f"{candidate['samples']} samples, "
                                f"{candidate['agreement']:.0%} agreement",
                }
            augmented[function] = FunctionSchema(
                name=function, properties=properties, required=schema.required)
        augmented_deltas[family] = {
            "params_added": {function: dict(counter)
                             for function, counter in additions.items()},
            "type_corrections": {function: dict(items)
                                 for function, items in corrections.items()},
        }

        # ---- rerun baseline and augmented on this fold's test failures ---
        for call in test:
            if call.recorded_ok:
                continue
            if "__raw__" in call.arguments or call.tool_name not in base_schemas:
                continue
            baseline_protocol["evaluable_failures"] += 1
            case_key = (family, call.trace_id, call.step, call.tool_name,
                        call.query_id, call.model)
            reference = test_successes.get(
                (call.group, call.query_id, call.tool_name)) or []
            kind = classify_error(call.error)
            baseline = repair_failure(call, base_schemas, memory, extractor_map,
                                      successful_arguments=reference)
            if baseline.proposed:
                baseline_protocol["proposed_repairs"] += 1
            if baseline.repairable:
                baseline_accepted_keys.add(case_key)
                baseline_protocol["accepted_repairs"] += 1
                if baseline.schema_valid:
                    baseline_protocol["schema_passed"] += 1
                else:
                    baseline_protocol["value_verified_repairs"] += 1
                if "schema_type_coercion" in baseline.rules:
                    baseline_protocol["coerced_repairs"] += 1
            augmented_result = repair_failure(call, augmented, memory, extractor_map,
                                              successful_arguments=reference)
            augmented_protocol["evaluable_failures"] += 1
            if augmented_result.proposed:
                augmented_protocol["proposed_repairs"] += 1
            if not augmented_result.repairable:
                continue
            accepted = augmented_result
            augmented_accepted_keys.add(case_key)
            augmented_protocol["accepted_repairs"] += 1
            status = "accepted_schema_valid" if accepted.schema_valid \
                else "accepted_value_verified"
            augmented_protocol["schema_passed" if accepted.schema_valid
                               else "value_verified_repairs"] += 1
            if "schema_type_coercion" in accepted.rules:
                augmented_protocol["coerced_repairs"] += 1
            function = accepted.function_name or call.tool_name
            arguments = accepted.arguments or {}
            payload = backend.call(function, arguments)
            changed = {name: value for name, value in arguments.items()
                       if call.arguments.get(name) != value}
            currency_false_positive = any(
                "country" in name.lower() and isinstance(value, str)
                and _CURRENCY_RE.search(call.query or "")
                for name, value in changed.items())
            accepted_rows.append({
                "fold": family,
                "model": call.model,
                "tool": call.tool_name,
                "kind": kind,
                "status": status,
                "rules": list(accepted.rules),
                "changed": changed,
                "cache_status": payload.status,
                "currency_false_positive": currency_false_positive,
                "training_memory_exact": any(
                    arguments == recorded
                    for recorded in memory.by_task_function.get(
                        (call.group, call.query_id, call.tool_name)) or []),
                "test_fold_match": any(arguments == recorded for recorded in reference),
            })

    computed = {key: int(baseline_protocol.get(key, 0)) for key in _EXPECTED_BASELINE}
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
    print(f"baseline protocol verification: {'MATCH' if protocol_match else 'MISMATCH'} "
          f"{computed}")

    # ---- candidate classification: keep one row per (function,param,kind) ----------
    folded: dict[tuple, dict] = {}
    for key, candidate in candidates.items():
        dedupe_key = (candidate["kind"], candidate["function"], candidate["param"])
        row = folded.setdefault(dedupe_key, {
            "kind": candidate["kind"],
            "function": candidate["function"],
            "param": candidate["param"],
            "folds": [],
            "applied": candidate["applied"],
        })
        row["folds"].append(candidate)
    candidate_rows = sorted(folded.values(), key=lambda row: (row["kind"], row["function"],
                                                              row["param"]))
    summary = Counter()
    for row in candidate_rows:
        summary["candidate_pairs"] += 1
        summary["candidate_" + row["kind"]] += 1
        summary["applied_pairs"] += int(row["applied"])
        summary["applied_" + row["kind"]] += int(row["applied"])
    summary["folds"] = len(families)

    augmented_counts = {key: int(augmented_protocol.get(key, 0))
                        for key in ("proposed_repairs", "schema_passed",
                                    "value_verified_repairs", "accepted_repairs",
                                    "coerced_repairs", "evaluable_failures")}
    currency_fps = [row for row in accepted_rows if row["currency_false_positive"]]
    cache_counts = Counter(row["cache_status"] for row in accepted_rows)
    falsy = {
        "accepted_but_no_cache_ok_no_value_verified": sum(
            1 for row in accepted_rows if row["cache_status"] != "ok"
            and not row["test_fold_match"] and not row["training_memory_exact"]),
        "currency_false_positives": len(currency_fps),
        "currency_false_positive_distinct_tools": len({row["tool"] for row in currency_fps}),
    }

    result = {
        "experiment": "ToolGuard B5 schema-fidelity extension audit",
        "protocol": {
            "scope": "diagnosis only; in-memory augmented schema; no committed change",
            "evidence": "training-side successful calls only (fixed-three minus the fold)",
            "evidence_bar": {
                "min_samples": _MIN_SAMPLES,
                "agreement": _AGREEMENT,
                "enum_unique_max": _ENUM_UNIQUE_MAX,
                "required_presence": _REQUIRED_PRESENCE,
                "type_rule": ("incompatible scalar types only; integer<->number widening "
                              "never performed; required lists never changed"),
            },
            "folds": families,
            "calls": len(calls),
            "schemas": schema_stats,
            "schema_fidelity_repair": {
                "type_corrections": schema_fidelity["type_corrections"],
                "params_added": schema_fidelity["params_added"],
            },
            "applied_policy": ("missing_param and type_conflict candidates are applied in "
                               "memory; enum_candidate and required_candidate are report-only"),
        },
        "headline_reproduction": {
            "computed": computed,
            "expected": expected,
            "match": protocol_match,
        },
        "summary": dict(summary),
        "candidates": candidate_rows,
        "augmented_schema_delta": augmented_deltas,
        "measurement": {
            "baseline": computed,
            "augmented": augmented_counts,
            "accepted_rows": accepted_rows,
            "cache_among_augmented_accepted": dict(cache_counts),
            "false_positive_audit": falsy,
            "currency_false_positive_examples": currency_fps[:20],
            "accepted_set_comparison": {
                "baseline_accepted": len(baseline_accepted_keys),
                "augmented_accepted": len(augmented_accepted_keys),
                "identical": baseline_accepted_keys == augmented_accepted_keys,
                "added_by_augmentation": len(augmented_accepted_keys - baseline_accepted_keys),
                "lost_by_augmentation": len(baseline_accepted_keys - augmented_accepted_keys),
            },
        },
        "limitations": [
            "the committed schema index was already refined using *all* calls, so the "
            "training-side audit can only add corrections the committed index does not "
            "already contain; a stale/extra correction would not be removed here",
            "enum and required candidates are report-only by design and are not applied",
            "schema-valid is structural only; cache_ok/cache_miss and the test-fold match "
            "are the same diagnostic evidence tiers as the headline evaluator",
            "a param added from training successes can unlock an extractor value that is "
            "structurally valid but semantically wrong; the country/currency audit counts "
            "the known 'US Dollars' pattern and nothing else",
        ],
    }
    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "schema_fidelity.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    report = _render_report(result)
    (output_dir / "schema_fidelity.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nwrote {output_dir / 'schema_fidelity.json'}")
    print(f"wrote {output_dir / 'schema_fidelity.md'}")
    return 0 if protocol_match else 1


def _render_report(result: dict) -> str:
    lines = ["# B5 schema-fidelity extension audit (training-side evidence only)\n"]
    protocol = result["protocol"]
    headline = result["headline_reproduction"]
    lines.append(f"Baseline protocol reproduction: "
                 f"**{'MATCH' if headline['match'] else 'MISMATCH'}** "
                 f"`{json.dumps(headline['computed'], ensure_ascii=False)}`\n")
    lines.append(f"Evidence: {protocol['evidence']}; bar: "
                 f"{json.dumps(protocol['evidence_bar'], ensure_ascii=False)}; folds: "
                 f"{protocol['folds']}.\n")
    summary = result["summary"]
    lines.append("## Candidate summary (one row per function/parameter/kind, all folds)\n")
    lines.append(_md_table(
        ["candidate kind", "pairs", "applied in memory?", "note"],
        [["missing_param", summary.get("candidate_missing_param", 0),
          summary.get("applied_missing_param", 0),
          "observed but undeclared parameter with >=3 samples / >=90% scalar type"],
         ["missing_param_insufficient", summary.get("candidate_missing_param_insufficient", 0),
          0, "observed but below the evidence bar -- not declared"],
         ["type_conflict", summary.get("candidate_type_conflict", 0),
          summary.get("applied_type_conflict", 0),
          "declared type incompatible with dominant observed type"],
         ["enum_candidate", summary.get("candidate_enum_candidate", 0), 0,
          "report only (<=10 unique, top-1 >=90%)"],
         ["required_candidate", summary.get("candidate_required_candidate", 0), 0,
          "report only (present in >=90% of successes, declared optional)"]]))
    lines.append(f"\ntotal candidate function/parameter/kind pairs: "
                 f"{summary.get('candidate_pairs', 0)}; applied pairs: "
                 f"{summary.get('applied_pairs', 0)}\n")
    measurement = result["measurement"]
    lines.append("## Impact on the 969 evaluable failures (in-memory schema, all folds)\n")
    baseline = measurement["baseline"]
    augmented = measurement["augmented"]
    lines.append(_md_table(
        ["metric", "baseline (committed schema)", "augmented (in-memory)",
         "delta"],
        [["proposed repairs", baseline["proposed_repairs"], augmented["proposed_repairs"],
          augmented["proposed_repairs"] - baseline["proposed_repairs"]],
         ["schema-valid", baseline["schema_passed"], augmented["schema_passed"],
          augmented["schema_passed"] - baseline["schema_passed"]],
         ["value-verified", baseline["value_verified_repairs"],
          augmented["value_verified_repairs"],
          augmented["value_verified_repairs"] - baseline["value_verified_repairs"]],
         ["accepted", baseline["accepted_repairs"], augmented["accepted_repairs"],
          augmented["accepted_repairs"] - baseline["accepted_repairs"]],
         ["coerced", baseline["coerced_repairs"], augmented["coerced_repairs"],
          augmented["coerced_repairs"] - baseline["coerced_repairs"]]]))
    cache = measurement["cache_among_augmented_accepted"]
    falsy = measurement["false_positive_audit"]
    accepted_sets = measurement["accepted_set_comparison"]
    lines.append(f"\naccepted set: baseline {accepted_sets['baseline_accepted']}, "
                 f"augmented {accepted_sets['augmented_accepted']}, identical="
                 f"{accepted_sets['identical']} (added {accepted_sets['added_by_augmentation']}, "
                 f"lost {accepted_sets['lost_by_augmentation']}); the delta is a tier "
                 "reclassification, not new coverage.\n")
    lines.append(f"\nAmong the augmented accepted repairs: cache ok "
                 f"{cache.get('ok', 0)}, cache error {cache.get('error', 0)}, cache miss "
                 f"{cache.get('miss', 0)} (diagnostic only, not a production success rate).\n")
    lines.append(f"\nFalse-positive audit: currency-pattern candidates "
                 f"**{falsy['currency_false_positives']}** "
                 f"(distinct tools {falsy['currency_false_positive_distinct_tools']}); "
                 f"accepted repairs with no cache-ok / no test-fold match / no training-side "
                 f"exact match: {falsy['accepted_but_no_cache_ok_no_value_verified']} "
                 f"of {augmented['accepted_repairs']}.\n")
    lines.append("\n## Applied candidates (in-memory)\n")
    applied = [row for row in result["candidates"] if row["applied"]]
    lines.append(_md_table(
        ["kind", "function", "param", "folds", "evidence"],
        [[row["kind"], row["function"], row["param"],
          len(row["folds"]),
          "; ".join(
              f"{fold['fold']}: {fold['samples']} samples, "
              f"{fold['agreement']:.0%} agreement"
              + (f" ({fold['declared_type']}->{fold['dominant_type']})"
                 if row["kind"] == "type_conflict" else "")
              for fold in row["folds"])]
         for row in applied]))
    report_only = [row for row in result["candidates"] if not row["applied"]]
    lines.append("\n## Report-only candidates (first 60 of "
                 f"{len(report_only)}; the full list is in the JSON artifact)\n")
    lines.append(_md_table(
        ["kind", "function", "param", "folds", "evidence"],
        [[row["kind"], row["function"], row["param"], len(row["folds"]),
          "; ".join(
              (f"{fold['fold']}: {fold['samples']} samples, "
               f"{fold.get('agreement', fold.get('presence_ratio', 0)):.0%}")
              for fold in row["folds"])]
         for row in report_only[:60]]))
    lines.append("\n## Limitations\n")
    for limitation in result["limitations"]:
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
