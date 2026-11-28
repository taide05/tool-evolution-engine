"""A1.8.2 experiment: value-verification closure for every accepted repair.

For every repair accepted by the default (headline) fixed-three LOMO configuration this
script records its **evidence type**, keeping independent evidence strictly separate from
the diagnostic test-fold match:

* ``training_memory_exact`` -- the accepted arguments are a verbatim argument set of a
  *training-side* successful call of the same task/function (independent evidence);
* ``cache_ok``              -- the released real-response cache has this exact call and its
  payload carries no error (independent evidence);
* ``cache_error``           -- cache hit whose payload carries an error;
* ``cache_miss``            -- no cache entry (a miss is *not* a failure);
* ``test_fold_match``       -- the accepted arguments are a verbatim argument set of a
  *test-fold* successful call of the same task/function.  This is the tier the headline
  counts as ``value_verified``; it is a *diagnostic* (the reference is the evaluated fold
  itself) and is never counted as independent verification;
* ``none``                  -- no evidence found.

``independent_verified = training_memory_exact or cache_ok``.

The A1.7 experiment's 77 accepted repairs (23 country false positives + 13 test-fold
replay oracle + 41 new-extractor structural) are reported on a separate table, with the
reported 0/77 cache hit rate recomputed.

Read-only: no committed schema/artifact is changed and the headline aggregate must match
``benchmarks/toolguard/metrics.json`` (969 / 583 / 519 / 38 / 131 / 557).

Usage:
    python scripts/experiment_a18_value_verification.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tool_evolution.toolguard.backend import RecordedBackend  # noqa: E402
from tool_evolution.toolguard.cache_index import CacheIndex  # noqa: E402
from tool_evolution.toolguard.engine import repair_failure  # noqa: E402
from tool_evolution.toolguard.extractors import ExtractorMap  # noqa: E402
from tool_evolution.toolguard.memory import SuccessMemory  # noqa: E402
from tool_evolution.toolguard.records import load_calls  # noqa: E402
from tool_evolution.toolguard.rules import classify_error  # noqa: E402
from tool_evolution.toolguard.schemas import build_schema_index, refine_schema_index  # noqa: E402

TRAIN_MODELS = {"chatgpt", "toolllama", "gpt-3.5-turbo-1106"}
_SENTINEL_CALLS = {"invalid_hallucination_function_name"}

_EXPECTED_BASELINE = {
    "evaluable_failures": 969,
    "proposed_repairs": 583,
    "schema_passed": 519,
    "value_verified_repairs": 38,
    "coerced_repairs": 131,
    "accepted_repairs": 557,
}
_EXPECTED_CACHE = {
    "cache_verified_recovered": 48,
    "cache_verified_still_error": 0,
    "cache_miss": 509,
}
_EVIDENCE_ORDER = ("training_memory_exact", "cache_ok", "test_fold_match", "none")


def _category(rules: list[str]) -> str:
    if not rules:
        return "unknown"
    if "recorded_success_match" in rules:
        rules = [rule for rule in rules if rule != "recorded_success_match"]
    return rules[0].split(":")[0] if rules else "unknown"


def _evidence_tier(row: dict) -> str:
    if row["training_memory_exact"]:
        return "training_memory_exact"
    if row["cache_status"] == "ok":
        return "cache_ok"
    if row["test_fold_match"]:
        return "test_fold_match"
    return "none"


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
    parser.add_argument("--a17-results",
                        default=".scratch/toolguard/a17/a17_results_unsafe_oracle_ff0d5ce.json",
                        help="A1.7 unsafe-oracle run (ff0d5ce behavior, 77 accepted); kept "
                             "as a stable input because the A1.7 script now defaults to the "
                             "training-only replay source")
    parser.add_argument("--output-dir", default=".scratch/toolguard/a18")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    calls = [call for call in load_calls(repo_root / args.calls)
             if call.tool_name not in _SENTINEL_CALLS]
    schemas, schema_stats = build_schema_index(repo_root / args.queries)
    schemas, schema_fidelity = refine_schema_index(schemas, calls)
    families = sorted({call.model for call in calls})
    cache_root = repo_root / args.cache
    backend = RecordedBackend(cache_root, CacheIndex.from_root(cache_root))
    print(f"calls: {len(calls)} | schemas: {len(schemas)} | "
          f"cache: {len(backend.index)} indexed responses")

    # ------------------------------------------------------------------
    # 1. baseline LOMO: collect every accepted repair with its evidence
    # ------------------------------------------------------------------
    protocol = Counter()
    rows: list[dict] = []
    for family in families:
        train = [call for call in calls
                 if call.model != family and call.model in TRAIN_MODELS]
        test = [call for call in calls if call.model == family]
        memory = SuccessMemory.build(train, schemas)
        extractor_map = ExtractorMap.learn(train, schemas)
        train_reference = memory.by_task_function
        test_reference: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
        for call in test:
            if call.recorded_ok:
                test_reference[(call.group, call.query_id, call.tool_name)].append(
                    dict(call.arguments))

        for call in test:
            if call.recorded_ok:
                continue
            if "__raw__" in call.arguments or call.tool_name not in schemas:
                protocol["skipped_no_schema"] += 1
                continue
            kind = classify_error(call.error)
            protocol["evaluable_failures"] += 1
            key = (call.group, call.query_id, call.tool_name)
            test_ref = test_reference.get(key) or []
            result = repair_failure(call, schemas, memory, extractor_map,
                                    successful_arguments=test_ref)
            if result.proposed:
                protocol["proposed_repairs"] += 1
            if not result.repairable:
                continue
            protocol["accepted_repairs"] += 1
            if result.schema_valid:
                protocol["schema_passed"] += 1
            else:
                protocol["value_verified_repairs"] += 1
            if "schema_type_coercion" in result.rules:
                protocol["coerced_repairs"] += 1

            arguments = result.arguments if result.arguments is not None else call.arguments
            function = result.function_name or call.tool_name
            train_ref = train_reference.get((call.group, call.query_id, call.tool_name)) or []
            training_memory_exact = any(
                ref == arguments for ref in train_ref if isinstance(ref, dict))
            test_fold_match = any(
                ref == arguments for ref in test_ref if isinstance(ref, dict))
            payload = backend.call(function, arguments)
            row = {
                "fold": family,
                "tool": call.tool_name,
                "function": function,
                "kind": kind,
                "tier": "schema_valid" if result.schema_valid else "value_verified",
                "rules": list(result.rules),
                "category": _category(list(result.rules)),
                "arguments": arguments,
                "changed_params": sorted(
                    name for name, value in arguments.items()
                    if call.arguments.get(name) != value),
                "training_memory_exact": training_memory_exact,
                "test_fold_match": test_fold_match,
                "test_fold_reference_count": len(test_ref),
                "training_memory_reference_count": len(train_ref),
                "cache_status": payload.status,
                "cache_error": payload.error,
            }
            row["evidence_tier"] = _evidence_tier(row)
            row["independent_verified"] = bool(
                row["training_memory_exact"] or row["cache_status"] == "ok")
            rows.append(row)

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
    # 2. aggregate evidence
    # ------------------------------------------------------------------
    summary: Counter[str] = Counter()
    by_category: dict[str, Counter] = defaultdict(Counter)
    by_kind: dict[str, Counter] = defaultdict(Counter)
    by_param: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        summary["accepted"] += 1
        summary[f"evidence_{row['evidence_tier']}"] += 1
        summary[f"cache_{row['cache_status']}"] += 1
        summary["training_memory_exact"] += int(row["training_memory_exact"])
        summary["test_fold_match"] += int(row["test_fold_match"])
        summary["independent_verified"] += int(row["independent_verified"])
        summary["unverified"] += int(not row["independent_verified"])
        summary["not_independent_test_fold_match_only"] += int(
            row["test_fold_match"] and not row["independent_verified"])
        if row["tier"] == "value_verified":
            summary["value_verified_tier_total"] += 1
            summary["value_verified_tier_test_fold_match"] += int(row["test_fold_match"])
        by_category[row["category"]][row["evidence_tier"]] += 1
        by_category[row["category"]]["accepted"] += 1
        by_kind[row["kind"]][row["evidence_tier"]] += 1
        by_kind[row["kind"]]["accepted"] += 1
        for name in row["changed_params"]:
            by_param[name][row["evidence_tier"]] += 1
            by_param[name]["accepted"] += 1

    cache_check = {
        "cache_verified_recovered": summary.get("cache_ok", 0),
        "cache_verified_still_error": summary.get("cache_error", 0),
        "cache_miss": summary.get("cache_miss", 0),
        "expected_headline_cache": _EXPECTED_CACHE,
        "match": {
            "cache_verified_recovered":
                summary.get("cache_ok", 0) == _EXPECTED_CACHE["cache_verified_recovered"],
            "cache_verified_still_error":
                summary.get("cache_error", 0) == _EXPECTED_CACHE["cache_verified_still_error"],
            "cache_miss": summary.get("cache_miss", 0) == _EXPECTED_CACHE["cache_miss"],
        },
    }
    print(f"cache cross-check: {json.dumps(cache_check['match'])}")
    print(f"evidence tiers: {dict(summary)}")

    # ------------------------------------------------------------------
    # 3. A1.7 accepted repairs (77): separate table + 0/77 cache recompute
    # ------------------------------------------------------------------
    a17_path = repo_root / args.a17_results
    a17_rows: list[dict] = []
    a17_summary: Counter[str] = Counter()
    if a17_path.is_file():
        a17 = json.loads(a17_path.read_text(encoding="utf-8"))
        for case in a17.get("cases", []):
            result = case.get("a17") or {}
            if not str(result.get("status", "")).startswith("accepted"):
                continue
            arguments = result.get("arguments") or {}
            rules = list(result.get("rules") or [])
            if "generic_success_replay" in rules:
                bucket = "replay (test-fold oracle)"
            elif any(rule.startswith("text_extract:") for rule in rules):
                new_extractor = any(
                    (case.get("param_evidence", {}).get(name) or {}).get("extractor_new")
                    for name in (case.get("resolved") or []))
                bucket = "new extractor (source/state/topic/ID)" if new_extractor \
                    else "core extractor (country)"
            else:
                bucket = "other"
            payload = backend.call(case.get("tool"), arguments)
            row = {
                "fold": case.get("fold"),
                "tool": case.get("tool"),
                "params": case.get("resolved"),
                "bucket": bucket,
                "rules": rules,
                "arguments": arguments,
                "cache_status": payload.status,
                "cache_error": payload.error,
                "suspected_country_currency_false_positive": bool(
                    bucket == "core extractor (country)"),
            }
            a17_rows.append(row)
            a17_summary[bucket] += 1
            a17_summary[f"cache_{payload.status}"] += 1
        a17_summary["accepted"] = len(a17_rows)
        a17_summary["cache_ok"] = sum(1 for row in a17_rows if row["cache_status"] == "ok")
        a17_summary["cache_error"] = sum(
            1 for row in a17_rows if row["cache_status"] == "error")
        a17_summary["cache_miss"] = sum(
            1 for row in a17_rows if row["cache_status"] == "miss")
    else:
        print(f"WARNING: A1.7 results not found at {a17_path}; A1.7 table is empty")

    # ------------------------------------------------------------------
    # 4. output
    # ------------------------------------------------------------------
    result = {
        "experiment": "ToolGuard A1.8.2 value-verification closure",
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
            "evidence_definition": {
                "training_memory_exact": "accepted arguments are a verbatim training-side "
                                         "successful argument set of the same task/function",
                "cache_ok": "released real-response cache has this exact function+arguments "
                            "call and its payload carries no error",
                "cache_error": "cache hit whose payload carries an error",
                "cache_miss": "no cache entry (not a failure)",
                "test_fold_match": "accepted arguments are a verbatim test-fold successful "
                                   "argument set of the same task/function (diagnostic only)",
                "independent_verified": "training_memory_exact or cache_ok",
            },
        },
        "headline_default_unchanged": {
            "computed": computed,
            "expected": expected,
            "match": protocol_match,
        },
        "cache_cross_check": cache_check,
        "summary": dict(summary),
        "by_category": {name: dict(counts) for name, counts in by_category.items()},
        "by_kind": {name: dict(counts) for name, counts in by_kind.items()},
        "by_param": {name: dict(counts) for name, counts in by_param.items()},
        "accepted_rows": rows,
        "a17": {
            "source_file": str(a17_path.relative_to(repo_root)) if a17_path.is_file() else None,
            "summary": dict(a17_summary),
            "rows": a17_rows,
            "note": ("A1.7 replay candidates came from test-fold recorded successes (oracle); "
                     "the 23 country rows are the 'US Dollars' false positives. The 0/77 cache "
                     "hit rate is recomputed here."),
        },
        "limitations": [
            "independent_verified means 'supported by an independent record (training-side "
            "success or released cache hit)', not 'semantically correct': a cache hit only "
            "proves that exact call succeeded once; a training-side match is still a recorded "
            "success, not a fresh verification",
            "the headline's 38 value_verified repairs are test-fold recorded-success matches "
            "(diagnostic), so none of them count as independent verification",
            "cache misses are not failures: the released cache only covers reference-solution "
            "inputs",
        ],
    }
    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "value_verification.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    report = _render_report(result)
    (output_dir / "value_verification.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nwrote {output_dir / 'value_verification.json'}")
    print(f"wrote {output_dir / 'value_verification.md'}")
    cache_ok = cache_check["match"]
    return 0 if protocol_match and all(cache_ok.values()) else 1


def _render_report(result: dict) -> str:
    lines = ["# A1.8.2 value-verification closure (557 accepted repairs)\n"]
    headline = result["headline_default_unchanged"]
    lines.append(f"Protocol verification against metrics.json: "
                 f"**{'MATCH' if headline['match'] else 'MISMATCH'}** "
                 f"`{json.dumps(headline['computed'], ensure_ascii=False)}`\n")
    summary = result["summary"]
    lines.append("## Evidence tiers (accepted = 557)\n")
    lines.append(_md_table(["evidence tier", "count", "independent?"], [
        ["training_memory_exact", summary.get("evidence_training_memory_exact", 0), "yes"],
        ["cache_ok", summary.get("evidence_cache_ok", 0), "yes"],
        ["test_fold_match (diagnostic only)",
         summary.get("evidence_test_fold_match", 0), "no"],
        ["none", summary.get("evidence_none", 0), "no"],
        ["**independent_verified (union)**", summary.get("independent_verified", 0), "—"],
        ["not independently verified", summary.get("unverified", 0), "—"],
        ["  of which test-fold match only",
         summary.get("not_independent_test_fold_match_only", 0), "—"],
    ]))
    lines.append("\n## Cache verification (cross-check vs headline)\n")
    cache = result["cache_cross_check"]
    lines.append(_md_table(
        ["metric", "recomputed", "headline", "match"],
        [["cache_verified_recovered", cache["cache_verified_recovered"],
          cache["expected_headline_cache"]["cache_verified_recovered"],
          cache["match"]["cache_verified_recovered"]],
         ["cache_verified_still_error", cache["cache_verified_still_error"],
          cache["expected_headline_cache"]["cache_verified_still_error"],
          cache["match"]["cache_verified_still_error"]],
         ["cache_miss", cache["cache_miss"],
          cache["expected_headline_cache"]["cache_miss"], cache["match"]["cache_miss"]]]))
    lines.append("\n## By rule category\n")
    lines.append(_md_table(
        ["rule category", "accepted", "training memory", "cache ok", "test-fold only", "none"],
        [[name, counts.get("accepted", 0),
          counts.get("training_memory_exact", 0), counts.get("cache_ok", 0),
          counts.get("test_fold_match", 0), counts.get("none", 0)]
         for name, counts in sorted(result["by_category"].items(),
                                    key=lambda item: -item[1].get("accepted", 0))]))
    lines.append("\n## By failure kind\n")
    lines.append(_md_table(
        ["kind", "accepted", "training memory", "cache ok", "test-fold only", "none"],
        [[name, counts.get("accepted", 0),
          counts.get("training_memory_exact", 0), counts.get("cache_ok", 0),
          counts.get("test_fold_match", 0), counts.get("none", 0)]
         for name, counts in sorted(result["by_kind"].items(),
                                    key=lambda item: -item[1].get("accepted", 0))]))
    lines.append("\n## By changed parameter (top 25)\n")
    items = sorted(result["by_param"].items(), key=lambda item: -item[1].get("accepted", 0))
    lines.append(_md_table(
        ["parameter", "accepted", "training memory", "cache ok", "test-fold only", "none"],
        [[name, counts.get("accepted", 0),
          counts.get("training_memory_exact", 0), counts.get("cache_ok", 0),
          counts.get("test_fold_match", 0), counts.get("none", 0)]
         for name, counts in items[:25]]))
    a17 = result["a17"]
    lines.append("\n## A1.7 accepted repairs (separate table)\n")
    if a17["source_file"]:
        lines.append(f"source: `{a17['source_file']}`\n")
        lines.append(_md_table(
            ["bucket", "count", "cache ok", "cache error", "cache miss"],
            [["replay (test-fold oracle)", a17["summary"].get("replay (test-fold oracle)", 0),
              "-", "-", "-"],
             ["core extractor (country, suspected false positive)",
              a17["summary"].get("core extractor (country)", 0), "-", "-", "-"],
             ["new extractor (source/state/topic/ID)",
              a17["summary"].get("new extractor (source/state/topic/ID)", 0), "-", "-", "-"],
             ["total", a17["summary"].get("accepted", 0),
              a17["summary"].get("cache_ok", 0), a17["summary"].get("cache_error", 0),
              a17["summary"].get("cache_miss", 0)]]))
        lines.append("\nThe 23 'core extractor (country)' rows are the 'US Dollars' false "
                     "positives (A1.6/A1.7 review): schema-valid is structural only and the "
                     "country value comes from a currency phrase, so they are not fixes. The 13 "
                     "replay rows are test-fold oracle; the 41 new-extractor rows are "
                     "structural, with 0 independent value verification.\n")
    else:
        lines.append("A1.7 results file not found; run `python "
                     "scripts/experiment_a17_value_recovery.py` first.\n")
    lines.append("\n## Limitations\n")
    for limitation in result["limitations"]:
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
