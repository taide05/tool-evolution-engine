"""B4: full offline A/B on the 969 evaluable failures with the released-recorded backend.

Reuses the headline fixed-three LOMO protocol and the **default** repair configuration
(no experimental flags).  For every evaluable failure:

* ``baseline``: the original call is a recorded failure (that is why it is in the 969);
* ``guard``: ``repair_failure`` with the default configuration.  If the repair is accepted,
  the repaired call is replayed through ``RecordedBackend`` against the released
  real-response cache and recorded as ``cache_ok`` / ``cache_error`` / ``cache_miss``.
  If the repair is not accepted the row is ``proposed_schema_rejected`` / ``no_proposal``
  ("no repair").

Honest interpretation (also written into the report):
* ``cache_ok`` only proves that this exact function+arguments call appears in the released
  cache with no error payload; it is **not** a production success rate;
* ``cache_miss`` is **not** a failure (the cache only covers reference-solution inputs);
* no real API and no API key is used; this is a deterministic offline replay.

Acceptance: the default configuration must reproduce the headline aggregate
(969 / 583 / 519 / 38 / 131 / 557) and the cache counts (48 / 0 / 509).  If it does not,
the script exits non-zero and the report says so instead of adjusting anything.

Usage:
    python scripts/experiment_b4_full_offline_ab.py
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

_EXPECTED = {
    "evaluable_failures": 969,
    "proposed_repairs": 583,
    "schema_passed": 519,
    "value_verified_repairs": 38,
    "coerced_repairs": 131,
    "accepted_repairs": 557,
}
_EXPECTED_CACHE = {"cache_ok": 48, "cache_error": 0, "cache_miss": 509}


def _rule_category(rules: list[str]) -> str:
    if not rules:
        return "unknown"
    rules = [rule for rule in rules if rule != "recorded_success_match"]
    return rules[0].split(":")[0] if rules else "unknown"


def _md_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def _aggregate(rows: list[dict]) -> dict:
    counts: Counter[str] = Counter()
    by_kind: dict[str, Counter] = defaultdict(Counter)
    by_rule: dict[str, Counter] = defaultdict(Counter)
    by_tool: dict[str, Counter] = defaultdict(Counter)
    by_model: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        status = row["status"]
        counts["evaluable_failures"] += 1
        accepted = status.startswith("accepted")
        if status == "accepted_schema_valid":
            counts["accepted_repairs"] += 1
            counts["schema_passed"] += 1
        elif status == "accepted_value_verified":
            counts["accepted_repairs"] += 1
            counts["value_verified_repairs"] += 1
        elif status == "proposed_schema_rejected":
            counts["proposed_repairs"] += 1
            counts["proposed_schema_rejected"] += 1
            counts["no_repair"] += 1
        else:
            counts["no_proposal"] += 1
            counts["no_repair"] += 1
        if accepted:
            counts["proposed_repairs"] += 1
            counts["coerced_repairs"] += int(row["coerced"])
            if row["cache_status"] == "ok":
                counts["cache_ok"] += 1
            elif row["cache_status"] == "error":
                counts["cache_error"] += 1
            elif row["cache_status"] == "miss":
                counts["cache_miss"] += 1
        for group_name, bucket in (("by_kind", by_kind[row["kind"]]),
                                   ("by_tool", by_tool[row["tool"]]),
                                   ("by_model", by_model[row["model"]])):
            bucket["failures"] += 1
            bucket[status] += 1
            if accepted and row["cache_status"] == "ok":
                bucket["cache_ok"] += 1
        if accepted:
            rule = row["rule_category"]
            by_rule[rule]["repairs"] += 1
            by_rule[rule][status] += 1
            if row["cache_status"] == "ok":
                by_rule[rule]["cache_ok"] += 1
    return {
        "counts": dict(counts),
        "by_kind": {key: dict(value) for key, value in by_kind.items()},
        "by_rule": {key: dict(value) for key, value in by_rule.items()},
        "by_tool": {key: dict(value) for key, value in by_tool.items()},
        "by_model": {key: dict(value) for key, value in by_model.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calls", default=".scratch/toolguard/tool_calls.jsonl")
    parser.add_argument("--queries", default=(
        ".scratch/stb/repo/intuit-ai-research-StableToolBench-74f76c1/"
        "solvable_queries/test_instruction"))
    parser.add_argument("--expected-metrics", default="benchmarks/toolguard/metrics.json")
    parser.add_argument("--cache", default=".scratch/stb/cache/tool_response_cache")
    parser.add_argument("--output-dir", default=".scratch/toolguard/b4")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    calls = [call for call in load_calls(repo_root / args.calls)
             if call.tool_name not in _SENTINEL_CALLS]
    schemas, schema_stats = build_schema_index(repo_root / args.queries)
    schemas, schema_fidelity = refine_schema_index(schemas, calls)
    cache_root = repo_root / args.cache
    backend = RecordedBackend(cache_root, CacheIndex.from_root(cache_root))
    families = sorted({call.model for call in calls})
    print(f"calls: {len(calls)} | schemas: {len(schemas)} | "
          f"cache entries: {len(backend.index)} | families: {families}")

    rows: list[dict] = []
    protocol: Counter[str] = Counter()
    for family in families:
        train = [call for call in calls
                 if call.model != family and call.model in TRAIN_MODELS]
        test = [call for call in calls if call.model == family]
        memory = SuccessMemory.build(train, schemas)
        extractor_map = ExtractorMap.learn(train, schemas)
        test_successes: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
        for call in test:
            if call.recorded_ok:
                test_successes[(call.group, call.query_id, call.tool_name)].append(
                    dict(call.arguments))
        for call in test:
            if call.recorded_ok:
                continue
            if "__raw__" in call.arguments or call.tool_name not in schemas:
                protocol["skipped_no_schema"] += 1
                continue
            protocol["evaluable_failures"] += 1
            kind = classify_error(call.error)
            result = repair_failure(
                call, schemas, memory, extractor_map,
                successful_arguments=test_successes.get(
                    (call.group, call.query_id, call.tool_name)) or [])
            if result.proposed:
                protocol["proposed_repairs"] += 1
            if result.repairable and result.schema_valid:
                status = "accepted_schema_valid"
                protocol["schema_passed"] += 1
            elif result.repairable:
                status = "accepted_value_verified"
                protocol["value_verified_repairs"] += 1
            elif result.proposed:
                status = "proposed_schema_rejected"
            else:
                status = "no_proposal"
            if result.repairable:
                protocol["accepted_repairs"] += 1
                if "schema_type_coercion" in result.rules:
                    protocol["coerced_repairs"] += 1
                function = result.function_name or call.tool_name
                arguments = result.arguments if result.arguments is not None else call.arguments
                payload = backend.call(function, arguments)
                cache_status = payload.status
                cache_error = payload.error
            else:
                arguments = None
                function = None
                cache_status = None
                cache_error = None
            rows.append({
                "fold": family,
                "model": call.model,
                "group": call.group,
                "query_id": call.query_id,
                "trace_id": call.trace_id,
                "tool": call.tool_name,
                "function": function,
                "kind": kind,
                "status": status,
                "rules": list(result.rules),
                "rule_category": _rule_category(list(result.rules)),
                "coerced": "schema_type_coercion" in result.rules,
                "arguments": arguments,
                "cache_status": cache_status,
                "cache_error": cache_error,
            })

    computed = {key: int(protocol.get(key, 0)) for key in _EXPECTED}
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
    aggregate = _aggregate(rows)
    counts = aggregate["counts"]
    cache_computed = {
        "cache_ok": counts.get("cache_ok", 0),
        "cache_error": counts.get("cache_error", 0),
        "cache_miss": counts.get("cache_miss", 0),
    }
    protocol_match = computed == expected
    cache_match = cache_computed == _EXPECTED_CACHE
    print(f"protocol verification: {'MATCH' if protocol_match else 'MISMATCH'} {computed}")
    print(f"cache verification: {'MATCH' if cache_match else 'MISMATCH'} {cache_computed}")

    result = {
        "experiment": "ToolGuard B4 full offline A/B on the 969 evaluable failures",
        "protocol": {
            "protocol": "lomo fixed-three (headline), default repair configuration",
            "no_network": True,
            "no_api_key": True,
            "backend": "RecordedBackend over the released ToolBench response cache",
            "train_models": sorted(TRAIN_MODELS),
            "folds": families,
            "calls": len(calls),
            "schemas": schema_stats,
            "schema_fidelity_repair": {
                "type_corrections": schema_fidelity["type_corrections"],
                "params_added": schema_fidelity["params_added"],
            },
            "skipped_no_schema": int(protocol.get("skipped_no_schema", 0)),
            "definitions": {
                "baseline": "the original call is a recorded failure (all 969 rows)",
                "guard": "default-configuration repair proposal, schema-gated",
                "cache_ok": "the repaired call appears in the released cache with no error "
                            "payload (exact function+arguments); not a production success rate",
                "cache_error": "cache hit whose payload carries an error",
                "cache_miss": "no cache entry -- NOT a failure (the cache only covers "
                              "reference-solution inputs)",
                "no_repair": "not accepted (proposed and schema-rejected, or no proposal)",
            },
        },
        "headline_reproduction": {
            "computed": computed,
            "expected": expected,
            "match": protocol_match,
        },
        "cache_cross_check": {
            "computed": cache_computed,
            "expected": _EXPECTED_CACHE,
            "match": cache_match,
        },
        "aggregate": aggregate["counts"],
        "by_kind": aggregate["by_kind"],
        "by_rule": aggregate["by_rule"],
        "by_tool": aggregate["by_tool"],
        "by_model": aggregate["by_model"],
        "rows": rows,
        "limitations": [
            "cache_ok only proves this exact call is recorded as successful in the released "
            "cache; the cache covers reference-solution inputs, so the 48/557 = 8.6% hit "
            "rate is a verification lower bound, not a production success rate",
            "cache_miss is not a failure and does not mean the repair is wrong",
            "the repaired call is never executed against a live API in this experiment; "
            "side effects and live availability are out of scope",
            "adds no new capability over the headline evaluator: it is the same default "
            "configuration, measured row by row with the recorded backend",
        ],
    }
    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "full_offline_ab.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    report = _render_report(result)
    (output_dir / "full_offline_ab.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nwrote {output_dir / 'full_offline_ab.json'}")
    print(f"wrote {output_dir / 'full_offline_ab.md'}")
    return 0 if (protocol_match and cache_match) else 1


def _render_report(result: dict) -> str:
    lines = ["# B4 full offline A/B (969 evaluable failures, released recorded backend)\n"]
    protocol = result["protocol"]
    lines.append(f"Protocol: {protocol['protocol']}; backend: {protocol['backend']}; "
                 f"network: {'none (offline)' if protocol['no_network'] else 'USED'}; "
                 f"API key: {'not used' if protocol['no_api_key'] else 'USED'}.\n")
    headline = result["headline_reproduction"]
    lines.append(f"Headline reproduction: **{'MATCH' if headline['match'] else 'MISMATCH'}** "
                 f"`{json.dumps(headline['computed'], ensure_ascii=False)}`\n")
    cache = result["cache_cross_check"]
    lines.append(f"Cache cross-check: **{'MATCH' if cache['match'] else 'MISMATCH'}** "
                 f"`{json.dumps(cache['computed'], ensure_ascii=False)}`\n")
    counts = result["aggregate"]
    lines.append("## Aggregate funnel\n")
    lines.append(_md_table(
        ["metric", "count", "denominator", "rate"],
        [["evaluable failures", counts.get("evaluable_failures", 0), 969, "100.0%"],
         ["guard proposed repairs", counts.get("proposed_repairs", 0),
          counts.get("evaluable_failures", 0),
          f"{100 * counts.get('proposed_repairs', 0) / max(counts.get('evaluable_failures', 1), 1):.1f}%"],
         ["accepted (schema-valid + value-verified)", counts.get("accepted_repairs", 0),
          counts.get("evaluable_failures", 0),
          f"{100 * counts.get('accepted_repairs', 0) / max(counts.get('evaluable_failures', 1), 1):.1f}%"],
         ["  schema-valid", counts.get("schema_passed", 0),
          counts.get("accepted_repairs", 0),
          f"{100 * counts.get('schema_passed', 0) / max(counts.get('accepted_repairs', 1), 1):.1f}%"],
         ["  value-verified", counts.get("value_verified_repairs", 0),
          counts.get("accepted_repairs", 0),
          f"{100 * counts.get('value_verified_repairs', 0) / max(counts.get('accepted_repairs', 1), 1):.1f}%"],
         ["  coerced repairs", counts.get("coerced_repairs", 0),
          counts.get("accepted_repairs", 0),
          f"{100 * counts.get('coerced_repairs', 0) / max(counts.get('accepted_repairs', 1), 1):.1f}%"],
         ["no repair (not accepted)", counts.get("no_repair", 0),
          counts.get("evaluable_failures", 0),
          f"{100 * counts.get('no_repair', 0) / max(counts.get('evaluable_failures', 1), 1):.1f}%"],
         ["  of which proposed but schema-rejected",
          counts.get("proposed_schema_rejected", 0), counts.get("no_repair", 0),
          f"{100 * counts.get('proposed_schema_rejected', 0) / max(counts.get('no_repair', 1), 1):.1f}%"],
         ["  of which no proposal at all", counts.get("no_proposal", 0),
          counts.get("no_repair", 0),
          f"{100 * counts.get('no_proposal', 0) / max(counts.get('no_repair', 1), 1):.1f}%"],
         ["cache ok (among accepted)", counts.get("cache_ok", 0),
          counts.get("accepted_repairs", 0),
          f"{100 * counts.get('cache_ok', 0) / max(counts.get('accepted_repairs', 1), 1):.1f}%"],
         ["cache error (among accepted)", counts.get("cache_error", 0),
          counts.get("accepted_repairs", 0), "—"],
         ["cache miss (among accepted)", counts.get("cache_miss", 0),
          counts.get("accepted_repairs", 0),
          f"{100 * counts.get('cache_miss', 0) / max(counts.get('accepted_repairs', 1), 1):.1f}%"],
         ["skipped (not evaluable)", protocol.get("skipped_no_schema", 0), 1033,
          f"{100 * protocol.get('skipped_no_schema', 0) / 1033:.1f}% of real failures"]]))
    lines.append("\n## By failure kind\n")
    kinds = result["by_kind"]
    lines.append(_md_table(
        ["kind", "failures", "schema-valid", "value-verified", "schema rejected",
         "no repair", "cache ok"],
        [[kind, counts.get("failures", 0),
          counts.get("accepted_schema_valid", 0), counts.get("accepted_value_verified", 0),
          counts.get("proposed_schema_rejected", 0),
          counts.get("no_proposal", 0) + counts.get("proposed_schema_rejected", 0),
          counts.get("cache_ok", 0)]
         for kind, counts in sorted(kinds.items(), key=lambda item: -item[1].get("failures", 0))]))
    lines.append("\n## By accepted rule\n")
    rules = result["by_rule"]
    lines.append(_md_table(
        ["rule category", "repairs", "schema-valid", "value-verified", "cache ok"],
        [[rule, counts.get("repairs", 0), counts.get("accepted_schema_valid", 0),
          counts.get("accepted_value_verified", 0), counts.get("cache_ok", 0)]
         for rule, counts in sorted(rules.items(), key=lambda item: -item[1].get("repairs", 0))]))
    lines.append("\n## By model family\n")
    models = result["by_model"]
    lines.append(_md_table(
        ["model", "failures", "schema-valid", "value-verified", "no repair", "cache ok"],
        [[model, counts.get("failures", 0), counts.get("accepted_schema_valid", 0),
          counts.get("accepted_value_verified", 0),
          counts.get("no_proposal", 0) + counts.get("proposed_schema_rejected", 0),
          counts.get("cache_ok", 0)]
         for model, counts in sorted(models.items())]))
    lines.append("\n## Top tools by failures (25)\n")
    tools = sorted(result["by_tool"].items(), key=lambda item: -item[1].get("failures", 0))[:25]
    lines.append(_md_table(
        ["tool", "failures", "schema-valid", "value-verified", "no repair", "cache ok"],
        [[tool, counts.get("failures", 0), counts.get("accepted_schema_valid", 0),
          counts.get("accepted_value_verified", 0),
          counts.get("no_proposal", 0) + counts.get("proposed_schema_rejected", 0),
          counts.get("cache_ok", 0)]
         for tool, counts in tools]))
    lines.append("\n## How to read this\n")
    lines.append("- `cache_ok` only proves that this exact repaired call appears in the "
                 "released cache with no error payload; it is not a production success rate "
                 "and must never be presented as one.\n"
                 "- `cache_miss` is not a failure: the released cache only covers "
                 "reference-solution inputs, so most valid repairs are simply unverified.\n"
                 "- This experiment adds no new capability over the headline evaluator: it "
                 "is the same default configuration, measured row by row against the recorded "
                 "backend, with no network and no API key.\n")
    lines.append("\n## Limitations\n")
    for limitation in result["limitations"]:
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
