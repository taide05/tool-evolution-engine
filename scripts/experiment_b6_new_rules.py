"""B6 experiment: two opt-in deterministic rules, measured on the 969 failures.

Rules (both default off; enabling them can never change an already-accepted repair because
they append proposals after the historical ladder):

1. ``enable_required_default_completion`` -- fill *every* missing required parameter from
   the schema-declared example/default and pass the full schema gate.  Reading the code
   shows the historical ``fill_required_from_schema`` rule already fills named/all-required
   defaults, so this rule is expected to add nothing; it is measured, not assumed;
2. ``enable_enum_normalization`` -- replace a value that differs from exactly one declared
   enum member only by case/space/hyphen/underscore; several matching members -> refuse.

Scope rule 3 ("boolean/numeric-string coercion") is deliberately NOT implemented: the
historical ladder already coerces unambiguous numeric strings for integer/number params and
"true"/"false" for boolean params.  The artifact reports the number of already-coerced
accepted repairs as the evidence for that decision.

Every configuration is measured with the same fixed-three LOMO protocol as the headline;
the default configuration must reproduce 969 / 583 / 519 / 38 / 131 / 557.  Newly accepted
repairs are listed per rule and cache-checked against the released response cache.

Usage:
    python scripts/experiment_b6_new_rules.py
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
_COERCION_RULE_MARKERS = ("coerce", "schema_type_coercion", "memory_value_normalize")

_EXPECTED = {
    "evaluable_failures": 969,
    "proposed_repairs": 583,
    "schema_passed": 519,
    "value_verified_repairs": 38,
    "coerced_repairs": 131,
    "accepted_repairs": 557,
}

_CONFIGS = (
    ("baseline", False, False),
    ("rule1_required_default", True, False),
    ("rule2_enum_normalization", False, True),
    ("both_rules", True, True),
)


def _key(call) -> tuple:
    return (call.model, call.trace_id, call.step, call.tool_name, call.query_id)


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
    parser.add_argument("--output-dir", default=".scratch/toolguard/b6")
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
          f"cache entries: {len(backend.index)}")

    config_counts: dict[str, Counter] = {name: Counter() for name, _r1, _r2 in _CONFIGS}
    config_results: dict[str, dict[tuple, dict]] = {name: {} for name, _r1, _r2 in _CONFIGS}
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
                continue
            key = _key(call)
            kind = classify_error(call.error)
            reference = test_successes.get(
                (call.group, call.query_id, call.tool_name)) or []
            for name, rule1, rule2 in _CONFIGS:
                counts = config_counts[name]
                counts["evaluable_failures"] += 1
                result = repair_failure(
                    call, schemas, memory, extractor_map,
                    successful_arguments=reference,
                    enable_required_default_completion=rule1,
                    enable_enum_normalization=rule2)
                if result.proposed:
                    counts["proposed_repairs"] += 1
                row = {
                    "fold": family,
                    "tool": call.tool_name,
                    "kind": kind,
                    "accepted": bool(result.repairable),
                    "rules": list(result.rules),
                    "arguments": result.arguments,
                }
                config_results[name][key] = row
                counts["by_kind_" + kind + "_failures"] += 1
                if not result.repairable:
                    continue
                counts["accepted_repairs"] += 1
                counts["by_kind_" + kind + "_accepted"] += 1
                if result.schema_valid:
                    counts["schema_passed"] += 1
                else:
                    counts["value_verified_repairs"] += 1
                if "schema_type_coercion" in result.rules:
                    counts["coerced_repairs"] += 1
                if any(any(marker in rule for marker in _COERCION_RULE_MARKERS)
                       for rule in result.rules):
                    counts["coercion_rule_repairs"] += 1
                for rule in result.rules:
                    category = rule.split(":")[0]
                    if category in ("required_default", "enum_normalize"):
                        counts["new_rule_" + category] += 1

    baseline = config_counts["baseline"]
    computed = {key: int(baseline.get(key, 0)) for key in _EXPECTED}
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

    # newly accepted repairs vs baseline, with cache verification
    baseline_accepted = {key for key, row in config_results["baseline"].items()
                         if row["accepted"]}
    for name, _rule1, _rule2 in _CONFIGS:
        if name == "baseline":
            continue
        counts = config_counts[name]
        accepted = {key for key, row in config_results[name].items() if row["accepted"]}
        added = sorted(accepted - baseline_accepted)
        lost = sorted(baseline_accepted - accepted)
        counts["newly_accepted"] = len(added)
        counts["lost_accepted"] = len(lost)
        added_rows = []
        for key in added:
            row = config_results[name][key]
            call = next(call for call in calls
                        if _key(call) == key and not call.recorded_ok)
            payload = backend.call(call.tool_name, row["arguments"] or {})
            added_rows.append({
                "fold": row["fold"],
                "tool": row["tool"],
                "kind": row["kind"],
                "rules": row["rules"],
                "arguments": row["arguments"],
                "cache_status": payload.status,
            })
            counts["newly_accepted_cache_" + payload.status] += 1
        config_counts[name]["newly_accepted_rows"] = added_rows

    difference = {
        name: {
            **{key: int(value) for key, value in config_counts[name].items()
               if not key.startswith("newly_accepted_rows")},
        }
        for name, _rule1, _rule2 in _CONFIGS
    }

    result = {
        "experiment": "ToolGuard B6 opt-in deterministic rules",
        "protocol": {
            "protocol": "lomo fixed-three (same as metrics.json results.lomo_aggregate)",
            "scope": ("two opt-in rules; rule 3 (boolean/numeric-string coercion) is not "
                      "implemented because the historical ladder already covers it"),
            "defaults": "all new flags default False; the default configuration is unchanged",
            "train_models": sorted(TRAIN_MODELS),
            "folds": families,
            "calls": len(calls),
            "schemas": schema_stats,
            "schema_fidelity_repair": {
                "type_corrections": schema_fidelity["type_corrections"],
                "params_added": schema_fidelity["params_added"],
            },
            "schema_enum_declarations": sum(
                1 for schema in schemas.values() for prop in schema.properties.values()
                if prop.get("enum")),
            "rules": {
                "rule1_required_default": (
                    "fill every missing required parameter from schema example/default and "
                    "pass the full schema gate; refuses when any required value is unknown"),
                "rule2_enum_normalization": (
                    "replace a value that differs from exactly one declared enum member "
                    "only by case/space/hyphen/underscore; several matches -> refuse"),
                "rule3_boolean_numeric_coercion": (
                    "skipped: existing rules._repair_value_coercion + "
                    "engine.coerce_values_to_schema already coerce unambiguous numeric "
                    "strings and 'true'/'false'; see coercion_rule_repairs"),
            },
        },
        "headline_default_unchanged": {
            "computed": computed,
            "expected": expected,
            "match": protocol_match,
        },
        "configurations": difference,
        "newly_accepted_details": {
            name: config_counts[name].get("newly_accepted_rows", [])
            for name, _rule1, _rule2 in _CONFIGS if name != "baseline"
        },
        "answer": {
            "rule1_gain": difference["rule1_required_default"]["newly_accepted"],
            "rule2_gain": difference["rule2_enum_normalization"]["newly_accepted"],
            "both_gain": difference["both_rules"]["newly_accepted"],
            "rule1_verdict": (
                "no gain: the historical fill_required_from_schema rule already fills "
                "named/all-required defaults, so the new rule is shadowed (0 newly accepted)"
                if difference["rule1_required_default"]["newly_accepted"] == 0 else
                "gain measured; see newly accepted rows"),
            "rule2_verdict": (
                "no gain on this dataset: no invalid_value failure carries a near-miss enum "
                "value that passes the full gate"
                if difference["rule2_enum_normalization"]["newly_accepted"] == 0 else
                "gain measured; see newly accepted rows"),
            "rule3_verdict": (
                "skipped as already covered; the baseline accepts "
                f"{int(baseline.get('coercion_rule_repairs', 0))} repairs through existing "
                "coercion rules"),
        },
        "limitations": [
            "the two new rules are default-off and do not change the headline; their gain on "
            "this dataset is zero, which is the honest result",
            "the evaluated schema index contains 0 enum declarations (build_schema_index "
            "copies only type/description/default), so rule 2 has no trigger opportunity on "
            "this dataset at all; it is covered by unit tests only, and its real-world value "
            "would need a schema index that declares enums",
            "rule 1 is shadowed by the historical ladder on every measured case: reading "
            "``engine._schema_default_completion`` in isolation suggests a gap, but "
            "``rules._repair_missing_required`` already fills the required defaults",
            "newly accepted repairs are cache-checked; a cache miss is not a failure",
        ],
    }
    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "new_rules.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    report = _render_report(result)
    (output_dir / "new_rules.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nwrote {output_dir / 'new_rules.json'}")
    print(f"wrote {output_dir / 'new_rules.md'}")
    return 0 if protocol_match else 1


def _render_report(result: dict) -> str:
    lines = ["# B6 opt-in deterministic rules (default off, measured on the 969 failures)\n"]
    headline = result["headline_default_unchanged"]
    lines.append(f"Default configuration reproduces the headline: "
                 f"**{'MATCH' if headline['match'] else 'MISMATCH'}** "
                 f"`{json.dumps(headline['computed'], ensure_ascii=False)}`\n")
    protocol = result["protocol"]
    lines.append("## Rules\n")
    lines.append(_md_table(
        ["rule", "flag (default off)", "behaviour"],
        [["required default completion", "`enable_required_default_completion`",
          protocol["rules"]["rule1_required_default"]],
         ["enum normalization", "`enable_enum_normalization`",
          protocol["rules"]["rule2_enum_normalization"]],
         ["boolean/numeric-string coercion", "— (not implemented)",
          protocol["rules"]["rule3_boolean_numeric_coercion"]]]))
    configs = result["configurations"]
    lines.append("\n## Before/after on the 969 failures\n")
    lines.append(_md_table(
        ["configuration", "proposed", "schema-valid", "value-verified", "accepted",
         "coerced", "newly accepted", "lost"],
        [[name, counts.get("proposed_repairs", 0), counts.get("schema_passed", 0),
          counts.get("value_verified_repairs", 0), counts.get("accepted_repairs", 0),
          counts.get("coerced_repairs", 0), counts.get("newly_accepted", 0),
          counts.get("lost_accepted", 0)]
         for name, counts in configs.items()]))
    lines.append("\n## Rule triggers (accepted repairs)\n")
    lines.append(_md_table(
        ["configuration", "required_default repairs", "enum_normalize repairs",
         "coercion-rule repairs (rule 3 evidence)"],
        [[name, counts.get("new_rule_required_default", 0),
          counts.get("new_rule_enum_normalize", 0),
          counts.get("coercion_rule_repairs", 0)]
         for name, counts in configs.items()]))
    details = result["newly_accepted_details"]
    lines.append("\n## Newly accepted repairs (if any)\n")
    for name, rows in details.items():
        lines.append(f"### {name}\n")
        if not rows:
            lines.append("none\n")
            continue
        lines.append(_md_table(
            ["fold", "tool", "kind", "rules", "arguments", "cache status"],
            [[row["fold"], row["tool"], row["kind"], ", ".join(row["rules"]),
              json.dumps(row["arguments"], ensure_ascii=False), row["cache_status"]]
             for row in rows]))
    answer = result["answer"]
    lines.append("\n## Answer\n")
    lines.append(f"- rule 1 (required default completion): gain "
                 f"**{answer['rule1_gain']}** — {answer['rule1_verdict']}.")
    lines.append(f"- rule 2 (enum normalization): gain **{answer['rule2_gain']}** — "
                 f"{answer['rule2_verdict']}.")
    lines.append(f"- rule 3 (boolean/numeric coercion): {answer['rule3_verdict']}.")
    lines.append(f"- both rules together: gain **{answer['both_gain']}**.")
    lines.append("\n## Limitations\n")
    for limitation in result["limitations"]:
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
