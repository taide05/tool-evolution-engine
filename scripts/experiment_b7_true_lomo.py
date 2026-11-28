"""B7 diagnostic: true five-family LOMO vs the headline fixed-three protocol.

Runs both LOMO train policies with the same default repair configuration and reports the
per-fold and per-failure-kind funnel.  The two committed commands are also the source of
the artifacts (they are re-run by this script's documented commands):

    python scripts/eval_toolguard_rules.py --lomo-train-policy exclude-fold \
        --output .scratch/toolguard/b7/true_lomo.json
    python scripts/eval_toolguard_rules.py \
        --output .scratch/toolguard/b7/fixed_three.json

Main headline protocol: fixed-three (57.5%).  True LOMO (59.7%) is a *sensitivity* upper
bound, not the main protocol: in true LOMO the two gpt-4 families enter training for the
fixed-three folds, which is why it is higher.  This diagnostic only explains the gap; it
does not change ``metrics.json``, any artifact or ``TOOLGUARD.md``.

Usage:
    python scripts/experiment_b7_true_lomo.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tool_evolution.toolguard.engine import repair_failure  # noqa: E402
from tool_evolution.toolguard.extractors import ExtractorMap  # noqa: E402
from tool_evolution.toolguard.memory import SuccessMemory  # noqa: E402
from tool_evolution.toolguard.records import load_calls  # noqa: E402
from tool_evolution.toolguard.rules import classify_error  # noqa: E402
from tool_evolution.toolguard.schemas import build_schema_index, refine_schema_index  # noqa: E402

TRAIN_MODELS = {"chatgpt", "toolllama", "gpt-3.5-turbo-1106"}
_SENTINEL_CALLS = {"invalid_hallucination_function_name"}


def _md_table(headers: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join("---" for _ in headers) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines)


def run_protocol(calls, schemas, families, train_policy: str) -> dict:
    """Replicate the evaluator's LOMO loop, adding per-kind accepted counts."""
    folds: dict[str, dict] = {}
    per_kind: dict[str, Counter] = defaultdict(Counter)
    totals: Counter[str] = Counter()
    for family in families:
        train = [call for call in calls if call.model != family]
        if train_policy == "fixed-three":
            valid_train = [call for call in train if call.model in TRAIN_MODELS]
        else:
            valid_train = train
        test = [call for call in calls if call.model == family]
        memory = SuccessMemory.build(valid_train, schemas)
        extractor_map = ExtractorMap.learn(valid_train, schemas)
        test_successes: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
        for call in test:
            if call.recorded_ok:
                test_successes[(call.group, call.query_id, call.tool_name)].append(
                    dict(call.arguments))
        fold_counts: Counter[str] = Counter()
        for call in test:
            if call.recorded_ok:
                continue
            if "__raw__" in call.arguments or call.tool_name not in schemas:
                fold_counts["skipped_no_schema"] += 1
                continue
            kind = classify_error(call.error)
            fold_counts["failures"] += 1
            per_kind[kind]["failures"] += 1
            result = repair_failure(
                call, schemas, memory, extractor_map,
                successful_arguments=test_successes.get(
                    (call.group, call.query_id, call.tool_name)) or [])
            if result.proposed:
                fold_counts["proposed"] += 1
                per_kind[kind]["proposed"] += 1
            if result.repairable:
                fold_counts["accepted"] += 1
                per_kind[kind]["accepted"] += 1
                if result.schema_valid:
                    fold_counts["schema_passed"] += 1
                    per_kind[kind]["schema_passed"] += 1
                else:
                    fold_counts["value_verified"] += 1
                    per_kind[kind]["value_verified"] += 1
                if "schema_type_coercion" in result.rules:
                    fold_counts["coerced"] += 1
        folds[family] = {
            "train_models_used": sorted({call.model for call in valid_train}),
            "train_calls_used": len(valid_train),
            "test_calls": len(test),
            **{key: int(fold_counts.get(key, 0)) for key in
               ("failures", "proposed", "schema_passed", "value_verified", "accepted")},
            "coverage": round(fold_counts["accepted"] / max(fold_counts["failures"], 1), 4),
        }
        for key in ("failures", "proposed", "schema_passed", "value_verified", "accepted"):
            totals[key] += int(fold_counts.get(key, 0))
    totals["coverage"] = round(totals["accepted"] / max(totals["failures"], 1), 4)
    return {"folds": folds, "aggregate": dict(totals),
            "per_kind": {kind: dict(counts) for kind, counts in per_kind.items()}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calls", default=".scratch/toolguard/tool_calls.jsonl")
    parser.add_argument("--queries", default=(
        ".scratch/stb/repo/intuit-ai-research-StableToolBench-74f76c1/"
        "solvable_queries/test_instruction"))
    parser.add_argument("--expected-metrics", default="benchmarks/toolguard/metrics.json")
    parser.add_argument("--true-lomo", default=".scratch/toolguard/b7/true_lomo.json")
    parser.add_argument("--fixed-three", default=".scratch/toolguard/b7/fixed_three.json")
    parser.add_argument("--output-dir", default=".scratch/toolguard/b7")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    calls = [call for call in load_calls(repo_root / args.calls)
             if call.tool_name not in _SENTINEL_CALLS]
    schemas, schema_stats = build_schema_index(repo_root / args.queries)
    schemas, schema_fidelity = refine_schema_index(schemas, calls)
    families = sorted({call.model for call in calls})
    print(f"calls: {len(calls)} | schemas: {len(schemas)} | families: {families}")

    metrics = json.loads((repo_root / args.expected_metrics).read_text(encoding="utf-8"))
    stored_sensitivity = metrics["results"]["lomo_sensitivity_exclude_fold"]["aggregate"]
    headline = metrics["results"]["lomo_aggregate"]

    fixed = run_protocol(calls, schemas, families, "fixed-three")
    true_lomo = run_protocol(calls, schemas, families, "exclude-fold")
    print(f"fixed-three aggregate: {fixed['aggregate']}")
    print(f"true-LOMO   aggregate: {true_lomo['aggregate']}")

    # cross-check against the evaluator artifacts (same protocol, no per-kind counters)
    eval_true = json.loads((repo_root / args.true_lomo).read_text(encoding="utf-8"))
    eval_fixed = json.loads((repo_root / args.fixed_three).read_text(encoding="utf-8"))
    eval_true_agg = eval_true["lomo"]["aggregate"]
    eval_fixed_agg = eval_fixed["lomo"]["aggregate"]

    def _match(own: dict, other: dict) -> bool:
        mapping = {"failures": "failures", "proposed": "proposed_repairs",
                   "schema_passed": "schema_passed",
                   "value_verified": "value_verified_repairs",
                   "accepted": "accepted_repairs"}
        return all(own[key] == other[mapping[key]] for key in mapping)

    fixed_matches_headline = all(
        fixed["aggregate"][own] == headline[other]
        for own, other in (("failures", "evaluable_failures"),
                           ("proposed", "proposed_repairs"),
                           ("schema_passed", "schema_passed"),
                           ("value_verified", "value_verified_repairs"),
                           ("accepted", "accepted_repairs")))
    true_matches_stored = all(
        true_lomo["aggregate"][own] == stored_sensitivity[other]
        for own, other in (("failures", "evaluable_failures"),
                           ("proposed", "proposed_repairs"),
                           ("schema_passed", "schema_passed"),
                           ("value_verified", "value_verified_repairs"),
                           ("accepted", "accepted_repairs")))
    print(f"fixed-three reproduces headline: {fixed_matches_headline}")
    print(f"true LOMO reproduces stored sensitivity (578/969): {true_matches_stored}")
    print(f"true LOMO matches evaluator artifact: {_match(true_lomo['aggregate'], eval_true_agg)}")

    gap = {
        "accepted_delta": true_lomo["aggregate"]["accepted"]
        - fixed["aggregate"]["accepted"],
        "coverage_delta_pp": round(100 * (true_lomo["aggregate"]["coverage"]
                                          - fixed["aggregate"]["coverage"]), 2),
        "proposed_delta": true_lomo["aggregate"]["proposed"]
        - fixed["aggregate"]["proposed"],
        "schema_passed_delta": true_lomo["aggregate"]["schema_passed"]
        - fixed["aggregate"]["schema_passed"],
        "value_verified_delta": true_lomo["aggregate"]["value_verified"]
        - fixed["aggregate"]["value_verified"],
        "explanation": (
            "true LOMO trains on every call outside the held-out fold: for the chatgpt / "
            "toolllama / gpt-3.5 folds that adds both gpt-4 families to training, and for "
            "each gpt-4 fold it adds the other gpt-4 family. The extra training support "
            "yields more proposals (+21), more schema-valid repairs (+14) and more accepted "
            "repairs (+21 at the aggregate), i.e. 59.7% vs the headline 57.5%. The "
            "gpt-4-0613 fold is unchanged (61 accepted). This is a sensitivity upper bound, "
            "not the main protocol."),
    }
    by_fold = {}
    for family in families:
        fixed_fold = fixed["folds"][family]
        true_fold = true_lomo["folds"][family]
        by_fold[family] = {
            "fixed_three": fixed_fold,
            "true_lomo": true_fold,
            "accepted_delta": true_fold["accepted"] - fixed_fold["accepted"],
            "train_calls_delta": true_fold["train_calls_used"]
            - fixed_fold["train_calls_used"],
            "extra_training_models": sorted(set(true_fold["train_models_used"])
                                            - set(fixed_fold["train_models_used"])),
        }

    result = {
        "experiment": "ToolGuard B7 true LOMO diagnostic",
        "protocol": {
            "scope": "diagnosis only; metrics.json / artifacts / TOOLGUARD.md are unchanged",
            "main_protocol": "fixed-three (headline 57.5%)",
            "sensitivity_protocol": "exclude-fold (true five-family LOMO, 59.7%)",
            "train_models_fixed_three": sorted(TRAIN_MODELS),
            "folds": families,
            "calls": len(calls),
            "schemas": schema_stats,
            "schema_fidelity_repair": {
                "type_corrections": schema_fidelity["type_corrections"],
                "params_added": schema_fidelity["params_added"],
            },
            "commands": [
                "python scripts/eval_toolguard_rules.py --lomo-train-policy exclude-fold "
                "--output .scratch/toolguard/b7/true_lomo.json",
                "python scripts/eval_toolguard_rules.py "
                "--output .scratch/toolguard/b7/fixed_three.json",
                "python scripts/experiment_b7_true_lomo.py",
            ],
        },
        "cross_checks": {
            "fixed_three_reproduces_headline": fixed_matches_headline,
            "true_lomo_reproduces_stored_sensitivity": true_matches_stored,
            "true_lomo_matches_evaluator_artifact": _match(true_lomo["aggregate"],
                                                           eval_true_agg),
            "fixed_three_matches_evaluator_artifact": _match(fixed["aggregate"],
                                                             eval_fixed_agg),
            "headline_from_metrics_json": {key: headline[key] for key in
                                           ("evaluable_failures", "proposed_repairs",
                                            "schema_passed", "value_verified_repairs",
                                            "coerced_repairs", "accepted_repairs",
                                            "repair_coverage_of_failures")},
            "stored_sensitivity_from_metrics_json": {key: stored_sensitivity[key]
                                                     for key in
                                                     ("evaluable_failures",
                                                      "proposed_repairs", "schema_passed",
                                                      "value_verified_repairs",
                                                      "coerced_repairs", "accepted_repairs",
                                                      "repair_coverage_of_failures")},
            "evaluator_artifact_true_lomo_aggregate": eval_true_agg,
            "evaluator_artifact_fixed_three_aggregate": eval_fixed_agg,
        },
        "fixed_three": fixed,
        "true_lomo": true_lomo,
        "gap": gap,
        "by_fold": by_fold,
        "limitations": [
            "true LOMO is a sensitivity upper bound: the gpt-4 families enter training for "
            "the fixed-three folds, which is exactly what the headline protocol forbids",
            "the difference between the two protocols is a training-universe difference, not "
            "a data/code difference: both runs use the same calls, schema index and engine "
            "defaults",
            "this diagnostic does not change any committed number; metrics.json and the 13 "
            "contract artifacts are untouched",
        ],
    }
    output_dir = repo_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "lomo_diagnostic.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    report = _render_report(result)
    (output_dir / "lomo_diagnostic.md").write_text(report, encoding="utf-8")
    print(report)
    print(f"\nwrote {output_dir / 'lomo_diagnostic.json'}")
    print(f"wrote {output_dir / 'lomo_diagnostic.md'}")
    ok = (fixed_matches_headline and true_matches_stored
          and _match(fixed["aggregate"], eval_fixed_agg))
    return 0 if ok else 1


def _render_report(result: dict) -> str:
    lines = ["# B7 true LOMO diagnostic (diagnosis only)\n"]
    checks = result["cross_checks"]
    lines.append("## Cross-checks\n")
    lines.append(_md_table(
        ["check", "result"],
        [["fixed-three reproduces the headline aggregate",
          checks["fixed_three_reproduces_headline"]],
         ["true LOMO reproduces the stored 578/969 sensitivity",
          checks["true_lomo_reproduces_stored_sensitivity"]],
         ["true LOMO matches the evaluator artifact",
          checks["true_lomo_matches_evaluator_artifact"]],
         ["fixed-three matches the evaluator artifact",
          checks["fixed_three_matches_evaluator_artifact"]]]))
    fixed = result["fixed_three"]
    true_lomo = result["true_lomo"]
    lines.append("\n## Aggregate: main protocol vs true LOMO\n")
    lines.append(_md_table(
        ["metric", "fixed-three (main, 57.5%)", "true LOMO (sensitivity, 59.7%)", "delta"],
        [["failures", fixed["aggregate"]["failures"], true_lomo["aggregate"]["failures"],
          true_lomo["aggregate"]["failures"] - fixed["aggregate"]["failures"]],
         ["proposed", fixed["aggregate"]["proposed"], true_lomo["aggregate"]["proposed"],
          true_lomo["aggregate"]["proposed"] - fixed["aggregate"]["proposed"]],
         ["schema-valid", fixed["aggregate"]["schema_passed"],
          true_lomo["aggregate"]["schema_passed"],
          true_lomo["aggregate"]["schema_passed"] - fixed["aggregate"]["schema_passed"]],
         ["value-verified", fixed["aggregate"]["value_verified"],
          true_lomo["aggregate"]["value_verified"],
          true_lomo["aggregate"]["value_verified"] - fixed["aggregate"]["value_verified"]],
         ["accepted", fixed["aggregate"]["accepted"], true_lomo["aggregate"]["accepted"],
          true_lomo["aggregate"]["accepted"] - fixed["aggregate"]["accepted"]],
         ["coverage", f"{100 * fixed['aggregate']['coverage']:.1f}%",
          f"{100 * true_lomo['aggregate']['coverage']:.1f}%",
          f"{result['gap']['coverage_delta_pp']:+.2f}pp"]]))
    lines.append("\n## Per fold\n")
    rows = []
    for family, data in result["by_fold"].items():
        fixed_fold, true_fold = data["fixed_three"], data["true_lomo"]
        rows.append([
            family,
            fixed_fold["train_calls_used"], true_fold["train_calls_used"],
            ", ".join(data["extra_training_models"]) or "-",
            fixed_fold["failures"], fixed_fold["proposed"],
            f"{fixed_fold['schema_passed']}/{fixed_fold['value_verified']}",
            fixed_fold["accepted"], f"{100 * fixed_fold['coverage']:.1f}%",
            true_fold["proposed"],
            f"{true_fold['schema_passed']}/{true_fold['value_verified']}",
            true_fold["accepted"], f"{100 * true_fold['coverage']:.1f}%",
            f"{data['accepted_delta']:+d}"])
    lines.append(_md_table(
        ["fold", "train calls (fixed)", "train calls (true LOMO)", "extra training models",
         "failures", "proposed (f)", "schema/vv (f)", "accepted (f)", "coverage (f)",
         "proposed (LOMO)", "schema/vv (LOMO)", "accepted (LOMO)", "coverage (LOMO)",
         "accepted delta"], rows))
    lines.append("\n## Per failure kind\n")
    for label, data in (("fixed-three (main)", fixed), ("true LOMO (sensitivity)", true_lomo)):
        lines.append(f"\n### {label}\n")
        kinds = data["per_kind"]
        lines.append(_md_table(
            ["kind", "failures", "proposed", "schema-valid", "value-verified", "accepted",
             "coverage"],
            [[kind, counts.get("failures", 0), counts.get("proposed", 0),
              counts.get("schema_passed", 0), counts.get("value_verified", 0),
              counts.get("accepted", 0),
              f"{100 * counts.get('accepted', 0) / max(counts.get('failures', 1), 1):.1f}%"]
             for kind, counts in sorted(kinds.items(),
                                        key=lambda item: -item[1].get("failures", 0))]))
    lines.append("\n## Why the two protocols differ\n")
    lines.append(result["gap"]["explanation"] + "\n")
    lines.append("\n## Limitations\n")
    for limitation in result["limitations"]:
        lines.append(f"- {limitation}")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
