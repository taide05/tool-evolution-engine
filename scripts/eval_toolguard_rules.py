"""Evaluate ToolGuard repairs on real recorded tool-call failures.

Held-out protocols:
* ``cross_model`` -- rules/memory/extractors built from the fixed three-model training set
  (chatgpt/toolllama/gpt-3.5), evaluated on the two gpt-4 families.
* ``cross_group`` -- built from G1/G2 queries of the fixed three-model training set,
  evaluated on the independently worded G3 instruction set (the hard cross-task case).
* ``lomo``       -- **fixed three-model training set, evaluated one model family at a time**
  (the historical "leave-one-model-out" label; with the default ``fixed-three`` policy the
  training set is the same for every fold, so it is not a true leave-one-model-out).
  ``--lomo-train-policy exclude-fold`` runs the true leave-one-model-out sensitivity
  version, which trains on the remaining four families for each fold.

Evidence beyond "we proposed a repair" (two honest evidence tiers):
* **schema-valid**: the repaired call passes full structural schema validation (types,
  enums, formats, ranges, required/extra parameters).  ``schema_valid_rate`` is
  schema-passed / proposed (never passed / kept).  Deterministic type coercion of proposed
  values to the declared scalar types is attempted first and reported separately
  (``coerced_repairs``);
* **value-verified**: a proposal that fails the declared schema is still admitted when it is
  an *exact* recorded successful invocation of the same task/function (empirical evidence,
  not schema validity; reported separately as ``value_verified_repairs``).  Accepted repair
  coverage = schema-passed + value-verified; the two are never mixed in one rate;
* cache verification against the released real-response cache (subset only);
* value agreement with a recorded successful call of the same task/function,
  which validates that extracted parameter values are the ones that actually
  worked (available whenever any model solved that task instance).

Usage:
    python scripts/eval_toolguard_rules.py
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
from tool_evolution.toolguard.records import ToolCall, load_calls  # noqa: E402
from tool_evolution.toolguard.rules import classify_error  # noqa: E402
from tool_evolution.toolguard.schemas import build_schema_index  # noqa: E402

TRAIN_MODELS = {"chatgpt", "toolllama", "gpt-3.5-turbo-1106"}
TEST_MODELS = {"gpt-4-0613", "gpt-4-turbo-preview"}
_SENTINEL_CALLS = {"invalid_hallucination_function_name"}


def evaluate(
    label: str,
    train_calls: list[ToolCall],
    test_calls: list[ToolCall],
    schemas,
    backend: RecordedBackend | None,
    train_models: set[str] | None = TRAIN_MODELS,
) -> dict:
    """Evaluate the repair ladder on held-out calls.

    ``train_models`` is the training-set policy: ``TRAIN_MODELS`` means "train only on the
    fixed three-model training set" (the historical protocol), ``None`` means "use every
    training call" (true leave-one-model-out when the caller already excluded the fold).
    The policy is written into the result so the artifact can state it honestly.
    """
    valid_train = [c for c in train_calls if train_models is None or c.model in train_models]
    memory = SuccessMemory.build(valid_train, schemas)
    extractor_map = ExtractorMap.learn(valid_train, schemas)

    # value-agreement reference: recorded successful calls for the same task/function
    successes: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for call in test_calls:
        if call.recorded_ok:
            successes[(call.group, call.query_id, call.tool_name)].append(call.arguments)

    failures = [c for c in test_calls if not c.recorded_ok]
    stats: Counter[str] = Counter()
    per_kind: dict[str, Counter] = defaultdict(Counter)
    per_category: dict[str, Counter] = defaultdict(Counter)
    rule_hits: Counter[str] = Counter()
    verified: Counter[str] = Counter()
    examples: dict[str, dict] = {}

    for call in failures:
        if "__raw__" in call.arguments or call.tool_name not in schemas:
            stats["skipped_no_schema"] += 1
            continue
        kind = classify_error(call.error)
        stats["failures"] += 1
        per_kind[kind]["failures"] += 1
        reference = successes.get((call.group, call.query_id, call.tool_name)) or []
        result = repair_failure(call, schemas, memory, extractor_map,
                                successful_arguments=reference)
        if result.proposed:
            stats["proposed_repairs"] += 1
            per_kind[kind]["proposed"] += 1
            rejected = len(result.rejected_proposals)
            stats["schema_rejected_proposals"] += rejected
            per_kind[kind]["schema_rejected_proposals"] += rejected

        if not result.repairable:
            per_kind[kind]["unrepairable"] += 1
            if result.proposed:
                per_kind[kind]["rejected_failures"] += 1
            continue

        # two evidence tiers, never mixed into one rate:
        #   tier 1 = declared-schema-valid (coercion is reported separately)
        #   tier 2 = exact recorded successful invocation of the same task/function
        if result.schema_valid:
            tier = "schema_valid"
            stats["schema_passed"] += 1
            per_kind[kind]["schema_passed"] += 1
        else:
            tier = "value_verified"
            stats["value_verified_repairs"] += 1
            per_kind[kind]["value_verified"] += 1
        accepted_rules = list(result.rules)
        if "schema_type_coercion" in accepted_rules:
            stats["coerced_repairs"] += 1
        repaired_function = result.function_name or call.tool_name
        repaired_arguments = result.arguments

        category = accepted_rules[0].split(":")[0] if accepted_rules else "unknown"
        stats["accepted_repairs"] += 1
        stats["rule_" + category] += 1
        per_category[category]["repairs"] += 1
        per_category[category][tier] += 1
        for rule in accepted_rules:
            rule_hits[rule.split(":")[0]] += 1
        examples.setdefault(kind, {
            "error": (call.error or "")[:160],
            "tool": call.tool_name,
            "tier": tier,
            "rules": accepted_rules,
            "repaired_args": repaired_arguments,
            "query_head": call.query[:120],
        })

        # verification 1: recorded successful invocation of the same task/function
        reference = successes.get((call.group, call.query_id, call.tool_name)) or []
        if reference and repaired_arguments is not None:
            changed = {name: value for name, value in repaired_arguments.items()
                       if call.arguments.get(name) != value}
            if changed:
                stats["value_verifiable"] += 1
                per_category[category]["value_verifiable"] += 1
                if any(all(ref.get(name) == value for name, value in changed.items()) for ref in reference):
                    stats["value_agreement"] += 1
                    per_category[category]["value_agreement"] += 1

        # verification 2: released real-response cache
        if backend is not None:
            payload = backend.call(repaired_function, repaired_arguments
                                   if repaired_arguments is not None else call.arguments)
            if payload.status == "miss":
                verified["cache_miss"] += 1
                per_category[category]["cache_miss"] += 1
            elif payload.ok:
                verified["verified_recovered"] += 1
                per_category[category]["cache_verified_recovered"] += 1
            else:
                verified["verified_still_error"] += 1
                per_category[category]["cache_verified_still_error"] += 1

    accepted = stats["accepted_repairs"]
    return {
        "label": label,
        "train_models_used": sorted({c.model for c in valid_train}),
        "train_calls_used": len(valid_train),
        "train_calls": len(train_calls),
        "train_successes": memory.size(),
        "extractor_map": extractor_map.size(),
        "test_calls": len(test_calls),
        "test_failures": stats["failures"],
        "test_failure_rate": round(stats["failures"] / max(len(test_calls), 1), 4),
        "skipped_no_schema": stats["skipped_no_schema"],
        "transient_retryable": per_kind.get("server_error", {}).get("failures", 0),
        # honest two-tier evidence funnel:
        #   proposed = schema_passed + schema_rejected_failures
        #   accepted = schema_passed + value_verified_repairs (empirically verified tier)
        #   schema_valid_rate stays < 1; it is NOT passed/kept.
        "proposed_repairs": stats["proposed_repairs"],
        "schema_passed": stats["schema_passed"],
        "schema_rejected_proposals": stats["schema_rejected_proposals"],
        "schema_rejected_failures": stats["proposed_repairs"] - stats["schema_passed"],
        "value_verified_repairs": stats["value_verified_repairs"],
        "coerced_repairs": stats["coerced_repairs"],
        "accepted_repairs": accepted,
        "repairable": accepted,  # legacy alias for the accepted count
        "schema_valid_rate": round(stats["schema_passed"] / max(stats["proposed_repairs"], 1), 4),
        "value_verified_rate": round(stats["value_verified_repairs"]
                                     / max(stats["proposed_repairs"], 1), 4),
        "accepted_rate": round(accepted / max(stats["proposed_repairs"], 1), 4),
        "repair_coverage_of_failures": round(accepted / max(stats["failures"], 1), 4),
        "repair_coverage_of_calls": round(accepted / max(len(test_calls), 1), 4),
        "rule_categories": {key: value for key, value in stats.items() if key.startswith("rule_")},
        "rule_hits": dict(rule_hits),
        "value_verifiable": stats["value_verifiable"],
        "value_agreement": stats["value_agreement"],
        "value_agreement_rate": round(stats["value_agreement"] / max(stats["value_verifiable"], 1), 4),
        "cache_verification": dict(verified),
        "per_category": {k: dict(v) for k, v in per_category.items()},
        "failures_by_kind": dict(Counter(classify_error(c.error) for c in failures)),
        "per_kind": {k: dict(v) for k, v in per_kind.items()},
        "examples": examples,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calls", default=".scratch/toolguard/tool_calls.jsonl")
    parser.add_argument("--queries", default=".scratch/stb/repo/intuit-ai-research-StableToolBench-74f76c1/solvable_queries/test_instruction")
    parser.add_argument("--cache", default=".scratch/stb/cache/tool_response_cache")
    parser.add_argument("--output", default=".scratch/toolguard/rules_eval.json")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--lomo-train-policy", choices=("fixed-three", "exclude-fold"),
                        default="fixed-three",
                        help="fixed-three: train only on TRAIN_MODELS (historical protocol); "
                             "exclude-fold: train on every call outside the held-out fold "
                             "(true leave-one-model-out)")
    args = parser.parse_args()

    calls = [c for c in load_calls(Path(args.calls)) if c.tool_name not in _SENTINEL_CALLS]
    schemas, schema_stats = build_schema_index(Path(args.queries))
    print(f"calls: {len(calls)} | schemas: {len(schemas)} ({schema_stats})")

    backend = None
    if not args.no_cache and Path(args.cache).is_dir():
        backend = RecordedBackend(Path(args.cache), CacheIndex.from_root(Path(args.cache)))

    train_models = [c for c in calls if c.model in TRAIN_MODELS]
    test_models = [c for c in calls if c.model in TEST_MODELS]
    train_groups = [c for c in calls if c.group in ("G1_category", "G1_instruction", "G1_tool", "G2_category", "G2_instruction")]
    test_group = [c for c in calls if c.group == "G3_instruction"]

    results = {
        "cross_model": evaluate("cross_model", train_models, test_models, schemas, backend),
        "cross_group": evaluate("cross_group", train_groups, test_group, schemas, backend),
    }

    families = sorted({c.model for c in calls})
    lomo_train_models = TRAIN_MODELS if args.lomo_train_policy == "fixed-three" else None
    lomo_folds = {}
    agg = Counter()
    for family in families:
        fold = evaluate(f"lomo:{family}",
                        [c for c in calls if c.model != family],
                        [c for c in calls if c.model == family], schemas, backend,
                        train_models=lomo_train_models)
        lomo_folds[family] = {k: fold[k] for k in (
            "test_calls", "test_failures", "proposed_repairs", "schema_rejected_proposals",
            "schema_passed", "value_verified_repairs", "coerced_repairs", "accepted_repairs",
            "repair_coverage_of_failures", "schema_valid_rate", "value_verified_rate",
            "accepted_rate", "value_agreement_rate", "cache_verification",
            "failures_by_kind", "train_models_used", "train_calls_used")}
        agg["test_calls"] += fold["test_calls"]
        agg["failures"] += fold["test_failures"]
        agg["proposed_repairs"] += fold["proposed_repairs"]
        agg["schema_rejected_proposals"] += fold["schema_rejected_proposals"]
        agg["schema_passed"] += fold["schema_passed"]
        agg["value_verified_repairs"] += fold["value_verified_repairs"]
        agg["coerced_repairs"] += fold["coerced_repairs"]
        agg["accepted_repairs"] += fold["accepted_repairs"]
        agg["value_verifiable"] += fold["value_verifiable"]
        agg["value_agreement"] += fold["value_agreement"]
        for key, value in fold["cache_verification"].items():
            agg["cache_verified_" + key] += value
    results["lomo"] = {
        "train_policy": args.lomo_train_policy,
        "folds": lomo_folds,
        "aggregate": {
            "test_calls": agg["test_calls"],
            "failures": agg["failures"],
            "proposed_repairs": agg["proposed_repairs"],
            "schema_passed": agg["schema_passed"],
            "schema_rejected_proposals": agg["schema_rejected_proposals"],
            "schema_rejected_failures": agg["proposed_repairs"] - agg["schema_passed"],
            "value_verified_repairs": agg["value_verified_repairs"],
            "coerced_repairs": agg["coerced_repairs"],
            "accepted_repairs": agg["accepted_repairs"],
            "repairable": agg["accepted_repairs"],
            "schema_valid": agg["schema_passed"],
            "repair_coverage_of_failures": round(agg["accepted_repairs"] / max(agg["failures"], 1), 4),
            "schema_valid_rate": round(agg["schema_passed"] / max(agg["proposed_repairs"], 1), 4),
            "value_verified_rate": round(agg["value_verified_repairs"]
                                         / max(agg["proposed_repairs"], 1), 4),
            "accepted_rate": round(agg["accepted_repairs"] / max(agg["proposed_repairs"], 1), 4),
            "value_verifiable": agg["value_verifiable"],
            "value_agreement": agg["value_agreement"],
            "value_agreement_rate": round(agg["value_agreement"] / max(agg["value_verifiable"], 1), 4),
            "cache_verified_recovered": agg["cache_verified_verified_recovered"],
            "cache_verified_still_error": agg["cache_verified_verified_still_error"],
            "cache_miss": agg["cache_verified_cache_miss"],
        },
    }

    compact = {k: {kk: vv for kk, vv in v.items() if kk not in ("examples", "per_kind")}
               for k, v in results.items() if k != "lomo"}
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    print("\n[lomo aggregate]", json.dumps(results["lomo"]["aggregate"], ensure_ascii=False))
    for family, fold in lomo_folds.items():
        print(f"  {family:26s} failures={fold['test_failures']:4d} "
              f"schema_passed={fold['schema_passed']:4d} "
              f"value_verified={fold['value_verified_repairs']:4d} "
              f"accepted={fold['accepted_repairs']:4d} "
              f"coverage={100 * fold['repair_coverage_of_failures']:5.1f}% "
              f"value_agree={100 * fold.get('value_agreement_rate', 0):5.1f}%")
    for label in ("cross_model", "cross_group"):
        result = results[label]
        print(f"\n[{label}] per-kind:")
        for kind, counts in result["per_kind"].items():
            print(f"  {kind:18s} failures={counts.get('failures', 0):4d} "
                  f"schema_passed={counts.get('schema_passed', 0):4d} "
                  f"value_verified={counts.get('value_verified', 0):4d} "
                  f"unrepairable={counts.get('unrepairable', 0):4d}")
        print(f"[{label}] per-rule-category:")
        for category, counts in result["per_category"].items():
            verifiable = counts.get("value_verifiable", 0)
            print(f"  {category:32s} repairs={counts.get('repairs', 0):4d} "
                  f"schema_valid={counts.get('schema_valid', 0):4d} "
                  f"value_verified={counts.get('value_verified', 0):4d} "
                  f"value_verifiable={verifiable:4d} agreement={counts.get('value_agreement', 0):4d} "
                  f"cache_verified={counts.get('cache_verified_recovered', 0):3d}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
