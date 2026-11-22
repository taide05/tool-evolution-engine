"""Evaluate ToolGuard repairs on real recorded tool-call failures.

Two held-out protocols (both supported by the public trajectory release):

* ``cross_model``  -- success memory/rules are built from chatgpt/toolllama/
  gpt-3.5 traces and evaluated on gpt-4-0613 + gpt-4-turbo-preview failures.
* ``cross_group``  -- memory is built from G1/G2 traces and evaluated on the
  indepently worded G3 instruction set.

Deterministic rules also fill schema defaults; experience memory replays a
previously successful invocation of the same task/function.  Cache verification
is reported separately (the released response cache only covers reference inputs,
so ``cache_miss`` is expected for many repaired calls).

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
) -> dict:
    memory = SuccessMemory.build([c for c in train_calls if c.model in TRAIN_MODELS], schemas)
    failures = [c for c in test_calls if not c.recorded_ok]
    stats: Counter[str] = Counter()
    per_kind: dict[str, Counter] = defaultdict(Counter)
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
        result = repair_failure(call, schemas, memory)
        if not result.repairable:
            per_kind[kind]["unrepairable"] += 1
            continue
        stats["repairable"] += 1
        per_kind[kind]["repairable"] += 1
        if result.schema_valid:
            stats["schema_valid"] += 1
            per_kind[kind]["schema_valid"] += 1
        for rule in result.rules:
            rule_hits[rule.split(":")[0]] += 1
        examples.setdefault(kind, {
            "error": (call.error or "")[:160],
            "tool": call.tool_name,
            "rules": result.rules,
            "repaired_args": result.arguments,
        })
        if backend is not None:
            payload = backend.call(result.function_name or call.tool_name,
                                   result.arguments if result.arguments is not None else call.arguments)
            if payload.status == "miss":
                verified["cache_miss"] += 1
            elif payload.ok:
                verified["verified_recovered"] += 1
            else:
                verified["verified_still_error"] += 1

    return {
        "label": label,
        "train_calls": len(train_calls),
        "train_successes": memory.size(),
        "test_calls": len(test_calls),
        "test_failures": stats["failures"],
        "test_failure_rate": round(stats["failures"] / max(len(test_calls), 1), 4),
        "skipped_no_schema": stats["skipped_no_schema"],
        "transient_retryable": per_kind.get("server_error", {}).get("failures", 0),
        "repairable": stats["repairable"],
        "repair_coverage_of_failures": round(stats["repairable"] / max(stats["failures"], 1), 4),
        "repair_coverage_of_calls": round(stats["repairable"] / max(len(test_calls), 1), 4),
        "schema_valid": stats["schema_valid"],
        "schema_valid_rate": round(stats["schema_valid"] / max(stats["repairable"], 1), 4),
        "rule_hits": dict(rule_hits),
        "memory_replays": rule_hits.get("memory_replay_successful_call", 0)
        + rule_hits.get("memory_param_fill", 0)
        + rule_hits.get("memory_value_normalize", 0),
        "cache_verification": dict(verified),
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

    # Leave-one-model-out: every model family gets a turn as the held-out split.
    families = sorted({c.model for c in calls})
    lomo_folds = {}
    agg = Counter()
    for family in families:
        fold = evaluate(f"lomo:{family}",
                        [c for c in calls if c.model != family],
                        [c for c in calls if c.model == family], schemas, backend)
        lomo_folds[family] = {k: fold[k] for k in (
            "test_calls", "test_failures", "repairable", "repair_coverage_of_failures",
            "schema_valid_rate", "memory_replays", "cache_verification", "failures_by_kind")}
        agg["test_calls"] += fold["test_calls"]
        agg["failures"] += fold["test_failures"]
        agg["repairable"] += fold["repairable"]
        agg["schema_valid"] += fold["schema_valid"]
        for key, value in fold["cache_verification"].items():
            agg["cache_verified_" + key] += value
    results["lomo"] = {
        "folds": lomo_folds,
        "aggregate": {
            "test_calls": agg["test_calls"],
            "failures": agg["failures"],
            "repairable": agg["repairable"],
            "repair_coverage_of_failures": round(agg["repairable"] / max(agg["failures"], 1), 4),
            "schema_valid": agg["schema_valid"],
            "schema_valid_rate": round(agg["schema_valid"] / max(agg["repairable"], 1), 4),
            "cache_verified_recovered": agg["cache_verified_verified_recovered"],
            "cache_verified_still_error": agg["cache_verified_verified_still_error"],
            "cache_miss": agg["cache_verified_cache_miss"],
        },
    }
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk not in ("examples", "per_kind")}
                      for k, v in results.items() if k != "lomo"}, ensure_ascii=False, indent=2))
    print("\n[lomo aggregate]", json.dumps(results["lomo"]["aggregate"], ensure_ascii=False))
    for family, fold in lomo_folds.items():
        print(f"  {family:26s} failures={fold['test_failures']:4d} repairable={fold['repairable']:4d} "
              f"coverage={100 * fold['repair_coverage_of_failures']:5.1f}%")
    for label in ("cross_model", "cross_group"):
        result = results[label]
        print(f"\n[{label}] per-kind:")
        for kind, counts in result["per_kind"].items():
            print(f"  {kind:18s} failures={counts.get('failures', 0):4d} repairable={counts.get('repairable', 0):4d}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
