"""Recompute ToolGuard artifact aggregates from their raw per-cell records.

Every committed artifact under ``benchmarks/toolguard/`` stores the raw per-cell records
*and* a pre-aggregated ``aggregate`` block.  The aggregate is what ``TOOLGUARD.md`` quotes,
so it counts as evidence only if it can be re-derived from the records.  This module is the
single source of truth for that derivation: ``artifact_contracts.py`` uses it to validate
the committed files, and ``scripts/toolguard_evidence_summary.py`` uses it to print the
reviewer-facing numbers.  The two therefore cannot drift apart.

Contract: the ``recompute_*`` functions read **only** raw records (plus, for the multi-model
artifact, the calibration block that anchors capability placement and the price table from
the protocol block); they never read a success rate, token count, turn count, Wilson
interval, McNemar p-value or any other derived field out of an ``aggregate`` block.

The recomputations mirror the aggregation code in ``scripts/run_live_demo_hard_v2.py`` /
``scripts/run_mcp_demo_v2.py`` / ``scripts/run_multimodel_live.py`` exactly (same rounding,
same pairing key, same Wilson/McNemar helpers), so a matching artifact produces an exact
match rather than an approximate one.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

from .verify import mcnemar_exact, wilson_interval

ARMS = ("baseline", "guard")
USAGE_KEYS = ("prompt_tokens", "completion_tokens", "total_tokens")


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------
def _successes(rows: Iterable[dict]) -> int:
    return sum(1 for row in rows if row.get("success"))


def _rate(success: int, cells: int) -> float:
    return round(success / max(cells, 1), 4)


def _sum_field(rows: Iterable[dict], field: str) -> float:
    return sum(row.get(field) or 0 for row in rows)


def _usage_totals(rows: Iterable[dict]) -> dict[str, int]:
    totals = {key: 0 for key in USAGE_KEYS}
    for row in rows:
        usage = row.get("usage") or {}
        for key in USAGE_KEYS:
            totals[key] += int(usage.get(key) or 0)
    return totals


def _fault_kinds(rows: Iterable[dict]) -> dict[str, int]:
    kinds: Counter[str] = Counter()
    for row in rows:
        for kind, count in (row.get("fault_kinds") or {}).items():
            kinds[kind] += int(count)
    return dict(kinds)


def pairing_key(rows: list[dict]) -> tuple[str, ...]:
    """Key that pairs a baseline cell with its guard cell.

    The V2/V3 runners repeat tight-budget cells with several fault seeds and pair on
    ``(task, repeat)``; the V1 artifacts have no repeat axis and pair on ``task``.
    """
    if rows and all("repeat" in row for row in rows):
        return ("task", "repeat")
    return ("task",)


def recompute_arm(rows: list[dict]) -> dict[str, Any]:
    """Recompute one arm's summary from its raw cell records."""
    rows = list(rows)
    cells = len(rows)
    successes = _successes(rows)
    turns = int(_sum_field(rows, "turns"))
    return {
        "cells": cells,
        "success": successes,
        "success_rate": _rate(successes, cells),
        "success_wilson_95": wilson_interval(successes, cells),
        "turns": turns,
        "turns_per_cell": round(turns / max(cells, 1), 3),
        "tool_calls": int(_sum_field(rows, "tool_calls")),
        "tool_errors": int(_sum_field(rows, "tool_errors")),
        "exhausted": int(_sum_field(rows, "exhausted")),
        "faults_injected": int(_sum_field(rows, "faults_injected")),
        "fault_kinds": _fault_kinds(rows),
        "guard_recoveries": int(_sum_field(rows, "guard_recoveries")),
        "guard_retry_sleep_ms": round(float(_sum_field(rows, "guard_retry_sleep_ms")), 2),
        "mcp_calls": int(_sum_field(rows, "mcp_calls")),
        "mcp_errors": int(_sum_field(rows, "mcp_errors")),
        "mcp_restarts": int(_sum_field(rows, "mcp_restarts")),
        "mcp_server_calls": int(_sum_field(rows, "mcp_server_calls")),
        "mcp_server_errors": int(_sum_field(rows, "mcp_server_errors")),
        "recoveries": int(_sum_field(rows, "guard_recoveries")),
        "usage": _usage_totals(rows),
        "elapsed_ms": round(float(_sum_field(rows, "elapsed_ms")), 2),
    }


def recompute_paired(rows: list[dict]) -> dict[str, Any]:
    """Recompute paired-cell counts and the exact McNemar p-value from records."""
    if not rows:
        return {"both_success": 0, "baseline_only": 0, "guard_only": 0, "neither": 0,
                "mcnemar_exact_p": 1.0}
    key_fields = pairing_key(rows)
    cells: dict[tuple, dict[str, bool]] = {}
    for row in rows:
        key = tuple(row.get(field) for field in key_fields)
        cells.setdefault(key, {})[row["arm"]] = bool(row["success"])
    both = sum(1 for pair in cells.values() if pair.get("baseline") and pair.get("guard"))
    baseline_only = sum(1 for pair in cells.values() if pair.get("baseline") and not pair.get("guard"))
    guard_only = sum(1 for pair in cells.values() if pair.get("guard") and not pair.get("baseline"))
    return {
        "both_success": both,
        "baseline_only": baseline_only,
        "guard_only": guard_only,
        "neither": len(cells) - both - baseline_only - guard_only,
        "mcnemar_exact_p": mcnemar_exact(baseline_only, guard_only),
    }


def _per_category(rows: list[dict]) -> dict[str, dict[str, list[int]]]:
    out: dict[str, dict[str, list[int]]] = {}
    for row in rows:
        category = row.get("category") or "unknown"
        entry = out.setdefault(category, {arm: [0, 0] for arm in ARMS})
        if row["arm"] not in entry:
            entry[row["arm"]] = [0, 0]
        entry[row["arm"]][1] += 1
        entry[row["arm"]][0] += int(bool(row["success"]))
    return out


def _failure_modes(rows: list[dict]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for row in rows:
        if row["success"]:
            continue
        mode = row.get("failure_mode") or "unknown"
        out.setdefault(row["arm"], {})
        out[row["arm"]][mode] = out[row["arm"]].get(mode, 0) + 1
    return out


def recompute_condition(rows: list[dict]) -> dict[str, Any]:
    """Recompute one budget condition: arms, pairing, budget mean, failure modes."""
    budgets = [row["budget"] for row in rows if row.get("budget") is not None]
    block: dict[str, Any] = {
        "arms": {arm: recompute_arm([row for row in rows if row["arm"] == arm]) for arm in ARMS},
        "paired_cells": recompute_paired(rows),
        "per_category": _per_category(rows),
        "failure_modes": _failure_modes(rows),
    }
    if budgets:
        block["mean_budget"] = round(sum(budgets) / len(budgets), 2)
    return block


def recompute_conditions(records: list[dict], with_budget_curve: bool = False) -> dict[str, Any]:
    """Recompute the whole ``aggregate`` block of a records-based live artifact."""
    conditions = list(dict.fromkeys(row.get("condition") or "default" for row in records))
    block: dict[str, Any] = {"conditions": {}}
    for condition in conditions:
        rows = [row for row in records
                if (row.get("condition") or "default") == condition]
        block["conditions"][condition] = recompute_condition(rows)
    if with_budget_curve:
        curve = []
        for condition in conditions:
            summary = block["conditions"][condition]
            curve.append({
                "condition": condition,
                "mean_budget": summary.get("mean_budget"),
                "baseline_success_rate": summary["arms"]["baseline"]["success_rate"],
                "guard_success_rate": summary["arms"]["guard"]["success_rate"],
                "baseline_cells": summary["arms"]["baseline"]["cells"],
                "guard_cells": summary["arms"]["guard"]["cells"],
            })
        block["budget_curve"] = curve
    return block


def recompute_guard_repairs_by_rule(records: list[dict]) -> dict[str, int]:
    """Recompute the guard's per-rule event counts from the recorded guard events."""
    rules: Counter[str] = Counter()
    for row in records:
        for event in row.get("guard_events") or []:
            rule = str(event.get("rule") or "unknown").split(":")[0]
            rules[rule] += 1
    return dict(rules)


# ---------------------------------------------------------------------------
# multi-model artifact
# ---------------------------------------------------------------------------
def _estimate_cost_raw(usage: dict[str, int], price_input_per_m: float,
                       price_output_per_m: float) -> float:
    return (usage["prompt_tokens"] / 1e6 * price_input_per_m
            + usage["completion_tokens"] / 1e6 * price_output_per_m)


def _estimate_cost(usage: dict[str, int], price_input_per_m: float,
                   price_output_per_m: float) -> float:
    return round(_estimate_cost_raw(usage, price_input_per_m, price_output_per_m), 4)


def _placement(calibration_model: dict, prices: dict[str, float]) -> dict[str, Any]:
    tasks = (calibration_model or {}).get("tasks") or {}
    workers = list(tasks.values())
    turns = [int(task.get("turns") or 0) for task in workers]
    successes = sum(1 for task in workers if task.get("success"))
    usage = _usage_totals(workers)
    return {
        "success": f"{successes}/{len(workers)}",
        "success_rate": round(successes / max(len(workers), 1), 4),
        "avg_turns": round(sum(turns) / max(len(turns), 1), 3),
        "max_turns": max(turns) if turns else 0,
        "usage": usage,
        "estimated_cost_usd": _estimate_cost(
            usage, prices["price_input_per_m"], prices["price_output_per_m"]),
    }


def recompute_multimodel(records: list[dict], calibration: dict, protocol_models: list[dict],
                         ) -> dict[str, Any]:
    """Recompute ``aggregate.cross_model`` and the total cost from records + calibration."""
    prices = {spec["key"]: spec for spec in protocol_models}
    model_keys = list(dict.fromkeys(row["model_key"] for row in records))
    cross: dict[str, Any] = {}
    total_cost = 0.0
    for key in model_keys:
        model_rows = [row for row in records if row["model_key"] == key]
        spec = prices.get(key) or {}
        calibration_model = (calibration.get("models") or {}).get(key) or {}
        condition_names = list(dict.fromkeys(row.get("condition") or "default"
                                             for row in model_rows))
        per_condition: dict[str, dict[str, Any]] = {}
        for condition in condition_names:
            rows = [row for row in model_rows
                    if (row.get("condition") or "default") == condition]
            baseline = recompute_arm([row for row in rows if row["arm"] == "baseline"])
            guard = recompute_arm([row for row in rows if row["arm"] == "guard"])
            paired = recompute_paired(rows)
            per_condition[condition] = {
                "baseline_success": baseline["success"],
                "guard_success": guard["success"],
                "cells": baseline["cells"],
                "baseline_success_rate": baseline["success_rate"],
                "guard_success_rate": guard["success_rate"],
                "guard_success_wilson_95": guard["success_wilson_95"],
                "mcnemar_exact_p": paired["mcnemar_exact_p"],
                "baseline_turns_per_cell": baseline["turns_per_cell"],
                "guard_turns_per_cell": guard["turns_per_cell"],
                "baseline_tokens": baseline["usage"]["total_tokens"],
                "guard_tokens": guard["usage"]["total_tokens"],
                "baseline_tool_errors": baseline["tool_errors"],
                "guard_tool_errors": guard["tool_errors"],
            }
        usage = _usage_totals(model_rows)
        total_cost += _estimate_cost_raw(usage, spec.get("price_input_per_m", 0.0),
                                         spec.get("price_output_per_m", 0.0))
        cross[key] = {
            "tier": spec.get("tier"),
            "label": spec.get("label"),
            "model": spec.get("model"),
            "provider": spec.get("provider"),
            "capability_placement": _placement(calibration_model, spec),
            "conditions": per_condition,
            "guard_never_worse": all(
                row["guard_success"] >= row["baseline_success"]
                for row in per_condition.values()),
            "estimated_cost_usd": _estimate_cost(
                usage, spec.get("price_input_per_m", 0.0), spec.get("price_output_per_m", 0.0)),
        }
    cross["consistency"] = {
        "guard_success_ge_baseline_on_all_models": all(
            row["guard_never_worse"] for row in cross.values()),
        "tight_budget_guard_success_rates": {
            key: row["conditions"].get("clean+1", {}).get("guard_success_rate")
            for key, row in cross.items() if key != "consistency"},
        "tight_budget_baseline_success_rates": {
            key: row["conditions"].get("clean+1", {}).get("baseline_success_rate")
            for key, row in cross.items() if key != "consistency"},
    }
    return {"cross_model": cross, "total_estimated_cost_usd": round(total_cost, 4),
            "model_usage": {key: _usage_totals([row for row in records if row["model_key"] == key])
                            for key in model_keys}}


# ---------------------------------------------------------------------------
# V1 artifacts (no repeat axis; live_demo.json only keeps per_task, no raw records)
# ---------------------------------------------------------------------------
def recompute_v1_live(results: dict) -> dict[str, Any]:
    """Recompute what the V1 live artifact keeps at task granularity (per_task)."""
    out: dict[str, Any] = {}
    for condition, block in results.items():
        per_task = block.get("per_task") or {}
        arms: dict[str, dict[str, Any]] = {}
        for arm in ARMS:
            successes = sum(int(bool((row.get(arm) or {}).get("success"))) for row in per_task.values())
            tool_calls = sum(int((row.get(arm) or {}).get("tool_calls") or 0) for row in per_task.values())
            tool_errors = sum(int((row.get(arm) or {}).get("tool_errors") or 0) for row in per_task.values())
            recoveries = sum(int((row.get(arm) or {}).get("recoveries") or 0) for row in per_task.values())
            arms[arm] = {
                "tasks": len(per_task),
                "success": successes,
                "success_rate": round(successes / max(len(per_task), 1), 4),
                "tool_calls": tool_calls,
                "tool_errors": tool_errors,
                "recoveries": recoveries,
            }
        cells: dict[str, dict[str, bool]] = {
            task_id: {arm: bool((row.get(arm) or {}).get("success")) for arm in ARMS}
            for task_id, row in per_task.items()}
        both = sum(1 for pair in cells.values() if pair["baseline"] and pair["guard"])
        baseline_only = sum(1 for pair in cells.values() if pair["baseline"] and not pair["guard"])
        guard_only = sum(1 for pair in cells.values() if pair["guard"] and not pair["baseline"])
        out[condition] = {
            "tasks": len(per_task),
            "arms": arms,
            "paired": {
                "both_success": both,
                "baseline_only": baseline_only,
                "guard_only": guard_only,
                "neither": len(cells) - both - baseline_only - guard_only,
                "mcnemar_exact_p": mcnemar_exact(baseline_only, guard_only),
            },
        }
    return out


def recompute_v1_mcp(records: list[dict]) -> dict[str, Any]:
    """Recompute the V1 MCP artifact aggregate (single condition, task-paired)."""
    return {
        "arms": {arm: recompute_arm([row for row in records if row["arm"] == arm]) for arm in ARMS},
        "paired": recompute_paired(records),
        "guard_repairs_by_rule": recompute_guard_repairs_by_rule(records),
    }


# ---------------------------------------------------------------------------
# offline metrics.json: no committed records, so recompute internal derivations
# ---------------------------------------------------------------------------
def _close(actual: Any, expected: float, tolerance: float = 1e-4) -> bool:
    return isinstance(actual, (int, float)) and not isinstance(actual, bool) \
        and abs(float(actual) - float(expected)) <= tolerance


def check_offline_metrics(data: dict, name: str = "metrics.json") -> list[str]:
    """Validate every derivable number inside the offline metrics artifact.

    The raw StableToolBench records are not committed (license/size), so the counts
    themselves cannot be re-derived here; everything that *is* a function of those counts
    (coverage, schema funnel, agreement rate, cache sums, dataset arithmetic, the
    skipped_no_schema reconciliation) is re-derived and compared, and the artifact must
    state the exclusions.
    """
    problems: list[str] = []
    dataset = data.get("dataset") or {}
    results = data.get("results") or {}
    protocol = data.get("protocol") or {}

    raw = dataset.get("tool_calls_raw")
    excluded = dataset.get("dfs_sentinel_calls_excluded")
    evaluated = dataset.get("tool_calls_evaluated")
    if isinstance(raw, int) and isinstance(excluded, int) and isinstance(evaluated, int):
        if raw - excluded != evaluated:
            problems.append(
                f"{name}: dataset arithmetic: tool_calls_raw - dfs_sentinel_calls_excluded "
                f"= {raw - excluded} != tool_calls_evaluated {evaluated}")
    failures = dataset.get("real_failures")
    if isinstance(failures, int) and isinstance(evaluated, int) and evaluated:
        if not _close(dataset.get("failure_rate"), failures / evaluated):
            problems.append(f"{name}: dataset failure_rate != real_failures/evaluated_calls")
    kinds = dataset.get("failures_by_kind")
    if isinstance(kinds, dict) and isinstance(failures, int):
        if sum(v for v in kinds.values() if isinstance(v, int)) != failures:
            problems.append(f"{name}: failures_by_kind does not sum to real_failures")
    if isinstance(dataset.get("unique_traces_with_calls"), int) and isinstance(
            dataset.get("trajectories_parsed"), int):
        if dataset["unique_traces_with_calls"] > dataset["trajectories_parsed"]:
            problems.append(f"{name}: unique_traces_with_calls > trajectories_parsed")

    def check_protocol(block: dict, label: str) -> None:
        proposed = block.get("proposed_repairs")
        passed = block.get("schema_passed")
        rejected = block.get("schema_rejected_failures")
        value_verified = block.get("value_verified_repairs")
        accepted = block.get("accepted_repairs", block.get("repairable"))
        evaluable = block.get("evaluable_failures", block.get("test_failures"))
        if all(isinstance(value, int) for value in (proposed, passed, rejected)):
            if proposed != passed + rejected:
                problems.append(
                    f"{name}: {label} evidence funnel: proposed {proposed} != schema_passed "
                    f"{passed} + schema_rejected_failures {rejected}")
        if all(isinstance(value, int) for value in (passed, value_verified, accepted)):
            if accepted != passed + value_verified:
                problems.append(
                    f"{name}: {label} accepted {accepted} != schema_passed {passed} + "
                    f"value_verified {value_verified}")
        coerced = block.get("coerced_repairs")
        if isinstance(coerced, int) and isinstance(passed, int) and coerced > passed:
            problems.append(
                f"{name}: {label} coerced_repairs {coerced} > schema_passed {passed}")
        if isinstance(proposed, int) and proposed:
            if not _close(block.get("schema_valid_rate"), (passed or 0) / proposed):
                problems.append(f"{name}: {label} schema_valid_rate != schema_passed/proposed")
            if not _close(block.get("value_verified_rate"), (value_verified or 0) / proposed):
                problems.append(f"{name}: {label} value_verified_rate != value_verified/proposed")
            if not _close(block.get("accepted_rate"), (accepted or 0) / proposed):
                problems.append(f"{name}: {label} accepted_rate != accepted/proposed")
        if isinstance(evaluable, int) and evaluable:
            if not _close(block.get("repair_coverage_of_failures"),
                          (accepted or 0) / evaluable):
                problems.append(
                    f"{name}: {label} repair coverage != accepted/evaluable_failures")
        verifiable = block.get("value_verifiable")
        if isinstance(verifiable, int) and verifiable:
            if not _close(block.get("value_agreement_rate"),
                          (block.get("value_agreement") or 0) / verifiable):
                problems.append(f"{name}: {label} value_agreement_rate mismatch")
        cache_total = sum(block.get(key) or 0 for key in
                          ("cache_verified_recovered", "cache_verified_still_error", "cache_miss"))
        cache_dict = block.get("cache_verification")
        if isinstance(cache_dict, dict) and cache_dict:
            cache_total = sum(v for v in cache_dict.values() if isinstance(v, int))
        has_cache_fields = any(key in block for key in
                               ("cache_verified_recovered", "cache_miss", "cache_verification"))
        if isinstance(accepted, int) and has_cache_fields and cache_total != accepted:
            problems.append(
                f"{name}: {label} cache buckets sum to {cache_total}, expected accepted repairs "
                f"{accepted}")
        per_rule = block.get("per_rule_category") or {}
        if per_rule and isinstance(accepted, int):
            if sum((row.get("repairs") or 0) for row in per_rule.values()) != accepted:
                problems.append(
                    f"{name}: {label} per_rule_category repairs do not sum to accepted repairs")
            agreement = sum((row.get("value_agreement") or 0) for row in per_rule.values())
            if agreement != block.get("value_agreement"):
                problems.append(
                    f"{name}: {label} per_rule_category agreement does not sum to aggregate")
        protocol_skipped = block.get("skipped_no_schema")
        protocol_kinds = block.get("failures_by_kind")
        if isinstance(protocol_skipped, int) and isinstance(protocol_kinds, dict) \
                and isinstance(block.get("test_failures"), int):
            if sum(v for v in protocol_kinds.values() if isinstance(v, int)) != (
                    block["test_failures"] + protocol_skipped):
                problems.append(
                    f"{name}: {label} failures_by_kind != test_failures + skipped_no_schema")

    lomo = results.get("lomo_aggregate") or {}
    if lomo:
        check_protocol(lomo, "LOMO")
        if lomo.get("accepted_repairs", 0) > lomo.get("evaluable_failures", 0):
            problems.append(f"{name}: LOMO accepted repairs > evaluable_failures")
    for label in ("cross_model", "cross_group"):
        block = results.get(label) or {}
        if block:
            check_protocol(block, label)
            if block.get("accepted_repairs", 0) > block.get("test_failures", 0):
                problems.append(f"{name}: {label} accepted repairs > test_failures")

    folds = results.get("lomo_per_fold") or {}
    families = dataset.get("model_families")
    if isinstance(families, int) and families and len(folds) != families:
        problems.append(f"{name}: {len(folds)} LOMO folds but dataset says {families} model families")
    fold_failures = 0
    fold_accepted = 0
    for family, fold in folds.items():
        fold_failures += fold.get("test_failures") or 0
        fold_accepted += fold.get("accepted_repairs", fold.get("repairable")) or 0
        check_protocol({
            "proposed_repairs": fold.get("proposed_repairs"),
            "schema_passed": fold.get("schema_passed"),
            "schema_rejected_failures": (fold.get("proposed_repairs") or 0)
                                        - (fold.get("schema_passed") or 0),
            "value_verified_repairs": fold.get("value_verified_repairs"),
            "accepted_repairs": fold.get("accepted_repairs", fold.get("repairable")),
            "evaluable_failures": fold.get("test_failures"),
            "test_failures": fold.get("test_failures"),
            "schema_valid_rate": fold.get("schema_valid_rate"),
            "value_verified_rate": fold.get("value_verified_rate"),
            "accepted_rate": fold.get("accepted_rate"),
            "repair_coverage_of_failures": fold.get("repair_coverage_of_failures"),
            "cache_verification": fold.get("cache_verification") or {},
        }, f"LOMO fold {family}")
        train_models = fold.get("train_models_used")
        if isinstance(train_models, list) and protocol.get("lomo_train_policy") == "fixed-three":
            fixed = protocol.get("lomo_train_models") or []
            expected = [model for model in fixed if model != family] if family in fixed else fixed
            if sorted(train_models) != sorted(expected):
                problems.append(
                    f"{name}: LOMO fold {family} trained on {sorted(train_models)} but the "
                    f"protocol claims the fixed training universe; expected {sorted(expected)}")
    if folds and isinstance(lomo.get("evaluable_failures"), int):
        if fold_failures != lomo["evaluable_failures"]:
            problems.append(f"{name}: LOMO fold failures sum to {fold_failures}, aggregate "
                            f"evaluable_failures {lomo['evaluable_failures']}")
        if fold_accepted != lomo.get("accepted_repairs"):
            problems.append(f"{name}: LOMO fold accepted sum to {fold_accepted}, aggregate "
                            f"accepted {lomo.get('accepted_repairs')}")
    if "lomo_sensitivity_exclude_fold" in results:
        sensitivity = (results["lomo_sensitivity_exclude_fold"] or {}).get("aggregate") or {}
        if sensitivity:
            check_protocol(sensitivity, "LOMO sensitivity")

    # P1-5: the artifact must explain why evaluable_failures < real_failures.
    skipped = results.get("skipped_no_schema")
    if not isinstance(skipped, int):
        problems.append(
            f"{name}: missing skipped_no_schema count explaining "
            "real_failures vs evaluable_failures (P1-5)")
    else:
        if isinstance(failures, int) and isinstance((results.get("lomo_aggregate") or {}).get(
                "evaluable_failures"), int):
            explained = skipped + results["lomo_aggregate"]["evaluable_failures"]
            if explained != failures:
                problems.append(
                    f"{name}: skipped_no_schema + evaluable_failures = {explained} != "
                    f"real_failures {failures}")
        breakdown = results.get("skipped_no_schema_breakdown") or {}
        if isinstance(breakdown, dict):
            if sum(v for v in breakdown.values() if isinstance(v, int)) != skipped:
                problems.append(f"{name}: skipped_no_schema_breakdown does not sum to skipped")
        by_model = results.get("skipped_no_schema_by_model") or {}
        if isinstance(by_model, dict):
            if sum(v for v in by_model.values() if isinstance(v, int)) != skipped:
                problems.append(f"{name}: skipped_no_schema_by_model does not sum to skipped")
        if not isinstance(results.get("skipped_no_schema_explanation"), str) or not \
                results["skipped_no_schema_explanation"]:
            problems.append(f"{name}: skipped_no_schema has no explanation text")

    # The LOMO label must be honest: a fixed-three protocol cannot be called a true LOMO.
    lomo_text = str(protocol.get("lomo") or "").lower()
    if protocol.get("lomo_train_policy") == "fixed-three" and "leave-one-model-out" in lomo_text \
            and "not a true" not in lomo_text:
        problems.append(f"{name}: protocol.lomo calls a fixed-three protocol leave-one-model-out")
    return problems
