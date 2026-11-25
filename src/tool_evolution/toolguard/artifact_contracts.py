"""Artifact contracts for the committed ToolGuard benchmark artifacts.

Every artifact under ``benchmarks/toolguard/`` is evidence for claims made in
``TOOLGUARD.md``; this module encodes the invariants those files must satisfy so that an
inconsistent artifact (aggregate numbers that do not match the raw records, a missing
reproduce/limitations block, a mismatched per-model ground-truth snapshot) fails the test
suite instead of silently contradicting the documentation.

Since 2026-11-24 the contract does not merely check cells/success: **every derived number
in an aggregate block is re-derived from the raw records** (success rate, Wilson interval,
turns, turns/cell, tokens, tool errors, guard recoveries, retry backoff, fault-kind counts,
paired counts, exact McNemar p, failure modes, capability placement).  The recomputation
lives in :mod:`tool_evolution.toolguard.artifact_metrics` so the reviewer-facing
``scripts/toolguard_evidence_summary.py`` and this validator cannot disagree.

No network and no benchmark data are needed: everything is checked against the committed
JSON files themselves.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifact_metrics import (
    ARMS,
    check_offline_metrics,
    recompute_arm,
    recompute_conditions,
    recompute_guard_repairs_by_rule,
    recompute_multimodel,
    recompute_paired,
    recompute_v1_live,
    recompute_v1_mcp,
)

ARTIFACT_DIR = Path("benchmarks/toolguard")

# Arm fields every committed aggregate arm must expose, so that no silently dropped
# metric can make the contract weaker.
REQUIRED_ARM_FIELDS = {
    "cells": ("cells", "tasks"),
    "success": ("success",),
    "success_rate": ("success_rate",),
    "turns": ("turns",),
    "tool_calls": ("tool_calls",),
    "tool_errors": ("tool_errors",),
    "usage": ("usage",),
}
# Required for the V2/V3 (repeat-aware, stats-rich) artifacts.
REQUIRED_V2_ARM_FIELDS = {
    "success_wilson_95": ("success_wilson_95",),
    "turns_per_cell": ("turns_per_cell",),
    "faults_injected": ("faults_injected",),
    "guard_recoveries": ("guard_recoveries",),
}
REQUIRED_CONDITION_FIELDS = ("arms", "paired_cells")
_ALIASES = {"cells": ("cells", "tasks"), "recoveries": ("recoveries", "guard_recoveries")}


# ---------------------------------------------------------------------------
# comparison helpers
# ---------------------------------------------------------------------------
def _same(actual: Any, expected: Any) -> bool:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return actual is expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return abs(float(actual) - float(expected)) <= 1e-9 * max(1.0, abs(float(expected)))
    return actual == expected


def _diff(actual: Any, expected: Any, label: str) -> list[str]:
    """Recursively compare a published value against its recomputed counterpart."""
    problems: list[str] = []
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [f"{label}: expected an object, found {type(actual).__name__}"]
        for key in expected:
            if key not in actual:
                problems.append(f"{label}.{key}: missing from artifact")
            else:
                problems.extend(_diff(actual[key], expected[key], f"{label}.{key}"))
        return problems
    if isinstance(expected, list):
        if not isinstance(actual, list):
            return [f"{label}: expected a list, found {type(actual).__name__}"]
        if len(actual) != len(expected):
            return [f"{label}: length {len(actual)} != recomputed {len(expected)}"]
        for index, (actual_item, expected_item) in enumerate(zip(actual, expected)):
            problems.extend(_diff(actual_item, expected_item, f"{label}[{index}]"))
        return problems
    if not _same(actual, expected):
        problems.append(f"{label}: artifact has {actual!r}, records recompute {expected!r}")
    return problems


def _pick(arm: dict, field: str) -> tuple[bool, Any]:
    for alias in _ALIASES.get(field, (field,)):
        if alias in arm:
            return True, arm[alias]
    return False, None


def _check_arm(name: str, label: str, published: dict, computed: dict,
               required: dict[str, tuple[str, ...]],
               allowed_unverified: set[str] | None = None) -> list[str]:
    problems: list[str] = []
    allowed_unverified = allowed_unverified or set()
    known = (set(computed) | allowed_unverified
             | {alias for aliases in _ALIASES.values() for alias in aliases})
    for key in published:
        if key not in known:
            problems.append(f"{name}: {label} has unexpected field {key!r}")
    for logical, aliases in {**REQUIRED_ARM_FIELDS, **required}.items():
        if not any(alias in published for alias in aliases):
            problems.append(f"{name}: {label} is missing required field {logical}")
    for key in published:
        if key not in known:
            continue
        logical = key
        for canonical, aliases in _ALIASES.items():
            if key in aliases:
                logical = canonical
                break
        if logical not in computed:
            # Not derivable from this artifact's records (e.g. the V1 live artifact keeps
            # only per-task counters, no per-cell turns/tokens); the field cannot be
            # re-derived here, so it is not compared rather than silently trusted.
            continue
        problems.extend(_diff(published[key], computed[logical], f"{label}.{key}"))
    return problems


def _check_condition_block(name: str, label: str, published: dict,
                           computed: dict, v2: bool) -> list[str]:
    problems: list[str] = []
    for field in REQUIRED_CONDITION_FIELDS:
        if field not in published:
            problems.append(f"{name}: {label} is missing required field {field}")
    published_arms = published.get("arms") or {}
    computed_arms = computed.get("arms") or {}
    for arm in sorted(ARMS):
        if arm not in published_arms:
            problems.append(f"{name}: {label} aggregate missing arm {arm}")
            continue
        required = REQUIRED_V2_ARM_FIELDS if v2 else {}
        problems.extend(_check_arm(name, f"{label}/{arm}", published_arms[arm],
                                   computed_arms.get(arm) or {}, required))
    for field, computed_key in (("paired_cells", "paired_cells"), ("paired", "paired_cells")):
        if field in published:
            problems.extend(_diff(published[field], computed.get(computed_key) or {},
                                  f"{label}.{field}"))
    for field in ("mean_budget", "failure_modes", "per_category"):
        if field in published and field in computed:
            problems.extend(_diff(published[field], computed[field], f"{label}.{field}"))
    return problems


# ---------------------------------------------------------------------------
# per-artifact checks
# ---------------------------------------------------------------------------
def _check_common(name: str, data: dict, problems: list[str]) -> None:
    if not isinstance(data.get("artifact_version"), int):
        problems.append(f"{name}: artifact_version must be an int")
    if not isinstance(data.get("generated_at"), str) or not data["generated_at"]:
        problems.append(f"{name}: missing generated_at")
    for field in ("limitations", "reproduce"):
        value = data.get(field)
        if not isinstance(value, list) or not value:
            problems.append(f"{name}: missing non-empty {field}")
    for command in data.get("reproduce") or []:
        if not isinstance(command, str):
            problems.append(f"{name}: reproduce entry is not a string: {command!r}")
            continue
        parts = command.split()
        if len(parts) >= 2 and parts[0] == "python" and parts[1].endswith(".py"):
            if not Path(parts[1]).exists():
                problems.append(f"{name}: reproduce script does not exist: {parts[1]}")


def _check_record(record: dict, name: str, problems: list[str]) -> None:
    if not isinstance(record.get("success"), bool):
        problems.append(f"{name}: record without boolean success: {record.get('task')}")
    if record.get("arm") not in ARMS:
        problems.append(f"{name}: record with unknown arm {record.get('arm')!r}")
    if not isinstance(record.get("turns"), int) or record["turns"] < 1:
        problems.append(f"{name}: record with invalid turns: {record.get('turns')!r}")
    if not isinstance(record.get("condition"), str) and "condition" in record:
        problems.append(f"{name}: record with non-string condition: {record.get('condition')!r}")
    for field in ("usage",):
        if field not in record:
            problems.append(f"{name}: record without {field}: {record.get('task')}")
    if "faults_injected" not in record:
        problems.append(f"{name}: record without faults_injected: {record.get('task')}")


def _check_task_ids(name: str, data: dict, records: list[dict], problems: list[str]) -> None:
    task_ids = {task["id"] for task in data.get("tasks") or []}
    if not task_ids:
        return
    for record in records:
        if record.get("task") not in task_ids:
            problems.append(f"{name}: record for task missing from tasks list: {record.get('task')}")
    record_conditions = {record.get("condition") for record in records}
    for task in data.get("tasks") or []:
        if not isinstance(task.get("expected"), dict):
            problems.append(f"{name}: task without expected block: {task.get('id')}")


def _check_conditions_artifact(name: str, data: dict, problems: list[str],
                               v2: bool = True) -> None:
    records = data["records"]
    aggregate = data["aggregate"]
    published = aggregate["conditions"]
    computed = recompute_conditions(records, with_budget_curve="budget_curve" in aggregate)
    if set(published) != set(computed["conditions"]):
        problems.append(
            f"{name}: aggregate conditions {sorted(published)} != recomputed "
            f"{sorted(computed['conditions'])}")
    for condition in published:
        if condition not in computed["conditions"]:
            continue
        problems.extend(_check_condition_block(
            name, f"conditions.{condition}", published[condition],
            computed["conditions"][condition], v2=v2))
    if "budget_curve" in aggregate:
        problems.extend(_diff(aggregate["budget_curve"], computed.get("budget_curve") or [],
                              "aggregate.budget_curve"))
    if "guard_repairs_by_rule" in aggregate:
        problems.extend(_diff(aggregate["guard_repairs_by_rule"],
                              recompute_guard_repairs_by_rule(records),
                              "aggregate.guard_repairs_by_rule"))
    if "guard_repairs_by_rule" in data:
        problems.extend(_diff(data["guard_repairs_by_rule"],
                              recompute_guard_repairs_by_rule(records),
                              "guard_repairs_by_rule"))
    _check_task_ids(name, data, records, problems)
    calibration = data.get("calibration")
    if isinstance(calibration, dict) and "tasks" in calibration:
        calibration_tasks = calibration["tasks"]
        artifact_task_ids = {task["id"] for task in data.get("tasks") or []}
        if artifact_task_ids and set(calibration_tasks) != artifact_task_ids:
            problems.append(f"{name}: calibration tasks do not match artifact tasks")


def _check_legacy_hard_conditions(name: str, data: dict, problems: list[str]) -> None:
    """V1 hard artifact: aggregate is {condition: {arms, paired}} without per-cell stats."""
    records = data["records"]
    for condition, published in data["aggregate"].items():
        rows = [record for record in records if record.get("condition") == condition]
        if not rows:
            problems.append(f"{name}: aggregate condition {condition} has no records")
            continue
        computed = recompute_conditions(rows)["conditions"][condition]
        for arm in sorted(ARMS):
            if arm not in (published.get("arms") or {}):
                problems.append(f"{name}: {condition} aggregate missing arm {arm}")
                continue
            problems.extend(_check_arm(name, f"{condition}/{arm}",
                                       published["arms"][arm], computed["arms"][arm], {}))
        if "paired" in published:
            problems.extend(_diff(published["paired"], computed["paired_cells"],
                                  f"{condition}.paired"))
    _check_task_ids(name, data, records, problems)


def _check_multimodel(name: str, data: dict, problems: list[str]) -> None:
    records = data["records"]
    calibration = data.get("calibration") or {}
    protocol_models = (data.get("protocol") or {}).get("models") or []
    if not protocol_models:
        problems.append(f"{name}: protocol.models missing (needed to recompute cost/placement)")
        return
    computed = recompute_multimodel(records, calibration, protocol_models)
    problems.extend(_diff(data["aggregate"]["cross_model"], computed["cross_model"],
                          "aggregate.cross_model"))
    if "total_estimated_cost_usd" in data:
        problems.extend(_diff(data["total_estimated_cost_usd"],
                              computed["total_estimated_cost_usd"],
                              "total_estimated_cost_usd"))
    _check_task_ids(name, data, records, problems)

    # Per-record `expected` snapshots must agree with the per-model ground truth the
    # artifact claims to have used (P2-2: --from-records used to drop tasks_per_model).
    # Artifacts written before the snapshot existed (V2) have no per-record `expected`;
    # for them the snapshot is only checked when present.
    carries_expected = any("expected" in record for record in records)
    tasks_per_model = data.get("tasks_per_model") or {}
    if carries_expected and not tasks_per_model:
        problems.append(f"{name}: tasks_per_model snapshot missing")
    for model_key, model_tasks in tasks_per_model.items():
        if len(model_tasks) != len(data.get("tasks") or []):
            problems.append(f"{name}: tasks_per_model[{model_key}] size mismatch")
        by_id = {task["id"]: task for task in model_tasks}
        for record in records:
            if record.get("model_key") != model_key:
                continue
            task = by_id.get(record.get("task"))
            if task is None:
                problems.append(f"{name}: {model_key} record task {record.get('task')} "
                                "missing from tasks_per_model")
                continue
            if "expected" in record and record["expected"] != task.get("expected"):
                problems.append(
                    f"{name}: {model_key}/{record.get('task')} record expected snapshot "
                    "does not match tasks_per_model")


def _check_v1_live(name: str, data: dict, problems: list[str]) -> None:
    computed = recompute_v1_live(data.get("results") or {})
    # The V1 artifact keeps only per-task counters, not raw per-cell records, so turns,
    # token usage and elapsed time cannot be re-derived; they are allowed but unverified.
    allowed = {"turns", "usage", "elapsed_ms", "exhausted"}
    for condition, block in (data.get("results") or {}).items():
        expected = computed.get(condition) or {}
        per_task = block.get("per_task") or {}
        if not per_task:
            problems.append(f"{name}: {condition} has no per_task records")
            continue
        if len(per_task) != block.get("tasks"):
            problems.append(f"{name}: {condition} per_task {len(per_task)} != tasks "
                            f"{block.get('tasks')}")
        for arm, stats in (block.get("aggregate") or {}).items():
            if arm not in ARMS:
                continue
            problems.extend(_check_arm(name, f"{condition}/{arm}", stats,
                                       expected["arms"][arm], {},
                                       allowed_unverified=allowed))
        published_paired = block.get("paired") or {}
        for key, value in published_paired.items():
            if key not in expected["paired"]:
                problems.append(f"{name}: {condition}.paired has unexpected field {key!r}")
                continue
            problems.extend(_diff(value, expected["paired"][key], f"{condition}.paired.{key}"))


def _check_v1_mcp(name: str, data: dict, problems: list[str]) -> None:
    records = data["records"]
    computed = recompute_v1_mcp(records)
    aggregate = data["aggregate"]
    for arm in sorted(ARMS):
        if arm not in (aggregate.get("arms") or {}):
            problems.append(f"{name}: aggregate missing arm {arm}")
            continue
        problems.extend(_check_arm(name, arm, aggregate["arms"][arm],
                                   computed["arms"][arm], {}))
    if "paired" in aggregate:
        problems.extend(_diff(aggregate["paired"], computed["paired"], "aggregate.paired"))
    if "guard_repairs_by_rule" in data:
        problems.extend(_diff(data["guard_repairs_by_rule"],
                              computed["guard_repairs_by_rule"], "guard_repairs_by_rule"))
    _check_task_ids(name, data, records, problems)


def _check_metrics(name: str, data: dict, problems: list[str]) -> None:
    results = data.get("results") or {}
    if "lomo_aggregate" not in results:
        problems.append(f"{name}: missing results.lomo_aggregate")
    if "lomo_per_fold" not in results:
        problems.append(f"{name}: missing results.lomo_per_fold (per-fold evidence)")
    protocol = data.get("protocol") or {}
    for field in ("lomo", "lomo_train_policy", "lomo_train_models", "schema_funnel"):
        if field not in protocol:
            problems.append(f"{name}: protocol missing {field!r}")
    problems.extend(check_offline_metrics(data, name))


def validate_artifacts(root: Path | None = None) -> list[str]:
    """Return a list of contract violations for every committed artifact (empty = all good)."""
    directory = Path(root) if root is not None else ARTIFACT_DIR
    problems: list[str] = []
    paths = sorted(directory.glob("*.json"))
    if not paths:
        return [f"no artifacts found under {directory}"]
    for path in paths:
        name = path.name
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:  # pragma: no cover - corrupt artifact
            problems.append(f"{name}: invalid JSON ({exc})")
            continue
        if not isinstance(data, dict):  # pragma: no cover - corrupt artifact
            problems.append(f"{name}: artifact root is not an object")
            continue
        _check_common(name, data, problems)
        try:
            if name == "metrics.json":
                _check_metrics(name, data, problems)
            elif name == "live_demo.json":
                _check_v1_live(name, data, problems)
            elif name == "mcp_demo.json":
                _check_v1_mcp(name, data, problems)
            elif isinstance(data.get("records"), list) and data["records"]:
                for record in data["records"]:
                    _check_record(record, name, problems)
                aggregate = data.get("aggregate")
                if not isinstance(aggregate, dict):
                    problems.append(f"{name}: records without an aggregate block")
                elif "cross_model" in aggregate:
                    _check_multimodel(name, data, problems)
                elif "conditions" in aggregate:
                    v2 = name.endswith("_v2.json") or name.endswith("_v3.json") \
                        or name.endswith("_v4.json")
                    _check_conditions_artifact(name, data, problems, v2=v2)
                elif aggregate and all(isinstance(block, dict) and "arms" in block
                                       for block in aggregate.values()):
                    _check_legacy_hard_conditions(name, data, problems)
                else:
                    problems.append(f"{name}: unrecognised aggregate shape")
            else:
                problems.append(f"{name}: unrecognised artifact shape")
        except (KeyError, TypeError, ValueError) as exc:  # pragma: no cover - malformed artifact
            problems.append(f"{name}: contract check crashed on malformed artifact ({exc})")
    return problems


def _artifact_count() -> int:
    return len(sorted(ARTIFACT_DIR.glob("*.json")))


def main() -> int:
    problems = validate_artifacts()
    print(f"artifact contract check: {_artifact_count()} artifacts")
    if problems:
        print("\nproblems:")
        for problem in problems:
            print(" -", problem)
        return 1
    print("all artifacts consistent with their aggregates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
