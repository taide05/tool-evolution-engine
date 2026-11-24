"""Artifact contracts for the committed ToolGuard benchmark artifacts.

Every artifact under ``benchmarks/toolguard/`` is evidence for claims made in
``TOOLGUARD.md``; this module encodes the structural invariants those files must satisfy so
that an inconsistent artifact (e.g. aggregate counts that do not match the raw records, or
a missing reproduce/limitations block) fails the test suite instead of silently
contradicting the documentation.

No network and no benchmark data are needed: everything is checked against the committed
JSON files themselves.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ARTIFACT_DIR = Path("benchmarks/toolguard")
ARMS = {"baseline", "guard"}


def _records_by(records: list[dict], *keys: str) -> dict[tuple, list[dict]]:
    grouped: dict[tuple, list[dict]] = {}
    for record in records:
        grouped.setdefault(tuple(record.get(key) for key in keys), []).append(record)
    return grouped


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


def _check_aggregate_conditions(name: str, data: dict, problems: list[str]) -> None:
    records = data["records"]
    aggregate = data["aggregate"]
    task_ids = {task["id"] for task in data.get("tasks") or []}
    for record in records:
        if task_ids and record.get("task") not in task_ids:
            problems.append(f"{name}: record for task missing from tasks list: {record.get('task')}")
    for condition, block in aggregate["conditions"].items():
        for arm in sorted(ARMS):
            if arm not in block["arms"]:
                problems.append(f"{name}: aggregate missing arm {arm} for condition {condition}")
                continue
            stats = block["arms"][arm]
            rows = [r for r in records if r.get("condition") == condition and r.get("arm") == arm]
            successes = sum(1 for r in rows if r["success"])
            if stats.get("cells") != len(rows):
                problems.append(
                    f"{name}: {condition}/{arm} cells {stats.get('cells')} != records {len(rows)}")
            if stats.get("success") != successes:
                problems.append(
                    f"{name}: {condition}/{arm} success {stats.get('success')} != records {successes}")
            if isinstance(stats.get("usage"), dict) and stats["usage"].get("total_tokens", 0) <= 0:
                problems.append(f"{name}: {condition}/{arm} has no token usage")
        paired = block.get("paired_cells") or {}
        if paired:
            cells: dict[tuple, dict[str, bool]] = {}
            for row in records:
                if row.get("condition") != condition:
                    continue
                key = (row.get("task"), row.get("repeat"))
                cells.setdefault(key, {})[row["arm"]] = row["success"]
            both = sum(1 for pair in cells.values() if pair.get("baseline") and pair.get("guard"))
            guard_only = sum(1 for pair in cells.values() if pair.get("guard") and not pair.get("baseline"))
            baseline_only = sum(1 for pair in cells.values() if pair.get("baseline") and not pair.get("guard"))
            if paired.get("both_success") != both or paired.get("guard_only") != guard_only \
                    or paired.get("baseline_only") != baseline_only:
                problems.append(f"{name}: {condition} paired counts do not match records")
            p_value = paired.get("mcnemar_exact_p")
            if not isinstance(p_value, (int, float)) or not 0.0 <= p_value <= 1.0:
                problems.append(f"{name}: {condition} invalid McNemar p value")


def _check_cross_model(name: str, data: dict, problems: list[str]) -> None:
    records = data["records"]
    cross = data["aggregate"]["cross_model"]
    consistency = cross.get("consistency") or {}
    all_models_worse: list[bool] = []
    for model_key, block in cross.items():
        if model_key == "consistency":
            continue
        model_rows = [r for r in records if r.get("model_key") == model_key]
        if not model_rows:
            problems.append(f"{name}: no records for model {model_key}")
            continue
        for condition, row in block["conditions"].items():
            baseline_rows = [r for r in model_rows
                             if r.get("condition") == condition and r.get("arm") == "baseline"]
            guard_rows = [r for r in model_rows
                          if r.get("condition") == condition and r.get("arm") == "guard"]
            if row.get("cells") != len(baseline_rows):
                problems.append(f"{name}: {model_key}/{condition} cells {row.get('cells')} "
                                f"!= records {len(baseline_rows)}")
            if row.get("baseline_success") != sum(1 for r in baseline_rows if r["success"]):
                problems.append(f"{name}: {model_key}/{condition} baseline_success mismatch")
            if row.get("guard_success") != sum(1 for r in guard_rows if r["success"]):
                problems.append(f"{name}: {model_key}/{condition} guard_success mismatch")
            all_models_worse.append(block.get("guard_never_worse", False))
    if "guard_success_ge_baseline_on_all_models" in consistency:
        expected = bool(all_models_worse) and all(all_models_worse)
        if consistency["guard_success_ge_baseline_on_all_models"] != expected:
            problems.append(f"{name}: consistency flag does not match per-model results")
    cost = data.get("total_estimated_cost_usd")
    if not isinstance(cost, (int, float)) or cost < 0:
        problems.append(f"{name}: invalid total_estimated_cost_usd {cost!r}")


def _check_live_demo_v1(name: str, data: dict, problems: list[str]) -> None:
    results = data.get("results") or {}
    for condition, block in results.items():
        task_count = block.get("tasks")
        if not isinstance(task_count, int) or task_count <= 0:
            problems.append(f"{name}: {condition} has no task count")
            continue
        per_task = block.get("per_task") or {}
        if len(per_task) != task_count:
            problems.append(f"{name}: {condition} per_task {len(per_task)} != tasks {task_count}")
        for arm, stats in (block.get("aggregate") or {}).items():
            if arm not in ARMS:
                continue
            if stats.get("tasks") != task_count:
                problems.append(f"{name}: {condition}/{arm} tasks {stats.get('tasks')} != {task_count}")
            if not 0 <= stats.get("success", -1) <= task_count:
                problems.append(f"{name}: {condition}/{arm} impossible success count")
        paired = block.get("paired") or {}
        if paired and sum(paired.get(key, 0) for key in
                          ("both_success", "baseline_only", "guard_only")) > task_count:
            problems.append(f"{name}: {condition} paired counts exceed task count")


def _check_metrics(name: str, data: dict, problems: list[str]) -> None:
    results = data.get("results") or {}
    lomo = results.get("lomo_aggregate") or {}
    if lomo:
        if lomo.get("repairable", 0) > lomo.get("evaluable_failures", 0):
            problems.append(f"{name}: LOMO repairable > evaluable_failures")
        if lomo.get("schema_valid") != lomo.get("repairable"):
            problems.append(f"{name}: LOMO schema_valid != repairable")
        if lomo.get("value_agreement", 0) > lomo.get("value_verifiable", 0):
            problems.append(f"{name}: LOMO value_agreement > value_verifiable")
        if lomo.get("cache_verified_still_error") != 0:
            problems.append(f"{name}: LOMO cache verification found false repairs")
        coverage = lomo.get("repair_coverage_of_failures")
        if not isinstance(coverage, (int, float)) or not 0.0 <= coverage <= 1.0:
            problems.append(f"{name}: LOMO coverage out of range")
    for protocol in ("cross_model", "cross_group"):
        block = results.get(protocol) or {}
        if block.get("repairable", 0) > block.get("test_failures", 0):
            problems.append(f"{name}: {protocol} repairable > test_failures")
        if not isinstance(block.get("schema_valid_rate"), (int, float)):
            problems.append(f"{name}: {protocol} missing schema_valid_rate")


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
        if name == "live_demo.json":
            _check_live_demo_v1(name, data, problems)
        elif name == "metrics.json":
            _check_metrics(name, data, problems)
        elif isinstance(data.get("records"), list) and data["records"]:
            for record in data["records"]:
                _check_record(record, name, problems)
            if isinstance(data.get("aggregate"), dict):
                if "conditions" in data["aggregate"]:
                    _check_aggregate_conditions(name, data, problems)
                elif "cross_model" in data["aggregate"]:
                    _check_cross_model(name, data, problems)
            calibration = data.get("calibration")
            if isinstance(calibration, dict) and "tasks" in calibration:
                calibration_tasks = calibration["tasks"]
                artifact_task_ids = {task["id"] for task in data.get("tasks") or []}
                if artifact_task_ids and set(calibration_tasks) != artifact_task_ids:
                    problems.append(f"{name}: calibration tasks do not match artifact tasks")
            tasks_per_model = data.get("tasks_per_model")
            if isinstance(tasks_per_model, dict):
                for model_key, model_tasks in tasks_per_model.items():
                    if len(model_tasks) != len(data.get("tasks") or []):
                        problems.append(f"{name}: tasks_per_model[{model_key}] size mismatch")
        else:
            problems.append(f"{name}: unrecognised artifact shape")
    return problems


def main() -> int:
    problems = validate_artifacts()
    directory = ARTIFACT_DIR
    print(f"artifact contract check: {len(sorted(directory.glob('*.json')))} artifacts")
    if problems:
        print("\nproblems:")
        for problem in problems:
            print(" -", problem)
        return 1
    print("all artifacts consistent with their aggregates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
