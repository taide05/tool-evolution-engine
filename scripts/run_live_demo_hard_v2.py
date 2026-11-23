"""Hard multi-step live demo, V2: budget sweep + repeats + richer metrics.

Why V2 (vs ``scripts/run_live_demo_hard.py``): the V1 hard-task artifact was a single
point per cell (20 tasks, one budget) with a budget calibrated to "fault-free + 1 turn".
That is easy to criticise as "the budget was chosen so the baseline fails". V2 turns the
budget into an explicit experimental variable and adds repeats:

* 32 tasks (20 V1 tasks + 12 harder ones: derived arithmetic, cross-source comparisons,
  multi-country aggregates, 5-8 dependent tool calls);
* budget sweep: per-task turn budgets ``clean+1`` / ``clean+4`` / ``10`` (clean = measured
  fault-free turn requirement), so the artifact contains a **success-vs-budget curve**;
* repeats: the tight ``clean+1`` condition is repeated with several fault seeds; every
  cell records its repeat index and seed, so we report Wilson 95% intervals and paired
  McNemar over (task, repeat) cells;
* derived metrics: failure-mode split (format miss vs missing value), retry backoff time,
  injected-fault-type breakdown, per-category breakdown;
* optional worker pool (``--workers``) to keep the wall-clock within a session.

Single model by decision: ``deepseek-v4-flash``.  The only synthetic element is the
deterministic transient-fault epoch injection (same model as V1); everything else (LLM,
HTTP, verification) is live.  Missing API key -> stop, never fall back to offline.

Usage
-----
    python scripts/run_live_demo_hard_v2.py --mode calibrate
    python scripts/run_live_demo_hard_v2.py --dry-run
    python scripts/run_live_demo_hard_v2.py --workers 2 --output .scratch/toolguard/live_demo_hard_v2.json
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from run_live_demo_hard import build_tasks  # noqa: E402 - shared V2 task pool

from tool_evolution.toolguard.agent import DeepSeekClient, LLMConfig, run_task  # noqa: E402
from tool_evolution.toolguard.guard import ToolGuard  # noqa: E402
from tool_evolution.toolguard.live import FaultyBackend  # noqa: E402
from tool_evolution.toolguard.live_hard import LiveHardBackend  # noqa: E402
from tool_evolution.toolguard.verify import (  # noqa: E402
    contains_number,
    contains_text,
    mcnemar_exact,
    wilson_interval,
)
from tool_evolution.utils.config import settings  # noqa: E402

MODEL = "deepseek-v4-flash"
DEFAULT_OUTPUT = ".scratch/toolguard/live_demo_hard_v2.json"
DEFAULT_CALIBRATION = ".scratch/toolguard/hard_calibration_v2.json"
ARMS = ("baseline", "guard")
DEFAULT_BUDGETS = "clean+1,clean+4,10"
DEFAULT_REPEATS = "clean+1:3,clean+4:1,10:1"


# ---------------------------------------------------------------------------
# experiment planning
# ---------------------------------------------------------------------------
def parse_budgets(spec: str) -> list[str]:
    conditions = [value.strip() for value in spec.split(",") if value.strip()]
    for condition in conditions:
        if condition.startswith("clean+"):
            int(condition.split("+", 1)[1])
        else:
            int(condition)
    return conditions


def parse_repeats(spec: str, conditions: list[str]) -> dict[str, int]:
    repeats = {condition: 1 for condition in conditions}
    for part in spec.split(","):
        if not part.strip():
            continue
        name, _, count = part.partition(":")
        name = name.strip()
        if name not in repeats:
            raise SystemExit(f"repeat spec mentions unknown condition {name!r}")
        repeats[name] = max(1, int(count or 1))
    return repeats


def resolve_budget(condition: str, clean_turns: int) -> int:
    if condition.startswith("clean+"):
        return clean_turns + int(condition.split("+", 1)[1])
    return int(condition)


# ---------------------------------------------------------------------------
# metrics helpers
# ---------------------------------------------------------------------------
def _expected_scalars(expected) -> list:
    scalars: list = []
    if isinstance(expected, dict):
        for value in expected.values():
            scalars.extend(_expected_scalars(value))
    elif isinstance(expected, (list, tuple)):
        for value in expected:
            scalars.extend(_expected_scalars(value))
    elif isinstance(expected, (int, float, str)):
        scalars.append(expected)
    return scalars


def classify_failure(answer: str, expected) -> str:
    """Split verifier failures into 'format miss' (expected value present) vs 'missing'."""
    if not (answer or "").strip():
        return "no_answer"
    for scalar in _expected_scalars(expected):
        if isinstance(scalar, bool):
            continue
        if isinstance(scalar, (int, float)):
            tolerance = max(0.01, abs(float(scalar)) * 0.01)
            if contains_number(answer, float(scalar), tolerance):
                return "format_miss"
        elif isinstance(scalar, str) and contains_text(answer, scalar):
            return "format_miss"
    return "no_expected_value"


def summarise_arm(rows: list[dict]) -> dict:
    successes = sum(int(row["success"]) for row in rows)
    fault_kinds: dict[str, int] = {}
    for row in rows:
        for kind, count in row.get("fault_kinds", {}).items():
            fault_kinds[kind] = fault_kinds.get(kind, 0) + count
    return {
        "cells": len(rows),
        "success": successes,
        "success_rate": round(successes / max(len(rows), 1), 4),
        "success_wilson_95": wilson_interval(successes, len(rows)),
        "turns": sum(row["turns"] for row in rows),
        "turns_per_cell": round(sum(row["turns"] for row in rows) / max(len(rows), 1), 3),
        "tool_calls": sum(row["tool_calls"] for row in rows),
        "tool_errors": sum(row["tool_errors"] for row in rows),
        "exhausted": sum(int(row["exhausted"]) for row in rows),
        "faults_injected": sum(row["faults_injected"] for row in rows),
        "fault_kinds": fault_kinds,
        "guard_recoveries": sum(row["guard_recoveries"] for row in rows),
        "guard_retry_sleep_ms": round(sum(row.get("guard_retry_sleep_ms", 0.0) for row in rows), 2),
        "mcp_calls": sum(row.get("mcp_calls", 0) for row in rows),
        "mcp_errors": sum(row.get("mcp_errors", 0) for row in rows),
        "mcp_restarts": sum(row.get("mcp_restarts", 0) for row in rows),
        "usage": {key: sum(row["usage"][key] for row in rows)
                  for key in ("prompt_tokens", "completion_tokens", "total_tokens")},
        "elapsed_ms": round(sum(row["elapsed_ms"] for row in rows), 2),
    }


def aggregate(records: list[dict]) -> dict:
    conditions = list(dict.fromkeys(row["condition"] for row in records))
    aggregate: dict[str, dict] = {"conditions": {}, "budget_curve": []}
    for condition in conditions:
        condition_rows = [row for row in records if row["condition"] == condition]
        arms = {arm: summarise_arm([row for row in condition_rows if row["arm"] == arm]) for arm in ARMS}
        cells: dict[tuple, dict[str, bool]] = {}
        for row in condition_rows:
            cells.setdefault((row["task"], row["repeat"]), {})[row["arm"]] = bool(row["success"])
        baseline_only = sum(1 for pair in cells.values() if pair.get("baseline") and not pair.get("guard"))
        guard_only = sum(1 for pair in cells.values() if pair.get("guard") and not pair.get("baseline"))
        both = sum(1 for pair in cells.values() if pair.get("baseline") and pair.get("guard"))
        per_category: dict[str, dict] = {}
        for row in condition_rows:
            category = row["category"]
            entry = per_category.setdefault(category, {"baseline": [0, 0], "guard": [0, 0]})
            entry[row["arm"]][1] += 1
            entry[row["arm"]][0] += int(row["success"])
        failure_modes: dict[str, dict[str, int]] = {}
        for row in condition_rows:
            if row["success"]:
                continue
            mode = row.get("failure_mode") or "unknown"
            failure_modes.setdefault(row["arm"], {})
            failure_modes[row["arm"]][mode] = failure_modes[row["arm"]].get(mode, 0) + 1
        mean_budget = round(
            sum(row["budget"] for row in condition_rows) / max(len(condition_rows), 1), 2)
        aggregate["conditions"][condition] = {
            "mean_budget": mean_budget,
            "arms": arms,
            "paired_cells": {
                "both_success": both,
                "baseline_only": baseline_only,
                "guard_only": guard_only,
                "neither": len(cells) - both - baseline_only - guard_only,
                "mcnemar_exact_p": mcnemar_exact(baseline_only, guard_only),
            },
            "per_category": per_category,
            "failure_modes": failure_modes,
        }
        aggregate["budget_curve"].append({
            "condition": condition,
            "mean_budget": mean_budget,
            "baseline_success_rate": arms["baseline"]["success_rate"],
            "guard_success_rate": arms["guard"]["success_rate"],
            "baseline_cells": arms["baseline"]["cells"],
            "guard_cells": arms["guard"]["cells"],
        })
    return aggregate


def key_findings(aggregate_block: dict) -> list[str]:
    findings = []
    for point in aggregate_block["budget_curve"]:
        condition = point["condition"]
        block = aggregate_block["conditions"][condition]
        paired = block["paired_cells"]
        findings.append(
            f"{condition} (mean budget {point['mean_budget']}): baseline "
            f"{block['arms']['baseline']['success']}/{block['arms']['baseline']['cells']} vs guard "
            f"{block['arms']['guard']['success']}/{block['arms']['guard']['cells']} "
            f"(McNemar p={paired['mcnemar_exact_p']:.3g})"
        )
        findings.append(
            f"{condition}: turns/cell baseline {block['arms']['baseline']['turns_per_cell']} -> guard "
            f"{block['arms']['guard']['turns_per_cell']}, tokens baseline "
            f"{block['arms']['baseline']['usage']['total_tokens']} -> guard "
            f"{block['arms']['guard']['usage']['total_tokens']}, agent-visible tool errors "
            f"{block['arms']['baseline']['tool_errors']} -> {block['arms']['guard']['tool_errors']}"
        )
    return findings


V2_LIMITATIONS = [
    "Deterministic transient-fault injection (first *attempts_to_fail* attempts of a "
    "faulted logical call fail), not observed production failures; the injection rate and "
    "epoch length are experimental parameters, not measured values.",
    "Single model (deepseek-v4-flash) by decision; repeat seeds vary the injected fault "
    "surface only, they do not re-sample model behaviour.",
    "The budget sweep uses per-task budgets calibrated from one fault-free run per task; "
    "the calibrated turn requirement is itself a point estimate (no repeat of calibration).",
    "The V1 artifact (benchmarks/toolguard/live_demo_hard.json) was produced by commit "
    "02ac71a with the 20-task pool; the task pool has since grown to 32 tasks, so reproduce "
    "V2 from this runner and the V1 numbers from that commit.",
    "Failure-mode classification ('format_miss' vs 'no_expected_value') is a heuristic over "
    "the expected scalars, not an LLM judgement.",
    "Free public APIs are not production grade; API drift can affect reruns, and the GitHub "
    "API is excluded because of the key-free rate limit.",
]


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def build_experiment(tasks: list[dict], conditions: list[str], repeats: dict[str, int],
                     calibration: dict[str, dict]) -> list[dict]:
    items: list[dict] = []
    for condition in conditions:
        for task in tasks:
            clean_turns = max(task["min_turns"], int(calibration.get(task["id"], {}).get("turns") or 0))
            budget = resolve_budget(condition, clean_turns)
            for repeat in range(repeats[condition]):
                for arm in ARMS:
                    items.append({
                        "task": task,
                        "condition": condition,
                        "budget": budget,
                        "clean_turns": clean_turns,
                        "repeat": repeat,
                        "arm": arm,
                    })
    return items


def run_one(client: DeepSeekClient, item: dict, args: argparse.Namespace) -> dict:
    task = item["task"]
    for attempt in range(2):
        try:
            backend = LiveHardBackend()
            faulty = FaultyBackend(
                backend,
                seed=args.seed + item["repeat"] * 101,
                p_timeout=args.p_timeout,
                p_rate_limit=args.p_rate_limit,
                p_server_error=args.p_server_error,
                attempts_to_fail=args.attempts_to_fail,
            )
            middleware = ToolGuard(faulty, sleep_scale=0.05) if item["arm"] == "guard" else None
            run = run_task(client, faulty, task["question"], middleware, max_turns=item["budget"])
            success = bool(task["verifier"](run.answer, task["expected"]))
            record = {
                "task": task["id"],
                "category": task["category"],
                "condition": item["condition"],
                "budget": item["budget"],
                "clean_turns": item["clean_turns"],
                "repeat": item["repeat"],
                "fault_seed": faulty.seed,
                "arm": item["arm"],
                "success": success,
                "failure_mode": None if success else classify_failure(run.answer, task["expected"]),
                "answer": run.answer[:400],
                "turns": run.turns,
                "tool_calls": run.tool_calls,
                "tool_errors": run.tool_errors,
                "exhausted": run.exhausted,
                "usage": run.usage,
                "elapsed_ms": run.elapsed_ms,
                "faults_injected": faulty.stats["injected"],
                "fault_kinds": dict(faulty.stats["kinds"]),
                "guard_recoveries": middleware.stats["recovered"] if middleware is not None else 0,
                "guard_retry_sleep_ms": round(middleware.stats["retry_sleep_s"] * 1000, 2)
                if middleware is not None else 0.0,
                "guard_events": run.guard_events,
            }
            return record
        except RuntimeError as exc:  # transient upstream API failure: retry the cell once
            if attempt == 1:
                raise
            print(f"  retrying cell after upstream error: {exc}", file=sys.stderr)
            time.sleep(5.0)
    raise RuntimeError("unreachable")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_calibration(client: DeepSeekClient, tasks: list[dict], args: argparse.Namespace) -> int:
    calibration: dict[str, dict] = {}
    print("\n=== calibration: fault-free turn requirement ===")
    for task in tasks:
        backend = LiveHardBackend()
        run = run_task(client, backend, task["question"], None, max_turns=args.clean_max_turns)
        success = bool(task["verifier"](run.answer, task["expected"]))
        calibration[task["id"]] = {"turns": run.turns, "success": success, "exhausted": run.exhausted}
        print(f"  {task['id']:40s} success={success!s:5s} turns={run.turns} calls={run.tool_calls} "
              f"errors={run.tool_errors}")
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "protocol": "fault-free runs (no middleware, no injected faults); the turn count is the "
                    "interaction requirement used to anchor the budget sweep",
        "tasks": calibration,
    }
    path = Path(args.calibration)
    write_json(path, payload)
    failed = [task_id for task_id, row in calibration.items() if not row["success"]]
    print(f"\ncalibration success {len(calibration) - len(failed)}/{len(calibration)}; wrote {path}")
    if failed:
        print(f"WARNING: tasks not solvable in a fault-free run: {failed}")
    return 0


def run_eval(client: DeepSeekClient, tasks: list[dict], skipped: list[str],
             conditions: list[str], repeats: dict[str, int], args: argparse.Namespace) -> int:
    calibration: dict[str, dict] = {}
    calibration_path = Path(args.calibration)
    if calibration_path.exists():
        calibration = json.loads(calibration_path.read_text(encoding="utf-8")).get("tasks", {})
        print(f"loaded calibration for {len(calibration)} tasks from {calibration_path}")
    else:
        print(f"WARNING: no calibration file at {calibration_path}; budgets fall back to min_turns")
    items = build_experiment(tasks, conditions, repeats, calibration)
    print(f"experiment: {len(items)} cells "
          f"({len(conditions)} budget conditions x {len(ARMS)} arms, repeats {repeats}, workers {args.workers})")

    records: list[dict] = []
    lock = threading.Lock()
    output = Path(args.output)
    started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_one, client, item, args): item for item in items}
            for done, future in enumerate(as_completed(futures), start=1):
                record = future.result()
                with lock:
                    records.append(record)
                    if done % 25 == 0 or done == len(items):
                        write_json(output, {"status": "running", "generated_at":
                                            datetime.now(timezone.utc).isoformat(),
                                            "records": records})
                print(f"  [{done:3d}/{len(items)}] {record['task']:40s} {record['condition']:8s} "
                      f"r{record['repeat']} {record['arm']:8s} success={record['success']!s:5s} "
                      f"turns={record['turns']}/{record['budget']} errors={record['tool_errors']} "
                      f"recoveries={record['guard_recoveries']}")
    except RuntimeError as exc:
        print(f"\nERROR: DeepSeek/live API failure: {exc}", file=sys.stderr)
        write_json(output, {"status": "partial_api_error", "records": records,
                            "error": str(exc), "generated_at": datetime.now(timezone.utc).isoformat()})
        print(f"wrote partial results to {output} (no offline fallback)", file=sys.stderr)
        return 2
    wall_seconds = round(time.perf_counter() - started, 1)

    aggregate_block = aggregate(records)
    summary = {
        "artifact_version": 2,
        "experiment": "ToolGuard_hard_multistep_live_ab_v2_budget_sweep",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "protocol": {
            "arms": list(ARMS),
            "budget_conditions": {
                condition: ("per-task turn budget = fault-free calibration turns + "
                            f"{condition.split('+', 1)[1]}" if condition.startswith("clean+")
                            else f"{condition} turns for every task")
                for condition in conditions
            },
            "repeats": repeats,
            "workers": args.workers,
            "wall_seconds": wall_seconds,
            "calibration": {
                "description": "one fault-free run per task measured the turn requirement that anchors "
                               "the budget sweep",
                "tasks": {task["id"]: calibration.get(task["id"], {"turns": task["min_turns"]})
                          for task in tasks},
            },
            "fault_injection": {
                "model": "deterministic transient fault epochs: the first N attempts of a faulted "
                         "logical call fail (timeout / HTTP 429 / HTTP 5xx), then the real call runs",
                "attempts_to_fail": args.attempts_to_fail,
                "p_timeout": args.p_timeout,
                "p_rate_limit": args.p_rate_limit,
                "p_server_error": args.p_server_error,
                "seed": args.seed,
                "repeat_seed_rule": "fault seed = seed + repeat * 101",
            },
            "tool_surface": {
                "tools": [tool["function"]["name"] for tool in LiveHardBackend().schemas()],
                "apis": ["api.worldbank.org", "geocoding-api.open-meteo.com", "api.open-meteo.com",
                         "pypi.org", "hacker-news.firebaseio.com", "open.er-api.com"],
            },
            "verification": "programmatic; ground truth computed from the same live APIs before the runs",
        },
        "tasks": [{key: value for key, value in task.items() if key != "verifier"} for task in tasks],
        "skipped": skipped,
        "records": records,
        "aggregate": aggregate_block,
        "key_findings": key_findings(aggregate_block),
        "limitations": V2_LIMITATIONS,
        "reproduce": [
            "python scripts/run_live_demo_hard_v2.py --mode calibrate",
            "python scripts/run_live_demo_hard_v2.py --workers 2 --output benchmarks/toolguard/live_demo_hard_v2.json",
            "python -m pytest tests/test_toolguard -q -p no:cacheprovider",
        ],
    }
    write_json(output, summary)
    print("\n=== budget curve ===")
    print(json.dumps(aggregate_block["budget_curve"], ensure_ascii=False, indent=2))
    print(f"\nwrote {output} (wall {wall_seconds}s)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("eval", "calibrate"), default="eval")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--budgets", default=DEFAULT_BUDGETS)
    parser.add_argument("--repeats", default=DEFAULT_REPEATS)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--calibration", default=DEFAULT_CALIBRATION)
    parser.add_argument("--clean-max-turns", type=int, default=14)
    parser.add_argument("--p-timeout", type=float, default=0.20)
    parser.add_argument("--p-rate-limit", type=float, default=0.10)
    parser.add_argument("--p-server-error", type=float, default=0.08)
    parser.add_argument("--attempts-to-fail", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    conditions = parse_budgets(args.budgets)
    repeats = parse_repeats(args.repeats, conditions)
    clean = LiveHardBackend()
    tasks, skipped = build_tasks(clean)
    if args.max_tasks is not None:
        tasks = tasks[:args.max_tasks]
    categories = {}
    for task in tasks:
        categories[task["category"]] = categories.get(task["category"], 0) + 1
    print(f"built {len(tasks)} multi-step tasks, categories {categories}; "
          f"skipped ground-truth groups: {len(skipped)}")
    for note in skipped:
        print(f"  skip: {note}")
    if args.dry_run:
        for task in tasks:
            print(f"  {task['id']:42s} {task['category']:10s} min_turns>={task['min_turns']} "
                  f"expected={json.dumps(task['expected'], ensure_ascii=False)[:110]}")
        return 0

    if not settings.deepseek_api_key:
        print("ERROR: TOOLEVO_DEEPSEEK_API_KEY is missing; cannot run the live demo. "
              "Please provide the API key (the demo will not fall back to offline mode).", file=sys.stderr)
        return 2

    client = DeepSeekClient(LLMConfig(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        model=MODEL,
        max_turns=args.clean_max_turns,
    ))
    try:
        if args.mode == "calibrate":
            return run_calibration(client, tasks, args)
        return run_eval(client, tasks, skipped, conditions, repeats, args)
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
