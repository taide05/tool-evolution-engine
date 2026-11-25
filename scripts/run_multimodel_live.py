"""Multi-model live A/B: cheap / mid / strong agent models over the same hard task pool.

The point of the multi-model axis is replication, not a between-model hypothesis test:
for every model we run the *same* protocol (same tasks, same fault seeds, same budget
semantics) and report a paired within-model baseline-vs-guard comparison.  Across models
we report whether the direction and magnitude of the middleware effect hold.

Models (all OpenAI-compatible chat completions, tool calling required)
---------------------------------------------------------------------
| tier   | model id                | access                     | key                        |
| cheap  | qwen3.8-flash           | Alibaba Bailian/DashScope  | TOOLEVO_QWEN_API_KEY       |
| mid    | deepseek-v4-flash       | DeepSeek                   | TOOLEVO_DEEPSEEK_API_KEY   |
| strong | qwen4-max               | Alibaba Bailian/DashScope  | TOOLEVO_QWEN_API_KEY       |

Protocol per model
------------------
* calibration: one fault-free run per task (baseline, no middleware) -> per-model clean
  turn requirement (also the capability placement evidence: success / turns / tokens);
* eval: 32 tasks x 2 arms x (clean+1 with 2 fault-seed repeats + clean+4 x 1) = 192 cells;
* same deterministic transient-fault epoch injection as the V2 single-model artifact;
* programmatic verification, no LLM judge; missing key/model -> stop, never mock.

Usage
-----
    python scripts/run_multimodel_live.py --dry-run
    python scripts/run_multimodel_live.py --mode calibrate
    python scripts/run_multimodel_live.py --mode eval --workers 2
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

from run_live_demo_hard import build_tasks  # noqa: E402 - shared 32-task pool
from run_live_demo_hard_v2 import (  # noqa: E402 - shared V2 metrics helpers
    aggregate,
    classify_failure,
    parse_budgets,
    parse_repeats,
    resolve_budget,
    write_json,
)

from tool_evolution.toolguard.agent import ChatCompletionsClient, LLMConfig, run_task  # noqa: E402
from tool_evolution.toolguard.guard import ToolGuard  # noqa: E402
from tool_evolution.toolguard.live import FaultyBackend  # noqa: E402
from tool_evolution.toolguard.live_hard import LiveHardBackend  # noqa: E402
from tool_evolution.utils.config import settings  # noqa: E402

ARMS = ("baseline", "guard")
DEFAULT_BUDGETS = "clean+1,clean+4"
DEFAULT_REPEATS = "clean+1:2,clean+4:1"
DEFAULT_OUTPUT = ".scratch/toolguard/live_demo_multimodel.json"
DEFAULT_CALIBRATION = ".scratch/toolguard/multimodel_calibration.json"


def model_specs() -> list[dict]:
    return [
        {
            "tier": "cheap",
            "key": "qwen-flash",
            "label": "qwen3.8-flash",
            "model": settings.qwen_model,
            "api_key": settings.qwen_api_key,
            "base_url": settings.qwen_base_url,
            "price_input_per_m": 0.15,
            "price_output_per_m": 0.47,
            "provider": "Alibaba Model Studio (Bailian/DashScope)",
        },
        {
            "tier": "mid",
            "key": "deepseek-flash",
            "label": "DeepSeek-V4.1-Flash-1017",
            "model": "deepseek-v4-flash",
            "api_key": settings.deepseek_api_key,
            "base_url": settings.deepseek_base_url,
            "price_input_per_m": 0.15,
            "price_output_per_m": 0.60,
            "provider": "DeepSeek",
        },
        {
            "tier": "strong",
            "key": "qwen4-max",
            "label": "qwen4-max",
            "model": settings.qwen_strong_model,
            "api_key": settings.qwen_api_key,
            "base_url": settings.qwen_base_url,
            "price_input_per_m": 1.413,
            "price_output_per_m": 4.24,
            "provider": "Alibaba Model Studio (Bailian/DashScope)",
        },
    ]


def make_client(spec: dict, max_turns: int) -> ChatCompletionsClient:
    return ChatCompletionsClient(LLMConfig(
        api_key=spec["api_key"],
        base_url=spec["base_url"],
        model=spec["model"],
        name=spec["key"],
        max_turns=max_turns,
        max_tokens=1200,
        timeout_s=120.0,
    ))


def estimate_cost(usage: dict, spec: dict) -> float:
    """Cost estimate for one model's measured token usage at published list prices."""
    return round(
        usage["prompt_tokens"] / 1e6 * spec["price_input_per_m"]
        + usage["completion_tokens"] / 1e6 * spec["price_output_per_m"], 4)


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------
def run_calibration(tasks: list[dict], specs: list[dict], args: argparse.Namespace) -> int:
    calibration: dict[str, dict] = {}
    for spec in specs:
        client = make_client(spec, args.clean_max_turns)
        usage_total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        print(f"\n=== calibration: {spec['tier']} / {spec['label']} ({spec['model']}) ===")
        tasks_calibration: dict[str, dict] = {}
        try:
            for task in tasks:
                backend = LiveHardBackend(hn_schema=getattr(args, "hn_schema", "consistent"))
                run = run_task(client, backend, task["question"], None, max_turns=args.clean_max_turns)
                success = bool(task["verifier"](run.answer, task["expected"]))
                tasks_calibration[task["id"]] = {
                    "turns": run.turns,
                    "success": success,
                    "exhausted": run.exhausted,
                    "tool_calls": run.tool_calls,
                    "tool_errors": run.tool_errors,
                    "usage": run.usage,
                }
                for key in usage_total:
                    usage_total[key] += run.usage[key]
                print(f"  {task['id']:42s} success={success!s:5s} turns={run.turns:2d} "
                      f"calls={run.tool_calls} errors={run.tool_errors}")
        except RuntimeError as exc:
            print(f"\nERROR: model {spec['key']} failed during calibration: {exc}", file=sys.stderr)
            print("Stopping (no offline/mock fallback). Check the key/model id and rerun.", file=sys.stderr)
            return 2
        finally:
            client.close()
        succeeded = sum(int(row["success"]) for row in tasks_calibration.values())
        turns = [row["turns"] for row in tasks_calibration.values()]
        calibration[spec["key"]] = {
            "tier": spec["tier"],
            "model": spec["model"],
            "label": spec["label"],
            "provider": spec["provider"],
            "tasks": tasks_calibration,
            "placement": {
                "success": f"{succeeded}/{len(tasks_calibration)}",
                "success_rate": round(succeeded / max(len(tasks_calibration), 1), 4),
                "avg_turns": round(sum(turns) / max(len(turns), 1), 3),
                "max_turns": max(turns) if turns else 0,
                "usage": usage_total,
                "estimated_cost_usd": estimate_cost(usage_total, spec),
            },
        }
        print(f"  placement: {calibration[spec['key']]['placement']}")

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": "one fault-free run per task per model (baseline, no middleware); the per-model "
                    "turn count anchors the budget sweep and doubles as capability placement evidence",
        "models": calibration,
    }
    path = Path(args.calibration)
    write_json(path, payload)
    failed = {key: row["placement"]["success"] for key, row in calibration.items()}
    print(f"\ncalibration placement: {failed}")
    print(f"wrote {path}")
    if any(row["placement"]["success_rate"] < 0.75 for row in calibration.values()):
        print("WARNING: at least one model solved <75% of tasks fault-free; check tool-calling quality")
    return 0


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------
def run_one(client: ChatCompletionsClient, spec: dict, item: dict, args: argparse.Namespace) -> dict:
    task = item["task"]
    for attempt in range(2):
        try:
            backend = LiveHardBackend(hn_schema=getattr(args, "hn_schema", "consistent"))
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
            return {
                "model_key": spec["key"],
                "model_tier": spec["tier"],
                "model": spec["model"],
                "task": task["id"],
                "category": task["category"],
                "condition": item["condition"],
                "budget": item["budget"],
                "clean_turns": item["clean_turns"],
                "repeat": item["repeat"],
                "fault_seed": faulty.seed,
                "arm": item["arm"],
                "success": success,
                "expected": task["expected"],
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
        except RuntimeError as exc:
            if attempt == 1:
                raise
            print(f"  retrying cell after {spec['key']} error: {exc}", file=sys.stderr)
            time.sleep(5.0)
    raise RuntimeError("unreachable")


def cross_model_summary(records: list[dict], specs: list[dict], aggregates: dict[str, dict],
                        calibrations: dict[str, dict]) -> dict:
    summary: dict[str, dict] = {}
    for spec in specs:
        key = spec["key"]
        aggregate_block = aggregates[key]
        placement = calibrations[key]["placement"]
        rows = [row for row in records if row["model_key"] == key]
        per_condition: dict[str, dict] = {}
        for condition, block in aggregate_block["conditions"].items():
            baseline = block["arms"]["baseline"]
            guard = block["arms"]["guard"]
            per_condition[condition] = {
                "baseline_success": baseline["success"],
                "guard_success": guard["success"],
                "cells": baseline["cells"],
                "baseline_success_rate": baseline["success_rate"],
                "guard_success_rate": guard["success_rate"],
                "guard_success_wilson_95": guard["success_wilson_95"],
                "mcnemar_exact_p": block["paired_cells"]["mcnemar_exact_p"],
                "baseline_turns_per_cell": baseline["turns_per_cell"],
                "guard_turns_per_cell": guard["turns_per_cell"],
                "baseline_tokens": baseline["usage"]["total_tokens"],
                "guard_tokens": guard["usage"]["total_tokens"],
                "baseline_tool_errors": baseline["tool_errors"],
                "guard_tool_errors": guard["tool_errors"],
            }
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        for row in rows:
            for field in usage:
                usage[field] += row["usage"][field]
        summary[key] = {
            "tier": spec["tier"],
            "label": spec["label"],
            "model": spec["model"],
            "provider": spec["provider"],
            "capability_placement": placement,
            "conditions": per_condition,
            "guard_never_worse": all(
                row["guard_success"] >= row["baseline_success"]
                for row in per_condition.values()),
            "estimated_cost_usd": estimate_cost(usage, spec),
        }
    summary["consistency"] = {
        "guard_success_ge_baseline_on_all_models": all(
            row["guard_never_worse"] for key, row in summary.items() if key != "consistency"),
        "tight_budget_guard_success_rates": {
            key: row["conditions"].get("clean+1", {}).get("guard_success_rate")
            for key, row in summary.items() if key != "consistency"},
        "tight_budget_baseline_success_rates": {
            key: row["conditions"].get("clean+1", {}).get("baseline_success_rate")
            for key, row in summary.items() if key != "consistency"},
    }
    return summary


def run_eval(tasks: list[dict], skipped: list[str], specs: list[dict], conditions: list[str],
             repeats: dict[str, int], args: argparse.Namespace) -> int:
    calibration_path = Path(args.calibration)
    if not calibration_path.exists():
        print(f"ERROR: calibration file missing at {calibration_path}; run --mode calibrate first",
              file=sys.stderr)
        return 2
    calibration_payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    calibrations = calibration_payload.get("models", {})

    output = Path(args.output)
    all_records: list[dict] = []
    tasks_per_model: dict[str, list[dict]] = {}
    started = time.perf_counter()
    lock = threading.Lock()
    wanted_ids = {task["id"] for task in tasks}

    for spec in specs:
        key = spec["key"]
        # Refresh the live ground truth immediately before each model's block: the
        # previous model run takes ~20 minutes, and values such as weather/FX can drift.
        fresh_tasks, fresh_skipped = build_tasks(
            LiveHardBackend(hn_schema=getattr(args, "hn_schema", "consistent")))
        if fresh_skipped:
            print(f"  warning: ground-truth refresh skips for {key}: {fresh_skipped}")
        fresh_tasks = [task for task in fresh_tasks if task["id"] in wanted_ids]
        missing_ids = wanted_ids - {task["id"] for task in fresh_tasks}
        if missing_ids:
            print(f"ERROR: ground-truth refresh for {key} missed tasks: {sorted(missing_ids)}",
                  file=sys.stderr)
            return 2
        tasks_per_model[key] = fresh_tasks
        calibration = calibrations.get(key, {}).get("tasks", {})
        missing = [task["id"] for task in tasks if task["id"] not in calibration]
        if missing:
            print(f"ERROR: calibration for {key} misses tasks: {missing[:3]}...", file=sys.stderr)
            return 2
        items: list[dict] = []
        for condition in conditions:
            for task in fresh_tasks:
                clean_turns = max(task["min_turns"], int(calibration.get(task["id"], {}).get("turns") or 0))
                budget = resolve_budget(condition, clean_turns)
                for repeat in range(repeats[condition]):
                    for arm in ARMS:
                        items.append({"task": task, "condition": condition, "budget": budget,
                                      "clean_turns": clean_turns, "repeat": repeat, "arm": arm})
        print(f"\n=== eval: {spec['tier']} / {spec['label']} ({spec['model']}) — {len(items)} cells ===")
        client = make_client(spec, args.clean_max_turns)
        try:
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(run_one, client, spec, item, args): item for item in items}
                for done, future in enumerate(as_completed(futures), start=1):
                    record = future.result()
                    with lock:
                        all_records.append(record)
                        if done % 32 == 0 or done == len(items):
                            write_json(output, {"status": "running", "records": all_records})
                    print(f"  [{done:3d}/{len(items)}] {record['task']:40s} {record['condition']:8s} "
                          f"r{record['repeat']} {record['arm']:8s} success={record['success']!s:5s} "
                          f"turns={record['turns']}/{record['budget']} errors={record['tool_errors']} "
                          f"recoveries={record['guard_recoveries']}")
        except RuntimeError as exc:
            print(f"\nERROR: model {key} failed during eval: {exc}", file=sys.stderr)
            write_json(output, {"status": "partial_api_error", "model": key,
                                "records": all_records, "error": str(exc),
                                "generated_at": datetime.now(timezone.utc).isoformat()})
            print(f"wrote partial results to {output} (no offline fallback)", file=sys.stderr)
            return 2
        finally:
            client.close()
    wall_seconds = round(time.perf_counter() - started, 1)
    return finalize(all_records, specs, calibrations, calibration_payload, tasks, skipped,
                    args, conditions, repeats, wall_seconds, output, tasks_per_model)


def finalize(records: list[dict], specs: list[dict], calibrations: dict, calibration_payload: dict,
             tasks: list[dict], skipped: list[str], args: argparse.Namespace,
             conditions: list[str], repeats: dict[str, int], wall_seconds: float | None,
             output: Path, tasks_per_model: dict[str, list[dict]] | None = None) -> int:
    """Aggregate records (fresh run or rebuilt from a checkpoint) and write the artifact."""
    aggregates: dict[str, dict] = {}
    for spec in specs:
        model_records = [row for row in records if row["model_key"] == spec["key"]]
        aggregates[spec["key"]] = aggregate(model_records)
    total_cost = 0.0
    for spec in specs:
        for condition in conditions:
            for arm in ARMS:
                usage = aggregates[spec["key"]]["conditions"][condition]["arms"][arm]["usage"]
                total_cost += usage["prompt_tokens"] / 1e6 * spec["price_input_per_m"]
                total_cost += usage["completion_tokens"] / 1e6 * spec["price_output_per_m"]
    total_cost = round(total_cost, 4)
    cross = cross_model_summary(records, specs, aggregates, calibrations)

    findings: list[str] = []
    for spec in specs:
        block = cross[spec["key"]]
        tight = block["conditions"].get("clean+1", {})
        findings.append(
            f"{spec['tier']} ({spec['label']}, placement {block['capability_placement']['success']}, "
            f"avg {block['capability_placement']['avg_turns']} turns): clean+1 baseline "
            f"{tight.get('baseline_success')}/{tight.get('cells')} vs guard "
            f"{tight.get('guard_success')}/{tight.get('cells')} (McNemar p="
            f"{tight.get('mcnemar_exact_p', 1.0):.3g}), tokens "
            f"{tight.get('baseline_tokens')} -> {tight.get('guard_tokens')}"
        )
    findings.append(
        "guard >= baseline on every model/condition: "
        f"{cross['consistency']['guard_success_ge_baseline_on_all_models']}"
    )

    summary = {
        "artifact_version": 1,
        "experiment": "ToolGuard_multimodel_live_ab",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": {
            "models": [{"tier": spec["tier"], "key": spec["key"], "label": spec["label"],
                        "model": spec["model"], "provider": spec["provider"],
                        "price_input_per_m": spec["price_input_per_m"],
                        "price_output_per_m": spec["price_output_per_m"]} for spec in specs],
            "arms": list(ARMS),
            "budget_conditions": {
                condition: ("per-task turn budget = per-model fault-free calibration turns + "
                            f"{condition.split('+', 1)[1]}" if condition.startswith("clean+")
                            else f"{condition} turns for every task")
                for condition in conditions
            },
            "repeats": repeats,
            "workers": args.workers,
            "wall_seconds": wall_seconds,
            "finalised_from_checkpoint": bool(getattr(args, "from_records", False)),
            "failure_modes_recomputed": bool(getattr(args, "from_records", False)),
            "hn_schema": getattr(args, "hn_schema", "consistent"),
            "hn_schema_note": ("consistent contract: get_hn_item.story_id is an integer in the "
                               "agent-visible schema, matching the backend and the real HN API"
                               if getattr(args, "hn_schema", "consistent") == "consistent" else
                               "legacy drifting contract: get_hn_item advertises story_id as a "
                               "string while the backend requires an integer (historical V2/V3 "
                               "protocol)"),
            "failure_modes_note": (
                "rebuilt from the checkpoint: failure-mode labels were recomputed from "
                "answer+expected with the fixed metadata-aware classifier (2026-11-24 review), "
                "and tasks_per_model was rebuilt from the per-record expected snapshots"
                if getattr(args, "from_records", False) else
                "failure modes classified at run time with the fixed metadata-aware classifier"),
            "fault_injection": {
                "model": "deterministic transient fault epochs (first N attempts fail) per logical call",
                "attempts_to_fail": args.attempts_to_fail,
                "p_timeout": args.p_timeout,
                "p_rate_limit": args.p_rate_limit,
                "p_server_error": args.p_server_error,
                "seed": args.seed,
            },
            "calibration": {
                "source": str(getattr(args, "calibration", "")),
                "description": (
                    "per-model fault-free calibration reused from the V3 run so that the HN "
                    "schema is the only manipulated variable between the V3 and V4 artifacts"
                    if getattr(args, "from_records", False) is False
                    and "multimodel_calibration" in str(getattr(args, "calibration", "")) else
                    "per-model fault-free calibration payload stored at the top level"),
            },
            "verification": "programmatic; ground truth computed from the same live APIs before the runs",
            "ground_truth_refresh": "rebuilt immediately before each model's block to bound live drift "
                                    "(weather/FX); every record carries the `expected` snapshot it was "
                                    "verified against",
        },
        "calibration": calibration_payload,
        "tasks": [{key: value for key, value in task.items() if key != "verifier"} for task in tasks],
        "tasks_per_model": {
            key: [{field: value for field, value in task.items() if field != "verifier"}
                  for task in model_tasks]
            for key, model_tasks in (tasks_per_model or {}).items()
        },
        "skipped": skipped,
        "records": records,
        "aggregate": {"models": aggregates, "cross_model": cross},
        "key_findings": findings,
        "total_estimated_cost_usd": total_cost,
        "limitations": [
            "Multi-model axis is replication: each model's baseline-vs-guard comparison is paired; "
            "no formal between-model hypothesis test (3 models only).",
            "Transient faults are deterministic injections, not observed production failures; the "
            "budget conditions are controlled stress conditions (clean+1 / clean+4).",
            "One calibration run per task per model anchors the budget; repeat seeds vary the fault "
            "surface, not model sampling (temperature 0 where the provider supports it).",
            "Cheap and strong tiers are both Qwen-family models served by Alibaba Model Studio; the "
            "measured fault-free placement does not show the expensive flagship outperforming the "
            "mid-tier model on this tool-task set (see capability_placement).",
            "Live-mutable ground truth: the HN top-story author's karma and current weather can change "
            "between task construction and a run; a few guard failures are ground-truth drift or answer "
            "formatting (failure_mode/answer in the records), not middleware failures.",
            "V3 task-set change: the three HN karma tasks were replaced by immutable account-creation-year "
            "tasks (same two-tool chain). HN karma is live-mutable: during the V2 multi-model run one "
            "active author's karma changed 112 -> 161 within an hour, which made that task's ground truth "
            "noise. The V2 multi-model artifact (commit fb13a1f) still contains the karma tasks.",
            "When the artifact is finalised from a checkpoint (no re-run), task definitions are rebuilt "
            "after the run; per-cell success flags were computed at run time with the then-fresh ground "
            "truth, so drifting values in the metadata may differ slightly.",
            "Prices are the published list prices at run time and are used only for the cost estimate.",
            "Free public APIs are not production grade; API drift can affect reruns.",
            ("HN contract: this is the V4 run with the consistent integer get_hn_item.story_id "
             "schema (2026-11-25 review decision); the V3 drift artifact is kept as the contract-"
             "repair capability demo. Per-model budget calibration is reused from the V3 run so "
             "the HN schema is the only manipulated variable."
             if getattr(args, "hn_schema", "consistent") == "consistent" else
             "HN contract drift mode: get_hn_item advertises story_id as string while the backend "
             "requires an integer (historical V3 protocol)."),
        ],
        "reproduce": [
            "python scripts/run_multimodel_live.py --mode calibrate",
            "python scripts/run_multimodel_live.py --mode eval --hn-schema consistent --workers 2 "
            "--calibration .scratch/toolguard/multimodel_calibration.json "
            "--output benchmarks/toolguard/live_demo_multimodel_v4.json",
            "python scripts/run_multimodel_live.py --from-records "
            "--output benchmarks/toolguard/live_demo_multimodel_v4.json",
            "python -m pytest tests/test_toolguard -q -p no:cacheprovider",
        ],
    }
    write_json(output, summary)
    print("\n=== cross-model summary ===")
    print(json.dumps(cross, ensure_ascii=False, indent=2))
    print(f"\nestimated experiment cost: ${total_cost}")
    print(f"wrote {output} (wall {wall_seconds}s)")
    return 0


def _tasks_per_model_from_records(records: list[dict], tasks: list[dict]) -> dict[str, list[dict]]:
    """Reconstruct each model's ground-truth snapshot from the checkpoint records.

    ``--from-records`` rebuilds the artifact from the checkpoint, so the per-model `expected`
    snapshots must come from the records themselves (each record stores the snapshot it was
    verified against) instead of being silently dropped to ``{}`` (2026-11-24 review P2-2).
    """
    by_id = {task["id"]: task for task in tasks}
    order = {task["id"]: index for index, task in enumerate(tasks)}
    per_model: dict[str, dict[str, dict]] = {}
    for row in records:
        key = row.get("model_key")
        if not key:
            continue
        bucket = per_model.setdefault(key, {})
        if row["task"] in bucket:
            continue
        base = by_id.get(row["task"]) or {"id": row["task"]}
        snapshot = {field: value for field, value in base.items() if field != "verifier"}
        if "expected" in row:
            snapshot["expected"] = row["expected"]
        if "category" in row:
            snapshot["category"] = row["category"]
        bucket[row["task"]] = snapshot
    return {
        key: sorted(bucket.values(), key=lambda task: order.get(task["id"], 10 ** 9))
        for key, bucket in per_model.items()
    }


def run_from_records(tasks: list[dict], skipped: list[str], specs: list[dict],
                     conditions: list[str], repeats: dict[str, int],
                     args: argparse.Namespace) -> int:
    """Rebuild the final artifact from a checkpoint records file (no new model calls)."""
    output = Path(args.output)
    checkpoint = json.loads(output.read_text(encoding="utf-8"))
    records = checkpoint.get("records", [])
    print(f"loaded {len(records)} records from checkpoint {output}")
    missing_models = sorted({spec["key"] for spec in specs} - {row["model_key"] for row in records})
    if missing_models:
        print(f"ERROR: checkpoint is missing records for models: {missing_models}", file=sys.stderr)
        return 2
    tasks_per_model = _tasks_per_model_from_records(records, tasks)
    # Recompute failure-mode labels from answer + expected with the fixed metadata-aware
    # classifier: the committed artifact's labels were produced by the old classifier, which
    # let expected-dict metadata (`story_id`, lat/lon, ...) masquerade as format misses.
    reclassified = 0
    for row in records:
        if row.get("success"):
            continue
        expected = row.get("expected")
        if expected is None:
            continue
        old = row.get("failure_mode")
        new = classify_failure(row.get("answer") or "", expected)
        if new != old:
            reclassified += 1
        row["failure_mode"] = new
    print(f"failure-mode labels recomputed from answer+expected; changed: {reclassified}")
    missing_snapshots = [key for key in tasks_per_model
                         if any("expected" not in task for task in tasks_per_model[key])]
    if missing_snapshots:
        print(f"WARNING: checkpoint records for {missing_snapshots} carry no per-record "
              "`expected` snapshot (older artifact); tasks_per_model will fall back to the "
              "current task metadata for those models.", file=sys.stderr)
    calibration_payload = json.loads(Path(args.calibration).read_text(encoding="utf-8"))
    calibrations = calibration_payload.get("models", {})
    return finalize(records, specs, calibrations, calibration_payload, tasks, skipped,
                    args, conditions, repeats, None, output, tasks_per_model)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("calibrate", "eval"), default="eval")
    parser.add_argument("--from-records", action="store_true",
                        help="rebuild the final artifact from the checkpoint records already at --output")
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
    parser.add_argument("--hn-schema", choices=("drifting", "consistent"), default="consistent",
                        help="consistent (default): get_hn_item.story_id integer in the "
                             "agent-visible schema; drifting: legacy mismatched schema kept only "
                             "for rebuilding the historical drift artifact")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if args.from_records and Path(args.output).exists():
        checkpoint = json.loads(Path(args.output).read_text(encoding="utf-8"))
        protocol = checkpoint.get("protocol") or {}
        if "hn_schema" in protocol:
            args.hn_schema = protocol["hn_schema"]
        else:  # legacy drift artifacts (V2/V3) predate the field
            args.hn_schema = "drifting"

    conditions = parse_budgets(args.budgets)
    repeats = parse_repeats(args.repeats, conditions)
    specs = model_specs()

    print("model axis:")
    for spec in specs:
        state = "key configured" if spec["api_key"] else "MISSING KEY"
        print(f"  {spec['tier']:6s} {spec['label']:26s} model={spec['model']:18s} "
              f"{spec['provider']:38s} {state}")

    clean = LiveHardBackend(hn_schema=args.hn_schema)
    tasks, skipped = build_tasks(clean)
    if args.max_tasks is not None:
        tasks = tasks[:args.max_tasks]
    print(f"\nbuilt {len(tasks)} multi-step tasks; skipped ground-truth groups: {len(skipped)}")
    for note in skipped:
        print(f"  skip: {note}")
    if args.dry_run:
        for task in tasks:
            print(f"  {task['id']:42s} {task['category']:10s} min_turns>={task['min_turns']} "
                  f"expected={json.dumps(task['expected'], ensure_ascii=False)[:100]}")
        return 0

    if args.from_records:
        return run_from_records(tasks, skipped, specs, conditions, repeats, args)

    missing = [spec["key"] for spec in specs if not spec["api_key"]]
    if missing:
        print(f"\nERROR: missing API keys for models: {missing}. "
              "Set TOOLEVO_QWEN_API_KEY / TOOLEVO_DEEPSEEK_API_KEY in .env "
              "(see .env.example). No offline/mock fallback.", file=sys.stderr)
        return 2

    if args.mode == "calibrate":
        return run_calibration(tasks, specs, args)
    return run_eval(tasks, skipped, specs, conditions, repeats, args)


if __name__ == "__main__":
    raise SystemExit(main())
