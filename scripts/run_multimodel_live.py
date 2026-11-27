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
# Shared read-only budget anchor of the committed V3/V4 artifacts.  It must never be
# rewritten by a reproduction step: a fresh calibration is an explicit, separate action that
# writes to its own *_fresh.json path (see _fresh_calibration_path).
DEFAULT_CALIBRATION = ".scratch/toolguard/multimodel_calibration.json"
FRESH_CALIBRATION_DIR = ".scratch/toolguard"
CALIBRATION_MODES = ("fresh", "reused")


def _posix(value) -> str:
    """Render a path for shell commands: forward slashes even when generated on Windows."""
    return str(value).replace("\\", "/")


def _fresh_calibration_path(output: Path) -> str:
    """Separate, explicitly-fresh calibration path derived from the artifact label.

    The committed artifacts share ``multimodel_calibration.json`` as a read-only input, so a
    fresh re-measurement must never default to that path.
    """
    label = Path(output).stem.removeprefix("live_demo_multimodel_") or "fresh"
    return f".scratch/toolguard/multimodel_calibration_{label}_fresh.json"


def calibration_write_error(path, overwrite: bool) -> str | None:
    """Refuse to silently overwrite an existing calibration file (read-only artifact input)."""
    target = Path(str(path))
    if target.exists() and not overwrite:
        return (f"refusing to overwrite existing calibration {target}: the V3/V4 artifacts use "
                "the per-model calibration as a read-only budget input. Pass "
                "--overwrite-calibration to replace it explicitly, or write a fresh "
                "calibration to a separate path (e.g. "
                f"{FRESH_CALIBRATION_DIR}\\multimodel_calibration_fresh.json).")
    return None


def _calibration_provenance(source: str | None, mode: str, reused_from: str | None,
                            hn_schema: str, output: Path, workers: int | None) -> dict:
    """Explicit calibration provenance (never inferred from the file name)."""
    mode = mode or "checkpoint"
    if mode == "reused":
        description = (
            f"per-model fault-free calibration reused from {reused_from}; the reproduce "
            "commands use it as a read-only input and never regenerate it, so the HN schema is "
            "the only manipulated variable between the V3 and V4 artifacts")
    elif mode == "fresh":
        description = (
            "per-model fault-free calibration measured once for this run (fresh). The reproduce "
            "commands use the recorded calibration as a read-only input and do not regenerate "
            "it; a fresh calibration is a separate explicit action (see fresh_calibration)")
    else:
        description = (
            "per-model calibration payload restored from the checkpoint; calibration provenance "
            "is recorded in mode/reused_from and is not re-derived from the file name")
    fresh_path = _fresh_calibration_path(output)
    return {
        "source": _posix(source) if source else "embedded in the artifact calibration payload",
        "mode": mode,
        "reused_from": reused_from if mode == "reused" else None,
        "required_input": True,
        "reproduce_regenerates_calibration": False,
        "description": description,
        "fresh_calibration": {
            "command": " ".join([
                "python scripts/run_multimodel_live.py --mode calibrate",
                f"--hn-schema {hn_schema}",
                f"--workers {workers or 2}",
                f"--calibration {fresh_path}"]),
            "note": ("fresh calibration, not the exact V3/V4 artifact reproduction: it "
                     "re-measures the per-model fault-free turn budget and must not overwrite "
                     f"the shared read-only anchor; the artifact itself is reproduced by the "
                     "eval/from-records commands"),
        },
    }


def _calibration_payload_with_provenance(payload: dict | None, provenance: dict) -> dict:
    """Attach provenance metadata to the top-level calibration payload (models untouched)."""
    updated = dict(payload or {})
    updated["provenance"] = {
        "mode": provenance.get("mode"),
        "reused_from": provenance.get("reused_from"),
        "required_input": provenance.get("required_input"),
        "reproduce_regenerates_calibration": provenance.get(
            "reproduce_regenerates_calibration"),
        "note": "see protocol.calibration for the full provenance and fresh-calibration command",
    }
    return updated


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
    write_error = calibration_write_error(args.calibration,
                                          getattr(args, "overwrite_calibration", False))
    if write_error:
        print(f"ERROR: {write_error}", file=sys.stderr)
        return 2
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
        "provenance": {
            "mode": "fresh",
            "hn_schema": getattr(args, "hn_schema", "consistent"),
            "generated_for": ("explicit --mode calibrate run (fresh calibration; not an artifact "
                              "reproduction)"),
        },
        "models": calibration,
    }
    path = Path(args.calibration)
    write_json(path, payload)
    failed = {key: row["placement"]["success"] for key, row in calibration.items()}
    print(f"\ncalibration placement: {failed}")
    print(f"wrote fresh calibration {path} (not an artifact reproduction; the committed "
          "artifacts take their calibration as a read-only input)")
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
        print(
            f"ERROR: calibration file missing at {calibration_path}. The per-model budget "
            "anchor is a REQUIRED input; --mode eval never calibrates automatically.\n"
            "  * reproduce the committed artifacts offline: --from-records --output <artifact>\n"
            "  * or create a fresh calibration explicitly (this is NOT the exact artifact "
            "reproduction):\n"
            "      python scripts/run_multimodel_live.py --mode calibrate "
            f"--calibration {_fresh_calibration_path(Path(args.output))}",
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
                    args, conditions, repeats, wall_seconds, output, tasks_per_model,
                    calibration_mode=getattr(args, "calibration_mode", None) or "fresh",
                    calibration_reused_from=getattr(args, "calibration_reused_from", None))


def _artifact_metadata(*, specs: list[dict], tasks: list[dict], args: argparse.Namespace,
                       conditions: list[str], repeats: dict[str, int],
                       wall_seconds: float | None, output: Path,
                       calibration_source: str | None,
                       calibration_provenance: dict,
                       finalised_from_checkpoint: bool) -> dict:
    """Metadata block (experiment/generated_at/protocol/limitations/reproduce).

    Every reproduce command is generated from the arguments that produced *this* artifact
    (actual output path, HN schema, budgets, repeats, workers, calibration input).  The
    calibration anchor is a **required read-only input**: the reproduce commands only replay
    the eval / rebuild the artifact from its checkpoint, they never regenerate the calibration
    (a fresh calibration is a separate explicit action recorded in
    ``protocol.calibration.fresh_calibration``).
    """
    output_path = _posix(output)
    workers = getattr(args, "workers", None)
    hn_schema = getattr(args, "hn_schema", "consistent")
    budgets_spec = ",".join(conditions) or DEFAULT_BUDGETS
    repeats_spec = ",".join(f"{condition}:{repeats.get(condition, 1)}"
                            for condition in conditions) or DEFAULT_REPEATS
    calibration = _posix(calibration_source) if calibration_source else None
    provenance = calibration_provenance
    reproduce: list[str] = []
    eval_command = ["python scripts/run_multimodel_live.py --mode eval",
                    f"--hn-schema {hn_schema}",
                    f"--budgets {budgets_spec}",
                    f"--repeats {repeats_spec}",
                    f"--workers {workers or 2}"]
    if calibration:
        eval_command.append(f"--calibration {calibration}")
    if provenance.get("mode") == "reused" and provenance.get("reused_from"):
        eval_command.extend(["--calibration-mode reused",
                             f"--calibration-reused-from {provenance['reused_from']}"])
    eval_command.append(f"--output {output_path}")
    reproduce.append(" ".join(eval_command))
    # An offline rebuild from the committed checkpoint records is always a valid
    # (zero-API-cost) reproduction path for this artifact, whether or not it was finalised
    # from a checkpoint in the first place.
    reproduce.append("python scripts/run_multimodel_live.py --from-records "
                     f"--output {output_path}")
    reproduce.append("python -m pytest tests/test_toolguard -q -p no:cacheprovider")
    protocol = {
        "models": [{"tier": spec["tier"], "key": spec["key"], "label": spec["label"],
                    "model": spec["model"], "provider": spec["provider"],
                    "price_input_per_m": spec["price_input_per_m"],
                    "price_output_per_m": spec["price_output_per_m"]} for spec in specs],
        "arms": list(ARMS),
        "task_ids": [task["id"] for task in tasks],
        "output_artifact": output_path,
        "budget_conditions": {
            condition: ("per-task turn budget = per-model fault-free calibration turns + "
                        f"{condition.split('+', 1)[1]}" if condition.startswith("clean+")
                        else f"{condition} turns for every task")
            for condition in conditions
        },
        "repeats": repeats,
        "workers": workers,
        "wall_seconds": wall_seconds,
        "finalised_from_checkpoint": finalised_from_checkpoint,
        "failure_modes_recomputed": finalised_from_checkpoint,
        "hn_schema": hn_schema,
        "hn_schema_note": (
            "consistent contract: get_hn_item.story_id is an integer in the "
            "agent-visible schema, matching the backend and the real HN API"
            if hn_schema == "consistent" else
            "legacy drifting contract: get_hn_item advertises story_id as a "
            "string while the backend requires an integer (historical V2/V3 "
            "protocol)"),
        "failure_modes_note": (
            "rebuilt from the checkpoint: failure-mode labels were recomputed from "
            "answer+expected with the fixed metadata-aware classifier (2026-11-24 review), "
            "and tasks_per_model was rebuilt from the per-record expected snapshots"
            if finalised_from_checkpoint else
            "failure modes classified at run time with the fixed metadata-aware classifier"),
        "fault_injection": {
            "model": "deterministic transient fault epochs (first N attempts fail) per logical call",
            "attempts_to_fail": getattr(args, "attempts_to_fail", 2),
            "p_timeout": getattr(args, "p_timeout", 0.20),
            "p_rate_limit": getattr(args, "p_rate_limit", 0.10),
            "p_server_error": getattr(args, "p_server_error", 0.08),
            "seed": getattr(args, "seed", 7),
        },
        "calibration": provenance,
        "verification": "programmatic; ground truth computed from the same live APIs before the runs",
        "ground_truth_refresh": "rebuilt immediately before each model's block to bound live drift "
                                "(weather/FX); every record carries the `expected` snapshot it was "
                                "verified against",
    }
    calibration_limitation = (
        f"Budget calibration provenance: reused from the {provenance['reused_from']} run as a "
        "read-only per-model input; the reproduce commands (eval / from-records) never "
        "regenerate it and a fresh calibration is a separate explicit action "
        "(protocol.calibration.fresh_calibration), so the HN schema is the only manipulated "
        "variable between the V3 and V4 artifacts."
        if provenance.get("mode") == "reused" else
        "Budget calibration provenance: per-model fault-free calibration stored in this "
        "artifact is a required read-only input; the reproduce commands (eval / from-records) "
        "take it as-is and never regenerate it. A fresh calibration is a separate explicit "
        "action (protocol.calibration.fresh_calibration), not an artifact reproduction.")
    hn_contract_limitation = (
        ("HN contract: this is the V4 run with the consistent integer get_hn_item.story_id "
         "schema (2026-11-25 review decision); the V3 drift artifact is kept as the contract-"
         "repair capability demo. Per-model budget calibration is reused from the V3 run so "
         "the HN schema is the only manipulated variable."
         if provenance.get("mode") == "reused" else
         "HN contract: the agent-visible get_hn_item.story_id schema is the consistent integer "
         "form (2026-11-25 review decision); the historical V3 drift artifact is kept as the "
         "contract-repair capability demo.")
        if hn_schema == "consistent" else
        "HN contract drift mode: get_hn_item advertises story_id as string while the backend "
        "requires an integer (historical V3 protocol).")
    limitations = [
        "Multi-model axis is replication: each model's baseline-vs-guard comparison is paired; "
        "no formal between-model hypothesis test (3 models only).",
        "Transient faults are deterministic injections, not observed production failures; the "
        "budget conditions are controlled stress conditions (clean+1 / clean+4).",
        calibration_limitation,
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
        hn_contract_limitation,
    ]
    return {
        "experiment": "ToolGuard_multimodel_live_ab",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol": protocol,
        "limitations": limitations,
        "reproduce": reproduce,
    }


def finalize(records: list[dict], specs: list[dict], calibrations: dict, calibration_payload: dict,
             tasks: list[dict], skipped: list[str], args: argparse.Namespace,
             conditions: list[str], repeats: dict[str, int], wall_seconds: float | None,
             output: Path, tasks_per_model: dict[str, list[dict]] | None = None,
             calibration_source: str | None = None,
             finalised_from_checkpoint: bool | None = None,
             calibration_mode: str | None = None,
             calibration_reused_from: str | None = None) -> int:
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

    calibration_source = (calibration_source
                          or (str(args.calibration) if getattr(args, "calibration", None) else None))
    finalised = (bool(getattr(args, "from_records", False))
                 if finalised_from_checkpoint is None else finalised_from_checkpoint)
    mode = calibration_mode or getattr(args, "calibration_mode", None) or "fresh"
    reused_from = (calibration_reused_from if calibration_reused_from is not None
                   else getattr(args, "calibration_reused_from", None))
    provenance = _calibration_provenance(
        source=calibration_source, mode=mode, reused_from=reused_from,
        hn_schema=getattr(args, "hn_schema", "consistent"), output=output,
        workers=getattr(args, "workers", None))
    metadata = _artifact_metadata(
        specs=specs, tasks=tasks, args=args, conditions=conditions, repeats=repeats,
        wall_seconds=wall_seconds, output=output,
        calibration_source=calibration_source,
        calibration_provenance=provenance,
        finalised_from_checkpoint=finalised)
    summary = {
        "artifact_version": 1,
        "experiment": metadata["experiment"],
        "generated_at": metadata["generated_at"],
        "protocol": metadata["protocol"],
        "calibration": _calibration_payload_with_provenance(calibration_payload, provenance),
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
        "limitations": metadata["limitations"],
        "reproduce": metadata["reproduce"],
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


def _load_checkpoint(output: Path) -> dict | None:
    if not output.exists():
        print(f"ERROR: checkpoint {output} does not exist", file=sys.stderr)
        return None
    checkpoint = json.loads(output.read_text(encoding="utf-8"))
    if not checkpoint.get("records") or not checkpoint.get("tasks"):
        print(f"ERROR: checkpoint {output} has no records/tasks", file=sys.stderr)
        return None
    return checkpoint


def _checkpoint_conditions(checkpoint: dict) -> tuple[list[str], dict[str, int]]:
    protocol = checkpoint.get("protocol") or {}
    conditions = list((protocol.get("budget_conditions") or {}).keys())
    repeats = dict(protocol.get("repeats") or {})
    if not conditions:
        conditions = sorted({row["condition"] for row in checkpoint["records"]})
    if not repeats:
        task_count = len(checkpoint.get("tasks") or [])
        for condition in conditions:
            rows = [row for row in checkpoint["records"] if row.get("condition") == condition]
            repeats[condition] = max(1, len(rows) // max(task_count * len(ARMS), 1))
    return conditions, repeats


def _inherit_hn_schema(checkpoint: dict, args: argparse.Namespace) -> None:
    protocol = checkpoint.get("protocol") or {}
    args.hn_schema = protocol.get("hn_schema") or "drifting"


def _checkpoint_calibration(checkpoint: dict, args: argparse.Namespace) -> dict:
    path = Path(getattr(args, "calibration", "") or "")
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    payload = checkpoint.get("calibration")
    if isinstance(payload, dict) and payload.get("models"):
        return payload
    return {"models": {}}


def _checkpoint_calibration_source(checkpoint: dict, args: argparse.Namespace) -> str | None:
    protocol = checkpoint.get("protocol") or {}
    block = protocol.get("calibration")
    source = block.get("source") if isinstance(block, dict) else None
    if not source:
        explicit = getattr(args, "calibration", None)
        if explicit and Path(str(explicit)).is_file():
            source = str(explicit)
    return source


def _checkpoint_calibration_provenance(
        checkpoint: dict, args: argparse.Namespace) -> tuple[str, str | None] | None:
    """Recover explicit calibration provenance from the checkpoint, never from file names.

    A recorded ``protocol.calibration.mode`` always wins (so ``--from-records`` /
    ``--update-metadata`` preserve the artifact's provenance).  Only when the checkpoint
    predates the provenance fields are the explicit CLI flags used; without them the mode is
    recorded as ``checkpoint`` rather than guessed from the calibration path.
    """
    protocol = checkpoint.get("protocol") or {}
    block = protocol.get("calibration")
    block = block if isinstance(block, dict) else {}
    mode = block.get("mode")
    reused_from = block.get("reused_from")
    if mode not in CALIBRATION_MODES + ("checkpoint",):
        cli_mode = getattr(args, "calibration_mode", None)
        mode = cli_mode or "checkpoint"
    if mode == "reused":
        reused_from = reused_from or getattr(args, "calibration_reused_from", None)
        if not reused_from:
            print("ERROR: checkpoint records calibration mode 'reused' without a reused_from "
                  "label; pass --calibration-reused-from (no file-name inference is performed)",
                  file=sys.stderr)
            return None
    else:
        reused_from = None
    return mode, reused_from


def run_from_checkpoint(specs: list[dict], args: argparse.Namespace) -> int:
    """Rebuild the artifact from its committed checkpoint records (offline, no API calls).

    The checkpoint is read straight from ``--output``: live task construction (which calls
    public APIs) is skipped entirely, the aggregate is recomputed from the records, the
    failure-mode labels are recomputed from ``answer`` + ``expected`` with the fixed
    classifier, and the reproduce block is regenerated for this artifact's actual path.
    """
    output = Path(args.output)
    checkpoint = _load_checkpoint(output)
    if checkpoint is None:
        return 2
    records = checkpoint["records"]
    tasks = checkpoint["tasks"]
    skipped = checkpoint.get("skipped") or []
    conditions, repeats = _checkpoint_conditions(checkpoint)
    calibration_payload = _checkpoint_calibration(checkpoint, args)
    calibrations = calibration_payload.get("models") or {}
    if not calibrations:
        print(f"ERROR: no per-model calibration available for {output}; pass --calibration "
              "or use a checkpoint that embeds the calibration payload", file=sys.stderr)
        return 2
    print(f"loaded {len(records)} records / {len(tasks)} tasks from checkpoint {output}")
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
    protocol = checkpoint.get("protocol") or {}
    args.workers = protocol.get("workers") or getattr(args, "workers", 2)
    provenance = _checkpoint_calibration_provenance(checkpoint, args)
    if provenance is None:
        return 2
    return finalize(records, specs, calibrations, calibration_payload, tasks, skipped,
                    args, conditions, repeats, None, output, tasks_per_model,
                    calibration_source=_checkpoint_calibration_source(checkpoint, args),
                    finalised_from_checkpoint=True,
                    calibration_mode=provenance[0],
                    calibration_reused_from=provenance[1])


def update_metadata_only(specs: list[dict], args: argparse.Namespace) -> int:
    """Refresh only metadata; records/aggregate/tasks/calibration payload stay byte-identical.

    Used to fix historical artifacts whose reproduce block named the wrong output path: the
    data payload is copied from the checkpoint verbatim and only experiment/protocol/
    limitations/reproduce are regenerated from the recorded protocol parameters.
    """
    output = Path(args.output)
    checkpoint = _load_checkpoint(output)
    if checkpoint is None:
        return 2
    protocol = checkpoint.get("protocol") or {}
    conditions, repeats = _checkpoint_conditions(checkpoint)
    args.workers = protocol.get("workers") or getattr(args, "workers", 2)
    provenance_mode = _checkpoint_calibration_provenance(checkpoint, args)
    if provenance_mode is None:
        return 2
    calibration_source = _checkpoint_calibration_source(checkpoint, args)
    provenance = _calibration_provenance(
        source=calibration_source, mode=provenance_mode[0],
        reused_from=provenance_mode[1], hn_schema=getattr(args, "hn_schema", "consistent"),
        output=output, workers=args.workers)
    metadata = _artifact_metadata(
        specs=specs, tasks=checkpoint["tasks"], args=args, conditions=conditions,
        repeats=repeats, wall_seconds=protocol.get("wall_seconds"), output=output,
        calibration_source=calibration_source,
        calibration_provenance=provenance,
        finalised_from_checkpoint=bool(protocol.get("finalised_from_checkpoint")))
    # keep the recorded generation time; only metadata content is refreshed
    metadata["generated_at"] = checkpoint.get("generated_at", metadata["generated_at"])
    payload = dict(checkpoint)
    payload.update(metadata)
    # refresh only the provenance block of the top-level calibration payload (models untouched)
    payload["calibration"] = _calibration_payload_with_provenance(
        checkpoint.get("calibration"), provenance)
    write_json(output, payload)
    changed = sorted(key for key in metadata if checkpoint.get(key) != metadata[key])
    print(f"refreshed metadata for {output}; changed fields: {', '.join(changed)}")
    print(f"calibration provenance: mode={provenance['mode']} "
          f"reused_from={provenance['reused_from']}")
    print("records/aggregate/tasks/tasks_per_model and calibration model payload were left untouched")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("calibrate", "eval"), default="eval")
    parser.add_argument("--from-records", action="store_true",
                        help="rebuild the final artifact from the checkpoint records already at --output "
                             "(offline: no live task construction / no API calls)")
    parser.add_argument("--update-metadata", action="store_true",
                        help="refresh ONLY experiment/protocol/limitations/reproduce of the "
                             "artifact at --output from its checkpoint; records/aggregate/tasks/"
                             "calibration payload are copied verbatim (offline)")
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
    parser.add_argument("--calibration-mode", choices=CALIBRATION_MODES, default=None,
                        help="explicit provenance of the calibration anchor: 'fresh' (measured "
                             "for this run) or 'reused' (imported from another artifact, requires "
                             "--calibration-reused-from). Never inferred from the file name; "
                             "--mode eval never recalibrates")
    parser.add_argument("--calibration-reused-from", default=None,
                        help="artifact label the calibration was reused from (required when "
                             "--calibration-mode reused, e.g. V3)")
    parser.add_argument("--overwrite-calibration", action="store_true",
                        help="explicitly allow --mode calibrate to overwrite an existing "
                             "calibration file (the shared V3/V4 anchor is a read-only input)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.calibration_mode == "reused" and not args.calibration_reused_from:
        parser.error("--calibration-mode reused requires --calibration-reused-from")
    if args.calibration_reused_from and args.calibration_mode != "reused" \
            and not (args.from_records or args.update_metadata):
        parser.error("--calibration-reused-from only applies with --calibration-mode reused")
    specs = model_specs()

    if args.from_records or args.update_metadata:
        checkpoint = _load_checkpoint(Path(args.output))
        if checkpoint is None:
            return 2
        _inherit_hn_schema(checkpoint, args)
        if args.update_metadata:
            return update_metadata_only(specs, args)
        return run_from_checkpoint(specs, args)

    conditions = parse_budgets(args.budgets)
    repeats = parse_repeats(args.repeats, conditions)

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
