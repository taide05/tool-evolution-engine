"""MCP integration demo, V2: two real MCP servers, real transport crash, catalog drift.

This is the V2 of ``scripts/run_mcp_demo.py``.  It upgrades the MCP story from "the
middleware's retry also works over MCP" to a transport/deployment-level validation:

* **two MCP (stdio) servers** are started as separate processes with disjoint tool groups
  (``--group worldbank`` / ``--group market``); the client adapter merges them, so tool
  routing happens across servers;
* **real transport failure**: the World Bank server process is configured to die abruptly
  during its Nth call.  The stdio session really breaks; the adapter reconnects a fresh
  server process and the middleware retries the call as a retryable ``request_error``;
* **catalog drift / unknown tool**: the client advertises a drifted tool name
  ``fetch_exchange_rate`` that no server registers (a stale client config after a rename).
  Calls fail with an MCP unknown-tool error and the middleware resolves the name against
  the real tool catalogue with the same similarity rule validated offline;
* **server-side transient fault epochs** on top (deterministic, disclosed);
* same multi-step task pool as the function-tool demo (16 representative tasks), a
  fault-free calibration for the budget anchor, budget sweep ``clean+1`` / ``clean+4`` and
  repeats on the tight condition; programmatic verification, no LLM judge.

Single model by decision: ``deepseek-v4-flash``.  MCP stdio needs real pipe handles, so
this runner needs full local access (the test suite uses the SDK in-memory transport).

Usage
-----
    python scripts/run_mcp_demo_v2.py --mode calibrate
    python scripts/run_mcp_demo_v2.py --dry-run
    python scripts/run_mcp_demo_v2.py --workers 2 --output benchmarks/toolguard/mcp_demo_v2.json
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
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
    key_findings,
    parse_budgets,
    parse_repeats,
    resolve_budget,
)

from tool_evolution.toolguard.agent import DeepSeekClient, LLMConfig, run_task  # noqa: E402
from tool_evolution.toolguard.guard import ToolGuard  # noqa: E402
from tool_evolution.toolguard.mcp_backend import MCPServer, MCPBridge, MCPMultiBackend  # noqa: E402
from tool_evolution.utils.config import settings  # noqa: E402

MODEL = "deepseek-v4-flash"
DEFAULT_OUTPUT = ".scratch/toolguard/mcp_demo_v2.json"
DEFAULT_CALIBRATION = ".scratch/toolguard/mcp_calibration_v2.json"
SERVER_SCRIPT = REPO / "scripts" / "mcp_toolguard_server.py"
ARMS = ("baseline", "guard")
DEFAULT_BUDGETS = "clean+1,clean+4"
DEFAULT_REPEATS = "clean+1:2,clean+4:1"
DRIFT_ALIAS = "fetch_exchange_rate"
DRIFT_REAL_TOOL = "get_exchange_rate"
# 16 representative multi-step tasks from the shared pool; two FX tasks force the agent
# through the drifted alias, the rest cover chains/routing/crash on the worldbank server.
TASK_IDS = (
    "capital_weather_japan",
    "capital_weather_norway",
    "capital_weather_compare_france_germany",
    "capital_weather_triple",
    "capital_population_canberra",
    "capital_population_ottawa",
    "capital_gdp_switzerland",
    "capital_income_brazil",
    "population_total_three_capitals",
    "population_compare_oslo_wellington",
    "pypi_release_date_httpx",
    "pypi_newer_requests_flask",
    "hn_item_1_author_account_year",
    "hn_item_8863_author_account_year",
    "fx_eur_jpy_cross",
    "fx_triangular_check",
)


def mcp_sdk_version() -> str:
    try:
        return importlib.metadata.version("mcp")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - defensive
        return "unknown"


def start_multi_backend(fault: dict | None, crash_after: int) -> MCPMultiBackend:
    """Start the two MCP servers (fresh processes) and merge their tool catalogues."""
    base_env = dict(os.environ)
    if fault:
        base_env.update({
            "TOOLGUARD_MCP_FAULT_RATE": str(fault["rate"]),
            "TOOLGUARD_MCP_FAULT_SEED": str(fault["seed"]),
            "TOOLGUARD_MCP_FAULT_EPOCH": str(fault["epoch"]),
        })
    if crash_after > 0:
        worldbank_env = dict(base_env)
        worldbank_env["TOOLGUARD_MCP_CRASH_AFTER"] = str(crash_after)
    else:
        worldbank_env = dict(base_env)
    # Crash-once semantics: the restarted process must not crash again.
    worldbank_restart_env = dict(base_env)
    worldbank_bridge = MCPBridge(
        [sys.executable, str(SERVER_SCRIPT), "--group", "worldbank"],
        cwd=REPO, env=worldbank_env, restart_env=worldbank_restart_env,
    )
    market_bridge = MCPBridge(
        [sys.executable, str(SERVER_SCRIPT), "--group", "market"],
        cwd=REPO, env=dict(base_env), restart_env=dict(base_env),
    )
    worldbank_bridge.start()
    market_bridge.start()
    multi = MCPMultiBackend([MCPServer("worldbank", worldbank_bridge), MCPServer("market", market_bridge)])
    multi.load_tools()
    # Catalog drift: advertise the old name, hide the real one; calls to the alias reach the
    # server as unknown tools and are resolved by the middleware.
    multi.hide(DRIFT_REAL_TOOL)
    multi.advertise_alias(DRIFT_ALIAS, DRIFT_REAL_TOOL, "market")
    return multi


def _bridge_restarts(multi: MCPMultiBackend) -> int:
    return sum(server.bridge.restarts for server in multi.servers)


def run_calibration(client: DeepSeekClient, tasks: list[dict], args: argparse.Namespace) -> int:
    calibration: dict[str, dict] = {}
    print("\n=== calibration: fault-free MCP turn requirement (middleware on) ===")
    multi = start_multi_backend(fault=None, crash_after=0)
    try:
        guard = ToolGuard(multi, sleep_scale=0.05)
        for task in tasks:
            run = run_task(client, multi, task["question"], guard, max_turns=args.clean_max_turns)
            success = bool(task["verifier"](run.answer, task["expected"]))
            calibration[task["id"]] = {"turns": run.turns, "success": success, "exhausted": run.exhausted}
            print(f"  {task['id']:40s} success={success!s:5s} turns={run.turns} calls={run.tool_calls} "
                  f"repairs={guard.stats['recovered']}")
    finally:
        multi.close()
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "protocol": "fault-free MCP run per task with the middleware enabled (repairs are free), "
                    "used as the turn-requirement anchor for the budget sweep",
        "tasks": calibration,
    }
    path = Path(args.calibration)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    failed = [task_id for task_id, row in calibration.items() if not row["success"]]
    print(f"\ncalibration success {len(calibration) - len(failed)}/{len(calibration)}; wrote {path}")
    if failed:
        print(f"WARNING: tasks not solvable in a fault-free run: {failed}")
    return 0


def run_one(client: DeepSeekClient, item: dict, args: argparse.Namespace) -> dict:
    task = item["task"]
    for attempt in range(2):
        multi = None
        try:
            multi = start_multi_backend(fault={
                "rate": args.p_fault, "seed": args.seed + item["repeat"] * 101,
                "epoch": args.attempts_to_fail,
            }, crash_after=args.crash_after)
            middleware = ToolGuard(multi, sleep_scale=0.05) if item["arm"] == "guard" else None
            run = run_task(client, multi, task["question"], middleware, max_turns=item["budget"])
            success = bool(task["verifier"](run.answer, task["expected"]))
            record = {
                "task": task["id"],
                "category": task["category"],
                "condition": item["condition"],
                "budget": item["budget"],
                "clean_turns": item["clean_turns"],
                "repeat": item["repeat"],
                "fault_seed": args.seed + item["repeat"] * 101,
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
                "faults_injected": 0,
                "fault_kinds": {},
                "guard_recoveries": middleware.stats["recovered"] if middleware is not None else 0,
                "guard_retry_sleep_ms": round(middleware.stats["retry_sleep_s"] * 1000, 2)
                if middleware is not None else 0.0,
                "guard_events": run.guard_events,
                "mcp_calls": multi.stats["calls"],
                "mcp_errors": multi.stats["errors"],
                "mcp_restarts": _bridge_restarts(multi),
            }
            return record
        except RuntimeError as exc:
            if attempt == 1:
                raise
            print(f"  retrying cell after MCP/LLM error: {exc}", file=sys.stderr)
            time.sleep(5.0)
        finally:
            if multi is not None:
                multi.close()
    raise RuntimeError("unreachable")


def run_eval(client: DeepSeekClient, tasks: list[dict], skipped: list[str],
             conditions: list[str], repeats: dict[str, int], tool_names: list[str],
             advertised: list[str], args: argparse.Namespace) -> int:
    calibration: dict[str, dict] = {}
    calibration_path = Path(args.calibration)
    if calibration_path.exists():
        calibration = json.loads(calibration_path.read_text(encoding="utf-8")).get("tasks", {})
        print(f"loaded calibration for {len(calibration)} tasks from {calibration_path}")
    else:
        print(f"WARNING: no calibration file at {calibration_path}; budgets fall back to min_turns")

    items: list[dict] = []
    for condition in conditions:
        for task in tasks:
            clean_turns = max(task["min_turns"], int(calibration.get(task["id"], {}).get("turns") or 0))
            budget = resolve_budget(condition, clean_turns)
            for repeat in range(repeats[condition]):
                for arm in ARMS:
                    items.append({"task": task, "condition": condition, "budget": budget,
                                  "clean_turns": clean_turns, "repeat": repeat, "arm": arm})
    print(f"experiment: {len(items)} cells ({len(conditions)} conditions x {len(ARMS)} arms, "
          f"repeats {repeats}, workers {args.workers})")

    records: list[dict] = []
    output = Path(args.output)
    started = time.perf_counter()

    def write_progress() -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"status": "running", "records": records}, ensure_ascii=False,
                                     indent=2), encoding="utf-8")

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(run_one, client, item, args): item for item in items}
            for done, future in enumerate(as_completed(futures), start=1):
                record = future.result()
                records.append(record)
                if done % 16 == 0 or done == len(items):
                    write_progress()
                print(f"  [{done:3d}/{len(items)}] {record['task']:40s} {record['condition']:8s} "
                      f"r{record['repeat']} {record['arm']:8s} success={record['success']!s:5s} "
                      f"turns={record['turns']}/{record['budget']} mcp_calls={record['mcp_calls']} "
                      f"restarts={record['mcp_restarts']} recoveries={record['guard_recoveries']}")
    except RuntimeError as exc:
        print(f"\nERROR: DeepSeek/live/MCP failure: {exc}", file=sys.stderr)
        output.write_text(json.dumps({"status": "partial_api_error", "records": records,
                                      "error": str(exc)}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote partial results to {output} (no offline fallback)", file=sys.stderr)
        return 2
    wall_seconds = round(time.perf_counter() - started, 1)

    aggregate_block = aggregate(records)
    repair_rules: dict[str, int] = {}
    for record in records:
        for event in record["guard_events"]:
            rule = event["rule"].split(":")[0]
            repair_rules[rule] = repair_rules.get(rule, 0) + 1
    summary = {
        "artifact_version": 2,
        "experiment": "ToolGuard_mcp_multiserver_live_ab_v2",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "protocol": {
            "transport": "MCP over stdio; two real server processes (worldbank + market groups) "
                         "merged by the client adapter",
            "mcp_sdk_version": mcp_sdk_version(),
            "servers": {
                "worldbank": f"{SERVER_SCRIPT.relative_to(REPO)} --group worldbank",
                "market": f"{SERVER_SCRIPT.relative_to(REPO)} --group market",
            },
            "arms": list(ARMS),
            "budget_conditions": {
                condition: ("per-task turn budget = fault-free MCP calibration turns + "
                            f"{condition.split('+', 1)[1]}" if condition.startswith("clean+")
                            else f"{condition} turns for every task")
                for condition in conditions
            },
            "repeats": repeats,
            "workers": args.workers,
            "wall_seconds": wall_seconds,
            "failure_modes_injected": {
                "server_transient_fault_epoch": {
                    "rate": args.p_fault,
                    "attempts_to_fail": args.attempts_to_fail,
                    "seed": args.seed,
                    "note": "deterministic server-side isError epochs (timeout / 429 / 5xx)",
                },
                "server_process_crash": {
                    "crash_after_call": args.crash_after,
                    "note": "the worldbank server process exits abruptly during that call; the stdio "
                            "session really breaks, the adapter reconnects a fresh process, and the "
                            "middleware retries the call as a retryable request_error",
                },
                "catalog_drift": {
                    "advertised_alias": DRIFT_ALIAS,
                    "real_tool": DRIFT_REAL_TOOL,
                    "note": "stale client config advertises an old tool name no server registers; the "
                            "middleware resolves it against the real catalogue (offline-validated "
                            "function-name similarity rule)",
                },
            },
            "calibration": "fault-free MCP run per task with middleware enabled; the turn count anchors "
                           "the budget sweep",
            "tool_surface": {
                "advertised_tools": sorted(advertised),
                "real_tools": sorted(tool_names),
                "hidden_behind_alias": [DRIFT_REAL_TOOL],
            },
            "verification": "programmatic; ground truth fetched live through the MCP servers before the runs",
        },
        "tasks": [{key: value for key, value in task.items() if key != "verifier"} for task in tasks],
        "calibration": calibration,
        "skipped": skipped,
        "records": records,
        "aggregate": aggregate_block,
        "guard_repairs_by_rule": repair_rules,
        "key_findings": key_findings(aggregate_block),
        "limitations": [
            "The server crash and the catalog drift are deterministic injected scenarios (call "
            f"#{args.crash_after} of the worldbank server; alias {DRIFT_ALIAS}), not observed "
            "production incidents; everything else (stdio transport, process death, reconnects, "
            "tool routing, HTTP, LLM) is real.",
            "Single model (deepseek-v4-flash) and repeat seeds vary only the injected fault "
            "surface, not model sampling.",
            "A fresh pair of MCP server processes is started per cell, so cold HTTP caches make "
            "runs slower and fault assignment depends on the calls the cell actually makes.",
            "MCP stdio needs real pipe handles, so this runner needs full local access; the test "
            "suite covers the same adapter logic with the SDK in-memory transport.",
            "Free public APIs are not production grade and API drift can affect reruns.",
        ],
        "reproduce": [
            "python scripts/run_mcp_demo_v2.py --mode calibrate",
            "python scripts/run_mcp_demo_v2.py --workers 2 --output benchmarks/toolguard/mcp_demo_v2.json",
            "python -m pytest tests/test_toolguard -q -p no:cacheprovider",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== budget curve ===")
    print(json.dumps(aggregate_block["budget_curve"], ensure_ascii=False, indent=2))
    print("\n=== guard repairs by rule ===")
    print(json.dumps(repair_rules, ensure_ascii=False))
    print(f"\nwrote {output} (wall {wall_seconds}s)")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("eval", "calibrate"), default="eval")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--task-ids", default=None,
                        help="comma separated task ids (debug/smoke subset); default: the 16 demo tasks")
    parser.add_argument("--budgets", default=DEFAULT_BUDGETS)
    parser.add_argument("--repeats", default=DEFAULT_REPEATS)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--crash-after", type=int, default=2)
    parser.add_argument("--p-fault", type=float, default=0.38)
    parser.add_argument("--attempts-to-fail", type=int, default=2)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--calibration", default=DEFAULT_CALIBRATION)
    parser.add_argument("--clean-max-turns", type=int, default=14)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    conditions = parse_budgets(args.budgets)
    repeats = parse_repeats(args.repeats, conditions)
    task_ids = TASK_IDS[:args.max_tasks] if args.max_tasks else TASK_IDS
    if args.task_ids:
        task_ids = tuple(value.strip() for value in args.task_ids.split(",") if value.strip())

    multi = start_multi_backend(fault=None, crash_after=0)
    try:
        tool_names = multi.available_tool_names()
        advertised = [schema["function"]["name"] for schema in multi.schemas()]
        print(f"MCP real tools ({len(tool_names)}): {sorted(tool_names)}")
        print(f"MCP advertised tools ({len(advertised)}): {sorted(advertised)}")
        tasks, skipped = build_tasks(multi)
    finally:
        multi.close()
    tasks = [task for task in tasks if task["id"] in task_ids]
    order = {task_id: index for index, task_id in enumerate(task_ids)}
    tasks.sort(key=lambda task: order[task["id"]])
    missing = [task_id for task_id in task_ids if task_id not in {task["id"] for task in tasks}]
    print(f"built {len(tasks)} MCP tasks (skipped ground-truth groups: {len(skipped)})")
    for note in skipped:
        print(f"  skip: {note}")
    if missing:
        print(f"ERROR: tasks missing from the ground-truth build: {missing}", file=sys.stderr)
        return 2
    if args.dry_run:
        for task in tasks:
            print(f"  {task['id']:40s} {task['category']:10s} min_turns>={task['min_turns']} "
                  f"expected={json.dumps(task['expected'], ensure_ascii=False)[:110]}")
        return 0

    if not settings.deepseek_api_key:
        print("ERROR: TOOLEVO_DEEPSEEK_API_KEY is missing; cannot run the live MCP demo. "
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
        return run_eval(client, tasks, skipped, conditions, repeats, tool_names, advertised, args)
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
