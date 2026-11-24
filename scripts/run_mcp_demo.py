"""MCP integration demo: the same ToolGuard middleware in front of a real MCP tool server.

The tool layer is a genuine Model Context Protocol server speaking stdio
(``scripts/mcp_toolguard_server.py``) over seven key-free public APIs.  A real DeepSeek
tool-calling agent drives it through :class:`MCPBackend`, which maps MCP ``tools/list``
schemas and ``tools/call`` results onto the interface ToolGuard was built against -- the
middleware itself is unchanged from the function-tool demo.

Protocol
--------
* 10 multi-step tasks (2-6 dependent MCP tool calls each); ground truth is fetched live
  through the MCP server before the runs and answers are verified programmatically.
* Calibration: one fault-free run per task measures the interaction requirement.
* Budgeted condition: turn budget = fault-free requirement + 1.
* Transient faults live in the **MCP server** (a deterministic first-N-attempts epoch of
  ``isError`` results), exactly where real outages live.  A fresh server process is
  started for every arm, so both arms face the same fresh fault surface.
* Real DeepSeek agent (``deepseek-v4-flash``); no offline/mock fallback.

MCP stdio needs real pipe handles, which are unavailable under the workspace sandbox --
run this script with full local access (``python scripts/run_mcp_demo.py``).

Usage
-----
    python scripts/run_mcp_demo.py --dry-run
    python scripts/run_mcp_demo.py --max-tasks 3
    python scripts/run_mcp_demo.py --output benchmarks/toolguard/mcp_demo.json
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from run_live_demo_hard import build_tasks  # noqa: E402 - shared multi-step task set

from tool_evolution.toolguard.agent import DeepSeekClient, LLMConfig, run_task  # noqa: E402
from tool_evolution.toolguard.guard import ToolGuard  # noqa: E402
from tool_evolution.toolguard.mcp_backend import MCPBackend, MCPBridge  # noqa: E402
from tool_evolution.toolguard.verify import mcnemar_exact  # noqa: E402
from tool_evolution.utils.config import settings  # noqa: E402

DEFAULT_OUTPUT = ".scratch/toolguard/mcp_demo.json"
MODEL = "deepseek-v4-flash"
ARMS = ("baseline", "guard")
SERVER_SCRIPT = REPO / "scripts" / "mcp_toolguard_server.py"
# A representative subset of the multi-step task set, covering every MCP tool family.
TASK_IDS = (
    "capital_weather_japan",
    "capital_weather_compare_france_germany",
    "capital_population_canberra",
    "capital_income_brazil",
    "pypi_release_date_httpx",
    "pypi_newer_requests_flask",
    "hn_item_1_author_account_year",
    "hn_item_8863_author_account_year",
    "fx_eur_jpy_cross",
    "population_sum_australia_canada",
)


def mcp_sdk_version() -> str:
    try:
        return importlib.metadata.version("mcp")
    except importlib.metadata.PackageNotFoundError:  # pragma: no cover - defensive
        return "unknown"


def start_mcp_backend(fault: dict | None = None) -> tuple[MCPBridge, MCPBackend]:
    """Start a fresh MCP server process and wrap it in a ToolGuard-compatible backend."""
    env = dict(os.environ)
    if fault:
        env.update({
            "TOOLGUARD_MCP_FAULT_RATE": str(fault["rate"]),
            "TOOLGUARD_MCP_FAULT_SEED": str(fault["seed"]),
            "TOOLGUARD_MCP_FAULT_EPOCH": str(fault["epoch"]),
        })
    bridge = MCPBridge([sys.executable, str(SERVER_SCRIPT)], cwd=REPO, env=env)
    bridge.start()
    backend = MCPBackend(bridge)
    backend.load_tools()
    return bridge, backend


def summarise_arm(rows: list[dict], client: DeepSeekClient) -> dict:
    return {
        "tasks": len(rows),
        "success": sum(int(row["success"]) for row in rows),
        "success_rate": round(sum(int(row["success"]) for row in rows) / max(len(rows), 1), 4),
        "turns": sum(row["turns"] for row in rows),
        "tool_calls": sum(row["tool_calls"] for row in rows),
        "tool_errors": sum(row["tool_errors"] for row in rows),
        "exhausted": sum(int(row["exhausted"]) for row in rows),
        "mcp_server_calls": sum(row["mcp_server_calls"] for row in rows),
        "mcp_server_errors": sum(row["mcp_server_errors"] for row in rows),
        "guard_recoveries": sum(row["guard_recoveries"] for row in rows),
        "usage": {key: sum(row["usage"][key] for row in rows) for key in client.usage},
    }


def aggregate(records: list[dict], client: DeepSeekClient) -> dict:
    arms = {arm: summarise_arm([row for row in records if row["arm"] == arm], client) for arm in ARMS}
    by_task: dict[str, dict[str, bool]] = {}
    for row in records:
        by_task.setdefault(row["task"], {})[row["arm"]] = bool(row["success"])
    baseline_only = sum(1 for pair in by_task.values() if pair.get("baseline") and not pair.get("guard"))
    guard_only = sum(1 for pair in by_task.values() if pair.get("guard") and not pair.get("baseline"))
    both = sum(1 for pair in by_task.values() if pair.get("baseline") and pair.get("guard"))
    repairs: dict[str, int] = {}
    for row in records:
        for event in row["guard_events"]:
            rule = event["rule"].split(":")[0]
            repairs[rule] = repairs.get(rule, 0) + 1
    return {
        "arms": arms,
        "paired": {
            "both_success": both,
            "baseline_only": baseline_only,
            "guard_only": guard_only,
            "neither": len(by_task) - both - baseline_only - guard_only,
            "mcnemar_exact_p": mcnemar_exact(baseline_only, guard_only),
        },
        "guard_repairs_by_rule": repairs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--budget-slack", type=int, default=1)
    parser.add_argument("--p-fault", type=float, default=0.38,
                        help="server-side probability that a logical MCP call gets a fault epoch")
    parser.add_argument("--attempts-to-fail", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--clean-max-turns", type=int, default=12)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    task_ids = TASK_IDS[:args.max_tasks] if args.max_tasks else TASK_IDS
    bridge, backend = start_mcp_backend()
    try:
        tool_names = backend.tool_names()
        print(f"MCP server: {len(tool_names)} tools -> {tool_names}")
        tasks, skipped = build_tasks(backend)
    finally:
        bridge.close()
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
            print(f"  {task['id']:40s} min_turns>={task['min_turns']} "
                  f"expected={json.dumps(task['expected'], ensure_ascii=False)[:120]}")
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
    calibration: dict[str, dict] = {}
    records: list[dict] = []
    try:
        # -- calibration: one fault-free MCP run per task ---------------------------
        print("\n=== calibration: fault-free MCP turn requirement ===")
        bridge, backend = start_mcp_backend()
        try:
            for task in tasks:
                run = run_task(client, backend, task["question"], None, max_turns=args.clean_max_turns)
                success = bool(task["verifier"](run.answer, task["expected"]))
                calibration[task["id"]] = {"turns": run.turns, "success": success,
                                           "exhausted": run.exhausted}
                print(f"  {task['id']:40s} success={success!s:5s} turns={run.turns} "
                      f"calls={run.tool_calls} errors={run.tool_errors}")
        finally:
            bridge.close()

        # -- A/B under the calibrated budget ---------------------------------------
        print("\n=== budgeted condition: fresh MCP server per arm ===")
        fault = {"rate": args.p_fault, "seed": args.seed, "epoch": args.attempts_to_fail}
        for task in tasks:
            budget = int(calibration.get(task["id"], {}).get("turns") or task["min_turns"]) + args.budget_slack
            for arm in ARMS:
                bridge, backend = start_mcp_backend(fault)
                try:
                    # Faults come from the MCP server itself; ToolGuard wraps the MCP backend
                    # directly.  A fresh server process per arm = a fresh fault surface.
                    middleware = ToolGuard(backend, sleep_scale=0.05) if arm == "guard" else None
                    before = dict(client.usage)
                    run = run_task(client, backend, task["question"], middleware, max_turns=budget)
                    delta = {key: client.usage[key] - before.get(key, 0) for key in client.usage}
                    success = bool(task["verifier"](run.answer, task["expected"]))
                    records.append({
                        "task": task["id"],
                        "arm": arm,
                        "budget": budget,
                        "success": success,
                        "answer": run.answer[:400],
                        "turns": run.turns,
                        "tool_calls": run.tool_calls,
                        "tool_errors": run.tool_errors,
                        "exhausted": run.exhausted,
                        "usage": delta,
                        "elapsed_ms": run.elapsed_ms,
                        "mcp_server_calls": backend.stats["calls"],
                        "mcp_server_errors": backend.stats["errors"],
                        "guard_recoveries": middleware.stats["recovered"] if middleware is not None else 0,
                        "guard_events": run.guard_events,
                    })
                    print(f"  {task['id']:40s} {arm:8s} success={success!s:5s} "
                          f"turns={run.turns}/{budget} calls={run.tool_calls} errors={run.tool_errors} "
                          f"mcp_calls={backend.stats['calls']} recoveries={records[-1]['guard_recoveries']}")
                finally:
                    bridge.close()
    except RuntimeError as exc:
        print(f"\nERROR: DeepSeek/live/MCP failure: {exc}", file=sys.stderr)
        print("Stopping (no offline fallback). Please check the API key / balance and rerun.", file=sys.stderr)
        return 2
    finally:
        client.close()

    aggregate_block = aggregate(records, client)
    arms = aggregate_block["arms"]
    paired = aggregate_block["paired"]
    n_tasks = arms["baseline"]["tasks"]
    summary = {
        "artifact_version": 1,
        "experiment": "ToolGuard_mcp_stdio_live_ab",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "protocol": {
            "transport": "MCP over stdio (real stdio pipe session to a local MCP server process)",
            "mcp_sdk_version": mcp_sdk_version(),
            "server": str(SERVER_SCRIPT.relative_to(REPO)),
            "arms": list(ARMS),
            "fresh_server_per_arm": True,
            "fault_injection": {
                "location": "MCP server side: deterministic first-N-attempts isError epoch per "
                            "logical call (timeout / HTTP 429 / HTTP 5xx)",
                "p_fault": args.p_fault,
                "attempts_to_fail": args.attempts_to_fail,
                "seed": args.seed,
            },
            "calibration": "one fault-free MCP run per task; budgeted turn budget = "
                           f"calibrated turns + {args.budget_slack}",
            "tool_surface": {
                "tools": tool_names,
                "apis": ["api.worldbank.org", "geocoding-api.open-meteo.com", "api.open-meteo.com",
                         "pypi.org", "hacker-news.firebaseio.com", "open.er-api.com"],
            },
            "verification": "programmatic; ground truth fetched live through the MCP server before the runs",
        },
        "tasks": [{key: value for key, value in task.items() if key != "verifier"} for task in tasks],
        "calibration": calibration,
        "skipped": skipped,
        "records": records,
        "aggregate": aggregate_block,
        "key_findings": [
            f"budgeted: task success baseline {arms['baseline']['success']}/{n_tasks} vs guard "
            f"{arms['guard']['success']}/{n_tasks} (McNemar exact p={paired['mcnemar_exact_p']:.4f})",
            f"budgeted: turns {arms['baseline']['turns']} -> {arms['guard']['turns']}, MCP calls "
            f"{arms['baseline']['tool_calls']} -> {arms['guard']['tool_calls']}, agent-visible tool "
            f"errors {arms['baseline']['tool_errors']} -> {arms['guard']['tool_errors']}, tokens "
            f"{arms['baseline']['usage']['total_tokens']} -> {arms['guard']['usage']['total_tokens']}",
            f"guard recovered {arms['guard']['guard_recoveries']} transient MCP errors inside the "
            "middleware (agent saw none of them)",
        ],
        "limitations": [
            "Deterministic server-side fault injection, not observed outages: each fault epoch fails "
            "the first 2 attempts of a logical MCP call.",
            "Single model (deepseek-v4-flash) and a single run per task per arm; McNemar is computed "
            "over the paired discordant outcomes only.",
            "A fresh MCP server process is started per arm; the fault surface is deterministic for a "
            "logical call, but fault assignment depends on the calls the arm actually makes.",
            "MCP stdio needs real pipe handles, so this demo runs outside the workspace sandbox; the "
            "test suite uses the MCP SDK in-memory transport instead.",
            "Free public APIs are not production grade and API drift can affect reruns.",
        ],
        "reproduce": [
            "python scripts/mcp_toolguard_server.py  # speak MCP to it over stdio",
            "python scripts/run_mcp_demo.py --output benchmarks/toolguard/mcp_demo.json",
            "python -m pytest tests/test_toolguard -q -p no:cacheprovider",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== aggregate ===")
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2))
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
