"""ToolGuard CLI: repair one failed tool call, or print the evaluation summary.

Examples:
    python scripts/toolguard_cli.py repair \
        --tool-name search_for_weather \
        --arguments '{"city": "Paris", "dayz": 3}' \
        --error "got an unexpected keyword argument 'dayz'" \
        --available-tools search_for_weather,search_for_news

    python scripts/toolguard_cli.py summary
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tool_evolution.toolguard.engine import repair_failure  # noqa: E402
from tool_evolution.toolguard.memory import SuccessMemory  # noqa: E402
from tool_evolution.toolguard.records import FunctionSchema, ToolCall, load_calls  # noqa: E402
from tool_evolution.toolguard.schemas import build_schema_index  # noqa: E402

DEFAULT_QUERIES = (
    ".scratch/stb/repo/intuit-ai-research-StableToolBench-74f76c1/solvable_queries/test_instruction"
)
DEFAULT_CALLS = ".scratch/toolguard/tool_calls.jsonl"
DEFAULT_EVAL = ".scratch/toolguard/rules_eval.json"


def _load_schemas(path: Path) -> dict[str, FunctionSchema]:
    if not path.is_dir():
        return {}
    schemas, _ = build_schema_index(path)
    return schemas


def cmd_repair(args: argparse.Namespace) -> int:
    schemas = _load_schemas(Path(args.queries))
    memory = None
    calls_path = Path(args.calls)
    if calls_path.is_file():
        calls = load_calls(calls_path)
        memory = SuccessMemory.build(calls, schemas)
    call = ToolCall(
        trace_id="cli",
        method="cli",
        model="cli",
        strategy="cli",
        group="G1_instruction",
        query_id="0",
        try_index=0,
        step=0,
        tool_name=args.tool_name,
        arguments=json.loads(args.arguments),
        error=args.error,
        recorded_ok=False,
        available_tools=[name for name in (args.available_tools or "").split(",") if name],
    )
    result = repair_failure(call, schemas, memory)
    print(json.dumps(
        {
            "repairable": result.repairable,
            "rules": result.rules,
            "function_name": result.function_name or call.tool_name,
            "arguments": result.arguments if result.arguments is not None else call.arguments,
            "schema_valid": result.schema_valid,
            "reason": result.reason,
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


def cmd_summary(args: argparse.Namespace) -> int:
    path = Path(args.eval)
    if not path.is_file():
        print(f"evaluation artifact not found: {path}; run scripts/eval_toolguard_rules.py first", file=sys.stderr)
        return 2
    report = json.loads(path.read_text(encoding="utf-8"))
    lomo = report.get("lomo", {})
    print(json.dumps({
        "dataset_calls": report.get("cross_model", {}).get("test_calls", 0)
        + report.get("cross_model", {}).get("train_calls", 0),
        "lomo_aggregate": lomo.get("aggregate", {}),
        "cross_model": {k: report.get("cross_model", {}).get(k) for k in (
            "test_calls", "test_failures", "repairable", "repair_coverage_of_failures",
            "schema_valid_rate", "cache_verification")},
        "cross_group": {k: report.get("cross_group", {}).get(k) for k in (
            "test_calls", "test_failures", "repairable", "repair_coverage_of_failures")},
        "limitations": [
            "cache-only replay: repaired calls that are not in the released cache are unverified",
            "schema-default completion proposals are not verified",
            "server-side/transient errors are classified as retryable, not repaired",
            "no live API evaluation in this artifact",
        ],
    }, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="ToolGuard tool-call repair middleware")
    sub = parser.add_subparsers(dest="command", required=True)

    repair = sub.add_parser("repair", help="repair one failed tool call")
    repair.add_argument("--tool-name", required=True)
    repair.add_argument("--arguments", default="{}")
    repair.add_argument("--error", required=True)
    repair.add_argument("--available-tools", default="")
    repair.add_argument("--schemas", default=DEFAULT_QUERIES)
    repair.add_argument("--queries", dest="queries", default=DEFAULT_QUERIES)
    repair.add_argument("--calls", default=DEFAULT_CALLS)
    repair.set_defaults(func=cmd_repair)

    summary = sub.add_parser("summary", help="print evaluation summary")
    summary.add_argument("--eval", default=DEFAULT_EVAL)
    summary.set_defaults(func=cmd_summary)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
