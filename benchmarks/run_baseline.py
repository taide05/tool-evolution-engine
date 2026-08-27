"""Tool Evolution Engine Baseline Runner.

Runs the TEE evaluation pipeline (run_eval.py) and wraps results in GSM metrics framework.

Usage:
    cd D:/tool-evolution-engine
    python -m benchmarks.run_baseline              # run full eval
    python -m benchmarks.run_baseline --report     # show latest benchmark
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

BENCHMARKS_DIR = Path(__file__).resolve().parent
RUNS_DIR = BENCHMARKS_DIR / "runs"
PROJECT_ROOT = BENCHMARKS_DIR.parent

REQUIRED_EVAL_FIELDS = (
    "schema_version", "failure_reduction", "dag_recall", "classifier",
    "template_coverage", "rule_precision", "governance", "throughput",
    "data_composition",
)


class EvalResultError(Exception):
    pass


def load_eval_results(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise EvalResultError(f"eval results not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise EvalResultError(f"invalid JSON in {path}: {e}")
    missing = [k for k in REQUIRED_EVAL_FIELDS if k not in data]
    if missing:
        raise EvalResultError(f"missing fields in {path}: {missing}")
    return data


async def run_seeded_eval() -> dict[str, Any]:
    """Run the full evaluation pipeline and collect results."""
    from benchmarks.metrics import compute_all_metrics, save_benchmark

    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RUNS_DIR / "eval_results.json"

    print("Running evaluation pipeline (run_eval.py)...")
    t0 = time.monotonic()
    result = subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "run_eval.py"),
         "--output", str(output_path)],
        capture_output=True, text=True, cwd=str(PROJECT_ROOT), timeout=1800,
    )
    elapsed = time.monotonic() - t0
    print(f"  Eval finished in {elapsed:.1f}s (exit {result.returncode})")
    if result.returncode != 0:
        print(f"Evaluation FAILED (exit {result.returncode}): {result.stderr[:500]}")
        sys.exit(1)

    try:
        eval_data = load_eval_results(output_path)
    except EvalResultError as e:
        print(f"Evaluation results invalid: {e}")
        sys.exit(1)

    report = compute_all_metrics(eval_data)

    s = report["summary"]
    for key, val in s.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.1%}" if val < 1 else f"  {key}: {val:.1f}")
        else:
            print(f"  {key}: {val}")
    composition = report.get("data_composition", {})
    print(f"  data_composition: {composition.get('sources', {})}")

    path = save_benchmark(report, RUNS_DIR)
    print(f"\nBenchmark saved to: {path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tool Evolution Engine Benchmark")
    parser.add_argument("--report", action="store_true",
                        help="Show latest benchmark report")
    args = parser.parse_args()

    if args.report:
        from benchmarks.metrics import load_benchmarks
        reports = load_benchmarks(RUNS_DIR)
        if reports:
            latest = reports[-1]
            print(json.dumps(latest, indent=2, ensure_ascii=False))
        else:
            print("No benchmarks found. Run without --report first.")
    else:
        report = asyncio.run(run_seeded_eval())
        print("\nDone. Report saved to benchmarks/runs/")
