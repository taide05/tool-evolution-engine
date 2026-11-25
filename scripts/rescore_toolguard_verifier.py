"""Offline rescore of the committed live artifacts with the relational verifier.

The 2026-11-24 review showed the old comparison verifiers only required candidate names to
appear.  ``tool_evolution.toolguard.verify`` now verifies the *relation* (see
``relation_consistent``).  This script re-judges every committed answer in a comparison
task with the new verifier and reports the success changes; it never modifies the artifacts.

Usage:
    python scripts/rescore_toolguard_verifier.py
    python scripts/rescore_toolguard_verifier.py --output .scratch/toolguard/verifier_rescore.json

The output is a diff list: every record whose success flag changed, with artifact, task,
model/arm, old vs new verdict and the answer (truncated answers are flagged because the
stored answer was cut at 400 characters, so a new failure cannot always be fully assessed).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tool_evolution.toolguard.verify import (  # noqa: E402
    verify_earliest_of_three,
    verify_population_compare_list,
    verify_population_compare_map,
    verify_pypi_newer_pair,
    verify_release_vs_hn,
    verify_weather_compare_pair,
    verify_weather_compare_scalar,
)

ARTIFACT_DIR = REPO / "benchmarks/toolguard"
ANSWER_TRUNCATION_LENGTH = 400

# task id -> (new verifier, notes). Only comparison tasks are re-judged; the other tasks'
# verifiers were not changed by this round.
COMPARISON_VERIFIERS = {
    "capital_weather_compare_france_germany": verify_weather_compare_pair,
    "capital_weather_compare_norway_portugal": verify_weather_compare_pair,
    "weather_compare_australia_canada": verify_weather_compare_scalar,
    "population_compare_oslo_wellington": verify_population_compare_list,
    "population_compare_france_germany_region": verify_population_compare_map,
    "pypi_newer_requests_flask": verify_pypi_newer_pair,
    "pypi_newer_numpy_pandas": verify_pypi_newer_pair,
    "release_vs_hn_item_date": verify_release_vs_hn,
    "package_earliest_of_three": verify_earliest_of_three,
}


def _judge(task_id: str, answer: str, expected: dict) -> bool | None:
    verifier = COMPARISON_VERIFIERS.get(task_id)
    if verifier is None:
        return None
    return bool(verifier(answer or "", expected))


def _expected_for(record: dict, tasks: dict[str, dict]) -> dict | None:
    if isinstance(record.get("expected"), dict):
        return record["expected"]
    task = tasks.get(record.get("task"))
    return (task or {}).get("expected")


def rescore_artifact(name: str) -> dict:
    path = ARTIFACT_DIR / name
    data = json.loads(path.read_text(encoding="utf-8"))
    tasks = {task["id"]: task for task in data.get("tasks") or []}
    records = data.get("records") or []
    changes: list[dict] = []
    judged = 0
    old_successes = 0
    new_successes = 0
    unjudged_truncated = 0
    per_task = Counter()
    for record in records:
        task_id = record.get("task")
        if task_id not in COMPARISON_VERIFIERS:
            continue
        expected = _expected_for(record, tasks)
        if expected is None:
            continue
        judged += 1
        answer = record.get("answer") or ""
        old = bool(record.get("success"))
        new = _judge(task_id, answer, expected)
        old_successes += int(old)
        new_successes += int(bool(new))
        if old != new:
            truncated = len(answer) >= ANSWER_TRUNCATION_LENGTH
            if truncated and old and not new:
                unjudged_truncated += 1
            changes.append({
                "artifact": name,
                "task": task_id,
                "model_key": record.get("model_key"),
                "arm": record.get("arm"),
                "condition": record.get("condition"),
                "repeat": record.get("repeat"),
                "old_success": old,
                "new_success": new,
                "answer_len": len(answer),
                "answer_truncated_at_capture": truncated,
                "answer": answer,
            })
        per_task[task_id] += 1
    return {
        "artifact": name,
        "comparison_records": judged,
        "old_successes": old_successes,
        "new_successes": new_successes,
        "changed": len(changes),
        "old_true_now_false": sum(1 for item in changes if item["old_success"]),
        "old_false_now_true": sum(1 for item in changes if item["new_success"]),
        "changed_but_answer_truncated": unjudged_truncated,
        "tasks_judged": dict(per_task),
        "changes": changes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=".scratch/toolguard/verifier_rescore.json")
    args = parser.parse_args()

    artifacts = ("live_demo_hard_v2.json", "live_demo_hard_v3.json",
                 "live_demo_multimodel.json", "live_demo_multimodel_v3.json")
    results = [rescore_artifact(name) for name in artifacts]
    totals = {
        "comparison_records": sum(item["comparison_records"] for item in results),
        "old_successes": sum(item["old_successes"] for item in results),
        "new_successes": sum(item["new_successes"] for item in results),
        "changed": sum(item["changed"] for item in results),
        "old_true_now_false": sum(item["old_true_now_false"] for item in results),
        "old_false_now_true": sum(item["old_false_now_true"] for item in results),
        "changed_but_answer_truncated": sum(item["changed_but_answer_truncated"]
                                            for item in results),
    }
    payload = {"totals": totals, "artifacts": results,
               "note": "Read-only rescore: the committed artifacts were not modified. "
                       "V1 live/mcp artifacts store no answers and cannot be rescored."}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"comparison records judged: {totals['comparison_records']}")
    print(f"old verifier successes:    {totals['old_successes']}")
    print(f"new verifier successes:    {totals['new_successes']}")
    print(f"changed verdicts:          {totals['changed']} "
          f"(old true -> new false: {totals['old_true_now_false']}, "
          f"old false -> new true: {totals['old_false_now_true']})")
    print(f"changed but answer truncated at capture: {totals['changed_but_answer_truncated']}")
    for item in results:
        print(f"\n[{item['artifact']}] judged={item['comparison_records']} "
              f"old={item['old_successes']} new={item['new_successes']} "
              f"changed={item['changed']} truncated-changes={item['changed_but_answer_truncated']}")
        for change in item["changes"]:
            label = change.get("model_key") or change.get("arm")
            print(f"  - {change['task']} [{label}/{change['condition']}/r{change['repeat']}] "
                  f"{change['old_success']} -> {change['new_success']}"
                  f"{' (TRUNCATED)' if change['answer_truncated_at_capture'] else ''}")
            print(f"    {change['answer'][:240]!r}")
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
