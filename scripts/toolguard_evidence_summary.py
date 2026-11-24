"""Print the headline evidence for every committed ToolGuard artifact.

This is the machine-readable bridge between the claims in TOOLGUARD.md / REVIEW.md and the
JSON artifacts: every number printed here is recomputed from the artifact files (not copied
from documentation), so an external reviewer can diff the two.

Usage:
    python scripts/toolguard_evidence_summary.py           # human-readable
    python scripts/toolguard_evidence_summary.py --json    # machine-readable dict

No network, no API keys, no benchmark data needed. Exit code is non-zero when an artifact
is missing or fails its contract check.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tool_evolution.toolguard.artifact_contracts import ARTIFACT_DIR, validate_artifacts  # noqa: E402


def _load(name: str) -> dict | None:
    path = ARTIFACT_DIR / name
    if not path.exists():
        print(f"MISSING artifact: {path}", file=sys.stderr)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _offline(data: dict) -> dict:
    results = data["results"]
    lomo = results["lomo_aggregate"]
    cross_model = results.get("cross_model") or {}
    cross_group = results.get("cross_group") or {}
    return {
        "dataset": data.get("dataset"),
        "evaluated_calls": lomo.get("evaluated_calls"),
        "real_failures": lomo.get("evaluable_failures"),
        "repair_coverage": lomo.get("repair_coverage_of_failures"),
        "schema_valid_rate": lomo.get("schema_valid_rate"),
        "cache_verified_recovered": lomo.get("cache_verified_recovered"),
        "cache_verified_still_error": lomo.get("cache_verified_still_error"),
        "cross_model": {
            "test_failures": cross_model.get("test_failures"),
            "repairable": cross_model.get("repairable"),
            "repair_coverage": cross_model.get("repair_coverage_of_failures"),
            "schema_valid_rate": cross_model.get("schema_valid_rate"),
            "value_agreement": cross_model.get("value_agreement"),
            "value_verifiable": cross_model.get("value_verifiable"),
        },
        "cross_group": {
            "test_failures": cross_group.get("test_failures"),
            "repairable": cross_group.get("repairable"),
            "repair_coverage": cross_group.get("repair_coverage_of_failures"),
            "instance_memory_only_coverage": cross_group.get("previous_coverage_instance_memory_only"),
            "schema_valid_rate": cross_group.get("schema_valid_rate"),
            "value_agreement": cross_group.get("value_agreement"),
            "value_verifiable": cross_group.get("value_verifiable"),
        },
    }


def _single_model(data: dict) -> dict:
    out = {"experiment": data.get("experiment"), "cells": len(data["records"]), "conditions": {}}
    for condition, block in data["aggregate"]["conditions"].items():
        row = {"mean_budget": block["mean_budget"], "fault_kinds": None}
        for arm in ("baseline", "guard"):
            stats = block["arms"][arm]
            row[arm] = {
                "success": stats["success"],
                "cells": stats["cells"],
                "success_rate": stats["success_rate"],
                "success_wilson_95": stats["success_wilson_95"],
                "turns_per_cell": stats["turns_per_cell"],
                "tokens": stats["usage"]["total_tokens"],
                "tool_errors": stats["tool_errors"],
                "guard_recoveries": stats.get("guard_recoveries"),
                "retry_sleep_ms": stats.get("guard_retry_sleep_ms"),
            }
        row["mcnemar_exact_p"] = block["paired_cells"]["mcnemar_exact_p"]
        row["paired"] = {key: block["paired_cells"].get(key) for key in
                         ("both_success", "baseline_only", "guard_only", "neither")}
        row["failure_modes"] = block.get("failure_modes")
        row["fault_kinds"] = block["arms"]["baseline"].get("fault_kinds")
        out["conditions"][condition] = row
    return out


def _mcp(data: dict) -> dict:
    out = {"experiment": data.get("experiment"), "cells": len(data["records"]), "conditions": {}}
    aggregate = data.get("aggregate") or {}
    for condition, block in (aggregate.get("conditions") or {}).items():
        row = {}
        for arm in ("baseline", "guard"):
            stats = block["arms"][arm]
            row[arm] = {"success": stats["success"], "cells": stats["cells"],
                        "success_rate": stats["success_rate"],
                        "tokens": stats["usage"]["total_tokens"],
                        "tool_errors": stats["tool_errors"]}
        out["conditions"][condition] = row
    out["guard_repairs_by_rule"] = data.get("guard_repairs_by_rule")
    return out


def _multimodel(data: dict) -> dict:
    cross = data["aggregate"]["cross_model"]
    out = {"cells": len(data["records"]), "cost_usd": data.get("total_estimated_cost_usd"),
           "models": {}, "consistency": cross.get("consistency")}
    for key, block in cross.items():
        if key == "consistency":
            continue
        model = {"tier": block["tier"], "model": block["model"],
                 "placement": block["capability_placement"]["success"],
                 "placement_rate": block["capability_placement"]["success_rate"],
                 "avg_clean_turns": block["capability_placement"]["avg_turns"],
                 "conditions": {}}
        for condition, row in block["conditions"].items():
            model["conditions"][condition] = {
                "baseline": f"{row['baseline_success']}/{row['cells']} "
                            f"({row['baseline_success_rate']:.1%})",
                "guard": f"{row['guard_success']}/{row['cells']} "
                         f"({row['guard_success_rate']:.1%}, CI {row['guard_success_wilson_95']})",
                "mcnemar_exact_p": row["mcnemar_exact_p"],
                "baseline_tokens": row["baseline_tokens"],
                "guard_tokens": row["guard_tokens"],
                "baseline_tool_errors": row["baseline_tool_errors"],
                "guard_tool_errors": row["guard_tool_errors"],
            }
        out["models"][key] = model
    return out


def collect() -> dict:
    summary: dict = {}
    offline = _load("metrics.json")
    if offline:
        summary["offline"] = _offline(offline)
    for name in ("live_demo_hard_v3.json", "live_demo_hard_v2.json"):
        data = _load(name)
        if data:
            summary[name.removesuffix(".json")] = _single_model(data)
    mcp = _load("mcp_demo_v2.json")
    if mcp:
        summary["mcp_v2"] = _mcp(mcp)
    for name in ("live_demo_multimodel_v3.json", "live_demo_multimodel.json"):
        data = _load(name)
        if data:
            summary[name.removesuffix(".json")] = _multimodel(data)
    return summary


def main() -> int:
    problems = validate_artifacts()
    summary = collect()
    if "--json" in sys.argv:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print("=" * 78)
        print("ToolGuard evidence summary — every number recomputed from JSON artifacts")
        print("=" * 78)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    if problems:
        print("\nCONTRACT PROBLEMS:", file=sys.stderr)
        for problem in problems:
            print(" -", problem, file=sys.stderr)
        return 1
    print("\nAll artifacts present and internally consistent with their records.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
