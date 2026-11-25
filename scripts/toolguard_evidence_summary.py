"""Print the headline evidence for every committed ToolGuard artifact.

This is the machine-readable bridge between the claims in TOOLGUARD.md / REVIEW.md and the
JSON artifacts.  Since 2026-11-24 every derived number printed here for the records-based
artifacts is **recomputed from the raw per-cell records** by
:mod:`tool_evolution.toolguard.artifact_metrics` — success counts, success rates, Wilson
95% intervals, turns, turns/cell, token usage, tool errors, guard recoveries, retry
backoff, injected-fault counts, paired counts and the exact McNemar p-value.  It never
reads those fields out of the artifact's ``aggregate`` block, so a tampered aggregate
cannot be laundered through this report: the report shows what the records say and the
contract check (below) flags the mismatch.

The offline ``metrics.json`` artifact is the exception: the StableToolBench trajectory data
is not committed (size/licence), so its counts cannot be re-derived here.  The artifact's
counts are printed but every ratio/cache-sum that is a function of them is re-derived
(see :func:`artifact_metrics.check_offline_metrics` for the contract).

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
from tool_evolution.toolguard.artifact_metrics import (  # noqa: E402
    recompute_conditions,
    recompute_guard_repairs_by_rule,
    recompute_multimodel,
    recompute_v1_live,
    recompute_v1_mcp,
)

RECORD_BASED = (
    "live_demo_hard_v4.json",
    "live_demo_hard_v3.json",
    "live_demo_hard_v3_hnconsistent.json",
    "live_demo_hard_v2.json",
    "live_demo_hard.json",
    "mcp_demo_v3.json",
    "mcp_demo_v2.json",
    "live_demo_multimodel_v4.json",
    "live_demo_multimodel_v3.json",
    "live_demo_multimodel.json",
    "live_demo.json",
    "mcp_demo.json",
)


def _root(root: Path | None) -> Path:
    return Path(root) if root is not None else ARTIFACT_DIR


def _load(name: str, root: Path | None = None) -> dict | None:
    path = _root(root) / name
    if not path.exists():
        print(f"MISSING artifact: {path}", file=sys.stderr)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# offline artifact: counts are inputs, ratios are re-derived from them
# ---------------------------------------------------------------------------
def _offline_protocol(block: dict) -> dict:
    evaluable = block.get("evaluable_failures") or 0
    proposed = block.get("proposed_repairs") or 0
    passed = block.get("schema_passed") or 0
    accepted = block.get("accepted_repairs", block.get("repairable")) or 0
    value_verified = block.get("value_verified_repairs") or 0
    return {
        "evaluable_failures": evaluable,
        "proposed_repairs": proposed,
        "schema_passed": passed,
        "schema_rejected_proposals": block.get("schema_rejected_proposals"),
        "schema_rejected_failures": block.get("schema_rejected_failures"),
        "value_verified_repairs": value_verified,
        "coerced_repairs": block.get("coerced_repairs"),
        "accepted_repairs": accepted,
        "accepted_rate": round(accepted / max(proposed, 1), 4),
        "schema_valid_rate": round(passed / max(proposed, 1), 4),
        "value_verified_rate": round(value_verified / max(proposed, 1), 4),
        "repair_coverage": round(accepted / max(evaluable, 1), 4),
        "value_agreement": block.get("value_agreement"),
        "value_verifiable": block.get("value_verifiable"),
        "value_agreement_rate": round((block.get("value_agreement") or 0)
                                      / max(block.get("value_verifiable") or 1, 1), 4),
        "cache_verified_recovered": block.get("cache_verified_recovered"),
        "cache_verified_still_error": block.get("cache_verified_still_error"),
        "cache_miss": block.get("cache_miss"),
        "train_models_used": block.get("train_models_used"),
    }


def _offline(data: dict) -> dict:
    results = data["results"]
    lomo = results["lomo_aggregate"]
    cross_model = results.get("cross_model") or {}
    cross_group = results.get("cross_group") or {}
    return {
        "_provenance": "offline artifact counts (raw StableToolBench data is not committed); "
                       "ratios/cache sums/schema funnel re-derived from the counts",
        "dataset": data.get("dataset"),
        "evaluated_calls": lomo.get("evaluated_calls"),
        "real_failures": lomo.get("evaluable_failures"),
        "repair_coverage": round((lomo.get("repairable") or 0)
                                 / max(lomo.get("evaluable_failures") or 1, 1), 4),
        "schema_valid_rate": round((lomo.get("schema_passed") or 0)
                                   / max(lomo.get("proposed_repairs") or 1, 1), 4),
        "schema_funnel": {
            "proposed_repairs": lomo.get("proposed_repairs"),
            "schema_rejected_proposals": lomo.get("schema_rejected_proposals"),
            "schema_passed": lomo.get("schema_passed"),
        },
        "skipped_no_schema": results.get("skipped_no_schema"),
        "skipped_no_schema_breakdown": results.get("skipped_no_schema_breakdown"),
        "cache_verified_recovered": lomo.get("cache_verified_recovered"),
        "cache_verified_still_error": lomo.get("cache_verified_still_error"),
        "cache_miss": lomo.get("cache_miss"),
        "protocol": {
            "lomo": (data.get("protocol") or {}).get("lomo"),
            "lomo_train_policy": (data.get("protocol") or {}).get("lomo_train_policy"),
        },
        "lomo_sensitivity_exclude_fold": (
            (results.get("lomo_sensitivity_exclude_fold") or {}).get("aggregate")),
        "cross_model": _offline_protocol(cross_model),
        "cross_group": _offline_protocol(cross_group),
    }


# ---------------------------------------------------------------------------
# records-based artifacts: all derived numbers re-derived from records
# ---------------------------------------------------------------------------
def _single_model(data: dict) -> dict:
    records = data["records"]
    computed = recompute_conditions(records)
    out: dict = {
        "experiment": data.get("experiment"),
        "cells": len(records),
        "source": "records (recomputed; aggregate values not read)",
        "conditions": {},
    }
    faults_total = {"baseline": 0, "guard": 0}
    for condition, block in computed["conditions"].items():
        row = {
            "mean_budget": block.get("mean_budget"),
            "fault_kinds": block["arms"]["baseline"]["fault_kinds"],
            "faults_injected": {arm: block["arms"][arm]["faults_injected"]
                                for arm in ("baseline", "guard")},
        }
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
                "guard_recoveries": stats["guard_recoveries"],
                "retry_sleep_ms": stats["guard_retry_sleep_ms"],
            }
            faults_total[arm] += stats["faults_injected"]
        row["mcnemar_exact_p"] = block["paired_cells"]["mcnemar_exact_p"]
        row["paired"] = {key: block["paired_cells"].get(key) for key in
                         ("both_success", "baseline_only", "guard_only", "neither")}
        row["failure_modes"] = block.get("failure_modes")
        out["conditions"][condition] = row
    out["faults_injected_total"] = faults_total
    return out


def _mcp(data: dict) -> dict:
    records = data["records"]
    computed = recompute_conditions(records)
    out: dict = {
        "experiment": data.get("experiment"),
        "cells": len(records),
        "source": "records (recomputed; aggregate values not read)",
        "conditions": {},
    }
    for condition, block in computed["conditions"].items():
        row: dict = {}
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
                "faults_injected": stats["faults_injected"],
            }
        row["mcnemar_exact_p"] = block["paired_cells"]["mcnemar_exact_p"]
        out["conditions"][condition] = row
    out["guard_repairs_by_rule"] = recompute_guard_repairs_by_rule(records)
    return out


def _multimodel(data: dict) -> dict:
    computed = recompute_multimodel(
        data["records"], data.get("calibration") or {},
        (data.get("protocol") or {}).get("models") or [])
    cross = computed["cross_model"]
    out: dict = {
        "cells": len(data["records"]),
        "cost_usd": computed["total_estimated_cost_usd"],
        "source": "records + calibration + protocol prices (recomputed; aggregate not read)",
        "models": {},
        "consistency": cross["consistency"],
    }
    for key, block in cross.items():
        if key == "consistency":
            continue
        model = {
            "tier": block["tier"],
            "model": block["model"],
            "placement": block["capability_placement"]["success"],
            "placement_rate": block["capability_placement"]["success_rate"],
            "avg_clean_turns": block["capability_placement"]["avg_turns"],
            "conditions": {},
        }
        for condition, row in block["conditions"].items():
            model["conditions"][condition] = {
                "baseline": f"{row['baseline_success']}/{row['cells']} "
                            f"({row['baseline_success_rate']:.1%})",
                "guard": f"{row['guard_success']}/{row['cells']} "
                         f"({row['guard_success_rate']:.1%}, CI "
                         f"{row['guard_success_wilson_95']})",
                "mcnemar_exact_p": row["mcnemar_exact_p"],
                "baseline_tokens": row["baseline_tokens"],
                "guard_tokens": row["guard_tokens"],
                "baseline_tool_errors": row["baseline_tool_errors"],
                "guard_tool_errors": row["guard_tool_errors"],
            }
        out["models"][key] = model
    return out


def _v1_live(data: dict) -> dict:
    """V1 live artifact: only per-task counters survive, not raw records."""
    computed = recompute_v1_live(data.get("results") or {})
    out: dict = {"experiment": data.get("experiment"), "source": "per_task counters"}
    for condition, block in computed.items():
        out[condition] = {"tasks": block["tasks"], "arms": {}, "paired": block["paired"]}
        for arm, stats in block["arms"].items():
            out[condition]["arms"][arm] = {
                "success": stats["success"],
                "success_rate": stats["success_rate"],
                "tool_calls": stats["tool_calls"],
                "tool_errors": stats["tool_errors"],
            }
    return out


def _v1_hard(data: dict) -> dict:
    computed = recompute_conditions(data["records"])
    out: dict = {"experiment": data.get("experiment"), "cells": len(data["records"]),
                 "source": "records (recomputed; aggregate values not read)", "conditions": {}}
    for condition, block in computed["conditions"].items():
        out["conditions"][condition] = {
            arm: {
                "success": block["arms"][arm]["success"],
                "cells": block["arms"][arm]["cells"],
                "success_rate": block["arms"][arm]["success_rate"],
                "tool_errors": block["arms"][arm]["tool_errors"],
            }
            for arm in ("baseline", "guard")
        }
        out["conditions"][condition]["paired"] = block["paired_cells"]
    return out


def _v1_mcp(data: dict) -> dict:
    computed = recompute_v1_mcp(data["records"])
    return {
        "experiment": data.get("experiment"),
        "cells": len(data["records"]),
        "source": "records (recomputed; aggregate values not read)",
        "arms": {
            arm: {
                "success": stats["success"],
                "cells": stats["cells"],
                "success_rate": stats["success_rate"],
                "tool_errors": stats["tool_errors"],
            }
            for arm, stats in computed["arms"].items()
        },
        "paired": computed["paired"],
        "guard_repairs_by_rule": computed["guard_repairs_by_rule"],
    }


def collect(root: Path | None = None) -> dict:
    summary: dict = {}
    offline = _load("metrics.json", root)
    if offline:
        summary["offline"] = _offline(offline)
    for name in ("live_demo_hard_v4.json", "live_demo_hard_v3.json",
                 "live_demo_hard_v3_hnconsistent.json", "live_demo_hard_v2.json"):
        data = _load(name, root)
        if data:
            summary[name.removesuffix(".json")] = _single_model(data)
    for name in ("mcp_demo_v2.json", "mcp_demo_v3.json"):
        data = _load(name, root)
        if data:
            summary[name.removesuffix(".json")] = _mcp(data)
    for name in ("live_demo_multimodel_v4.json", "live_demo_multimodel_v3.json",
                 "live_demo_multimodel.json"):
        data = _load(name, root)
        if data:
            summary[name.removesuffix(".json")] = _multimodel(data)
    hard_v1 = _load("live_demo_hard.json", root)
    if hard_v1:
        summary["live_demo_hard"] = _v1_hard(hard_v1)
    live_v1 = _load("live_demo.json", root)
    if live_v1:
        summary["live_demo"] = _v1_live(live_v1)
    mcp_v1 = _load("mcp_demo.json", root)
    if mcp_v1:
        summary["mcp_demo"] = _v1_mcp(mcp_v1)
    return summary


def main() -> int:
    root = Path(sys.argv[sys.argv.index("--root") + 1]) if "--root" in sys.argv else None
    problems = validate_artifacts(root)
    summary = collect(root)
    if "--json" in sys.argv:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print("=" * 78)
        print("ToolGuard evidence summary — derived numbers recomputed from raw records")
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
