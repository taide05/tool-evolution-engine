"""Negative tests: a tampered aggregate must fail the artifact contract.

The point of the evidence chain is that the artifacts can catch a doctored headline number.
These tests copy the committed artifacts into a scratch directory, change exactly one
derived field in an ``aggregate`` block, and require ``validate_artifacts`` to report it.
The fields covered are the ones the 2026-11-24 review found could be tampered without
detection: token counts, success rates, turns/cell, tool-error counts, Wilson intervals and
the McNemar p-value (plus the same fields for the multi-model and MCP artifacts).
"""

from __future__ import annotations

import json
import shutil
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from tool_evolution.toolguard.artifact_contracts import validate_artifacts  # noqa: E402

BENCH = ROOT / "benchmarks" / "toolguard"


@pytest.fixture()
def artifact_copy() -> Path:
    """A writable copy of the committed artifacts (pytest tmp_path ACL is restricted here)."""
    workdir = ROOT / ".scratch" / "test-toolguard" / f"tamper-{uuid.uuid4().hex}"
    workdir.mkdir(parents=True, exist_ok=True)
    for path in BENCH.glob("*.json"):
        shutil.copy2(path, workdir / path.name)
    try:
        yield workdir
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _tamper(workdir: Path, artifact: str, mutate) -> list[str]:
    path = workdir / artifact
    data = json.loads(path.read_text(encoding="utf-8"))
    mutate(data)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return validate_artifacts(workdir)


def _guard_arm(data: dict, condition: str = "clean+1") -> dict:
    return data["aggregate"]["conditions"][condition]["arms"]["guard"]


def _baseline_arm(data: dict, condition: str = "clean+1") -> dict:
    return data["aggregate"]["conditions"][condition]["arms"]["baseline"]


def _condition(data: dict, condition: str = "clean+1") -> dict:
    return data["aggregate"]["conditions"][condition]


def test_untampered_copy_passes_contract(artifact_copy: Path) -> None:
    assert validate_artifacts(artifact_copy) == []


def test_tampered_total_tokens_is_detected(artifact_copy: Path) -> None:
    def mutate(data: dict) -> None:
        _guard_arm(data)["usage"]["total_tokens"] = 1

    problems = _tamper(artifact_copy, "live_demo_hard_v3.json", mutate)
    assert any("total_tokens" in problem for problem in problems), problems


def test_tampered_success_rate_is_detected(artifact_copy: Path) -> None:
    def mutate(data: dict) -> None:
        _baseline_arm(data)["success_rate"] = 0.0

    problems = _tamper(artifact_copy, "live_demo_hard_v3.json", mutate)
    assert any("success_rate" in problem for problem in problems), problems


def test_tampered_turns_per_cell_is_detected(artifact_copy: Path) -> None:
    def mutate(data: dict) -> None:
        _baseline_arm(data)["turns_per_cell"] = 999.0

    problems = _tamper(artifact_copy, "live_demo_hard_v3.json", mutate)
    assert any("turns_per_cell" in problem for problem in problems), problems


def test_tampered_tool_errors_is_detected(artifact_copy: Path) -> None:
    def mutate(data: dict) -> None:
        _guard_arm(data)["tool_errors"] = 0

    problems = _tamper(artifact_copy, "live_demo_hard_v3.json", mutate)
    assert any("tool_errors" in problem for problem in problems), problems


def test_tampered_wilson_interval_is_detected(artifact_copy: Path) -> None:
    def mutate(data: dict) -> None:
        _baseline_arm(data)["success_wilson_95"] = [0.0, 1.0]

    problems = _tamper(artifact_copy, "live_demo_hard_v3.json", mutate)
    assert any("success_wilson_95" in problem for problem in problems), problems


def test_tampered_mcnemar_p_is_detected(artifact_copy: Path) -> None:
    def mutate(data: dict) -> None:
        _condition(data)["paired_cells"]["mcnemar_exact_p"] = 0.5

    problems = _tamper(artifact_copy, "live_demo_hard_v3.json", mutate)
    assert any("mcnemar_exact_p" in problem for problem in problems), problems


def test_tampered_multimodel_tokens_and_rate_are_detected(artifact_copy: Path) -> None:
    def mutate(data: dict) -> None:
        row = data["aggregate"]["cross_model"]["deepseek-flash"]["conditions"]["clean+1"]
        row["guard_tokens"] = 1
        row["baseline_success_rate"] = 0.0

    problems = _tamper(artifact_copy, "live_demo_multimodel_v3.json", mutate)
    assert any("guard_tokens" in problem for problem in problems), problems
    assert any("baseline_success_rate" in problem for problem in problems), problems


def test_tampered_multimodel_mcnemar_and_cost_are_detected(artifact_copy: Path) -> None:
    def mutate(data: dict) -> None:
        row = data["aggregate"]["cross_model"]["qwen-flash"]["conditions"]["clean+1"]
        row["mcnemar_exact_p"] = 1.0
        data["total_estimated_cost_usd"] = 0.01

    problems = _tamper(artifact_copy, "live_demo_multimodel_v3.json", mutate)
    assert any("mcnemar_exact_p" in problem for problem in problems), problems
    assert any("total_estimated_cost_usd" in problem for problem in problems), problems


def test_tampered_mcp_rate_and_errors_are_detected(artifact_copy: Path) -> None:
    def mutate(data: dict) -> None:
        arm = data["aggregate"]["conditions"]["clean+1"]["arms"]["baseline"]
        arm["success_rate"] = 1.0
        arm["tool_errors"] = 0

    problems = _tamper(artifact_copy, "mcp_demo_v2.json", mutate)
    assert any("success_rate" in problem for problem in problems), problems
    assert any("tool_errors" in problem for problem in problems), problems


def test_tampered_offline_cache_derivation_is_detected(artifact_copy: Path) -> None:
    def mutate(data: dict) -> None:
        lomo = data["results"]["lomo_aggregate"]
        lomo["cache_miss"] = lomo["cache_miss"] - 1  # no longer sums to repairable

    problems = _tamper(artifact_copy, "metrics.json", mutate)
    assert any("cache" in problem for problem in problems), problems


def test_evidence_summary_reports_records_not_tampered_aggregate(artifact_copy: Path) -> None:
    """The reviewer-facing summary must keep reporting the record-derived value."""
    import toolguard_evidence_summary as summary

    before = summary.collect(artifact_copy)["live_demo_hard_v3"]["conditions"]["clean+1"]["guard"]

    def mutate(data: dict) -> None:
        _guard_arm(data)["usage"]["total_tokens"] = 1
        _guard_arm(data)["success_rate"] = 0.0

    problems = _tamper(artifact_copy, "live_demo_hard_v3.json", mutate)
    assert problems, "tampering must be detected"

    after = summary.collect(artifact_copy)["live_demo_hard_v3"]["conditions"]["clean+1"]["guard"]
    assert before["tokens"] == after["tokens"]
    assert after["tokens"] != 1
    assert before["success_rate"] == after["success_rate"]
    assert after["success_rate"] != 0.0


# ---------------------------------------------------------------------------
# two-tier offline evidence artifact: schema funnel / value-verified fields
# ---------------------------------------------------------------------------
def test_tampered_metrics_accepted_tier_is_detected(artifact_copy: Path) -> None:
    def mutate(data: dict) -> None:
        lomo = data["results"]["lomo_aggregate"]
        lomo["accepted_repairs"] = lomo["accepted_repairs"] + 1

    problems = _tamper(artifact_copy, "metrics.json", mutate)
    assert any("accepted" in problem for problem in problems), problems


def test_tampered_metrics_schema_funnel_is_detected(artifact_copy: Path) -> None:
    def mutate(data: dict) -> None:
        lomo = data["results"]["lomo_aggregate"]
        lomo["schema_rejected_failures"] = lomo["schema_rejected_failures"] + 1

    problems = _tamper(artifact_copy, "metrics.json", mutate)
    assert any("funnel" in problem or "proposed" in problem for problem in problems), problems


def test_tampered_metrics_value_verified_tier_is_detected(artifact_copy: Path) -> None:
    def mutate(data: dict) -> None:
        lomo = data["results"]["lomo_aggregate"]
        lomo["value_verified_repairs"] = lomo["value_verified_repairs"] + 1

    problems = _tamper(artifact_copy, "metrics.json", mutate)
    assert any("accepted" in problem or "value_verified" in problem for problem in problems), problems


def test_tampered_metrics_coerced_count_is_detected(artifact_copy: Path) -> None:
    def mutate(data: dict) -> None:
        lomo = data["results"]["lomo_aggregate"]
        lomo["coerced_repairs"] = lomo["schema_passed"] + 1

    problems = _tamper(artifact_copy, "metrics.json", mutate)
    assert any("coerced" in problem for problem in problems), problems


def test_tampered_metrics_schema_valid_rate_is_detected(artifact_copy: Path) -> None:
    def mutate(data: dict) -> None:
        data["results"]["lomo_aggregate"]["schema_valid_rate"] = 1.0

    problems = _tamper(artifact_copy, "metrics.json", mutate)
    assert any("schema_valid_rate" in problem for problem in problems), problems
