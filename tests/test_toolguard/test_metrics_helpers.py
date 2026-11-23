"""Tests for the statistical helpers used by the V2 / multi-model live artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_live_demo_hard_v2 import parse_budgets, parse_repeats, resolve_budget  # noqa: E402
from run_multimodel_live import estimate_cost  # noqa: E402
from tool_evolution.toolguard.verify import mcnemar_exact, wilson_interval  # noqa: E402


def test_wilson_interval_contains_point_estimate() -> None:
    low, high = wilson_interval(8, 20)
    assert 0.0 <= low < 0.4 < high <= 1.0
    assert high - low > 0.2


def test_wilson_interval_edge_cases() -> None:
    assert wilson_interval(0, 0) == [0.0, 1.0]
    assert wilson_interval(0, 10)[0] == 0.0
    assert wilson_interval(10, 10)[1] == 1.0


def test_mcnemar_exact_matches_known_values() -> None:
    assert mcnemar_exact(0, 0) == 1.0
    assert abs(mcnemar_exact(1, 12) - 0.00341796875) < 1e-9
    assert mcnemar_exact(6, 0) == 0.03125
    assert mcnemar_exact(2, 2) == 1.0


def test_budget_spec_parsing() -> None:
    conditions = parse_budgets("clean+1,clean+4,10")
    assert conditions == ["clean+1", "clean+4", "10"]
    repeats = parse_repeats("clean+1:2,clean+4:1", conditions)
    assert repeats == {"clean+1": 2, "clean+4": 1, "10": 1}
    assert resolve_budget("clean+4", 5) == 9
    assert resolve_budget("10", 5) == 10


def test_multimodel_cost_estimate() -> None:
    spec = {"price_input_per_m": 1.0, "price_output_per_m": 4.0}
    usage = {"prompt_tokens": 2_000_000, "completion_tokens": 500_000, "total_tokens": 2_500_000}
    assert estimate_cost(usage, spec) == 4.0
