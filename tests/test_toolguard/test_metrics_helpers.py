"""Tests for the statistical helpers used by the V2 live artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

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
