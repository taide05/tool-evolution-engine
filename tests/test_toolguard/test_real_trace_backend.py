"""Real-trace fault source for the live A/B (2026-11-25 follow-up).

The headline live A/B uses synthetic fault epochs; this wrapper is the optional
``--fault-source real-trace`` mode that replays captured real API errors for selected
logical calls.  Both A/B arms share the same deterministic selection, so pairing stays
valid.  These tests pin determinism, tool matching, kind filtering and delegation.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tool_evolution.toolguard.backend import CallResult  # noqa: E402
from tool_evolution.toolguard.live import RealTraceFaultBackend  # noqa: E402

TRACES = [
    {"trace_id": "t1", "tool": "tool_a", "outcome": "error",
     "error_kind": "invalid_argument", "error_message": "x must be an integer",
     "arguments": {"x": "1"}},
    {"trace_id": "t2", "tool": "tool_a", "outcome": "error",
     "error_kind": "request_error", "error_message": "HTTP 400: bad request",
     "arguments": {"x": 2}},
    {"trace_id": "t3", "tool": "tool_a", "outcome": "error",
     "error_kind": "not_found", "error_message": "not found",
     "arguments": {"x": 3}},
    {"trace_id": "t4", "tool": "tool_b", "outcome": "unexpected_success",
     "error_kind": None, "error_message": None, "arguments": {}},
]


class _FakeBackend:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def schemas(self):
        return [{"type": "function", "function": {"name": "tool_a", "parameters": {"type": "object"}}},
                {"type": "function", "function": {"name": "tool_b", "parameters": {"type": "object"}}}]

    def available_tool_names(self):
        return ["tool_a", "tool_b"]

    def call(self, function_name, arguments):
        self.calls.append((function_name, arguments))
        return CallResult("ok", response={"function": function_name})


def test_real_trace_backend_replays_matching_trace_then_passes_through() -> None:
    backend = _FakeBackend()
    wrapper = RealTraceFaultBackend(backend, TRACES, seed=7, p_fault=1.0, attempts_to_fail=2)
    first = wrapper.call("tool_a", {"x": 1})
    second = wrapper.call("tool_a", {"x": 1})
    third = wrapper.call("tool_a", {"x": 1})
    assert not first.ok and first.error.startswith(("invalid_argument:", "request_error:"))
    assert not second.ok
    assert third.ok
    assert backend.calls == [("tool_a", {"x": 1})]
    assert wrapper.stats["injected"] == 2
    assert wrapper.stats["faulted_calls"] == 1
    assert set(wrapper.stats["real_trace_ids"]) <= {"t1", "t2"}
    assert sum(wrapper.stats["kinds"].values()) == 2


def test_real_trace_backend_filters_non_injectable_kinds_and_unknown_tools() -> None:
    wrapper = RealTraceFaultBackend(_FakeBackend(), TRACES, seed=7, p_fault=1.0,
                                    attempts_to_fail=1)
    # not_found is filtered out; tool_a only has t1/t2 -> t3's kind never replays
    assert wrapper.traces_by_tool["tool_a"] == [TRACES[0], TRACES[1]]
    assert "not_found" not in wrapper.stats["kinds"]
    # a tool with no injected traces passes straight through
    backend = _FakeBackend()
    wrapper = RealTraceFaultBackend(backend, TRACES, seed=7, p_fault=1.0, attempts_to_fail=1)
    result = wrapper.call("tool_b", {"y": 1})
    assert result.ok and backend.calls == [("tool_b", {"y": 1})]
    # delegations used by ToolGuard / live runner
    assert wrapper.available_tool_names() == ["tool_a", "tool_b"]
    assert len(wrapper.schemas()) == 2


def test_real_trace_backend_selection_is_deterministic() -> None:
    def sequence() -> list[str]:
        backend = _FakeBackend()
        wrapper = RealTraceFaultBackend(backend, TRACES, seed=11, p_fault=1.0,
                                        attempts_to_fail=2)
        errors = []
        for _ in range(6):
            result = wrapper.call("tool_a", {"x": 5})
            if not result.ok:
                errors.append(result.error)
        return errors

    assert sequence() == sequence()
