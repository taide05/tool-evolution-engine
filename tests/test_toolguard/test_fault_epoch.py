"""Tests for the deterministic transient-fault model used by the live demos.

The live demos inject *fault epochs*: the first N attempts of a faulted logical call fail.
These tests pin that behaviour down without any network: a baseline agent loses one turn
per failed attempt, while ToolGuard's in-middleware retry absorbs the whole epoch inside
a single call.
"""

from __future__ import annotations

from typing import Any

from tool_evolution.toolguard.backend import CallResult
from tool_evolution.toolguard.guard import ToolGuard
from tool_evolution.toolguard.live import FaultyBackend


class CountingBackend:
    """In-memory backend that records how often the real tool was actually hit."""

    def __init__(self) -> None:
        self.calls = 0

    def schemas(self) -> list[dict[str, Any]]:
        return [{
            "type": "function",
            "function": {
                "name": "echo",
                "description": "echo text",
                "parameters": {"type": "object", "properties": {"text": {"type": "string"}}},
            },
        }]

    def call(self, function_name: str, arguments: dict[str, Any]) -> CallResult:
        self.calls += 1
        return CallResult("ok", response={"text": arguments.get("text"), "backend_calls": self.calls})


def test_fault_epoch_fails_the_first_attempts_then_reaches_the_backend() -> None:
    backend = CountingBackend()
    faulty = FaultyBackend(backend, seed=1, p_timeout=1.0, p_rate_limit=0.0,
                           p_server_error=0.0, attempts_to_fail=2)
    first = faulty.call("echo", {"text": "a"})
    second = faulty.call("echo", {"text": "a"})
    third = faulty.call("echo", {"text": "a"})
    assert not first.ok and (first.error or "").startswith("timeout")
    assert not second.ok and (second.error or "").startswith("timeout")
    assert third.ok
    assert backend.calls == 1
    assert faulty.stats["injected"] == 2
    assert faulty.stats["faulted_calls"] == 1


def test_guard_absorbs_a_whole_epoch_inside_one_agent_call() -> None:
    backend = CountingBackend()
    faulty = FaultyBackend(backend, seed=1, p_timeout=1.0, p_rate_limit=0.0,
                           p_server_error=0.0, attempts_to_fail=2)
    guard = ToolGuard(faulty, sleep_scale=0.0)
    result = guard.call("echo", {"text": "b"})
    assert result.ok
    assert guard.stats["recovered"] == 1
    assert guard.events[0].rule == "transient_retry:2"
    assert backend.calls == 1


def test_fault_surface_is_deterministic_for_the_same_logical_call() -> None:
    def first_attempt(seed: int) -> str:
        faulty = FaultyBackend(CountingBackend(), seed=seed, p_timeout=0.3,
                               p_rate_limit=0.3, p_server_error=0.0, attempts_to_fail=2)
        result = faulty.call("echo", {"text": "same arguments"})
        return result.error or "ok"

    assert first_attempt(11) == first_attempt(11)
    assert first_attempt(11) == first_attempt(11)


def test_non_faulted_logical_call_goes_straight_to_the_backend() -> None:
    backend = CountingBackend()
    faulty = FaultyBackend(backend, seed=3, p_timeout=0.0, p_rate_limit=0.0,
                           p_server_error=0.0, attempts_to_fail=2)
    result = faulty.call("echo", {"text": "c"})
    assert result.ok
    assert backend.calls == 1
    assert faulty.stats["injected"] == 0


def test_baseline_retry_costs_one_attempt_per_turn_until_epoch_ends() -> None:
    backend = CountingBackend()
    faulty = FaultyBackend(backend, seed=5, p_timeout=1.0, p_rate_limit=0.0,
                           p_server_error=0.0, attempts_to_fail=2)
    # What a baseline agent does: one attempt per agent turn.
    attempts = [faulty.call("echo", {"text": "d"}) for _ in range(3)]
    assert [result.ok for result in attempts] == [False, False, True]
    assert backend.calls == 1
