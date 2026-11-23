"""Tests for the live guard middleware using a fake backend (no network)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tool_evolution.toolguard.backend import CallResult  # noqa: E402
from tool_evolution.toolguard.guard import ToolGuard, coerce_arguments, live_error_kind  # noqa: E402


class FakeBackend:
    def __init__(self, behaviours):
        self.behaviours = list(behaviours)
        self.calls = []

    def schemas(self):
        return [
            {"type": "function", "function": {"name": "get_hn_item",
                                              "parameters": {"type": "object", "properties": {
                                                  "story_id": {"type": "string"}}}}},
            {"type": "function", "function": {"name": "get_weather",
                                              "parameters": {"type": "object", "properties": {
                                                  "latitude": {"type": "number"},
                                                  "longitude": {"type": "number"}}}}},
        ]

    def call(self, function_name, arguments):
        self.calls.append((function_name, arguments))
        behaviour = self.behaviours.pop(0) if self.behaviours else CallResult("ok", response={"ok": True})
        return behaviour if isinstance(behaviour, CallResult) else behaviour(arguments)


def test_live_error_kind_parses_prefixed_messages():
    assert live_error_kind("timeout: request timed out") == "timeout"
    assert live_error_kind("rate_limit: HTTP 429") == "rate_limit"
    assert live_error_kind("invalid_argument: latitude must be a number") == "invalid_argument"
    assert live_error_kind("No such function name: x") == "invalid_argument"


def test_transient_error_is_retried():
    backend = FakeBackend([CallResult("error", error="timeout: request timed out")])
    guard = ToolGuard(backend, sleep_scale=0.0)
    result = guard.call("get_weather", {"latitude": 1.0, "longitude": 2.0})
    assert result.ok
    assert guard.stats["recovered"] == 1
    assert guard.events[0].rule == "transient_retry:1"


def test_error_hint_coercion_repairs_string_integer():
    backend = FakeBackend([
        CallResult("error", error="invalid_argument: story_id must be an integer"),
        CallResult("ok", response={"title": "hello"}),
    ])
    guard = ToolGuard(backend, sleep_scale=0.0)
    result = guard.call("get_hn_item", {"story_id": "8863"})
    assert result.ok
    assert backend.calls[-1][1] == {"story_id": 8863}
    assert guard.events[0].rule == "error_hint_coercion"


def test_schema_coercion_helper():
    parameters = {"properties": {"latitude": {"type": "number"}, "longitude": {"type": "number"}}}
    assert coerce_arguments("get_weather", {"latitude": "52.52", "longitude": "13.41"}, parameters) == {
        "latitude": 52.52, "longitude": 13.41}


def test_repaired_call_retries_transient_fault():
    """Regression: a transient fault on the *repaired* call must not reach the agent.

    Found in the live hard-task traces: the HN story_id type repair (string -> int) hit an
    injected transient fault on the repaired call and the old code surfaced it to the
    agent, which spent two extra turns.
    """
    backend = FakeBackend([
        CallResult("error", error="invalid_argument: story_id must be an integer"),
        CallResult("error", error="timeout: request timed out"),
        CallResult("ok", response={"title": "hello"}),
    ])
    guard = ToolGuard(backend, sleep_scale=0.0)
    result = guard.call("get_hn_item", {"story_id": "8863"})
    assert result.ok
    assert backend.calls[-1][1] == {"story_id": 8863}
    assert guard.stats["recovered"] == 1
    assert guard.events[0].rule == "error_hint_coercion+transient_retry:1"
    assert guard.events[0].recovered is True


def test_repaired_call_transient_retry_exhausted_returns_error():
    backend = FakeBackend([
        CallResult("error", error="invalid_argument: story_id must be an integer"),
        CallResult("error", error="timeout: request timed out"),
        CallResult("error", error="timeout: request timed out"),
        CallResult("error", error="timeout: request timed out"),
    ])
    guard = ToolGuard(backend, sleep_scale=0.0)
    result = guard.call("get_hn_item", {"story_id": "8863"})
    assert not result.ok
    assert guard.stats["recovered"] == 0
    assert guard.stats["failed_after_repair"] == 1
