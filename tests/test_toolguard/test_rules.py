"""Unit tests for deterministic repair rules and the experience-memory engine."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tool_evolution.toolguard.engine import repair_failure  # noqa: E402
from tool_evolution.toolguard.memory import SuccessMemory  # noqa: E402
from tool_evolution.toolguard.records import FunctionSchema, ToolCall  # noqa: E402
from tool_evolution.toolguard.rules import classify_error, repair_call  # noqa: E402

SCHEMAS = {
    "search_for_weather": FunctionSchema(
        name="search_for_weather",
        properties={
            "city": {"type": "string"},
            "days": {"type": "integer", "example_value": 3},
        },
        required=("city",),
    ),
    "search_for_news": FunctionSchema(
        name="search_for_news",
        properties={"source": {"type": "string"}, "limit": {"type": "integer"}},
        required=("source",),
    ),
}


def make_call(tool: str, args: dict, error: str | None, ok: bool = False, valid: bool = True,
              group: str = "G1_instruction", query_id: str = "1") -> ToolCall:
    return ToolCall(
        trace_id="m/G1_instruction/1/try0",
        method="gpt-3.5-turbo-1106_cot",
        model="gpt-3.5-turbo-1106",
        strategy="cot",
        group=group,
        query_id=query_id,
        try_index=0,
        step=1,
        tool_name=tool,
        arguments=args,
        error=error,
        recorded_ok=ok,
        available_tools=list(SCHEMAS),
        valid_data=valid,
    )


def test_classify_error_covers_real_patterns():
    assert classify_error("Missing required parameters: 'source'") == "missing_required"
    assert classify_error("No such function name: finish") == "no_such_function"
    assert classify_error("function got an unexpected keyword argument 'x'") == "unexpected_kwarg"
    assert classify_error("The country code 'ZZ' is not recognized") == "invalid_value"
    assert classify_error("Failed to generate fake response") == "server_error"
    assert classify_error("") == "other"


def test_missing_required_filled_from_schema_defaults():
    result = repair_call("search_for_weather", {}, "Missing required parameters: 'city'",
                         list(SCHEMAS), SCHEMAS)
    # city has no default -> not repairable
    assert not result.repairable


def test_missing_required_filled_when_schema_has_example_value():
    result = repair_call("search_for_weather", {"city": "Paris"},
                         "Missing required parameters: 'days'", list(SCHEMAS), SCHEMAS)
    # 'days' has an example_value in the schema -> the rule fills it from the schema.
    assert result.repairable
    assert result.schema_valid
    assert result.arguments["days"] == 3


def test_unexpected_keyword_is_dropped_or_renamed():
    result = repair_call("search_for_weather", {"city": "Paris", "dayz": 3},
                         "got an unexpected keyword argument 'dayz'", list(SCHEMAS), SCHEMAS)
    assert result.repairable
    assert result.schema_valid
    assert "days" in result.arguments


def test_function_name_resolution_picks_available_tool():
    result = repair_call("search_for_wether", {}, "No such function name: search_for_wether",
                         ["search_for_weather", "search_for_news"], SCHEMAS)
    assert result.repairable
    assert result.function_name == "search_for_weather"


def test_memory_replay_uses_previous_success():
    success = make_call("search_for_news", {"source": "coindesk", "limit": 5}, None, ok=True)
    memory = SuccessMemory.build([success], SCHEMAS)
    failed = make_call("search_for_news", {}, "Missing required parameters: 'source'")
    result = repair_failure(failed, SCHEMAS, memory)
    assert result.repairable
    assert result.schema_valid
    assert result.arguments["source"] == "coindesk"
    assert "memory_replay_successful_call" in result.rules


def test_memory_respects_trace_validity():
    bad_trace = make_call("search_for_news", {"source": "x", "limit": 1}, None, ok=True, valid=False)
    memory = SuccessMemory.build([bad_trace], SCHEMAS)
    assert memory.size()["successful_calls"] == 0


def test_invalid_value_memory_normalization():
    successes = [
        make_call("search_for_news", {"source": "coindesk", "limit": 5}, None, ok=True),
        make_call("search_for_news", {"source": "cnn", "limit": 5}, None, ok=True),
    ]
    memory = SuccessMemory.build(successes, SCHEMAS)
    failed = make_call("search_for_news", {"source": "unknown-source", "limit": 5},
                       "The source 'unknown-source' is not recognized")
    result = repair_failure(failed, SCHEMAS, memory)
    assert result.repairable
    assert result.arguments["source"] in {"coindesk", "cnn"}
