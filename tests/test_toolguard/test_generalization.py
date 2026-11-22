"""Tests for cross-task generalisation: missing-name parsing, extractors and text repair."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tool_evolution.toolguard.engine import repair_failure  # noqa: E402
from tool_evolution.toolguard.extractors import ExtractorMap  # noqa: E402
from tool_evolution.toolguard.records import FunctionSchema, ToolCall  # noqa: E402
from tool_evolution.toolguard.rules import parse_missing_params  # noqa: E402


def make_call(tool: str, args: dict, error: str | None = None, query: str = "",
              ok: bool = False) -> ToolCall:
    return ToolCall(
        trace_id="m/G3/try0", method="m", model="gpt-4-0613", strategy="cot",
        group="G3_instruction", query_id="1", try_index=0, step=1,
        tool_name=tool, arguments=args, error=error, recorded_ok=ok,
        available_tools=[tool], valid_data=True, query=query,
    )


DATE_SCHEMA = FunctionSchema(
    name="new_titles_for_watchmode",
    properties={
        "start_date": {"type": "string"},
        "end_date": {"type": "string"},
        "page": {"type": "string", "example_value": "1"},
    },
    required=(),
)


def test_parse_missing_params_supports_unquoted_names():
    assert parse_missing_params("Missing required parameters: start_date, end_date") == ["start_date", "end_date"]
    assert parse_missing_params("Missing required parameters: 'source'") == ["source"]


def test_extractor_map_learns_date_extractor():
    successes = [
        make_call("new_titles_for_watchmode",
                  {"start_date": "2022-06-01", "end_date": "2022-06-30"},
                  query="Provide titles added between June 1, 2022, and June 30, 2022.", ok=True),
        make_call("new_titles_for_watchmode",
                  {"start_date": "2022-06-01", "end_date": "2022-06-30"},
                  query="Provide titles added between June 1, 2022, and June 30, 2022.", ok=True),
        make_call("new_titles_for_watchmode",
                  {"start_date": "2021-01-01", "end_date": "2021-01-31"},
                  query="Titles added between January 1, 2021, and January 31, 2021.", ok=True),
        make_call("new_titles_for_watchmode",
                  {"start_date": "2021-01-01", "end_date": "2021-01-31"},
                  query="Titles added between January 1, 2021, and January 31, 2021.", ok=True),
    ]
    mapping = ExtractorMap.learn(successes, {"new_titles_for_watchmode": DATE_SCHEMA})
    assert mapping.extractor_for("start_date") == "dates"
    assert "start_date" in mapping.by_param


def test_text_repair_recovers_dates_from_unseen_task():
    mapping = ExtractorMap()
    # even without learning, the deterministic name fallback handles date parameters
    call = make_call(
        "new_titles_for_watchmode", {},
        error="Missing required parameters: start_date, end_date",
        query="Provide titles added between June 1, 2022, and June 30, 2022.",
    )
    result = repair_failure(call, {"new_titles_for_watchmode": DATE_SCHEMA}, None, mapping)
    assert result.repairable
    assert result.arguments["start_date"] == "2022-06-01"
    assert result.arguments["end_date"] == "2022-06-30"
    assert "text_extract:start_date" in result.rules


def test_text_repair_expands_month_year_range():
    call = make_call(
        "new_titles_for_watchmode", {},
        error="Missing required parameters: start_date, end_date",
        query="List titles added in June 2022.",
    )
    result = repair_failure(call, {"new_titles_for_watchmode": DATE_SCHEMA}, None, ExtractorMap())
    assert result.repairable
    assert result.arguments["start_date"] == "2022-06-01"
    assert result.arguments["end_date"] == "2022-06-30"


def test_schema_default_completion_for_unnamed_missing_parameters():
    schema = FunctionSchema(
        name="advanced_search_for_ott_details",
        properties={
            "start_year": {"type": "string", "example_value": "1970"},
            "type": {"type": "string", "example_value": "movie"},
        },
        required=(),
    )
    call = make_call("advanced_search_for_ott_details", {}, error="Missing parameters")
    result = repair_failure(call, {"advanced_search_for_ott_details": schema}, None, ExtractorMap())
    assert result.repairable
    assert result.arguments == {"start_year": "1970", "type": "movie"}
    assert result.rules[0].startswith("schema_default")
