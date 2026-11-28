"""A1.7 unit tests: generic missing-parameter error replay from same task/function successes.

The replay rule is opt-in (``allow_recorded_success_replay=True``), only applies to
``missing_required`` errors that name no parameter, and only when the regular ladder
produced no schema-valid proposal.  It reuses the normal schema gate / value-verified
tier, so a schema-invalid replay is still reported honestly.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tool_evolution.toolguard.engine import repair_failure  # noqa: E402
from tool_evolution.toolguard.extractors import ExtractorMap  # noqa: E402
from tool_evolution.toolguard.records import FunctionSchema, ToolCall  # noqa: E402

TRENDING_SCHEMA = FunctionSchema(
    name="get_trending_for_url_content_extractor",
    properties={"category": {"type": "string"}, "region": {"type": "string"}},
    required=(),
)
EMPTY_SCHEMA = FunctionSchema(name="get_user_orders_for_onboarding_project_v3",
                              properties={}, required=())


def _call(tool: str, error: str, arguments: dict | None = None,
          query: str = "") -> ToolCall:
    return ToolCall(
        trace_id="m/G3/try0", method="m", model="gpt-4-0613", strategy="cot",
        group="G3_instruction", query_id="1", try_index=0, step=1,
        tool_name=tool, arguments=arguments or {}, error=error, recorded_ok=False,
        available_tools=[tool], valid_data=True, query=query,
    )


def test_generic_replay_is_off_by_default():
    call = _call("get_trending_for_url_content_extractor", "No input parameters provided")
    result = repair_failure(call, {"get_trending_for_url_content_extractor": TRENDING_SCHEMA},
                            None, ExtractorMap(),
                            successful_arguments=[{"category": "news"}])
    assert not result.proposed, "the historical ladder must be unchanged by default"


def test_generic_replay_proposes_the_recorded_success_and_passes_the_schema_gate():
    call = _call("get_trending_for_url_content_extractor", "No input parameters provided")
    result = repair_failure(call, {"get_trending_for_url_content_extractor": TRENDING_SCHEMA},
                            None, ExtractorMap(),
                            successful_arguments=[{"category": "news"}],
                            allow_recorded_success_replay=True)
    assert result.repairable and result.schema_valid
    assert result.arguments == {"category": "news"}
    assert result.rules == ["generic_success_replay"]


def test_generic_replay_schema_invalid_candidate_is_not_silently_passed():
    """With no declared parameters the candidate fails the schema; it is admitted only
    through the existing value-verified tier (exact recorded success), reported as such."""
    call = _call("get_user_orders_for_onboarding_project_v3",
                 "Missing input parameters for retrieving user orders.")
    result = repair_failure(call, {"get_user_orders_for_onboarding_project_v3": EMPTY_SCHEMA},
                            None, ExtractorMap(),
                            successful_arguments=[{"user_id": "123456789"}],
                            allow_recorded_success_replay=True)
    assert result.repairable
    assert result.value_verified is True
    assert result.schema_valid is False
    # the schema violation is exposed, not swallowed
    assert any("user_id" in error for error in result.schema_errors)
    assert "recorded_success_match" in result.rules


def test_generic_replay_only_applies_to_unnamed_errors():
    schema = FunctionSchema(
        name="coindesk_for_cryptocurrency_news",
        properties={"source": {"type": "string"}},
        required=("source",),
    )
    call = _call("coindesk_for_cryptocurrency_news",
                 "Missing required parameters: 'source'")
    result = repair_failure(call, {"coindesk_for_cryptocurrency_news": schema},
                            None, ExtractorMap(),
                            successful_arguments=[{"source": "CoinDesk"}],
                            allow_recorded_success_replay=True)
    assert not result.proposed
    assert "generic_success_replay" not in result.rules


def test_generic_replay_is_skipped_when_the_ladder_already_has_a_schema_valid_repair():
    schema = FunctionSchema(
        name="advanced_search_for_ott_details",
        properties={"type": {"type": "string", "example_value": "movie"}},
        required=(),
    )
    call = _call("advanced_search_for_ott_details", "Missing parameters")
    result = repair_failure(call, {"advanced_search_for_ott_details": schema},
                            None, ExtractorMap(),
                            successful_arguments=[{"type": "show"}],
                            allow_recorded_success_replay=True)
    assert result.repairable and result.schema_valid
    assert result.rules[0].startswith("schema_default")
    assert "generic_success_replay" not in result.rules


def test_generic_replay_does_not_apply_to_other_error_kinds():
    call = _call("get_trending_for_url_content_extractor", "API not working")
    result = repair_failure(call, {"get_trending_for_url_content_extractor": TRENDING_SCHEMA},
                            None, ExtractorMap(),
                            successful_arguments=[{"category": "news"}],
                            allow_recorded_success_replay=True)
    assert not result.proposed


def test_generic_replay_dedupes_and_stops_at_the_first_valid_candidate():
    call = _call("get_trending_for_url_content_extractor", "No input parameters provided")
    result = repair_failure(
        call, {"get_trending_for_url_content_extractor": TRENDING_SCHEMA}, None, ExtractorMap(),
        successful_arguments=[
            {"category": "news"}, {"category": "news"}, {"category": "technology"}],
        allow_recorded_success_replay=True)
    assert result.repairable and result.schema_valid
    assert result.arguments == {"category": "news"}
