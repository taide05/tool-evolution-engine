"""A1.8.3 unit tests: replay safety (opt-in replay must be read-only, allowlisted, schema-gated).

Requirements covered:

* a tool that defines the fix must be on an explicit replay allowlist (default empty);
* side-effectful tool names are denied even when allowlisted;
* an empty declared schema cannot self-verify a replay (no schema-valid / recorded-success
  admission);
* a read-only allowlisted tool with a non-empty schema can replay;
* the replay candidate source can be training-side successes without falling back to the
  test-fold reference (the default ``successful_arguments``);
* duplicate recorded successes are deduped;
* the default configuration (``allow_recorded_success_replay=False``) is unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tool_evolution.toolguard.engine import (  # noqa: E402
    is_side_effect_tool, repair_failure, replay_tool_allowed,
)
from tool_evolution.toolguard.extractors import ExtractorMap  # noqa: E402
from tool_evolution.toolguard.records import FunctionSchema, ToolCall  # noqa: E402

TRENDING_TOOL = "get_trending_for_url_content_extractor"
TRENDING_SCHEMA = FunctionSchema(
    name=TRENDING_TOOL,
    properties={"category": {"type": "string"}, "region": {"type": "string"}},
    required=(),
)
TRENDING_ALLOWLIST = frozenset({TRENDING_TOOL})
CREATE_TOOL = "create_invoice_for_acme"
CREATE_SCHEMA = FunctionSchema(
    name=CREATE_TOOL,
    properties={"amount": {"type": "number"}},
    required=("amount",),
)
EMPTY_TOOL = "get_user_orders_for_onboarding_project_v3"
EMPTY_SCHEMA = FunctionSchema(name=EMPTY_TOOL, properties={}, required=())


def _call(tool: str, error: str = "No input parameters provided",
          arguments: dict | None = None) -> ToolCall:
    return ToolCall(
        trace_id="m/G3/try0", method="m", model="gpt-4-0613", strategy="cot",
        group="G3_instruction", query_id="1", try_index=0, step=1,
        tool_name=tool, arguments=arguments or {}, error=error, recorded_ok=False,
        available_tools=[tool], valid_data=True, query="",
    )


# ---------------------------------------------------------------------------
# allowlist / side-effect / empty-schema gates
# ---------------------------------------------------------------------------
def test_side_effect_classifier_is_conservative():
    assert is_side_effect_tool("create_invoice_for_acme")
    assert is_side_effect_tool("delete_user_for_admin")
    assert is_side_effect_tool("place_order_for_shop")
    assert is_side_effect_tool("send_message_for_chat")
    assert not is_side_effect_tool(TRENDING_TOOL)
    assert not is_side_effect_tool("search_for_astronomy")
    assert not is_side_effect_tool("get_user_orders_for_onboarding_project_v3")


def test_replay_tool_allowed_matrix():
    assert replay_tool_allowed(TRENDING_TOOL, TRENDING_SCHEMA, TRENDING_ALLOWLIST)
    assert not replay_tool_allowed(TRENDING_TOOL, TRENDING_SCHEMA, None)
    assert not replay_tool_allowed(TRENDING_TOOL, TRENDING_SCHEMA, frozenset())
    assert not replay_tool_allowed(TRENDING_TOOL, TRENDING_SCHEMA, frozenset({"other_tool"}))
    assert not replay_tool_allowed(CREATE_TOOL, CREATE_SCHEMA, frozenset({CREATE_TOOL}))
    assert not replay_tool_allowed(EMPTY_TOOL, EMPTY_SCHEMA, frozenset({EMPTY_TOOL}))


def test_replay_requires_allowlist_even_when_opted_in():
    call = _call(TRENDING_TOOL)
    result = repair_failure(call, {TRENDING_TOOL: TRENDING_SCHEMA}, None, ExtractorMap(),
                            successful_arguments=[{"category": "news"}],
                            allow_recorded_success_replay=True)
    assert not result.proposed
    assert "generic_success_replay" not in result.rules


def test_replay_tool_not_in_allowlist_is_denied():
    call = _call(TRENDING_TOOL)
    result = repair_failure(call, {TRENDING_TOOL: TRENDING_SCHEMA}, None, ExtractorMap(),
                            successful_arguments=[{"category": "news"}],
                            allow_recorded_success_replay=True,
                            replay_allowlist=frozenset({"some_other_tool"}))
    assert not result.proposed
    assert "generic_success_replay" not in result.rules


def test_side_effect_tool_is_denied_even_when_allowlisted():
    call = _call(CREATE_TOOL, "Missing input parameters.")
    result = repair_failure(call, {CREATE_TOOL: CREATE_SCHEMA}, None, ExtractorMap(),
                            successful_arguments=[{"amount": 10.0}],
                            allow_recorded_success_replay=True,
                            replay_allowlist=frozenset({CREATE_TOOL}))
    assert not result.proposed, "a side-effectful tool must never be replayed"
    assert "generic_success_replay" not in result.rules


def test_empty_schema_replay_is_denied():
    call = _call(EMPTY_TOOL, "Missing input parameters for retrieving user orders.")
    result = repair_failure(call, {EMPTY_TOOL: EMPTY_SCHEMA}, None, ExtractorMap(),
                            successful_arguments=[{"user_id": "123456789"}],
                            allow_recorded_success_replay=True,
                            replay_allowlist=frozenset({EMPTY_TOOL}))
    assert not result.proposed
    assert result.value_verified is False


def test_allowlisted_read_only_tool_with_non_empty_schema_can_replay():
    call = _call(TRENDING_TOOL)
    result = repair_failure(call, {TRENDING_TOOL: TRENDING_SCHEMA}, None, ExtractorMap(),
                            successful_arguments=[{"category": "news"}],
                            allow_recorded_success_replay=True,
                            replay_allowlist=TRENDING_ALLOWLIST)
    assert result.repairable and result.schema_valid
    assert result.arguments == {"category": "news"}
    assert result.rules == ["generic_success_replay"]


# ---------------------------------------------------------------------------
# candidate source: training-side successes, not the test-fold oracle
# ---------------------------------------------------------------------------
def test_replay_source_can_be_training_side_successes():
    call = _call(TRENDING_TOOL)
    result = repair_failure(
        call, {TRENDING_TOOL: TRENDING_SCHEMA}, None, ExtractorMap(),
        successful_arguments=[{"category": "oracle_from_test_fold"}],
        allow_recorded_success_replay=True,
        replay_allowlist=TRENDING_ALLOWLIST,
        replay_successes=[{"category": "training_side"}])
    assert result.repairable and result.schema_valid
    assert result.arguments == {"category": "training_side"}


def test_empty_training_replay_source_does_not_fall_back_to_test_fold_successes():
    call = _call(TRENDING_TOOL)
    result = repair_failure(
        call, {TRENDING_TOOL: TRENDING_SCHEMA}, None, ExtractorMap(),
        successful_arguments=[{"category": "oracle_from_test_fold"}],
        allow_recorded_success_replay=True,
        replay_allowlist=TRENDING_ALLOWLIST,
        replay_successes=[])
    assert not result.proposed, "no training-side success means no replay (no oracle fallback)"


def test_replay_candidates_are_deduped():
    call = _call(TRENDING_TOOL)
    result = repair_failure(
        call, {TRENDING_TOOL: TRENDING_SCHEMA}, None, ExtractorMap(),
        successful_arguments=[],
        allow_recorded_success_replay=True,
        replay_allowlist=TRENDING_ALLOWLIST,
        replay_successes=[{"category": "news"}, {"category": "news"},
                          {"category": "technology"}])
    replay_candidates = [candidate for candidate in result.candidates
                         if "generic_success_replay" in candidate["rules"]]
    assert len(replay_candidates) == 2
    assert result.arguments == {"category": "news"}


# ---------------------------------------------------------------------------
# default configuration is unchanged
# ---------------------------------------------------------------------------
def test_default_configuration_is_unchanged():
    call = _call(TRENDING_TOOL)
    result = repair_failure(call, {TRENDING_TOOL: TRENDING_SCHEMA}, None, ExtractorMap(),
                            successful_arguments=[{"category": "news"}])
    assert not result.proposed
    assert not result.value_verified
