"""B6 unit tests: two opt-in deterministic rules (default off) + the coercion no-op.

The rules are:
1. ``enable_required_default_completion`` -- fill every missing required parameter from
   the schema-declared example/default and pass the full schema gate;
2. ``enable_enum_normalization`` -- replace a value that differs from exactly one declared
   enum member only by case/space/hyphen/underscore; refuse when several members match.

Rule 3 of the B6 scope ("boolean/numeric-string coercion") is deliberately *not* a new
rule: the historical ladder already coerces unambiguous numeric strings for
integer/number parameters and "true"/"false" for boolean parameters
(``rules._repair_value_coercion`` + ``engine.coerce_values_to_schema``); the last test
pins that existing coverage.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tool_evolution.toolguard.engine import (  # noqa: E402
    _enum_normalization, _required_default_completion, repair_failure,
)
from tool_evolution.toolguard.extractors import ExtractorMap  # noqa: E402
from tool_evolution.toolguard.records import FunctionSchema, ToolCall  # noqa: E402

ENUM_SCHEMA = FunctionSchema(
    name="get_rate_for_fx",
    properties={
        "currency": {"type": "string", "enum": ["USD", "EUR", "GBP"]},
        "amount": {"type": "integer"},
    },
    required=("currency",),
)
AMBIGUOUS_ENUM_SCHEMA = FunctionSchema(
    name="get_rate_for_fx",
    properties={"currency": {"type": "string", "enum": ["US-D", "USD"]}},
    required=("currency",),
)
DEFAULT_SCHEMA = FunctionSchema(
    name="search_for_books",
    properties={
        "start_date": {"type": "string", "example_value": "2023-01-01"},
        "end_date": {"type": "string", "example_value": "2023-12-31"},
    },
    required=("start_date", "end_date"),
)


def _call(tool: str, error: str, arguments: dict | None = None) -> ToolCall:
    return ToolCall(
        trace_id="m/G3/try0", method="m", model="gpt-4-0613", strategy="cot",
        group="G3_instruction", query_id="1", try_index=0, step=1,
        tool_name=tool, arguments=arguments or {}, error=error, recorded_ok=False,
        available_tools=[tool], valid_data=True, query="",
    )


# ---------------------------------------------------------------------------
# rule 1: required default completion
#
# Reading the code showed the *historical* ``fill_required_from_schema`` rule already
# fills named / all-required defaults for missing_required errors, so this new rule is
# shadowed in the repair ladder.  The tests below pin both facts: the historical coverage
# and the new opt-in helper itself (which the B6 experiment measures on the real data).
# ---------------------------------------------------------------------------
def test_required_default_completion_is_already_covered_by_the_historical_rule():
    call = _call("search_for_books", "No input parameters provided")
    result = repair_failure(call, {"search_for_books": DEFAULT_SCHEMA}, None, ExtractorMap())
    assert result.repairable and result.schema_valid
    assert result.rules == ["fill_required_from_schema:end_date,start_date"]
    assert result.arguments == {"start_date": "2023-01-01", "end_date": "2023-12-31"}


def test_required_default_completion_helper_fills_every_missing_required_param():
    call = _call("search_for_books", "Missing parameters")
    result = _required_default_completion(call, DEFAULT_SCHEMA, "missing_required")
    assert result is not None
    assert result.repairable and result.schema_valid
    assert result.arguments == {"start_date": "2023-01-01", "end_date": "2023-12-31"}
    assert "required_default:start_date" in result.rules


def test_new_required_default_rule_is_shadowed_by_the_historical_rule():
    call = _call("search_for_books", "No input parameters provided")
    result = repair_failure(call, {"search_for_books": DEFAULT_SCHEMA}, None, ExtractorMap(),
                            enable_required_default_completion=True)
    assert result.repairable and result.schema_valid
    # the historical rule comes first in the ladder, so the new rule never wins here
    assert result.rules == ["fill_required_from_schema:end_date,start_date"]


def test_required_default_completion_refuses_when_a_required_param_has_no_default():
    schema = FunctionSchema(
        name="search_for_books",
        properties={"start_date": {"type": "string", "example_value": "2023-01-01"},
                    "end_date": {"type": "string"}},
        required=("start_date", "end_date"),
    )
    call = _call("search_for_books", "Missing parameters")
    assert _required_default_completion(call, schema, "missing_required") is None
    result = repair_failure(call, {"search_for_books": schema}, None, ExtractorMap(),
                            enable_required_default_completion=True)
    # the historical named/default rule also refuses (one required value is unknown)
    assert not result.proposed


# ---------------------------------------------------------------------------
# rule 2: unique enum normalization
# ---------------------------------------------------------------------------
def test_enum_normalization_is_off_by_default():
    call = _call("get_rate_for_fx", "The currency code 'usd' is not recognized",
                 {"currency": "usd"})
    result = repair_failure(call, {"get_rate_for_fx": ENUM_SCHEMA}, None, ExtractorMap())
    assert not result.proposed


def test_enum_normalization_replaces_case_space_hyphen_variants():
    for variant in ("usd", "Usd", "us d", "us-d", "US_D"):
        call = _call("get_rate_for_fx", "The currency code is not recognized",
                     {"currency": variant})
        result = repair_failure(call, {"get_rate_for_fx": ENUM_SCHEMA}, None, ExtractorMap(),
                                enable_enum_normalization=True)
        assert result.repairable and result.schema_valid, variant
        assert result.arguments["currency"] == "USD"
        assert result.rules == ["enum_normalize:currency"]


def test_enum_normalization_refuses_when_multiple_members_match():
    # "US_D" normalizes to "usd", which matches both "US-D" and "USD": refuse, do not guess
    call = _call("get_rate_for_fx", "The currency code is not recognized",
                 {"currency": "US_D"})
    assert _enum_normalization(call, AMBIGUOUS_ENUM_SCHEMA) is None
    result = repair_failure(call, {"get_rate_for_fx": AMBIGUOUS_ENUM_SCHEMA}, None,
                            ExtractorMap(), enable_enum_normalization=True)
    assert "generic_success_replay" not in result.rules
    assert not any(rule.startswith("enum_normalize") for rule in result.rules)


def test_enum_normalization_ignores_valid_values_and_non_declared_enums():
    call = _call("get_rate_for_fx", "The currency code is not recognized",
                 {"currency": "USD"})
    result = repair_failure(call, {"get_rate_for_fx": ENUM_SCHEMA}, None, ExtractorMap(),
                            enable_enum_normalization=True)
    assert not any(rule.startswith("enum_normalize") for rule in result.rules)


# ---------------------------------------------------------------------------
# rule 3: already covered by the historical coercion -- pin the existing behaviour
# ---------------------------------------------------------------------------
def test_numeric_string_coercion_is_already_covered_without_new_rules():
    schema = FunctionSchema(
        name="get_rate_for_fx",
        properties={"currency": {"type": "string", "enum": ["USD"]},
                    "amount": {"type": "integer"}},
        required=("currency",),
    )
    call = _call("get_rate_for_fx", "The amount is invalid: expected integer",
                 {"currency": "USD", "amount": "100"})
    result = repair_failure(call, {"get_rate_for_fx": schema}, None, ExtractorMap())
    assert result.repairable and result.schema_valid
    assert any("coerce" in rule for rule in result.rules)
    assert result.arguments["amount"] == 100


def test_boolean_string_coercion_is_already_covered_without_new_rules():
    schema = FunctionSchema(
        name="get_rate_for_fx",
        properties={"available": {"type": "boolean"}},
        required=("available",),
    )
    call = _call("get_rate_for_fx", "The value is invalid: expected boolean",
                 {"available": "true"})
    result = repair_failure(call, {"get_rate_for_fx": schema}, None, ExtractorMap())
    assert result.repairable and result.schema_valid
    assert any("coerce" in rule for rule in result.rules)
    assert result.arguments["available"] is True
