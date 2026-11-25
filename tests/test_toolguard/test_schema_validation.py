"""Structural schema validation (P0-2) — positive and negative cases.

The 2026-11-24 review showed the old ``schema_valid`` accepted ``{"year": "2023"}`` for an
``integer`` schema and knew nothing about enums, formats or ranges.  These tests pin the
stricter contract, the honest proposal funnel in the offline evaluator, and the live guard's
validate-before-send behaviour (including that a rejected repair is recorded, not swallowed).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from tool_evolution.toolguard.backend import CallResult  # noqa: E402
from tool_evolution.toolguard.engine import coerce_values_to_schema, repair_failure  # noqa: E402
from tool_evolution.toolguard.extractors import ExtractorMap  # noqa: E402
from tool_evolution.toolguard.guard import ToolGuard  # noqa: E402
from tool_evolution.toolguard.memory import SuccessMemory  # noqa: E402
from tool_evolution.toolguard.records import FunctionSchema, ToolCall  # noqa: E402
from tool_evolution.toolguard.rules import repair_call  # noqa: E402
from tool_evolution.toolguard.schema_check import (  # noqa: E402
    validate_against_parameters,
    validate_arguments,
)


# ---------------------------------------------------------------------------
# validator unit tests: type / enum / format / range / required / extra
# ---------------------------------------------------------------------------
def _schema(properties: dict, required: tuple[str, ...] = ()) -> FunctionSchema:
    return FunctionSchema(name="f", properties=properties, required=required)


def test_string_pretending_to_be_integer_is_rejected() -> None:
    schema = _schema({"year": {"type": "integer"}}, required=("year",))
    errors = validate_arguments({"year": "2023"}, schema)
    assert errors and "integer" in errors[0]
    assert validate_arguments({"year": 2023}, schema) == []


def test_boolean_is_not_an_integer() -> None:
    schema = _schema({"count": {"type": "integer"}})
    assert validate_arguments({"count": True}, schema)


def test_enum_out_of_range_is_rejected() -> None:
    schema = _schema({"unit": {"type": "string", "enum": ["metric", "imperial"]}})
    assert validate_arguments({"unit": "kelvin"}, schema)
    assert validate_arguments({"unit": "metric"}, schema) == []


def test_invalid_format_is_rejected() -> None:
    schema = _schema({"release_date": {"type": "string", "format": "date"}})
    errors = validate_arguments({"release_date": "06/12/2024"}, schema)
    assert errors and "date" in errors[0]
    assert validate_arguments({"release_date": "2024-12-06"}, schema) == []
    date_time = _schema({"at": {"type": "string", "format": "date-time"}})
    assert validate_arguments({"at": "2024-12-06 10:00:00"}, date_time) == []
    assert validate_arguments({"at": "not a timestamp"}, date_time)


def test_unknown_format_is_not_invented_as_a_failure() -> None:
    """Only formats the schema actually declares are checked; unknown ones are annotations."""
    schema = _schema({"x": {"type": "string", "format": "some-future-format"}})
    assert validate_arguments({"x": "anything"}, schema) == []


def test_numeric_range_is_enforced() -> None:
    schema = _schema({"latitude": {"type": "number", "minimum": -90, "maximum": 90}})
    assert validate_arguments({"latitude": 91}, schema)
    assert validate_arguments({"latitude": -91}, schema)
    assert validate_arguments({"latitude": 45.5}, schema) == []
    integer = _schema({"page": {"type": "integer", "minimum": 1}})
    assert validate_arguments({"page": 0}, integer)


def test_required_and_extra_parameters() -> None:
    schema = _schema({"city": {"type": "string"}}, required=("city",))
    assert validate_arguments({}, schema)
    assert validate_arguments({"city": "Paris", "dayz": 3}, schema)
    assert validate_arguments({"city": "Paris"}, schema) == []


def test_nested_object_and_array_validation() -> None:
    schema = _schema({
        "window": {
            "type": "object",
            "properties": {"start": {"type": "string", "format": "date"}},
            "required": ["start"],
        },
        "years": {"type": "array", "items": {"type": "integer"}, "maxItems": 2},
    })
    assert validate_arguments({"window": {"start": "2024-01-01"}, "years": [2020, 2021]}, schema) == []
    assert validate_arguments({"window": {"start": "01/01/2024"}}, schema)
    assert validate_arguments({"years": [2020, "2021"]}, schema)
    assert validate_arguments({"years": [2020, 2021, 2022]}, schema)


def test_validate_against_raw_parameters_block() -> None:
    parameters = {
        "type": "object",
        "properties": {"story_id": {"type": "string"}},
        "required": ["story_id"],
    }
    # advertised schema says string; the backend's error hint says integer
    assert validate_against_parameters({"story_id": 1}, parameters)
    assert validate_against_parameters({"story_id": 1}, parameters,
                                       type_overrides={"story_id": "integer"}) == []
    assert validate_against_parameters({"story_id": "1"}, parameters) == []


# ---------------------------------------------------------------------------
# offline engine: proposed vs schema-rejected funnel
# ---------------------------------------------------------------------------
def test_invalid_rule_proposal_is_kept_as_proposed_and_rejected() -> None:
    schemas = {
        "search_for_weather": _schema(
            {"city": {"type": "string"}, "days": {"type": "integer"}},
            required=("city", "days")),
    }
    # resolving the hallucinated name leaves the required arguments missing -> the proposal
    # exists but must be rejected by the structural validator.
    result = repair_call("search_for_wether", {}, "No such function name: search_for_wether",
                         ["search_for_weather"], schemas)
    assert result.proposed
    assert not result.repairable
    assert not result.schema_valid
    assert result.schema_errors


def test_engine_records_rejected_proposals_in_funnel() -> None:
    from tool_evolution.toolguard.engine import repair_failure
    from tool_evolution.toolguard.extractors import ExtractorMap

    schemas = {
        "search_for_weather": _schema(
            {"city": {"type": "string"}, "days": {"type": "integer"}}, required=("city", "days")),
    }
    call = ToolCall(
        trace_id="t", method="m", model="gpt-4-0613", strategy="cot", group="G1",
        query_id="1", try_index=0, step=1, tool_name="search_for_wether",
        arguments={}, error="No such function name: search_for_wether",
        recorded_ok=False, available_tools=["search_for_weather"], valid_data=True,
    )
    result = repair_failure(call, schemas, None, ExtractorMap())
    assert result.proposed
    assert not result.repairable
    assert result.rejected_proposals
    assert any("city" in item or "days" in item for item in result.rejected_proposals)


def test_offline_evaluator_reports_proposal_funnel() -> None:
    """schema_valid_rate now has the *proposed repairs* denominator, not repairable."""
    from eval_toolguard_rules import evaluate

    schemas = {
        "search_for_weather": _schema(
            {"city": {"type": "string"}, "days": {"type": "integer"},
             "token": {"type": "string"}},
            required=("city", "days", "token")),
    }
    base = dict(trace_id="t", method="m", strategy="cot", group="G1_instruction",
                query_id="1", try_index=0, step=1, valid_data=True, query="")
    train = [ToolCall(**base, model="chatgpt", tool_name="search_for_weather",
                      arguments={"city": "Paris", "days": 3}, error=None, recorded_ok=True,
                      available_tools=["search_for_weather"])]
    # the tool *is* in the schema index (otherwise the failure is skipped as no-schema);
    # the rule drops the unexpected argument and the proposal is then rejected for missing
    # required "days" (nothing else can fill it: no query text, no memory).
    rejected = ToolCall(**base, model="gpt-4-0613", tool_name="search_for_weather",
                        arguments={"city": "Paris", "bogus_flag": 3},
                        error="function got an unexpected keyword argument 'bogus_flag'",
                        recorded_ok=False, available_tools=["search_for_weather"])
    test = [rejected]
    result = evaluate("unit", train, test, schemas, None)
    assert result["test_failures"] == 1
    assert result["proposed_repairs"] == 1
    assert result["schema_rejected_proposals"] >= 1
    assert result["schema_rejected_failures"] == 1
    assert result["schema_passed"] == 0
    assert result["value_verified_repairs"] == 0
    assert result["accepted_repairs"] == 0
    assert result["repairable"] == 0
    assert result["schema_valid_rate"] == 0.0


# ---------------------------------------------------------------------------
# schema-type coercion repair (fixes type-invalid proposals, does not relax schema)
# ---------------------------------------------------------------------------
def test_coerce_values_to_schema_types() -> None:
    schema = _schema({
        "start_year": {"type": "number"},
        "page": {"type": "integer"},
        "label": {"type": "string"},
        "flag": {"type": "boolean"},
    })
    coerced = coerce_values_to_schema(
        {"start_year": "1970", "page": "1", "label": 42, "flag": "true"}, schema)
    assert coerced == {"start_year": 1970.0, "page": 1, "label": "42", "flag": True}


def test_coerce_values_leaves_unconvertible_values_untouched() -> None:
    schema = _schema({"start_date": {"type": "number"}})
    assert coerce_values_to_schema({"start_date": "2022-06-01"}, schema) == {
        "start_date": "2022-06-01"}
    # a float cannot be losslessly coerced to integer -> left for the validator to reject
    schema_int = _schema({"count": {"type": "integer"}})
    assert coerce_values_to_schema({"count": 1.5}, schema_int) == {"count": 1.5}


def test_engine_coercion_repairs_typed_schema_default() -> None:
    """schema default "1970" for a number param becomes a schema-valid repair via coercion."""
    schemas = {
        "titles_between_years": _schema(
            {"start_year": {"type": "number", "example_value": "1970"},
             "end_year": {"type": "number", "example_value": "2020"}},
            required=("start_year", "end_year")),
    }
    call = ToolCall(
        trace_id="t", method="m", model="gpt-4-0613", strategy="cot", group="G1",
        query_id="1", try_index=0, step=1, tool_name="titles_between_years",
        arguments={}, error="Missing required parameters: start_year, end_year",
        recorded_ok=False, available_tools=["titles_between_years"], valid_data=True,
    )
    result = repair_failure(call, schemas, None, ExtractorMap())
    assert result.repairable
    assert result.schema_valid
    assert "schema_type_coercion" in result.rules
    assert result.arguments == {"start_year": 1970.0, "end_year": 2020.0}


# ---------------------------------------------------------------------------
# value-verified evidence tier (exact recorded successful invocation only)
# ---------------------------------------------------------------------------
def _recorded_success_call() -> ToolCall:
    return ToolCall(
        trace_id="t", method="m", model="chatgpt", strategy="cot", group="G1",
        query_id="1", try_index=0, step=0, tool_name="titles_between_years",
        arguments={"start_date": "2022-06-01", "end_date": "2022-06-30"},
        error=None, recorded_ok=True, available_tools=["titles_between_years"],
        valid_data=True,
    )


def _date_schema() -> dict:
    # the upstream schema wrongly declares date parameters as number (fidelity defect)
    return {
        "titles_between_years": _schema(
            {"start_date": {"type": "number"}, "end_date": {"type": "number"}},
            required=("start_date", "end_date")),
    }


def _date_failure() -> ToolCall:
    return ToolCall(
        trace_id="t", method="m", model="gpt-4-0613", strategy="cot", group="G1",
        query_id="1", try_index=0, step=1, tool_name="titles_between_years",
        arguments={}, error="Missing required parameters: start_date, end_date",
        recorded_ok=False, available_tools=["titles_between_years"], valid_data=True,
        query="titles added between June 1, 2022 and June 30, 2022",
    )


def test_engine_value_verified_tier_accepts_exact_recorded_success() -> None:
    schemas = _date_schema()
    memory = SuccessMemory.build([_recorded_success_call()], schemas)
    recorded = [{"start_date": "2022-06-01", "end_date": "2022-06-30"}]
    result = repair_failure(_date_failure(), schemas, memory, ExtractorMap(),
                            successful_arguments=recorded)
    assert result.repairable
    assert result.schema_valid is False  # never mixed into schema validity
    assert result.value_verified is True
    assert "recorded_success_match" in result.rules
    assert result.arguments == recorded[0]


def test_engine_value_verified_tier_rejects_unverified_candidate() -> None:
    schemas = _date_schema()
    memory = SuccessMemory.build([_recorded_success_call()], schemas)
    result = repair_failure(_date_failure(), schemas, memory, ExtractorMap(),
                            successful_arguments=[])
    assert not result.repairable
    assert not result.value_verified
    assert result.proposed



# ---------------------------------------------------------------------------
# live guard: validate repaired call before sending it
# ---------------------------------------------------------------------------
class _FakeBackend:
    def __init__(self, behaviours, schemas):
        self.behaviours = list(behaviours)
        self._schemas = schemas
        self.calls: list[tuple[str, dict]] = []

    def schemas(self):
        return self._schemas

    def available_tool_names(self):
        return [schema["function"]["name"] for schema in self._schemas]

    def call(self, function_name, arguments):
        self.calls.append((function_name, arguments))
        return self.behaviours.pop(0)


def test_guard_records_schema_validation_and_backend_hint_override() -> None:
    backend = _FakeBackend(
        [
            CallResult("error", error="invalid_argument: story_id must be an integer"),
            CallResult("ok", response={"title": "hello"}),
        ],
        [{"type": "function", "function": {"name": "get_hn_item", "parameters": {
            "type": "object",
            "properties": {"story_id": {"type": "string"}},
            "required": ["story_id"]}}}],
    )
    guard = ToolGuard(backend, sleep_scale=0.0)
    result = guard.call("get_hn_item", {"story_id": "8863"})
    assert result.ok
    event = guard.events[0]
    assert event.schema_validated is True
    assert event.schema_overrides == {"story_id": "integer"}
    assert guard.stats["schema_validated"] == 1
    assert guard.stats["schema_rejected"] == 0


def test_guard_rejects_schema_invalid_repair_and_records_it() -> None:
    backend = _FakeBackend(
        [CallResult("error", error="tool_error: Unknown tool: get_weather_forecast")],
        [
            {"type": "function", "function": {"name": "get_weather", "parameters": {
                "type": "object",
                "properties": {"latitude": {"type": "number"}, "longitude": {"type": "number"}},
                "required": ["latitude", "longitude"]}}},
            {"type": "function", "function": {"name": "get_weather_forecast", "parameters": {
                "type": "object", "properties": {"latitude": {"type": "number"}},
                "required": ["latitude"]}}},
        ],
    )
    guard = ToolGuard(backend, sleep_scale=0.0)
    # resolution finds get_weather, but {"longitude": 2.0} alone is missing latitude, and the
    # extra longitude... the repaired call must be rejected before it reaches the backend.
    result = guard.call("get_weather_forecast", {"longitude": 2.0})
    assert not result.ok
    assert result.error == "tool_error: Unknown tool: get_weather_forecast"
    event = guard.events[0]
    assert "schema_rejected" in event.rule
    assert event.schema_errors
    assert event.recovered is False
    assert guard.stats["schema_rejected"] == 1
    # the backend only ever saw the original (failing) call, never the invalid repair
    assert all(name == "get_weather_forecast" for name, _ in backend.calls)


def test_guard_records_unavailable_schema_instead_of_silently_passing() -> None:
    """A hidden real tool has no agent-visible schema: recorded as schema_unavailable."""
    backend = _FakeBackend(
        [
            CallResult("error", error="tool_error: Unknown tool: fetch_fx_rate"),
            CallResult("ok", response={"rate": 1.25}),
        ],
        [{"type": "function", "function": {"name": "fetch_fx_rate", "parameters": {
            "type": "object", "properties": {"base": {"type": "string"}},
            "required": ["base"]}}}],
    )
    backend.available_tool_names = lambda: ["get_fx_rate"]  # type: ignore[method-assign]
    guard = ToolGuard(backend, sleep_scale=0.0)
    result = guard.call("fetch_fx_rate", {"base": "USD"})
    assert result.ok
    event = guard.events[0]
    assert event.rule == "function_name_resolution"
    assert event.schema_unavailable is True
    assert event.schema_validated is False
    assert guard.stats["schema_unchecked"] == 1


# ---------------------------------------------------------------------------
# HN contract: consistent by default, drifting only as an explicit demo mode
# ---------------------------------------------------------------------------
def test_live_hard_backend_hn_schema_modes() -> None:
    """The main protocol advertises the consistent integer HN contract (2026-11-25 fix).

    The drifting string schema is still reachable only when explicitly requested, so the
    targeted contract-repair demo remains reproducible without contaminating the main run.
    """
    import pytest

    from tool_evolution.toolguard.live_hard import LiveHardBackend

    def story_id_type(backend: LiveHardBackend) -> str:
        for schema in backend.schemas():
            function = schema["function"]
            if function["name"] == "get_hn_item":
                properties = function["parameters"]["properties"]
                return properties["story_id"]["type"]
        raise AssertionError("get_hn_item not advertised")

    assert story_id_type(LiveHardBackend()) == "integer"
    assert story_id_type(LiveHardBackend(hn_schema="consistent")) == "integer"
    assert story_id_type(LiveHardBackend(hn_schema="drifting")) == "string"
    with pytest.raises(ValueError):
        LiveHardBackend(hn_schema="bogus")
