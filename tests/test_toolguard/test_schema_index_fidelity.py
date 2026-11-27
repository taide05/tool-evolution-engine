"""Schema-index fidelity repair (2026-11-25).

The released ``api_list`` is the agent-visible schema but is demonstrably wrong for some
parameters (e.g. date parameters declared NUMBER while every recorded successful call
passes an ISO string).  These tests pin the evidence-based repair rules and their limits:
corrections need >=3 samples and >=90% agreement, integer/number widening is never applied,
and required lists are never changed.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tool_evolution.toolguard.records import FunctionSchema  # noqa: E402
from tool_evolution.toolguard.schemas import refine_schema_index  # noqa: E402


def _call(tool: str, arguments: dict, ok: bool = True):
    return SimpleNamespace(tool_name=tool, arguments=arguments, recorded_ok=ok)


def _schema(properties: dict, required=()) -> FunctionSchema:
    return FunctionSchema(name="f", properties=properties, required=tuple(required))


def test_number_declaration_corrected_to_string_from_successful_calls() -> None:
    schema = _schema({"start_date": {"type": "number"}}, required=("start_date",))
    calls = [_call("f", {"start_date": value})
             for value in ("2022-06-01", "2022-06-30", "2022-07-01", "2022-07-02")]
    refined, stats = refine_schema_index({"f": schema}, calls)
    prop = refined["f"].properties["start_date"]
    assert prop["type"] == "string"
    assert prop["type_corrected_from"] == "number"
    assert refined["f"].required == ("start_date",)
    assert stats["type_corrections"] == 1
    example = stats["type_correction_examples"][0]
    assert example["function"] == "f" and example["param"] == "start_date"
    assert example["declared"] == "number" and example["observed_type"] == "string"
    assert example["observations"] == 4


def test_string_declaration_corrected_to_integer_from_successful_calls() -> None:
    schema = _schema({"story_id": {"type": "string"}})
    calls = [_call("f", {"story_id": value}) for value in (1, 2, 3, 4)]
    refined, stats = refine_schema_index({"f": schema}, calls)
    assert refined["f"].properties["story_id"]["type"] == "integer"
    assert stats["type_corrections"] == 1


def test_correction_needs_enough_consistent_evidence() -> None:
    schema = _schema({"start_date": {"type": "number"}})
    # only two successful observations -> below min_samples
    two = [_call("f", {"start_date": "2022-06-01"}), _call("f", {"start_date": "2022-06-02"})]
    refined, stats = refine_schema_index({"f": schema}, two)
    assert refined["f"].properties["start_date"]["type"] == "number"
    assert stats["type_corrections"] == 0
    # mixed evidence (50/50) -> below agreement threshold
    mixed = ([_call("f", {"start_date": "2022-06-01"})] * 2
             + [_call("f", {"start_date": 123})] * 2)
    refined, stats = refine_schema_index({"f": schema}, mixed)
    assert refined["f"].properties["start_date"]["type"] == "number"
    assert stats["type_corrections"] == 0


def test_integer_number_widening_is_never_applied() -> None:
    schema = _schema({"count": {"type": "integer"}})
    calls = [_call("f", {"count": value}) for value in (1.5, 2.5, 3.5)]
    refined, stats = refine_schema_index({"f": schema}, calls)
    assert refined["f"].properties["count"]["type"] == "integer"
    assert stats["type_corrections"] == 0


def test_empty_parameter_list_is_back_filled_as_optional() -> None:
    schema = _schema({}, required=())
    calls = [_call("f", {"city": "Paris", "days": 3}) for _ in range(3)]
    refined, stats = refine_schema_index({"f": schema}, calls)
    props = refined["f"].properties
    assert props["city"]["type"] == "string"
    assert props["days"]["type"] == "integer"
    assert refined["f"].required == ()
    assert stats["params_added"] == 2


def test_single_successful_sample_does_not_back_fill_parameters() -> None:
    """One recorded success is not enough evidence for a schema change (min_samples=3)."""
    schema = _schema({}, required=())
    refined, stats = refine_schema_index({"f": schema}, [_call("f", {"user_id": "u1"})])
    assert refined["f"].properties == {}
    assert stats["params_added"] == 0
    assert stats["param_additions"] == []


def test_three_consistent_successful_samples_back_fill_parameters() -> None:
    schema = _schema({}, required=())
    calls = [_call("f", {"user_id": value}) for value in ("u1", "u2", "u3")]
    refined, stats = refine_schema_index({"f": schema}, calls)
    assert refined["f"].properties["user_id"]["type"] == "string"
    assert stats["params_added"] == 1
    addition = stats["param_additions"][0]
    assert addition == {"function": "f", "param": "user_id", "observed_type": "string",
                        "observations": 3, "agreement": 1.0}
    # the full list is recorded, not just the first ten examples
    assert stats["param_additions"] == stats["param_addition_examples"]


def test_below_min_samples_does_not_back_fill() -> None:
    """Fewer than ``min_samples`` recorded successful values never change the schema.

    This test validates the *sample-count threshold*, not an "error-envelope filter": the
    real one-call example is ``getsponsorships_for_diablo4_smartable`` called with
    ``{"error": "Function not available", "response": ""}``; with only one successful sample
    (any argument names, including these pseudo-parameters) nothing is back-filled.  There is
    no explicit error-envelope filtering in ``refine_schema_index``; three consistent
    observations of the same argument shape would still be back-filled.
    """
    function = "getsponsorships_for_diablo4_smartable"
    schema = _schema({}, required=())
    calls = [
        _call(function, {"error": "Function not available", "response": ""}),
        _call(function, {"error": "Function not available"}),
    ]
    refined, stats = refine_schema_index({function: schema}, calls)
    assert refined[function].properties == {}
    assert stats["params_added"] == 0


def test_failed_calls_are_not_evidence() -> None:
    schema = _schema({"start_date": {"type": "number"}})
    calls = [_call("f", {"start_date": "2022-06-01"}, ok=False) for _ in range(5)]
    refined, stats = refine_schema_index({"f": schema}, calls)
    assert refined["f"].properties["start_date"]["type"] == "number"
    assert stats["type_corrections"] == 0
    assert stats["params_added"] == 0
