"""Relational verifier tests (P0-3): a reversed comparison answer must fail.

The 2026-11-24 review found that the old verifiers only checked that candidate names
appeared, so answers that asserted the *opposite* relation were scored as successes.  These
tests pin the four review cases (weather / population / PyPI newer / earliest) plus
negation, unit words and version comparison.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tool_evolution.toolguard.verify import (  # noqa: E402
    compare_versions,
    contains_text_asserted,
    relation_consistent,
    verify_earliest_of_three,
    verify_higher_version,
    verify_population_compare_list,
    verify_population_compare_map,
    verify_pypi_newer_pair,
    verify_release_vs_hn,
    verify_weather_compare_pair,
    verify_weather_compare_scalar,
)

WEATHER_EXPECTED = {
    "a": {"country": "France", "capital": "Paris", "temperature_c": 3.6},
    "b": {"country": "Germany", "capital": "Berlin", "temperature_c": 0.9},
    "warmer_capital": "Paris",
    "require_warmer": True,
}

POPULATION_EXPECTED = {
    "france": 68_720_337,
    "germany": 83_491_249,
    "larger": "Germany",
    "region_france": "Europe & Central Asia",
    "region_germany": "Europe & Central Asia",
}

NEWER_EXPECTED = {
    "a": {"package": "requests", "version": "2.34.2", "release_date": "2026-05-14"},
    "b": {"package": "flask", "version": "3.1.3", "release_date": "2026-02-19"},
    "newer": "requests",
}

EARLIEST_EXPECTED = {
    "dates": {"requests": "2026-05-14", "flask": "2026-02-19", "httpx": "2024-12-06"},
    "earliest": "httpx",
}


# ---------------------------------------------------------------------------
# review P0-3: the four contradictory answers that used to pass
# ---------------------------------------------------------------------------
def test_weather_compare_reverse_answer_fails() -> None:
    correct = "Paris is warmer at 3.6 °C, compared with Berlin at 0.9 °C."
    reverse = "Berlin is warmer at 0.9 °C; Paris is 3.6 °C."
    assert verify_weather_compare_pair(correct, WEATHER_EXPECTED) is True
    assert verify_weather_compare_pair(reverse, WEATHER_EXPECTED) is False


def test_population_compare_reverse_answer_fails() -> None:
    correct = ("Germany has the larger population (83,491,249 vs. France's 68,720,337), "
               "and both belong to the World Bank region Europe & Central Asia.")
    reverse = ("France has the larger population: France 68,720,337, Germany 83,491,249, "
               "and both belong to the World Bank region Europe & Central Asia.")
    assert verify_population_compare_map(correct, POPULATION_EXPECTED) is True
    assert verify_population_compare_map(reverse, POPULATION_EXPECTED) is False


def test_pypi_newer_reverse_answer_fails() -> None:
    correct = ("requests was updated more recently — requests 2.34.2 released 2026-05-14 "
               "vs Flask 3.1.3 released 2026-02-19.")
    reverse = ("Flask was updated more recently: requests 2.34.2 was released 2026-05-14, "
               "Flask 3.1.3 was released 2026-02-19.")
    assert verify_pypi_newer_pair(correct, NEWER_EXPECTED) is True
    assert verify_pypi_newer_pair(reverse, NEWER_EXPECTED) is False


def test_earliest_reverse_answer_fails() -> None:
    correct = ("httpx was released earliest on 2024-12-06 (requests: 2026-05-14, "
               "Flask: 2026-02-19).")
    reverse = ("requests was released earliest on 2026-05-14; httpx was released 2024-12-06 "
               "and Flask 2026-02-19.")
    assert verify_earliest_of_three(correct, EARLIEST_EXPECTED) is True
    assert verify_earliest_of_three(reverse, EARLIEST_EXPECTED) is False


# ---------------------------------------------------------------------------
# the answer must *assert* the relation, not merely mention the candidate
# ---------------------------------------------------------------------------
def test_mentioning_the_winner_without_asserting_the_relation_fails() -> None:
    expected = {"a": {"capital": "Paris", "temperature_c": 3.6},
                "b": {"capital": "Berlin", "temperature_c": 0.9},
                "warmer_capital": "Paris", "require_warmer": True}
    numbers_only = "Paris: 3.6 °C. Berlin: 0.9 °C."
    assert verify_weather_compare_pair(numbers_only, expected) is False


def test_negated_winner_assertion_fails() -> None:
    expected = {"a": {"capital": "Paris", "temperature_c": 3.6},
                "b": {"capital": "Berlin", "temperature_c": 0.9},
                "warmer_capital": "Paris", "require_warmer": True}
    answer = "Paris is not warmer than Berlin: 3.6 °C versus 0.9 °C."
    assert verify_weather_compare_pair(answer, expected) is False


def test_negated_loser_mention_does_not_create_contradiction() -> None:
    expected = {"a": {"capital": "Paris", "temperature_c": 3.6},
                "b": {"capital": "Berlin", "temperature_c": 0.9},
                "warmer_capital": "Paris", "require_warmer": True}
    answer = "Paris is warmer at 3.6 °C, not Berlin (0.9 °C)."
    assert verify_weather_compare_pair(answer, expected) is True


def test_contains_text_asserted_rejects_negation() -> None:
    assert contains_text_asserted("both are in Europe & Central Asia", "Europe & Central Asia")
    assert not contains_text_asserted("neither is in Europe & Central Asia",
                                      "Europe & Central Asia")


# ---------------------------------------------------------------------------
# unit words (million) and multiple candidates
# ---------------------------------------------------------------------------
def test_population_millions_are_normalised_for_the_relation_check() -> None:
    expected = {"france": 68_720_337, "germany": 83_491_249, "larger": "Germany",
                "region_germany": "Europe & Central Asia"}
    millions = ("Germany has the larger population: 83.5 million versus France's 68.7 million. "
                "Both are in Europe & Central Asia.")
    assert verify_population_compare_map(millions, expected) is True
    reverse_millions = ("France has the larger population: 68.7 million versus Germany's "
                        "83.5 million. Both are in Europe & Central Asia.")
    assert verify_population_compare_map(reverse_millions, expected) is False


def test_oslo_wellington_population_reverse_fails() -> None:
    expected = {"countries": ["Norway", "New Zealand"],
                "populations": [5_610_870, 5_324_700], "larger_country": "Norway"}
    correct = ("Norway (capital Oslo) has the larger population at 5,610,870 versus "
               "New Zealand (capital Wellington) at 5,324,700.")
    reverse = ("New Zealand has the larger population: New Zealand (capital Wellington) "
               "5,324,700 versus Norway 5,610,870.")
    assert verify_population_compare_list(correct, expected) is True
    assert verify_population_compare_list(reverse, expected) is False


def test_release_vs_hn_reverse_fails() -> None:
    expected = {"httpx_release": "2024-12-06", "hn_item": "2006-10-09", "newer": "httpx"}
    correct = ("The httpx release (2024-12-06) happened more recently than HN item 1 "
               "(2006-10-09).")
    reverse = ("Hacker News item 1 (2006-10-09) was posted more recently than the httpx "
               "release (2024-12-06).")
    assert verify_release_vs_hn(correct, expected) is True
    assert verify_release_vs_hn(reverse, expected) is False


def test_weather_scalar_comparison_reverse_fails() -> None:
    expected = {"australia": 22.5, "canada": -0.5, "gap": 23.0, "warmer": "Canberra",
                "colder": "Ottawa", "warmer_temp": 22.5, "colder_temp": -0.5,
                "require_warmer": True}
    correct = ("Canberra (Australia's capital) is warmer at 22.5 °C versus Ottawa "
               "(Canada's capital) at -0.5 °C, a difference of 23.0 °C.")
    reverse = ("Ottawa is warmer at -0.5 °C versus Canberra at 22.5 °C, a difference of "
               "23.0 °C.")
    assert verify_weather_compare_scalar(correct, expected, loser_names=["Ottawa"]) is True
    assert verify_weather_compare_scalar(reverse, expected, loser_names=["Ottawa"]) is False


# ---------------------------------------------------------------------------
# version comparison (V1 pypi_newer_package used to only check both strings appear)
# ---------------------------------------------------------------------------
def test_compare_versions() -> None:
    assert compare_versions("2.31.0", "2.5.0") == 1
    assert compare_versions("2.5.10", "2.5.9") == 1
    assert compare_versions("3.1.3", "3.1.3") == 0


def test_higher_version_reverse_fails() -> None:
    expected = {"packages": {"numpy": "2.5.0", "requests": "2.31.0"},
                "higher_version": "requests"}
    correct = ("requests 2.31.0 has the higher version number than numpy 2.5.0.")
    reverse = ("numpy 2.5.0 has the higher version number than requests 2.31.0.")
    assert verify_higher_version(correct, expected) is True
    assert verify_higher_version(reverse, expected) is False


def test_relation_consistent_is_strict_about_assertion() -> None:
    answer = "Paris: 3.6 °C. Berlin: 0.9 °C."
    assert relation_consistent(answer, ["Paris"], "warmer", 3.6,
                               {"Berlin": 0.9}, absolute_tolerance=3.0) is False


# ---------------------------------------------------------------------------
# real committed ground truth: hand-built reversed answers must fail
# ---------------------------------------------------------------------------
def _committed_expected() -> dict[str, dict]:
    import json

    data = json.loads((ROOT / "benchmarks/toolguard/live_demo_hard_v3.json")
                      .read_text(encoding="utf-8"))
    return {task["id"]: task["expected"] for task in data["tasks"]}


def test_review_reverse_answers_fail_on_real_committed_ground_truth() -> None:
    expected = _committed_expected()
    cases = [
        ("capital_weather_compare_france_germany", verify_weather_compare_pair,
         "Berlin is warmer at 0.9 °C; Paris is 3.6 °C."),
        ("population_compare_france_germany_region", verify_population_compare_map,
         "France has the larger population: France 68,720,337, Germany 83,491,249, "
         "and both are in the World Bank region Europe & Central Asia."),
        ("pypi_newer_requests_flask", verify_pypi_newer_pair,
         "Flask was updated more recently: requests 2.34.2 released 2026-05-14, "
         "Flask 3.1.3 released 2026-02-19."),
        ("package_earliest_of_three", verify_earliest_of_three,
         "requests was released earliest (2026-05-14); httpx 2024-12-06 and "
         "Flask 2026-02-19 are later."),
        ("population_compare_oslo_wellington", verify_population_compare_list,
         "New Zealand has the larger population: New Zealand 5,324,700 versus "
         "Norway 5,610,870."),
    ]
    for task_id, verifier, answer in cases:
        assert task_id in expected, f"missing committed expected for {task_id}"
        assert verifier(answer, expected[task_id]) is False, task_id


def test_correct_committed_answers_still_pass() -> None:
    """At least one recorded real answer per family must still be accepted (no over-reject)."""
    import json

    data = json.loads((ROOT / "benchmarks/toolguard/live_demo_hard_v3.json")
                      .read_text(encoding="utf-8"))
    expected = {task["id"]: task["expected"] for task in data["tasks"]}
    answers: dict[str, str] = {}
    for record in data["records"]:
        if record["success"] and record["task"] in (
                "capital_weather_compare_france_germany", "population_compare_oslo_wellington",
                "pypi_newer_requests_flask", "release_vs_hn_item_date",
                "package_earliest_of_three", "weather_compare_australia_canada"):
            answers.setdefault(record["task"], record["answer"])
    verifiers = {
        "capital_weather_compare_france_germany": verify_weather_compare_pair,
        "population_compare_oslo_wellington": verify_population_compare_list,
        "pypi_newer_requests_flask": verify_pypi_newer_pair,
        "release_vs_hn_item_date": verify_release_vs_hn,
        "package_earliest_of_three": verify_earliest_of_three,
        "weather_compare_australia_canada": verify_weather_compare_scalar,
    }
    for task_id, verifier in verifiers.items():
        assert task_id in answers, f"no recorded success for {task_id}"
        assert verifier(answers[task_id], expected[task_id]) is True, task_id
