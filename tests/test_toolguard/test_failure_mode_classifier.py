"""Failure-mode classifier: metadata scalars in ``expected`` must not fabricate format misses.

The 2026-11-24 review found ``classify_failure()`` labelled a "no answer at all" as
``format_miss`` whenever the answer echoed *any* scalar inside ``expected`` — including task
metadata such as the Hacker News ``story_id`` or weather ``latitude``/``longitude``.  The
multi-model V3 artifact contains 17 such mislabels (13 HN, 2 weather, 2 GDP).  The classifier
now only considers the answer-bearing fields.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_live_demo_hard_v2 import classify_failure  # noqa: E402

HN_EXPECTED = {"story_id": 8863, "username": "someuser", "created_year": 2011}
WEATHER_EXPECTED = {"country": "Japan", "capital": "Tokyo", "latitude": 35.6895,
                    "longitude": 139.69171, "temperature_c": 7.0}
POPULATION_EXPECTED = {"countries": ["Norway", "New Zealand"],
                       "populations": [5_610_870, 5_324_700], "larger_country": "Norway"}


def test_story_id_echo_is_not_a_format_miss() -> None:
    answer = ('I couldn\'t retrieve this: the `get_hn_item` call for item 8863 repeatedly '
              'returned "story_id must be an integer", so the creation year cannot be determined.')
    assert classify_failure(answer, HN_EXPECTED) == "no_expected_value"


def test_coordinate_echo_is_not_a_format_miss() -> None:
    answer = "Tokyo's coordinates are 35.6895 latitude, 139.69171 longitude."
    assert classify_failure(answer, WEATHER_EXPECTED) == "no_expected_value"


def test_metadata_only_country_echo_is_not_a_format_miss() -> None:
    answer = "Switzerland's GDP per capita is $106,588"
    expected = {"capital": "Bern", "country": "Switzerland", "gdp_per_capita_usd": 103_342}
    assert classify_failure(answer, expected) == "no_expected_value"


def test_actual_answer_value_present_is_a_format_miss() -> None:
    # population rendered in millions instead of the exact integer: expected value is there
    assert classify_failure("Norway has about 5.61 million people.",
                            POPULATION_EXPECTED) == "format_miss"
    # year present but the answer is wrong in other ways
    assert classify_failure("The account was created in 2011, I think.",
                            HN_EXPECTED) == "format_miss"


def test_empty_answer_is_no_answer() -> None:
    assert classify_failure("", HN_EXPECTED) == "no_answer"
    assert classify_failure("   ", WEATHER_EXPECTED) == "no_answer"
