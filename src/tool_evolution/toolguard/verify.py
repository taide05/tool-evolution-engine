"""Programmatic answer verification shared by the live end-to-end demos.

The demos never use an LLM judge: ground truth is fetched from the same live APIs before
the agent runs and answers are checked with these deterministic helpers.

Since the 2026-11-24 review, comparison tasks are verified with *relational* assertions
(``relation_consistent``): the winning candidate must be asserted as the winner (with
negation handling) and the numbers/dates the answer itself reports must order the
candidates the same way as the ground truth.  Merely mentioning the candidate name is not
enough — a reversed answer ("Germany has the larger population: France 68,720,337, Germany
83,491,249" when Germany is not the expected winner) must fail.
"""

from __future__ import annotations

import math
import re
import unicodedata

# Unicode dash/minus variants that LLM answers commonly emit instead of ASCII '-'
# (e.g. "Oslo −3.4 °C"), plus the fullwidth minus.
_MINUS_TRANSLATION = str.maketrans({
    "\u2212": "-",  # minus sign
    "\u2013": "-",  # en dash
    "\u2014": "-",  # em dash
    "\u2012": "-",  # figure dash
    "\u2010": "-",  # hyphen
    "\u2011": "-",  # non-breaking hyphen
    "\uff0d": "-",  # fullwidth hyphen-minus
})


def norm_text(text: str) -> str:
    """Lowercase, ASCII-folded, punctuation-free form used for substring matching."""
    decomposed = unicodedata.normalize("NFKD", str(text))
    # "replace" (not "ignore") so that non-ASCII punctuation becomes a separator instead of
    # gluing neighbouring words together ("recently—requests" must not become
    # "recentlyrequests", which would hide the relation keyword from the verifier).
    folded = decomposed.encode("ascii", "replace").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", folded.lower()).strip()


def digits(text: str) -> str:
    return re.sub(r"\D", "", str(text))


def numbers(text: str) -> list[float]:
    normalized = str(text).translate(_MINUS_TRANSLATION).replace(",", "")
    return [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", normalized)]


def contains_text(answer: str, expected: str) -> bool:
    return norm_text(expected)[:40] in norm_text(answer)


def contains_number(answer: str, expected: float, tolerance: float = 0.0) -> bool:
    return any(abs(value - expected) <= tolerance for value in numbers(answer))


def contains_digits(answer: str, expected: int | str) -> bool:
    return digits(expected) in digits(answer)


def mcnemar_exact(baseline_only: int, guard_only: int) -> float:
    """Two-sided exact binomial p-value for discordant paired outcomes."""
    n = baseline_only + guard_only
    if n == 0:
        return 1.0
    k = min(baseline_only, guard_only)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    """Wilson score 95% interval for a binomial proportion (used for success rates)."""
    if total <= 0:
        return [0.0, 1.0]
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return [round(max(0.0, centre - margin), 4), round(min(1.0, centre + margin), 4)]


# ---------------------------------------------------------------------------
# negation-aware containment (the generic helpers above are pure substring checks)
# ---------------------------------------------------------------------------
_NEGATION_RE = re.compile(
    r"\b(?:not|never|isn t|wasn t|aren t|weren t|doesn t|didn t|does not|did not|is not|"
    r"was not|rather than|instead of|neither|nor|cannot|can t|no longer|far from)\b"
)


def _context_is_negated(text: str, start: int, end: int, window: int = 45) -> bool:
    prefix = text[max(0, start - window):start]
    suffix = text[end:end + 20]
    return bool(_NEGATION_RE.search(prefix)) or bool(_NEGATION_RE.search(suffix))


def contains_text_asserted(answer: str, expected: str) -> bool:
    """Like :func:`contains_text` but a negated mention ("not Europe & Central Asia") fails."""
    norm = norm_text(answer)
    needle = norm_text(expected)[:40]
    if not needle:
        return False
    start = norm.find(needle)
    while start != -1:
        if not _context_is_negated(norm, start, start + len(needle)):
            return True
        start = norm.find(needle, start + 1)
    return False


# ---------------------------------------------------------------------------
# relational verification for comparison tasks
# ---------------------------------------------------------------------------
_RELATION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "larger": ("larger", "bigger", "greater", "more populous", "higher population"),
    "warmer": ("warmer", "hotter"),
    "newer": ("more recently", "more recent", "newer", "most recently", "most recent"),
    "earliest": ("earliest", "oldest"),
    "higher": ("higher", "newer", "greater"),
}
_SCALE_WORDS = {"hundred": 1e2, "thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}


def _occurrences(text: str, names: tuple[str, ...] | list[str]) -> list[tuple[int, int, str]]:
    occurrences: list[tuple[int, int, str]] = []
    for canonical in names:
        needle = norm_text(canonical)
        if not needle:
            continue
        for match in re.finditer(r"\b" + re.escape(needle) + r"\b", text):
            occurrences.append((match.start(), match.end(), canonical))
    occurrences.sort()
    return occurrences


def _asserted_winner(answer: str, winner_names: tuple[str, ...] | list[str],
                     loser_names: tuple[str, ...] | list[str], relation: str) -> bool | None:
    """True if the answer asserts the expected winner, False if it asserts another candidate.

    Returns ``None`` when no comparison direction is stated (which must not count as a
    success for a comparison task).
    """
    norm = norm_text(answer)
    winner_norm = {norm_text(name) for name in winner_names}
    loser_norm = {norm_text(name) for name in loser_names}
    names = list(winner_names) + list(loser_names)
    occurrences = _occurrences(norm, names)
    verdicts: set[str] = set()
    for keyword in _RELATION_KEYWORDS[relation]:
        for match in re.finditer(r"\b" + re.escape(keyword) + r"\b", norm):
            keyword_start, keyword_end = match.span()
            subject: tuple[int, int, str] | None = None
            for occ_start, occ_end, name in reversed(occurrences):
                if occ_end <= keyword_start and keyword_start - occ_end <= 140:
                    subject = (occ_start, occ_end, name)
                    break
            if subject is not None:
                between = norm[subject[1]:keyword_start]
                before = norm[max(0, subject[0] - 25):subject[0]]
                if _NEGATION_RE.search(between) or re.search(
                        r"\b(?:not|never|no)\s*$", before):
                    continue  # "Paris is not warmer": not a winner assertion
                canonical = norm_text(subject[2])
                if canonical in winner_norm:
                    verdicts.add("winner")
                elif canonical in loser_norm:
                    verdicts.add("loser")
                else:  # pragma: no cover - defensively treat unknown entity as a claimant
                    verdicts.add("other")
                continue
            # "The warmer capital is Paris": no candidate before the keyword, first one after
            for occ_start, occ_end, name in occurrences:
                if keyword_end < occ_start <= keyword_end + 80:
                    if _NEGATION_RE.search(norm[keyword_end:occ_start]):
                        break
                    canonical = norm_text(name)
                    if canonical in winner_norm:
                        verdicts.add("winner")
                    elif canonical in loser_norm:
                        verdicts.add("loser")
                    break
    if "winner" in verdicts and "loser" not in verdicts and "other" not in verdicts:
        return True
    if verdicts and "winner" not in verdicts:
        return False
    if "winner" in verdicts and ({"loser", "other"} & verdicts):
        return False  # both directions asserted: contradictory, do not credit it
    return None


_ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def _answer_numbers_with_units(text: str) -> list[float]:
    normalized = str(text).translate(_MINUS_TRANSLATION).replace(",", "")
    values: list[float] = []
    for match in re.finditer(
            r"(-?\d+(?:\.\d+)?)\s*(hundred|thousand|million|billion|trillion)?", normalized):
        scale = _SCALE_WORDS.get((match.group(2) or "").lower(), 1.0)
        try:
            values.append(float(match.group(1)) * scale)
        except ValueError:  # pragma: no cover - regex guarantees a float
            continue
    return values


def _match_score(number: float, expected: float,
                 absolute_tolerance: float | None) -> float | None:
    """Distance score for a number matching the expected value (unit words included)."""
    best: float | None = None
    if absolute_tolerance is not None and abs(number - expected) <= absolute_tolerance:
        best = abs(number - expected) / max(abs(expected), 1.0)
    for scale in (1e3, 1e6, 1e9):
        candidate = expected / scale
        if abs(candidate) < 1e-12:
            continue
        if abs(number - candidate) <= 0.01 * abs(candidate) + 1e-9:
            score = abs(number - candidate) / abs(candidate)
            best = score if best is None else min(best, score)
    return best


def _value_near(answer: str, name: str, expected: float, *,
                absolute_tolerance: float | None = None) -> float | None:
    """Find the number nearest ``name`` that matches ``expected`` (unit words handled).

    When a tolerance is wide (e.g. ±3 °C for live-drifted temperatures) several numbers can
    "match"; the closest one wins so that a neighbouring candidate's value is not reused.
    """
    lowered = answer.lower()
    needle = name.lower()
    start = lowered.find(needle)
    best_number: float | None = None
    best_score = float("inf")
    while start != -1:
        window = answer[start:start + 180]
        for number in _answer_numbers_with_units(window):
            score = _match_score(number, expected, absolute_tolerance)
            if score is not None and score < best_score:
                best_number, best_score = number, score
        start = lowered.find(needle, start + 1)
    return best_number


def _date_near(answer: str, name: str, other_names: tuple[str, ...] | list[str] = ()
               ) -> str | None:
    """Return the ISO date adjacent to ``name``, stopping at the next candidate mention.

    This matters for answers like "httpx was released earliest — requests 2.34.2:
    2026-05-14, Flask ...", where a plain 180-char window around "httpx" would grab the
    *other* package's date.
    """
    lowered = answer.lower()
    needle = name.lower()
    boundaries = [other.lower() for other in other_names if other and other.lower() != needle]
    start = lowered.find(needle)
    while start != -1:
        window_end = start + len(needle) + 140
        for boundary in boundaries:
            next_boundary = lowered.find(boundary, start + len(needle))
            if next_boundary != -1:
                window_end = min(window_end, next_boundary)
        match = _ISO_DATE_RE.search(answer[start:window_end])
        if match:
            return match.group(0)
        start = lowered.find(needle, start + 1)
    return None


def _numeric_consistent(answer: str, winner: str, loser: str, relation: str,
                        winner_value: float | str, loser_value: float | str,
                        absolute_tolerance: float | None,
                        other_names: tuple[str, ...] | list[str] = ()) -> bool | None:
    """Return False only when the answer's own numbers contradict the ground-truth order."""
    if isinstance(winner_value, str) and isinstance(loser_value, str):
        winner_seen = _date_near(answer, winner, list(other_names) + [loser])
        loser_seen = _date_near(answer, loser, list(other_names) + [winner])
        if winner_seen is None or loser_seen is None:
            return None
        if relation == "newer":
            return winner_seen > loser_seen
        if relation == "earliest":
            return winner_seen <= loser_seen
        return None
    if isinstance(winner_value, (int, float)) and isinstance(loser_value, (int, float)):
        winner_seen = _value_near(answer, winner, float(winner_value),
                                  absolute_tolerance=absolute_tolerance)
        loser_seen = _value_near(answer, loser, float(loser_value),
                                 absolute_tolerance=absolute_tolerance)
        if winner_seen is None or loser_seen is None:
            return None
        expected_gap = float(winner_value) - float(loser_value)
        seen_gap = winner_seen - loser_seen
        if expected_gap == 0 or seen_gap == 0:
            return None
        return (expected_gap > 0) == (seen_gap > 0)
    return None


def relation_consistent(answer: str, winner_names: tuple[str, ...] | list[str],
                        relation: str,
                        winner_value: float | str | None = None,
                        losers: dict[str, float | str] | None = None,
                        absolute_tolerance: float | None = None) -> bool:
    """Verify a comparison answer as a *relation*, not a name occurrence.

    * the expected winner must be asserted as the winner (negation-aware);
    * an assertion of any other candidate as the winner fails the answer;
    * when the expected values are known and the answer supplies them, the answer's own
      numbers/dates must order the candidates the same way as the ground truth
      (this is where "83.5 million"-style unit words are normalised).
    """
    losers = losers or {}
    verdict = _asserted_winner(answer, winner_names, list(losers), relation)
    if verdict is False:
        return False
    if winner_value is not None:
        other_names = [name for name in losers] + list(winner_names[1:])
        for loser_name, loser_value in losers.items():
            numeric = _numeric_consistent(answer, winner_names[0], loser_name, relation,
                                          winner_value, loser_value, absolute_tolerance,
                                          other_names=other_names)
            if numeric is False:
                return False
    return verdict is True


# ---------------------------------------------------------------------------
# task-family verifiers built on the relational checker
# ---------------------------------------------------------------------------
def verify_weather_compare_pair(answer: str, expected: dict) -> bool:
    a, b = expected["a"], expected["b"]
    if not (contains_number(answer, a["temperature_c"], 3.0)
            and contains_number(answer, b["temperature_c"], 3.0)):
        return False
    if not expected.get("require_warmer", True):
        return True
    winner = expected["warmer_capital"]
    if winner == a["capital"]:
        loser, winner_value, loser_value = b["capital"], a["temperature_c"], b["temperature_c"]
    else:
        loser, winner_value, loser_value = a["capital"], b["temperature_c"], a["temperature_c"]
    return relation_consistent(answer, [winner], "warmer", winner_value,
                               {loser: loser_value}, absolute_tolerance=3.0)


def verify_weather_compare_scalar(answer: str, expected: dict,
                                  loser_names: tuple[str, ...] | list[str] = ()) -> bool:
    """Weather comparison stored in the flat Australia/Canada shape.

    The committed V2/V3 artifacts store only the two country temperatures, the winning
    capital and ``require_warmer``.  New runs also store ``warmer_temp``/``colder_temp``
    and ``colder_capital`` so the numeric ordering can be cross-checked; for the committed
    artifacts the check degrades to the (already strict) relation assertion.
    """
    temps = [value for key, value in expected.items()
             if key in ("australia", "canada") and isinstance(value, (int, float))]
    if len(temps) != 2 or not all(contains_number(answer, value, 3.0) for value in temps):
        return False
    if "gap" in expected and not contains_number(answer, expected["gap"], 3.5):
        return False
    if not expected.get("require_warmer", True):
        return True
    winner = expected["warmer"]
    if "warmer_temp" in expected and "colder_temp" in expected:
        losers = {name: expected["colder_temp"] for name in loser_names}
        return relation_consistent(answer, [winner], "warmer", expected["warmer_temp"],
                                   losers, absolute_tolerance=3.0)
    return relation_consistent(answer, [winner], "warmer")


def verify_population_compare_map(answer: str, expected: dict) -> bool:
    values = {key.lower(): expected[key] for key in ("france", "germany") if key in expected}
    if len(values) != 2:
        return False
    if not all(contains_digits(answer, value) or contains_number(answer, value / 1e6, 0.05)
               for value in values.values()):
        return False
    winner = expected["larger"].lower()
    if winner not in values:
        return False
    loser = next(name for name in values if name != winner)
    for region_key in ("region_germany", "region_france"):
        if region_key in expected and not contains_text_asserted(answer, expected[region_key]):
            return False
    return relation_consistent(answer, [expected["larger"]], "larger", values[winner],
                               {loser.title(): values[loser]})


def verify_population_compare_list(answer: str, expected: dict) -> bool:
    countries = expected["countries"]
    populations = expected["populations"]
    if not all(contains_digits(answer, value) for value in populations):
        return False
    winner = expected["larger_country"]
    values = dict(zip(countries, populations))
    losers = {name: value for name, value in values.items() if name != winner}
    return relation_consistent(answer, [winner], "larger", values[winner], losers,
                               absolute_tolerance=None)


def verify_pypi_newer_pair(answer: str, expected: dict) -> bool:
    a, b = expected["a"], expected["b"]
    if not (contains_text(answer, a["release_date"]) and contains_text(answer, b["release_date"])):
        return False
    winner = expected["newer"]
    if winner == a["package"]:
        loser, winner_value, loser_value = b["package"], a["release_date"], b["release_date"]
    else:
        loser, winner_value, loser_value = a["package"], b["release_date"], a["release_date"]
    return relation_consistent(answer, [winner], "newer", winner_value, {loser: loser_value})


def verify_release_vs_hn(answer: str, expected: dict) -> bool:
    if not (contains_text(answer, expected["httpx_release"])
            and contains_text(answer, expected["hn_item"])):
        return False
    winner = expected["newer"]
    if winner == "httpx":
        return relation_consistent(answer, ["httpx"], "newer", expected["httpx_release"],
                                   {"hacker news": expected["hn_item"]})
    return relation_consistent(answer, ["hacker news", "HN item"], "newer", expected["hn_item"],
                               {"httpx": expected["httpx_release"]})


def verify_earliest_of_three(answer: str, expected: dict) -> bool:
    dates = expected["dates"]
    if not all(contains_text(answer, date) for date in dates.values()):
        return False
    winner = expected["earliest"]
    losers = {name: date for name, date in dates.items() if name != winner}
    return relation_consistent(answer, [winner], "earliest", dates[winner], losers)


def compare_versions(left: str, right: str) -> int:
    """Compare dotted version strings; returns -1/0/1 (non-numeric tails compare as text)."""
    def parts(version: str) -> list[int | str]:
        return [int(part) if part.isdigit() else part
                for part in re.split(r"[.\-+]", str(version))]

    left_parts, right_parts = parts(left), parts(right)
    for left_part, right_part in zip(left_parts, right_parts):
        if left_part == right_part:
            continue
        if isinstance(left_part, int) and isinstance(right_part, int):
            return 1 if left_part > right_part else -1
        return 1 if str(left_part) > str(right_part) else -1
    if len(left_parts) == len(right_parts):
        return 0
    return 1 if len(left_parts) > len(right_parts) else -1


def _version_near(answer: str, package: str) -> str | None:
    match = re.search(r"\b" + re.escape(package.lower()) + r"\b[^\n]{0,120}?"
                      r"(\d+\.\d+(?:\.\d+)?)", answer.lower())
    return match.group(1) if match else None


def verify_higher_version(answer: str, expected: dict) -> bool:
    """V1 pypi_newer_package: assert which package has the higher *version number*.

    The old V1 verifier only required both version strings to appear, so it did not verify
    the comparison at all.  New task runs store ``packages`` and ``higher_version``; the
    answer must assert that winner and the version numbers it reports must order correctly.
    """
    packages = expected["packages"]
    winner = expected["higher_version"]
    if not all(contains_text(answer, version) for version in packages.values()):
        return False
    verdict = _asserted_winner(answer, [winner], [name for name in packages if name != winner],
                               "higher")
    if verdict is False:
        return False
    loser = next(name for name in packages if name != winner)
    winner_version = _version_near(answer, winner)
    loser_version = _version_near(answer, loser)
    if winner_version and loser_version:
        if compare_versions(winner_version, loser_version) <= 0:
            return False
    return verdict is True
