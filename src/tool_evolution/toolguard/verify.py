"""Programmatic answer verification shared by the live end-to-end demos.

The demos never use an LLM judge: ground truth is fetched from the same live APIs before
the agent runs and answers are checked with these deterministic helpers.
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
    folded = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")
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
