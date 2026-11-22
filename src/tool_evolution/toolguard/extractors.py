"""Task-text parameter extractors: the cross-task generalisation strategy.

Instead of replaying an earlier invocation of the *same task instance*, ToolGuard
learns which extractor recovers each parameter name from the task text.  Training
data only: for every successful training call we check whether the actual argument
value appears in the task text under one of a small set of deterministic extractors
(ISO dates, years, quoted keywords, numbers, countries).  The best-supported
extractor per parameter name is then applied to unseen tasks.

This turns "remember one invocation" into "learn a parameter-name -> extraction
pattern" rule, which transfers to task sets the middleware has never seen.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable

from .records import FunctionSchema, ToolCall

_MONTHS = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9,
    "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}
_MONTH_RE = "|".join(sorted(_MONTHS, key=len, reverse=True))
_DATE_MDY_RE = re.compile(rf"\b({_MONTH_RE})\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(\d{{4}})\b", re.I)
_DATE_ISO_RE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_MONTH_YEAR_RE = re.compile(rf"\b({_MONTH_RE})\.?\s+(\d{{4}})\b", re.I)
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
_QUOTED_RE = re.compile(r"['\"\u2018\u2019\u201c\u201d]([^'\"\u2018\u2019\u201c\u201d]{2,80})['\"\u2018\u2019\u201c\u201d]")
_NUMBER_RE = re.compile(r"\b(\d{1,4})\b")

_COUNTRY_CODES = {
    "united states": "US", "usa": "US", "us": "US", "america": "US", "australia": "AU",
    "canada": "CA", "united kingdom": "GB", "uk": "GB", "germany": "DE", "france": "FR",
    "japan": "JP", "china": "CN", "india": "IN", "brazil": "BR", "spain": "ES",
    "italy": "IT", "netherlands": "NL", "sweden": "SE", "norway": "NO", "denmark": "DK",
    "mexico": "MX", "singapore": "SG", "south korea": "KR", "ireland": "IE", "poland": "PL",
}


def extract_dates(text: str) -> list[str]:
    values: list[str] = []
    for month, day, year in _DATE_MDY_RE.findall(text):
        values.append(f"{int(year):04d}-{_MONTHS[month.lower()]:02d}-{int(day):02d}")
    for year, month, day in _DATE_ISO_RE.findall(text):
        if 1 <= int(month) <= 12 and 1 <= int(day) <= 31:
            values.append(f"{year}-{month}-{day}")
    return _unique(values)


def extract_month_years(text: str) -> list[str]:
    return _unique([f"{int(year):04d}-{_MONTHS[month.lower()]:02d}" for month, year in _MONTH_YEAR_RE.findall(text)])


def extract_years(text: str) -> list[str]:
    return _unique(_YEAR_RE.findall(text))


def extract_quoted(text: str) -> list[str]:
    return _unique(_QUOTED_RE.findall(text))


def extract_numbers(text: str) -> list[str]:
    return _unique(_NUMBER_RE.findall(text))


def extract_countries(text: str) -> list[str]:
    lowered = text.lower()
    values: list[str] = []
    for name, code in _COUNTRY_CODES.items():
        if re.search(rf"\b{re.escape(name)}\b", lowered):
            values.extend([code, name.upper() if len(name) == 2 else name.title()])
    return _unique(values)


EXTRACTORS: dict[str, Callable[[str], list[str]]] = {
    "dates": extract_dates,
    "month_years": extract_month_years,
    "years": extract_years,
    "quoted": extract_quoted,
    "numbers": extract_numbers,
    "countries": extract_countries,
}


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        norm = str(value).strip()
        if norm and norm.lower() not in seen:
            seen.add(norm.lower())
            out.append(norm)
    return out


@dataclass
class ExtractorMap:
    """Learned parameter-name -> extractor mapping plus deterministic name fallbacks."""

    by_param: dict[str, str] = field(default_factory=dict)
    support: dict[str, dict[str, float]] = field(default_factory=dict)

    def extractor_for(self, param_name: str) -> str | None:
        name = param_name.lower()
        if name in self.by_param:
            return self.by_param[name]
        if "date" in name or name in {"from", "to", "since", "until", "start", "end"}:
            return "dates"
        if "year" in name and not name.endswith("years"):
            return "years"
        if name in {"keyword", "keywords", "q", "query", "search", "term", "title"}:
            return "quoted"
        if "country" in name:
            return "countries"
        if name in {"limit", "count", "page", "per_page", "max", "max_results"}:
            return "numbers"
        return None

    def candidates(self, param_name: str, text: str) -> list[str]:
        extractor_name = self.extractor_for(param_name)
        if not extractor_name or not text:
            return []
        extractor = EXTRACTORS.get(extractor_name)
        return extractor(text) if extractor else []

    def size(self) -> dict[str, int]:
        return {"learned_params": len(self.by_param), "extractor_kinds": len(set(self.by_param.values()))}

    @classmethod
    def learn(cls, calls: list[ToolCall], schemas: dict[str, FunctionSchema],
              min_support: float = 0.6, min_samples: int = 4) -> "ExtractorMap":
        """Learn the best extractor per parameter name from successful training calls."""
        samples: dict[str, list[tuple[str, str]]] = {}
        for call in calls:
            if not call.recorded_ok or not call.valid_data or not call.query:
                continue
            for name, value in call.arguments.items():
                if isinstance(value, (dict, list)):
                    continue
                samples.setdefault(name, []).append((str(value), call.query))
        mapping = cls()
        scores: dict[str, dict[str, float]] = {}
        for param, rows in samples.items():
            if len(rows) < min_samples:
                continue
            support: dict[str, float] = {}
            for extractor_name, extractor in EXTRACTORS.items():
                hits = 0
                for value, text in rows:
                    candidates = extractor(text)
                    if any(_norm(value) == _norm(candidate) for candidate in candidates):
                        hits += 1
                support[extractor_name] = hits / len(rows)
            scores[param] = support
            best = max(support.items(), key=lambda item: item[1])
            if best[1] >= min_support:
                mapping.by_param[param.lower()] = best[0]
        mapping.support = scores
        return mapping


def _norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())
