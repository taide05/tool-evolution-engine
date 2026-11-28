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

# ---------------------------------------------------------------------------
# A1.7 experimental deterministic extractors (opt-in; the default extractor set,
# the default learned map and the committed metrics/artifacts are unchanged).
#
# These cover parameter names the core set has no extractor for: ``source``,
# ``state``, ``topic`` and ``ID``.  They are deterministic (regex / curated
# vocabulary, no LLM) and deliberately conservative: when the task text carries
# no explicit value they return no candidate instead of guessing.
# ---------------------------------------------------------------------------

_US_STATE_CODES = {
    "alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR",
    "california": "CA", "colorado": "CO", "connecticut": "CT", "delaware": "DE",
    "florida": "FL", "georgia": "GA", "hawaii": "HI", "idaho": "ID",
    "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
    "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD",
    "massachusetts": "MA", "michigan": "MI", "minnesota": "MN", "mississippi": "MS",
    "missouri": "MO", "montana": "MT", "nebraska": "NE", "nevada": "NV",
    "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM", "new york": "NY",
    "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
    "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
    "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT",
    "vermont": "VT", "virginia": "VA", "washington": "WA", "west virginia": "WV",
    "wisconsin": "WI", "wyoming": "WY",
}

# "fetch the top articles from CoinDesk, BSC News, and Decrypt?" -> a publisher list
# introduced by "from".  Only a capitalised list item is accepted, so prose such as
# "extract the content from a specific URL" yields no candidate.
_SOURCE_FROM_RE = re.compile(r"\bfrom\s+([^.;!?]+)")
_SOURCE_LIST_SPLIT_RE = re.compile(r",\s*(?:and\s+)?|\s+and\s+")

# "cryptocurrency blog", "sports news" -> the lowercase noun that modifies a
# content-type noun.  The task text must contain the phrase; nothing is inferred.
_TOPIC_NOUN_RE = re.compile(
    r"\b([a-z][a-z0-9'\-]{2,30})\s+"
    r"(?:news|blog|articles?|content|information|updates?|stories|topics?|headlines?)\b",
    re.I,
)
_TOPIC_STOPWORDS = {
    "the", "a", "an", "and", "or", "for", "with", "from", "into", "about", "over",
    "latest", "top", "new", "recent", "breaking", "daily", "weekly", "some", "more",
    "other", "another", "interesting", "specific", "all", "any", "this", "that",
    "these", "those", "need", "needs", "want", "wants", "fetch", "get", "find",
    "provide", "show", "give", "read", "search", "related", "various", "different",
    "news", "article", "articles",
}

# IDs that appear verbatim in task text: IMDb ids, UUIDs and explicitly labelled
# identifiers ("ID: abc123", "id is xyz", "track_id=42").  A bare word that merely
# says "a specific music ID" carries no value and yields nothing.
_ID_RES = (
    re.compile(r"\btt\d{7,}\b"),
    re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-"
               r"[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"),
    re.compile(r"\b(?:id|identifier|track_id|music_id|image_id|item_id|story_id)\s*"
               r"(?:is|:|=)\s*['\"]?([A-Za-z0-9][A-Za-z0-9_\-]{1,63})", re.I),
)


def extract_states(text: str) -> list[str]:
    """US state names present in the text, returned as (2-letter code, full name)."""
    lowered = text.lower()
    values: list[str] = []
    for name, code in _US_STATE_CODES.items():
        if re.search(rf"\b{re.escape(name)}\b", lowered):
            values.extend([code, name.title()])
    return _unique(values)


def extract_sources(text: str) -> list[str]:
    """Publisher/source names in a comma/"and" list introduced by "from"."""
    values: list[str] = []
    for chunk in _SOURCE_FROM_RE.findall(text):
        for part in _SOURCE_LIST_SPLIT_RE.split(chunk):
            words: list[str] = []
            for token in part.split():
                cleaned = token.strip("'\"()[]:")
                if not cleaned or not cleaned[0].isalpha():
                    break
                if not cleaned[0].isupper():
                    break
                words.append(cleaned)
            if words:
                values.append(" ".join(words))
    return _unique(values)


def extract_topics(text: str) -> list[str]:
    """Lowercase topic keywords that modify a content noun ("cryptocurrency blog")."""
    values = [word.lower() for word in _TOPIC_NOUN_RE.findall(text)
              if word.lower() not in _TOPIC_STOPWORDS]
    return _unique(values)


def extract_identifiers(text: str) -> list[str]:
    """Identifiers written verbatim in the text (IMDb id / UUID / labelled id)."""
    values: list[str] = []
    for pattern in _ID_RES:
        for match in pattern.findall(text):
            if isinstance(match, tuple):
                match = next((item for item in match if item), "")
            if match:
                values.append(match)
    return _unique(values)


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

# A1.7 additions.  Kept separate from ``EXTRACTORS`` so that the default learned map
# (and therefore the committed metrics) is bit-for-bit unchanged unless explicitly
# enabled with ``ExtractorMap.learn(..., include_experimental=True)``.
EXPERIMENTAL_EXTRACTORS: dict[str, Callable[[str], list[str]]] = {
    "sources": extract_sources,
    "states": extract_states,
    "topics": extract_topics,
    "identifiers": extract_identifiers,
}

ALL_EXTRACTORS: dict[str, Callable[[str], list[str]]] = {**EXTRACTORS, **EXPERIMENTAL_EXTRACTORS}

# deterministic parameter-name fallbacks for the experimental extractors; only active
# on an ExtractorMap that was built with ``include_experimental=True``.
_EXPERIMENTAL_PARAM_FALLBACKS: dict[str, str] = {
    "source": "sources", "sources": "sources", "news_source": "sources",
    "state": "states", "states": "states", "province": "states",
    "topic": "topics", "topics": "topics", "subject": "topics",
    "id": "identifiers", "ids": "identifiers", "identifier": "identifiers",
    "music_id": "identifiers", "track_id": "identifiers", "image_id": "identifiers",
    "item_id": "identifiers", "story_id": "identifiers", "video_id": "identifiers",
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
    """Learned parameter-name -> extractor mapping plus deterministic name fallbacks.

    ``include_experimental`` opts into the A1.7 extractors (source/state/topic/ID).
    The default is ``False`` so the default learned map — and therefore the committed
    metrics/artifacts — is unchanged; the A1.7 experiment script enables it explicitly.
    """

    by_param: dict[str, str] = field(default_factory=dict)
    support: dict[str, dict[str, float]] = field(default_factory=dict)
    include_experimental: bool = False

    def __post_init__(self) -> None:
        self._pool = ALL_EXTRACTORS if self.include_experimental else EXTRACTORS

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
        if self.include_experimental:
            return _EXPERIMENTAL_PARAM_FALLBACKS.get(name)
        return None

    def candidates(self, param_name: str, text: str) -> list[str]:
        extractor_name = self.extractor_for(param_name)
        if not extractor_name or not text:
            return []
        extractor = self._pool.get(extractor_name)
        return extractor(text) if extractor else []

    def size(self) -> dict[str, int]:
        return {"learned_params": len(self.by_param), "extractor_kinds": len(set(self.by_param.values()))}

    @classmethod
    def learn(cls, calls: list[ToolCall], schemas: dict[str, FunctionSchema],
              min_support: float = 0.6, min_samples: int = 4,
              include_experimental: bool = False) -> "ExtractorMap":
        """Learn the best extractor per parameter name from successful training calls."""
        pool = ALL_EXTRACTORS if include_experimental else EXTRACTORS
        samples: dict[str, list[tuple[str, str]]] = {}
        for call in calls:
            if not call.recorded_ok or not call.valid_data or not call.query:
                continue
            for name, value in call.arguments.items():
                if isinstance(value, (dict, list)):
                    continue
                samples.setdefault(name, []).append((str(value), call.query))
        mapping = cls(include_experimental=include_experimental)
        scores: dict[str, dict[str, float]] = {}
        for param, rows in samples.items():
            if len(rows) < min_samples:
                continue
            support: dict[str, float] = {}
            for extractor_name, extractor in pool.items():
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
