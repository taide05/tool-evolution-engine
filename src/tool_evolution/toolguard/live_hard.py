"""Extended live tool surface for the harder multi-step demo.

The six-tool surface used by the first live demo (``live.py``) is intentionally frozen so
that the committed artifact stays reproducible.  This module adds the extra key-free
tools that make *dependent* multi-step chains possible:

* ``find_country_by_capital`` -- reverse World Bank lookup (capital -> country)
* ``get_country_population`` / ``get_country_gdp_per_capita`` -- World Bank indicators
* ``geocode_city`` -- Open-Meteo geocoding (city name -> coordinates)
* ``get_package_release_date`` -- PyPI release metadata
* ``get_hn_user`` / ``get_hn_top_story`` -- Hacker News users and current top story

Every handler talks to a real public API.  Two failure classes measured in the offline
trace data are modelled on purpose: the HN schema/backend type mismatch
(``get_hn_item`` can advertise ``story_id`` as a string while the backend requires an
integer, selected explicitly with ``LiveHardBackend(hn_schema="drifting")``), and
transient HTTP failures (injected by the runner, not here).  The default is the
*consistent* contract (``story_id`` integer), which is the main protocol after the
2026-11-25 review decision; the drifting variant exists only for the targeted
contract-repair demonstration.
"""

from __future__ import annotations

import urllib.parse
from typing import Any

from .live import (
    ToolError,
    ToolSpec,
    fold_text,
    get_country_info,
    get_exchange_rate,
    get_hn_item,
    get_weather,
    http_get_json,
    resolve_country_row,
    world_bank_country_rows,
)

__all__ = ["HARD_TOOLS", "LiveHardBackend", "get_package_release_date", "geocode_city"]

# ---------------------------------------------------------------------------
# extra handlers
# ---------------------------------------------------------------------------
_PYPI_CACHE: dict[str, dict[str, Any]] = {}
_WB_INDICATOR_CACHE: dict[tuple[str, str], list[dict[str, Any]]] = {}


def _pypi_payload(package: str) -> dict[str, Any]:
    if not isinstance(package, str) or not package.strip():
        raise ToolError("invalid_argument", "package must be a non-empty string")
    name = package.strip().lower()
    if " " in name:
        raise ToolError("invalid_argument", f"invalid package name: {package!r}")
    cached = _PYPI_CACHE.get(name)
    if cached is None:
        payload = http_get_json(f"https://pypi.org/pypi/{urllib.parse.quote(name)}/json")
        if not isinstance(payload, dict) or not (payload.get("info") or {}).get("version"):
            raise ToolError("not_found", f"package not found: {package}")
        _PYPI_CACHE[name] = payload
        cached = payload
    return cached


def get_package_version(package: str) -> dict[str, Any]:
    """Cached variant of the six-tool demo handler (same response shape)."""
    payload = _pypi_payload(package)
    info = payload.get("info") or {}
    return {"package": info.get("name"), "version": info.get("version"), "summary": (info.get("summary") or "")[:300]}


def get_package_release_date(package: str, version: str) -> dict[str, Any]:
    if not isinstance(version, str) or not version.strip():
        raise ToolError("invalid_argument", "version must be a non-empty string")
    payload = _pypi_payload(package)
    info = payload.get("info") or {}
    releases = payload.get("releases") or {}
    files = releases.get(version.strip())
    if not files:
        for key, candidate in releases.items():
            if key.replace("-", ".") == version.strip().replace("-", "."):
                files = candidate
                break
    if not files:
        raise ToolError("not_found", f"no release {version} for package {package}")
    stamp = ""
    for file_info in files:
        stamp = file_info.get("upload_time_iso_8601") or file_info.get("upload_time") or ""
        if stamp:
            break
    if not stamp:
        raise ToolError("server_error", f"release metadata missing upload time for {package} {version}")
    return {"package": info.get("name") or package, "version": version, "release_date": stamp[:10]}


def _indicator_rows(code: str, indicator: str) -> list[dict[str, Any]]:
    cache_key = (code, indicator)
    cached = _WB_INDICATOR_CACHE.get(cache_key)
    if cached is not None:
        return cached
    payload = http_get_json(
        f"https://api.worldbank.org/v2/country/{urllib.parse.quote(code)}/indicator/{indicator}",
        {"format": "json", "per_page": 300},
    )
    rows: list[dict[str, Any]] = []
    if isinstance(payload, list) and len(payload) > 1 and isinstance(payload[1], list):
        rows = [row for row in payload[1] if isinstance(row, dict)]
    if not rows:
        raise ToolError("not_found", f"no {indicator} data for country code {code}")
    _WB_INDICATOR_CACHE[cache_key] = rows
    return rows


def _pick_indicator(rows: list[dict[str, Any]], year: int | None) -> tuple[float | None, str]:
    parsed = []
    for row in rows:
        date = str(row.get("date") or "")
        if not date.isdigit():
            continue
        parsed.append((int(date), row.get("value")))
    parsed.sort(reverse=True)
    if year is not None:
        for date, value in parsed:
            if date == int(year) and value is not None:
                return float(value), str(date)
        return None, str(year)
    for date, value in parsed:
        if value is not None:
            return float(value), str(date)
    return None, ""


def get_country_population(country: str, year: int | None = None) -> dict[str, Any]:
    if year is not None and (not isinstance(year, int) or isinstance(year, bool)):
        raise ToolError("invalid_argument", "year must be an integer")
    row = resolve_country_row(country)
    if row is None:
        raise ToolError("not_found", f"country not found: {country}")
    rows = _indicator_rows(str(row["id"]), "SP.POP.TOTL")
    value, picked_year = _pick_indicator(rows, year)
    if value is None:
        raise ToolError("not_found", f"no population data for {row.get('name')} in {picked_year}")
    return {"country": row.get("name"), "code": row.get("id"), "year": int(picked_year), "population": int(value)}


def get_country_gdp_per_capita(country: str, year: int | None = None) -> dict[str, Any]:
    if year is not None and (not isinstance(year, int) or isinstance(year, bool)):
        raise ToolError("invalid_argument", "year must be an integer")
    row = resolve_country_row(country)
    if row is None:
        raise ToolError("not_found", f"country not found: {country}")
    rows = _indicator_rows(str(row["id"]), "NY.GDP.PCAP.CD")
    value, picked_year = _pick_indicator(rows, year)
    if value is None:
        raise ToolError("not_found", f"no GDP per capita data for {row.get('name')} in {picked_year}")
    return {"country": row.get("name"), "code": row.get("id"), "year": int(picked_year),
            "gdp_per_capita_usd": round(value, 2)}


def geocode_city(name: str) -> dict[str, Any]:
    if not isinstance(name, str) or not name.strip():
        raise ToolError("invalid_argument", "name must be a non-empty string")
    payload = http_get_json(
        "https://geocoding-api.open-meteo.com/v1/search",
        {"name": name.strip(), "count": 1, "language": "en", "format": "json"},
    )
    results = (payload or {}).get("results") or []
    if not results:
        raise ToolError("not_found", f"city not found: {name}")
    hit = results[0]
    return {
        "name": hit.get("name"),
        "latitude": hit.get("latitude"),
        "longitude": hit.get("longitude"),
        "country": hit.get("country"),
    }


def find_country_by_capital(capital: str) -> dict[str, Any]:
    if not isinstance(capital, str) or not capital.strip():
        raise ToolError("invalid_argument", "capital must be a non-empty string")
    key = fold_text(capital)
    rows = list(world_bank_country_rows().values())
    exact = [row for row in rows if key and fold_text(row.get("capitalCity")) == key]
    if not exact:
        exact = [row for row in rows if key and key in fold_text(row.get("capitalCity"))]
    if not exact:
        raise ToolError("not_found", f"no country found for capital: {capital}")
    row = exact[0]
    return {"country": row.get("name"), "code": row.get("id"), "capital": row.get("capitalCity")}


def get_hn_user(username: str) -> dict[str, Any]:
    if not isinstance(username, str) or not username.strip():
        raise ToolError("invalid_argument", "username must be a non-empty string")
    payload = http_get_json(
        "https://hacker-news.firebaseio.com/v0/user/" + urllib.parse.quote(username.strip()) + ".json"
    )
    if not isinstance(payload, dict) or not payload.get("id"):
        raise ToolError("not_found", f"no Hacker News user: {username}")
    return {"username": payload.get("id"), "karma": payload.get("karma"), "created": payload.get("created")}


def get_hn_top_story() -> dict[str, Any]:
    ids = http_get_json("https://hacker-news.firebaseio.com/v0/topstories.json")
    if not isinstance(ids, list) or not ids:
        raise ToolError("server_error", "Hacker News top stories unavailable")
    story_id = ids[0]
    payload = http_get_json(f"https://hacker-news.firebaseio.com/v0/item/{int(story_id)}.json")
    if not isinstance(payload, dict) or not payload.get("title"):
        raise ToolError("not_found", f"no Hacker News item: {story_id}")
    return {"id": int(story_id), "title": payload.get("title"), "by": payload.get("by"), "score": payload.get("score")}


# ---------------------------------------------------------------------------
# registry used by the hard multi-step demo (GitHub is deliberately excluded: the
# key-free API rate limit makes it non-deterministic for a 20-task A/B run)
# ---------------------------------------------------------------------------
HARD_TOOLS: dict[str, ToolSpec] = {
    "get_country_info": ToolSpec(
        name="get_country_info",
        description="Get a country's capital city, region and income level (World Bank data).",
        parameters={
            "type": "object",
            "properties": {"country": {"type": "string", "description": "Country name, e.g. Japan"}},
            "required": ["country"],
        },
        handler=get_country_info,
    ),
    "find_country_by_capital": ToolSpec(
        name="find_country_by_capital",
        description="Find the country that has a given capital city (World Bank data).",
        parameters={
            "type": "object",
            "properties": {"capital": {"type": "string", "description": "Capital city name, e.g. Canberra"}},
            "required": ["capital"],
        },
        handler=find_country_by_capital,
    ),
    "get_country_population": ToolSpec(
        name="get_country_population",
        description="Get a country's population from the World Bank (latest available year, or a given year).",
        parameters={
            "type": "object",
            "properties": {
                "country": {"type": "string", "description": "Country name, e.g. Australia"},
                "year": {"type": "integer", "description": "Optional year, e.g. 2023"},
            },
            "required": ["country"],
        },
        handler=get_country_population,
    ),
    "get_country_gdp_per_capita": ToolSpec(
        name="get_country_gdp_per_capita",
        description="Get a country's GDP per capita in current US dollars (World Bank).",
        parameters={
            "type": "object",
            "properties": {
                "country": {"type": "string", "description": "Country name, e.g. Switzerland"},
                "year": {"type": "integer", "description": "Optional year, e.g. 2023"},
            },
            "required": ["country"],
        },
        handler=get_country_gdp_per_capita,
    ),
    "geocode_city": ToolSpec(
        name="geocode_city",
        description="Resolve a city name to latitude/longitude coordinates (Open-Meteo geocoding).",
        parameters={
            "type": "object",
            "properties": {"name": {"type": "string", "description": "City name, e.g. Tokyo"}},
            "required": ["name"],
        },
        handler=geocode_city,
    ),
    "get_weather": ToolSpec(
        name="get_weather",
        description="Get current temperature (Celsius) for a coordinate.",
        parameters={
            "type": "object",
            "properties": {
                "latitude": {"type": "number", "description": "Decimal latitude"},
                "longitude": {"type": "number", "description": "Decimal longitude"},
            },
            "required": ["latitude", "longitude"],
        },
        handler=get_weather,
    ),
    "get_package_version": ToolSpec(
        name="get_package_version",
        description="Get the latest published version of a Python package on PyPI.",
        parameters={
            "type": "object",
            "properties": {"package": {"type": "string", "description": "PyPI package name"}},
            "required": ["package"],
        },
        handler=get_package_version,
    ),
    "get_package_release_date": ToolSpec(
        name="get_package_release_date",
        description="Get the release date (YYYY-MM-DD) of one published version of a PyPI package.",
        parameters={
            "type": "object",
            "properties": {
                "package": {"type": "string", "description": "PyPI package name"},
                "version": {"type": "string", "description": "Exact version string, e.g. 0.27.2"},
            },
            "required": ["package", "version"],
        },
        handler=get_package_release_date,
    ),
    "get_hn_item": ToolSpec(
        name="get_hn_item",
        description="Get a Hacker News item (story) by numeric id.",
        parameters={
            "type": "object",
            "properties": {"story_id": {"type": "integer", "description": "Numeric HN item id"}},
            "required": ["story_id"],
        },
        handler=get_hn_item,
    ),
    "get_hn_user": ToolSpec(
        name="get_hn_user",
        description="Get a Hacker News user's karma score.",
        parameters={
            "type": "object",
            "properties": {"username": {"type": "string", "description": "Hacker News username"}},
            "required": ["username"],
        },
        handler=get_hn_user,
    ),
    "get_hn_top_story": ToolSpec(
        name="get_hn_top_story",
        description="Get the current top story on Hacker News (id, title, author, score).",
        parameters={"type": "object", "properties": {}},
        handler=get_hn_top_story,
    ),
    "get_exchange_rate": ToolSpec(
        name="get_exchange_rate",
        description="Get the current FX rate from a base currency to a target currency (open.er-api.com).",
        parameters={
            "type": "object",
            "properties": {"base": {"type": "string"}, "target": {"type": "string"}},
            "required": ["base", "target"],
        },
        handler=get_exchange_rate,
    ),
}


class LiveHardBackend:
    """Calls real public APIs over the extended tool set.

    ``hn_schema`` selects the advertised HN contract:

    * ``"consistent"`` (default, main protocol since the 2026-11-25 review decision):
      ``story_id`` is an integer, matching the backend and the real Hacker News API.
    * ``"drifting"``: the schema advertises ``story_id`` as a string while the backend
      requires an integer.  This reproduces a real contract-mismatch failure class and is
      kept only for the targeted contract-repair demonstration (same tasks / same fault
      seeds), so the contribution of the drift to the baseline-vs-guard gap can be
      measured against the consistent run.
    """

    def __init__(self, tools: dict[str, ToolSpec] | None = None,
                 hn_schema: str = "consistent"):
        if hn_schema not in ("consistent", "drifting"):
            raise ValueError(f"hn_schema must be 'consistent' or 'drifting', got {hn_schema!r}")
        self.tools = tools or HARD_TOOLS
        self.hn_schema = hn_schema
        self.stats = {"calls": 0, "ok": 0, "errors": 0}

    def schemas(self) -> list[dict[str, Any]]:
        schemas = [tool.openai_schema() for tool in self.tools.values()]
        if self.hn_schema == "drifting":
            for schema in schemas:
                function = schema.get("function") or {}
                if function.get("name") == "get_hn_item":
                    properties = function.setdefault("parameters", {}).setdefault(
                        "properties", {})
                    properties["story_id"] = {
                        "type": "string",
                        "description": "Numeric HN item id (as string)",
                    }
        return schemas

    def call(self, function_name: str, arguments: dict[str, Any]):
        from .backend import CallResult

        self.stats["calls"] += 1
        tool = self.tools.get(function_name)
        if tool is None:
            self.stats["errors"] += 1
            return CallResult("error", error=f"No such function name: {function_name}")
        try:
            payload = tool.handler(**arguments)
        except ToolError as exc:
            self.stats["errors"] += 1
            return CallResult("error", error=f"{exc.kind}: {exc.message}")
        except TypeError as exc:
            self.stats["errors"] += 1
            return CallResult("error", error=f"invalid_argument: {exc}")
        self.stats["ok"] += 1
        return CallResult("ok", response=payload, source="live")
