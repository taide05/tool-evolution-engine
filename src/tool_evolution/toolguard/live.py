"""Live tool backend over real public APIs (no API keys required).

The demo uses six independent public services so the failure surface is real HTTP:
REST Countries, PyPI, Open-Meteo, GitHub, Hacker News and an exchange-rate API.
Wrappers are intentionally strict about argument types (the middleware repairs type
errors) and raise typed ``ToolError``s mirroring real status codes.  ``get_hn_item``
declares ``story_id`` as a string in its schema while the backend requires an integer,
reproducing the schema/backend-mismatch failure class measured in the offline data.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
import unicodedata
import urllib.parse
from dataclasses import dataclass, field
from typing import Any, Callable

import httpx

TIMEOUT_S = 12.0
_HEADERS = {"User-Agent": "toolguard-live-demo/0.1 (personal project)"}


@dataclass
class ToolError(Exception):
    kind: str  # invalid_argument | not_found | rate_limit | server_error | timeout | request_error
    message: str

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.kind}: {self.message}"


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., dict[str, Any]]

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }


@dataclass
class _Response:
    json: Any


def http_get_json(url: str, params: dict[str, Any] | None = None) -> Any:
    """GET JSON with bounded retries for transient transport failures.

    Timeouts / connection errors are retried (the public APIs used by the live demos are
    occasionally slow); this keeps ground-truth construction from silently dropping tasks.
    """
    last_status = None
    for attempt in range(3):
        try:
            response = httpx.get(url, params=params, timeout=TIMEOUT_S, headers=_HEADERS, follow_redirects=True)
        except httpx.TimeoutException as exc:
            if attempt < 2:
                time.sleep(1.0 * (attempt + 1))
                continue
            raise ToolError("timeout", f"request timed out after {TIMEOUT_S}s") from exc
        except httpx.HTTPError as exc:
            if attempt < 2:
                time.sleep(0.5 * (attempt + 1))
                continue
            raise ToolError("request_error", str(exc)[:200]) from exc
        last_status = response.status_code
        if response.status_code == 429 or response.status_code >= 500:
            if attempt < 2:
                time.sleep(1.0 * (attempt + 1))  # transient API limit/outage: back off briefly
                continue
        break
    if last_status == 429:
        raise ToolError("rate_limit", "HTTP 429 rate limited")
    if last_status in (401, 403):
        raise ToolError("rate_limit", f"HTTP {last_status} forbidden/rate-limited")
    if last_status == 404:
        raise ToolError("not_found", "HTTP 404 not found")
    if last_status >= 500:
        raise ToolError("server_error", f"HTTP {last_status} server error")
    if last_status >= 400:
        raise ToolError("request_error", f"HTTP {last_status}: {response.text[:160]}")
    try:
        return response.json()
    except json.JSONDecodeError as exc:
        raise ToolError("server_error", "invalid JSON response") from exc


# ---------------------------------------------------------------------------
# tool handlers
# ---------------------------------------------------------------------------
_WB_COUNTRIES: dict[str, dict[str, Any]] | None = None


def _world_bank_countries() -> dict[str, dict[str, Any]]:
    global _WB_COUNTRIES
    if _WB_COUNTRIES is None:
        payload = http_get_json("https://api.worldbank.org/v2/country", {"format": "json", "per_page": 400})
        rows = payload[1] if isinstance(payload, list) and len(payload) > 1 else []
        mapping: dict[str, dict[str, Any]] = {}
        for row in rows or []:
            name = str(row.get("name") or "").strip()
            if name and row.get("id"):
                mapping[name.lower()] = row
        _WB_COUNTRIES = mapping
    return _WB_COUNTRIES


def fold_text(text: Any) -> str:
    """Accent-insensitive lowercase form used for name matching."""
    folded = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")
    return " ".join(folded.lower().split())


def resolve_country_row(country: str) -> dict[str, Any] | None:
    """Resolve a free-form country name to a World Bank country row."""
    if not isinstance(country, str) or not country.strip():
        return None
    mapping = _world_bank_countries()
    key = fold_text(country)
    row = mapping.get(country.strip().lower())
    if row is None:
        for name, candidate in mapping.items():
            if key and (key in fold_text(name) or fold_text(name) in key):
                row = candidate
                break
    return row


def world_bank_country_rows() -> dict[str, dict[str, Any]]:
    """Access the cached World Bank country rows (name -> row)."""
    return _world_bank_countries()


def get_country_info(country: str) -> dict[str, Any]:
    if not isinstance(country, str) or not country.strip():
        raise ToolError("invalid_argument", "country must be a non-empty string")
    row = resolve_country_row(country)
    if row is None:
        raise ToolError("not_found", f"country not found: {country}")
    detail = http_get_json(f"https://api.worldbank.org/v2/country/{row['id']}", {"format": "json"})
    if not isinstance(detail, list) or len(detail) < 2 or not detail[1]:
        raise ToolError("not_found", f"country details unavailable: {country}")
    item = detail[1][0]
    return {
        "country": item.get("name"),
        "capital": item.get("capitalCity"),
        "region": (item.get("region") or {}).get("value"),
        "income_level": (item.get("incomeLevel") or {}).get("value"),
        "code": item.get("id"),
    }


def get_package_version(package: str) -> dict[str, Any]:
    if not isinstance(package, str) or not package.strip():
        raise ToolError("invalid_argument", "package must be a non-empty string")
    name = package.strip().lower()
    if " " in name:
        raise ToolError("invalid_argument", f"invalid package name: {package!r}")
    payload = http_get_json(f"https://pypi.org/pypi/{urllib.parse.quote(name)}/json")
    info = (payload or {}).get("info") or {}
    if not info.get("version"):
        raise ToolError("not_found", f"package not found: {package}")
    return {"package": info.get("name"), "version": info.get("version"), "summary": (info.get("summary") or "")[:300]}


def get_weather(latitude: float, longitude: float) -> dict[str, Any]:
    if not isinstance(latitude, (int, float)) or isinstance(latitude, bool):
        raise ToolError("invalid_argument", "latitude must be a number")
    if not isinstance(longitude, (int, float)) or isinstance(longitude, bool):
        raise ToolError("invalid_argument", "longitude must be a number")
    payload = http_get_json(
        "https://api.open-meteo.com/v1/forecast",
        {"latitude": latitude, "longitude": longitude, "current": "temperature_2m", "timezone": "auto"},
    )
    current = (payload or {}).get("current") or {}
    if current.get("temperature_2m") is None:
        raise ToolError("server_error", "weather response missing temperature")
    return {
        "latitude": latitude,
        "longitude": longitude,
        "temperature_c": current.get("temperature_2m"),
        "time": current.get("time"),
    }


def get_github_repo(owner: str, repo: str) -> dict[str, Any]:
    if not isinstance(owner, str) or not isinstance(repo, str) or not owner.strip() or not repo.strip():
        raise ToolError("invalid_argument", "owner and repo must be non-empty strings")
    payload = http_get_json(f"https://api.github.com/repos/{urllib.parse.quote(owner)}/{urllib.parse.quote(repo)}")
    return {
        "full_name": payload.get("full_name"),
        "stargazers_count": payload.get("stargazers_count"),
        "description": (payload.get("description") or "")[:300],
    }


def get_hn_item(story_id: int) -> dict[str, Any]:
    if not isinstance(story_id, int) or isinstance(story_id, bool):
        raise ToolError("invalid_argument", "story_id must be an integer")
    payload = http_get_json(f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json")
    if not isinstance(payload, dict) or not payload.get("title"):
        raise ToolError("not_found", f"no Hacker News item: {story_id}")
    return {
        "id": story_id,
        "title": payload.get("title"),
        "score": payload.get("score"),
        "type": payload.get("type"),
        "by": payload.get("by"),
        "time": payload.get("time"),
    }


def get_exchange_rate(base: str, target: str) -> dict[str, Any]:
    if not isinstance(base, str) or not isinstance(target, str):
        raise ToolError("invalid_argument", "base and target must be strings")
    payload = http_get_json(f"https://open.er-api.com/v6/latest/{urllib.parse.quote(base.strip().upper())}")
    rates = (payload or {}).get("rates") or {}
    rate = rates.get(target.strip().upper())
    if rate is None:
        raise ToolError("not_found", f"no rate for {base}->{target}")
    return {"base": base.upper(), "target": target.upper(), "rate": rate}


TOOLS: dict[str, ToolSpec] = {
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
    "get_github_repo": ToolSpec(
        name="get_github_repo",
        description="Get a GitHub repository's star count and description.",
        parameters={
            "type": "object",
            "properties": {"owner": {"type": "string"}, "repo": {"type": "string"}},
            "required": ["owner", "repo"],
        },
        handler=get_github_repo,
    ),
    "get_hn_item": ToolSpec(
        name="get_hn_item",
        description="Get a Hacker News item (story) by numeric id.",
        parameters={
            "type": "object",
            "properties": {"story_id": {"type": "string", "description": "Numeric HN item id (as string)"}},
            "required": ["story_id"],
        },
        handler=get_hn_item,
    ),
    "get_exchange_rate": ToolSpec(
        name="get_exchange_rate",
        description="Get the current FX rate from a base currency to a target currency.",
        parameters={
            "type": "object",
            "properties": {"base": {"type": "string"}, "target": {"type": "string"}},
            "required": ["base", "target"],
        },
        handler=get_exchange_rate,
    ),
}


class LiveBackend:
    """Calls real public APIs; returns typed results/errors like a real tool layer."""

    def __init__(self, tools: dict[str, ToolSpec] | None = None):
        self.tools = tools or TOOLS
        self.stats = {"calls": 0, "ok": 0, "errors": 0}

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.openai_schema() for tool in self.tools.values()]

    def call(self, function_name: str, arguments: dict[str, Any]):
        from .backend import CallResult  # local import to keep module import light

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


def _stable_rng(*parts: Any) -> random.Random:
    digest = hashlib.blake2b(repr(parts).encode("utf-8"), digest_size=8).digest()
    return random.Random(int.from_bytes(digest, "big"))


@dataclass
class FaultyBackend:
    """Deterministically injects transient fault *epochs* into a tool backend.

    Each unique logical call (function name + canonical arguments) is rolled once with a
    seed-derived RNG.  A faulted call fails its first ``attempts_to_fail`` attempts with
    transient errors (timeout / rate-limit / server error), modelling a short outage,
    and only then reaches the real backend.  Retries of the same logical call hit the
    same outage, so recovery requires either in-middleware retry (one agent turn) or
    extra agent retries (one wasted agent turn each).  Deterministic in
    ``(seed, function, arguments)`` across processes, so both A/B arms see the same
    fault surface for identical calls.
    """

    backend: Any
    seed: int = 42
    p_timeout: float = 0.12
    p_rate_limit: float = 0.06
    p_server_error: float = 0.0
    attempts_to_fail: int = 1
    _remaining: dict[str, int] = field(default_factory=dict)
    stats: dict[str, Any] = field(default_factory=lambda: {
        "injected": 0, "calls": 0, "faulted_calls": 0, "kinds": {},
    })

    def schemas(self):
        return self.backend.schemas()

    def call(self, function_name: str, arguments: dict[str, Any]):
        from .backend import CallResult

        self.stats["calls"] += 1
        key = f"{function_name}:{json.dumps(arguments, sort_keys=True, ensure_ascii=False, default=str)}"
        if key not in self._remaining:
            total = self.p_timeout + self.p_rate_limit + self.p_server_error
            roll = _stable_rng(self.seed, key).random()
            self._remaining[key] = self.attempts_to_fail if total > 0 and roll < total else 0
            if self._remaining[key]:
                self.stats["faulted_calls"] += 1
        remaining = self._remaining[key]
        if remaining > 0:
            self._remaining[key] = remaining - 1
            attempt_index = self.attempts_to_fail - remaining
            self.stats["injected"] += 1
            selector = _stable_rng(self.seed, key, "kind").random()
            total = self.p_timeout + self.p_rate_limit + self.p_server_error
            if selector < self.p_timeout / total:
                kind, message = "timeout", "timeout: request timed out"
            elif selector < (self.p_timeout + self.p_rate_limit) / total:
                kind, message = "rate_limit", "rate_limit: HTTP 429 rate limited"
            else:
                kind, message = "server_error", f"server_error: HTTP 503 server error (attempt {attempt_index + 1})"
            self.stats["kinds"][kind] = self.stats["kinds"].get(kind, 0) + 1
            return CallResult("error", error=message)
        return self.backend.call(function_name, arguments)


def sleep_s(seconds: float) -> None:  # patchable for tests
    time.sleep(seconds)
