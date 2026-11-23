"""Real MCP (stdio) server exposing the key-free public-API tool surface.

This is the tool layer for the MCP integration demo: a standard MCP server that any MCP
client can drive.  Every tool calls a real public API (World Bank, Open-Meteo, PyPI,
Hacker News, open.er-api.com); there is no mock data.

Errors are reported through MCP ``isError`` results with a ``TOOL_ERROR[kind]: message``
marker, so a middleware client can map them back onto the failure taxonomy it learned
from the offline traces (``invalid_argument`` / ``timeout`` / ``rate_limit`` / ...).

Fault injection lives *in the server*, i.e. where real outages live: when
``TOOLGUARD_MCP_FAULT_RATE`` is set, a deterministic subset of logical calls fails its
first ``TOOLGUARD_MCP_FAULT_EPOCH`` attempts with transient MCP tool errors (timeout /
HTTP 429 / HTTP 5xx).  The demo runner starts a fresh server per arm so both A/B arms
face the same fresh fault surface.  With the env vars unset the server never injects
faults.

Run it directly to speak MCP over stdio::

    python scripts/mcp_toolguard_server.py

or point any MCP client at it.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mcp.server.fastmcp import FastMCP  # noqa: E402

from tool_evolution.toolguard import live_hard  # noqa: E402
from tool_evolution.toolguard.live import ToolError  # noqa: E402

# Keep stderr quiet: the MCP session owns stdout, and per-request INFO logs would drown
# the demo runner's output.
for _logger in ("httpx", "httpcore", "mcp", "mcp.server", "mcp.server.lowlevel.server"):
    logging.getLogger(_logger).setLevel(logging.WARNING)

mcp = FastMCP("toolguard-live-tools", log_level="WARNING")

_FAULT_RATE = float(os.environ.get("TOOLGUARD_MCP_FAULT_RATE") or 0.0)
_FAULT_SEED = int(os.environ.get("TOOLGUARD_MCP_FAULT_SEED") or 7)
_FAULT_EPOCH = max(1, int(os.environ.get("TOOLGUARD_MCP_FAULT_EPOCH") or 2))
_FAULT_STATE: dict[str, int] = {}


def _stable_roll(*parts: object) -> float:
    digest = hashlib.blake2b(repr(parts).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(2**64)


def _maybe_fault(tool: str, arguments: dict) -> None:
    """Server-side deterministic transient fault epochs (disabled unless configured)."""
    if _FAULT_RATE <= 0:
        return
    key = tool + ":" + json.dumps(arguments, sort_keys=True, ensure_ascii=False, default=str)
    if key not in _FAULT_STATE:
        _FAULT_STATE[key] = _FAULT_EPOCH if _stable_roll(_FAULT_SEED, key) < _FAULT_RATE else 0
    remaining = _FAULT_STATE[key]
    if remaining <= 0:
        return
    _FAULT_STATE[key] = remaining - 1
    attempt = _FAULT_EPOCH - remaining + 1
    selector = _stable_roll(_FAULT_SEED, key, "kind")
    if selector < 0.5:
        raise ValueError(f"TOOL_ERROR[timeout]: simulated upstream timeout (attempt {attempt})")
    if selector < 0.8:
        raise ValueError(f"TOOL_ERROR[rate_limit]: simulated HTTP 429 (attempt {attempt})")
    raise ValueError(f"TOOL_ERROR[server_error]: simulated HTTP 503 (attempt {attempt})")


def _call(tool: str, handler, **arguments):
    _maybe_fault(tool, arguments)
    try:
        return handler(**arguments)
    except ToolError as exc:
        raise ValueError(f"TOOL_ERROR[{exc.kind}]: {exc.message}") from exc


@mcp.tool()
def get_country_info(country: str) -> dict:
    """Get a country's capital city, region and income level (World Bank data)."""
    return _call("get_country_info", live_hard.get_country_info, country=country)


@mcp.tool()
def find_country_by_capital(capital: str) -> dict:
    """Find the country that has a given capital city (World Bank data)."""
    return _call("find_country_by_capital", live_hard.find_country_by_capital, capital=capital)


@mcp.tool()
def get_country_population(country: str) -> dict:
    """Get a country's latest population from the World Bank."""
    return _call("get_country_population", live_hard.get_country_population, country=country)


@mcp.tool()
def get_country_gdp_per_capita(country: str) -> dict:
    """Get a country's GDP per capita in current US dollars (World Bank)."""
    return _call("get_country_gdp_per_capita", live_hard.get_country_gdp_per_capita, country=country)


@mcp.tool()
def geocode_city(name: str) -> dict:
    """Resolve a city name to latitude/longitude coordinates (Open-Meteo geocoding)."""
    return _call("geocode_city", live_hard.geocode_city, name=name)


@mcp.tool()
def get_weather(latitude: float, longitude: float) -> dict:
    """Get the current temperature in Celsius for a coordinate (Open-Meteo)."""
    return _call("get_weather", live_hard.get_weather, latitude=latitude, longitude=longitude)


@mcp.tool()
def get_package_version(package: str) -> dict:
    """Get the latest published version of a Python package on PyPI."""
    return _call("get_package_version", live_hard.get_package_version, package=package)


@mcp.tool()
def get_package_release_date(package: str, version: str) -> dict:
    """Get the release date (YYYY-MM-DD) of one published version of a PyPI package."""
    return _call("get_package_release_date", live_hard.get_package_release_date,
                 package=package, version=version)


@mcp.tool()
def get_hn_item(story_id: int) -> dict:
    """Get a Hacker News item (story) by its numeric id."""
    return _call("get_hn_item", live_hard.get_hn_item, story_id=story_id)


@mcp.tool()
def get_hn_user(username: str) -> dict:
    """Get a Hacker News user's karma score."""
    return _call("get_hn_user", live_hard.get_hn_user, username=username)


@mcp.tool()
def get_hn_top_story() -> dict:
    """Get the current top story on Hacker News (id, title, author, score)."""
    return _call("get_hn_top_story", live_hard.get_hn_top_story)


@mcp.tool()
def get_exchange_rate(base: str, target: str) -> dict:
    """Get the current FX rate from a base currency to a target currency."""
    return _call("get_exchange_rate", live_hard.get_exchange_rate, base=base, target=target)


if __name__ == "__main__":
    mcp.run(transport="stdio")
