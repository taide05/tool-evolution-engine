"""Collect **real online fault traces** from the live demo's public APIs.

2026-11-25 evidence-chain follow-up: the live A/B experiments inject deterministic
faults for reproducibility; this collector captures *genuine* error responses from the
real public APIs (bad arguments, unknown ids, invalid coordinates) so the middleware's
repair layer can be exercised against real faults instead of only synthetic ones.

The collected traces are evidence input for ``scripts/run_real_fault_repair.py``; each
trace stores the tool, the exact bad arguments, the real error kind/message, the API host
and the capture timestamp.  No LLM and no paid API is used.

Usage
-----
    python scripts/collect_real_fault_traces.py \
        --output benchmarks/toolguard/real_traces/real_fault_traces.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tool_evolution.toolguard import live_hard  # noqa: E402
from tool_evolution.toolguard.live import ToolError  # noqa: E402

# (trace id, tool, bad arguments, API host, description)
CASES = [
    ("country_info_unknown_country", "get_country_info",
     {"country": "Zzzland-Not-A-Country"}, "api.worldbank.org",
     "unknown country name against the real World Bank country table"),
    ("find_country_unknown_capital", "find_country_by_capital",
     {"capital": "Atlantis"}, "api.worldbank.org",
     "unknown capital city"),
    ("country_population_unknown", "get_country_population",
     {"country": "Zzzland"}, "api.worldbank.org",
     "unknown country code for an indicator query"),
    ("country_gdp_unknown", "get_country_gdp_per_capita",
     {"country": "Zzzland"}, "api.worldbank.org",
     "unknown country code for a GDP query"),
    ("weather_invalid_coordinates", "get_weather",
     {"latitude": 999.0, "longitude": 999.0}, "api.open-meteo.com",
     "out-of-range coordinates"),
    ("weather_non_numeric_latitude", "get_weather",
     {"latitude": "not-a-number", "longitude": 0.0}, "api.open-meteo.com",
     "type mismatch: latitude arrives as a string"),
    ("weather_numeric_strings", "get_weather",
     {"latitude": "48.85", "longitude": "2.35"}, "api.open-meteo.com",
     "type mismatch: coordinates arrive as numeric strings (coercible contract error)"),
    ("country_population_year_as_string", "get_country_population",
     {"country": "Brazil", "year": "2020"}, "api.worldbank.org",
     "type mismatch: year arrives as a string (coercible contract error)"),
    ("hn_story_id_float", "get_hn_item",
     {"story_id": 8863.0}, "hacker-news.firebaseio.com",
     "type mismatch: story_id arrives as an integral float"),
    ("geocode_empty_city", "geocode_city",
     {"name": ""}, "geocoding-api.open-meteo.com",
     "empty city name"),
    ("pypi_unknown_package", "get_package_version",
     {"package": "this-package-does-not-exist-xyzzz"}, "pypi.org",
     "unknown PyPI package"),
    ("pypi_unknown_version", "get_package_release_date",
     {"package": "numpy", "version": "999.0.0"}, "pypi.org",
     "known package with an unknown version"),
    ("hn_missing_item", "get_hn_item",
     {"story_id": 0}, "hacker-news.firebaseio.com",
     "non-existent HN item id"),
    ("hn_story_id_as_string", "get_hn_item",
     {"story_id": "8863"}, "hacker-news.firebaseio.com",
     "type mismatch: story_id arrives as a string (real backend contract error)"),
    ("hn_unknown_user", "get_hn_user",
     {"username": "user-that-does-not-exist-xyzzz"}, "hacker-news.firebaseio.com",
     "unknown HN username"),
    ("fx_unknown_currency", "get_exchange_rate",
     {"base": "ZZZ", "target": "ZZZ"}, "open.er-api.com",
     "unknown currency codes"),
]


def _endpoint(tool: str) -> str:
    handler = getattr(live_hard, tool, None) or getattr(
        __import__("tool_evolution.toolguard.live", fromlist=["*"]), tool)
    return f"{handler.__module__}.{tool}" if handler else tool


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output",
                        default="benchmarks/toolguard/real_traces/real_fault_traces.json")
    args = parser.parse_args()

    records: list[dict] = []
    for trace_id, tool, bad_arguments, host, description in CASES:
        handler = getattr(live_hard, tool, None)
        if handler is None:
            from tool_evolution.toolguard import live
            handler = getattr(live, tool)
        captured_at = datetime.now(timezone.utc).isoformat()
        try:
            payload = handler(**bad_arguments)
        except ToolError as exc:
            record = {
                "trace_id": trace_id,
                "tool": tool,
                "arguments": bad_arguments,
                "description": description,
                "api_host": host,
                "captured_at": captured_at,
                "outcome": "error",
                "error_kind": exc.kind,
                "error_message": exc.message,
                "source": "real public API call (captured, not injected)",
            }
            print(f"[{trace_id:32s}] {tool:28s} real error: {exc.kind}: {exc.message[:110]}")
        except Exception as exc:  # noqa: BLE001 - capture any genuine transport failure
            record = {
                "trace_id": trace_id,
                "tool": tool,
                "arguments": bad_arguments,
                "description": description,
                "api_host": host,
                "captured_at": captured_at,
                "outcome": "error",
                "error_kind": "unexpected_exception",
                "error_message": f"{type(exc).__name__}: {exc}"[:400],
                "source": "real public API call (captured, not injected)",
            }
            print(f"[{trace_id:32s}] {tool:28s} exception: {type(exc).__name__}: {exc}"[:180])
        else:
            record = {
                "trace_id": trace_id,
                "tool": tool,
                "arguments": bad_arguments,
                "description": description,
                "api_host": host,
                "captured_at": captured_at,
                "outcome": "unexpected_success",
                "error_kind": None,
                "error_message": None,
                "response_keys": sorted(payload)[:8] if isinstance(payload, dict) else None,
                "source": "real public API call (captured, not injected)",
            }
            print(f"[{trace_id:32s}] {tool:28s} unexpectedly succeeded (kept with outcome=unexpected_success)")
        records.append(record)

    errors = [record for record in records if record["outcome"] == "error"]
    payload = {
        "artifact_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "description": (
            "Real online fault traces captured by calling the live demo's real public APIs "
            "with bad/edge arguments. Used by scripts/run_real_fault_repair.py to exercise "
            "the middleware repair layer against genuine API errors (not injected faults)."),
        "collector": "scripts/collect_real_fault_traces.py",
        "cost_usd": 0.0,
        "summary": {
            "cases": len(records),
            "real_errors_captured": len(errors),
            "unexpected_success": len(records) - len(errors),
            "by_error_kind": _count(errors, "error_kind"),
            "by_tool": _count(errors, "tool"),
        },
        "records": records,
    }
    output = REPO / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\ncaptured {len(errors)} real faults from {len(records)} cases -> {output}")
    return 0


def _count(records: list[dict], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.get(key))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    raise SystemExit(main())
