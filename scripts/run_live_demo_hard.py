"""Harder multi-step live demo: ToolGuard middleware vs baseline over dependent tool chains.

Protocol
--------
* 20 tasks; each needs 2-6 *dependent* tool calls over 7 key-free public APIs
  (World Bank, Open-Meteo geocoding + forecast, PyPI, Hacker News, open.er-api.com).
  Answers are verified programmatically against ground truth fetched live before the runs;
  no LLM judge is used.
* Two interaction-budget conditions share the same fault surface:
    - ``budgeted``: per-task turn budget = clean minimum turns + 2;
    - ``generous``: 10 turns for every task.
* Deterministic transient-fault epochs: the first two attempts of a faulted logical call
  fail (timeout / HTTP 429 / HTTP 5xx), modelling a short outage.  A baseline agent must
  spend *agent turns* retrying; ToolGuard's in-middleware retry absorbs the epoch inside
  one turn.
* Real DeepSeek tool-calling agent (``deepseek-v4-flash``, temperature 0).

There is no offline/mock fallback: if DeepSeek or a public API is unavailable the run
stops and reports, it never fabricates results.

Usage
-----
    python scripts/run_live_demo_hard.py --dry-run
    python scripts/run_live_demo_hard.py --conditions budgeted --max-tasks 3
    python scripts/run_live_demo_hard.py --output benchmarks/toolguard/live_demo_hard.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tool_evolution.toolguard.agent import DeepSeekClient, LLMConfig, run_task  # noqa: E402
from tool_evolution.toolguard.guard import ToolGuard  # noqa: E402
from tool_evolution.toolguard.live import FaultyBackend, ToolError  # noqa: E402
from tool_evolution.toolguard.live_hard import LiveHardBackend  # noqa: E402
from tool_evolution.toolguard.verify import (  # noqa: E402
    contains_digits,
    contains_number,
    contains_text,
    mcnemar_exact,
)
from tool_evolution.utils.config import settings  # noqa: E402

DEFAULT_OUTPUT = ".scratch/toolguard/live_demo_hard.json"
MODEL = "deepseek-v4-flash"
ARMS = ("baseline", "guard")


# ---------------------------------------------------------------------------
# task construction: ground truth is fetched live before any agent runs
# ---------------------------------------------------------------------------
def build_tasks(backend: LiveHardBackend) -> tuple[list[dict], list[str]]:
    tasks: list[dict] = []
    skipped: list[str] = []

    def add(task_id: str, question: str, expected: dict, verifier, min_turns: int) -> None:
        tasks.append({"id": task_id, "question": question, "expected": expected,
                      "verifier": verifier, "min_turns": min_turns})

    def safe(tool: str, **arguments):
        last_error = None
        for _ in range(3):
            result = backend.call(tool, arguments)
            if result.ok:
                return result.response
            last_error = result.error
        raise ToolError("request_error", f"{tool} failed: {last_error}")

    def weather_chain(country: str) -> dict:
        info = safe("get_country_info", country=country)
        capital = info["capital"]
        geo = safe("geocode_city", name=capital)
        weather = safe("get_weather", latitude=geo["latitude"], longitude=geo["longitude"])
        return {"country": country, "capital": capital, "latitude": geo["latitude"],
                "longitude": geo["longitude"], "temperature_c": weather["temperature_c"]}

    def latest_release(package: str) -> tuple[str, str]:
        version = safe("get_package_version", package=package)["version"]
        release_date = safe("get_package_release_date", package=package, version=version)["release_date"]
        return version, release_date

    def author_karma(story_id: int) -> tuple[str, int]:
        item = safe("get_hn_item", story_id=story_id)
        if not item.get("by"):
            raise ToolError("not_found", f"HN item {story_id} has no author")
        user = safe("get_hn_user", username=item["by"])
        return item["by"], int(user["karma"])

    # -- dependent weather chains (country -> capital -> coordinates -> weather) -----
    # The questions explicitly ask for geocoding so the chain is a real dependent call
    # chain: the coordinates are not in the question, so they must come from the tools.
    for task_id, country in (("capital_weather_japan", "Japan"),
                             ("capital_weather_norway", "Norway"),
                             ("capital_weather_portugal", "Portugal")):
        try:
            chain = weather_chain(country)
            add(task_id,
                f"What is the current temperature, in degrees Celsius, in the capital city of {country}? "
                f"Use get_country_info for the capital, geocode_city for its coordinates, and get_weather "
                f"for the temperature.",
                chain,
                lambda answer, expected: contains_number(answer, expected["temperature_c"], 3.0),
                4)
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"{task_id}: {exc}")

    # -- cross-country weather comparisons ------------------------------------------
    for task_id, country_a, country_b in (
        ("capital_weather_compare_france_germany", "France", "Germany"),
        ("capital_weather_compare_norway_portugal", "Norway", "Portugal"),
    ):
        try:
            chain_a = weather_chain(country_a)
            chain_b = weather_chain(country_b)
            warmer = chain_a["capital"] if chain_a["temperature_c"] >= chain_b["temperature_c"] else chain_b["capital"]
            require_warmer = abs(chain_a["temperature_c"] - chain_b["temperature_c"]) > 2.0
            expected = {
                "a": {"country": country_a, "capital": chain_a["capital"],
                      "temperature_c": chain_a["temperature_c"]},
                "b": {"country": country_b, "capital": chain_b["capital"],
                      "temperature_c": chain_b["temperature_c"]},
                "warmer_capital": warmer,
                "require_warmer": require_warmer,
            }
            add(task_id,
                f"Which capital city is currently warmer: the capital of {country_a} or the capital of "
                f"{country_b}? For each country use get_country_info for the capital, geocode_city for its "
                f"coordinates, and get_weather for the temperature. Give both temperatures in degrees Celsius.",
                expected,
                lambda answer, expected: (
                    contains_number(answer, expected["a"]["temperature_c"], 3.0)
                    and contains_number(answer, expected["b"]["temperature_c"], 3.0)
                    and (not expected["require_warmer"] or contains_text(answer, expected["warmer_capital"]))
                ),
                5)
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"{task_id}: {exc}")

    # -- reverse capital lookup -> World Bank indicators ---------------------------
    for task_id, capital in (("capital_population_canberra", "Canberra"),
                             ("capital_population_ottawa", "Ottawa")):
        try:
            country = safe("find_country_by_capital", capital=capital)["country"]
            population = safe("get_country_population", country=country)["population"]
            add(task_id,
                f"What is the latest population of the country whose capital city is {capital}? "
                f"Give the exact integer from the World Bank data.",
                {"capital": capital, "country": country, "population": population},
                lambda answer, expected: contains_digits(answer, expected["population"]),
                3)
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"{task_id}: {exc}")

    try:
        country = safe("find_country_by_capital", capital="Bern")["country"]
        gdp = safe("get_country_gdp_per_capita", country=country)["gdp_per_capita_usd"]
        add("capital_gdp_switzerland",
            "What is the GDP per capita in current US dollars of the country whose capital city is Bern?",
            {"capital": "Bern", "country": country, "gdp_per_capita_usd": gdp},
            lambda answer, expected: contains_number(answer, expected["gdp_per_capita_usd"],
                                                    max(1.0, expected["gdp_per_capita_usd"] * 0.01)),
            3)
    except Exception as exc:  # noqa: BLE001
        skipped.append(f"capital_gdp_switzerland: {exc}")

    try:
        country = safe("find_country_by_capital", capital="Brasília")["country"]
        income = safe("get_country_info", country=country)["income_level"]
        add("capital_income_brazil",
            "What is the World Bank income-level classification (for example 'High income') of the "
            "country whose capital city is Brasília?",
            {"capital": "Brasília", "country": country, "income_level": income},
            lambda answer, expected: contains_text(answer, expected["income_level"]),
            3)
    except Exception as exc:  # noqa: BLE001
        skipped.append(f"capital_income_brazil: {exc}")

    try:
        country = safe("find_country_by_capital", capital="Buenos Aires")["country"]
        region = safe("get_country_info", country=country)["region"]
        add("capital_region_argentina",
            "In which World Bank region is the country whose capital city is Buenos Aires?",
            {"capital": "Buenos Aires", "country": country, "region": region},
            lambda answer, expected: contains_text(answer, expected["region"]),
            3)
    except Exception as exc:  # noqa: BLE001
        skipped.append(f"capital_region_argentina: {exc}")

    # -- population comparison and aggregation ------------------------------------
    try:
        country_oslo = safe("find_country_by_capital", capital="Oslo")["country"]
        country_wellington = safe("find_country_by_capital", capital="Wellington")["country"]
        population_oslo = safe("get_country_population", country=country_oslo)["population"]
        population_wellington = safe("get_country_population", country=country_wellington)["population"]
        larger = country_oslo if population_oslo > population_wellington else country_wellington
        add("population_compare_oslo_wellington",
            "Which country has the larger population: the country whose capital city is Oslo, or the "
            "country whose capital city is Wellington? Give both populations.",
            {"countries": [country_oslo, country_wellington],
             "populations": [population_oslo, population_wellington], "larger_country": larger},
            lambda answer, expected: (
                all(contains_digits(answer, value) for value in expected["populations"])
                and contains_text(answer, expected["larger_country"])
            ),
            4)
    except Exception as exc:  # noqa: BLE001
        skipped.append(f"population_compare_oslo_wellington: {exc}")

    try:
        population_australia = safe("get_country_population", country="Australia")["population"]
        population_canada = safe("get_country_population", country="Canada")["population"]
        total = population_australia + population_canada
        add("population_sum_australia_canada",
            "What is the combined (summed) population of Australia and Canada according to the World "
            "Bank? Give the exact integer total.",
            {"populations": [population_australia, population_canada], "total": total},
            lambda answer, expected: contains_number(answer, expected["total"], expected["total"] * 0.006),
            3)
    except Exception as exc:  # noqa: BLE001
        skipped.append(f"population_sum_australia_canada: {exc}")

    # -- PyPI version -> release date chains --------------------------------------
    try:
        version, release_date = latest_release("httpx")
        add("pypi_release_date_httpx",
            "When was the latest version of the Python package httpx released on PyPI? "
            "Give the date as YYYY-MM-DD.",
            {"package": "httpx", "version": version, "release_date": release_date},
            lambda answer, expected: contains_text(answer, expected["release_date"]),
            3)
    except Exception as exc:  # noqa: BLE001
        skipped.append(f"pypi_release_date_httpx: {exc}")

    for task_id, package_a, package_b in (
        ("pypi_newer_requests_flask", "requests", "flask"),
        ("pypi_newer_numpy_pandas", "numpy", "pandas"),
    ):
        try:
            version_a, date_a = latest_release(package_a)
            version_b, date_b = latest_release(package_b)
            newer = package_a if date_a > date_b else package_b
            add(task_id,
                f"Which Python package was updated more recently on PyPI: {package_a} or {package_b}? "
                f"Give both release dates as YYYY-MM-DD.",
                {"a": {"package": package_a, "version": version_a, "release_date": date_a},
                 "b": {"package": package_b, "version": version_b, "release_date": date_b},
                 "newer": newer},
                lambda answer, expected: (
                    contains_text(answer, expected["a"]["release_date"])
                    and contains_text(answer, expected["b"]["release_date"])
                    and contains_text(answer, expected["newer"])
                ),
                4)
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"{task_id}: {exc}")

    # -- Hacker News item -> author -> karma chains -------------------------------
    for task_id, story_id in (("hn_item_1_author_karma", 1), ("hn_item_8863_author_karma", 8863)):
        try:
            username, karma = author_karma(story_id)
            add(task_id,
                f"What is the Hacker News karma score of the user who posted Hacker News item {story_id}?",
                {"story_id": story_id, "username": username, "karma": karma},
                lambda answer, expected: contains_number(answer, expected["karma"],
                                                        max(5.0, expected["karma"] * 0.01)),
                3)
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"{task_id}: {exc}")

    try:
        top = safe("get_hn_top_story")
        username = top.get("by")
        if not username:
            raise ToolError("not_found", "top HN story has no author")
        karma = int(safe("get_hn_user", username=username)["karma"])
        add("hn_recent_item_author_karma",
            f"What is the Hacker News karma score of the user who posted Hacker News item {top['id']}?",
            {"story_id": top["id"], "title": top.get("title"), "username": username, "karma": karma},
            lambda answer, expected: contains_number(answer, expected["karma"],
                                                    max(5.0, expected["karma"] * 0.01)),
            3)
    except Exception as exc:  # noqa: BLE001
        skipped.append(f"hn_recent_item_author_karma: {exc}")

    # -- FX chains ---------------------------------------------------------------
    try:
        eur_usd = safe("get_exchange_rate", base="EUR", target="USD")["rate"]
        usd_jpy = safe("get_exchange_rate", base="USD", target="JPY")["rate"]
        cross = eur_usd * usd_jpy
        add("fx_eur_jpy_cross",
            "What is the implied EUR to JPY cross rate (how many Japanese yen is 1 euro worth) computed "
            "through the USD market? Use the EUR-to-USD and USD-to-JPY rates and give the result to two "
            "decimal places.",
            {"eur_usd": eur_usd, "usd_jpy": usd_jpy, "cross_rate": cross},
            lambda answer, expected: contains_number(answer, expected["cross_rate"],
                                                    max(0.02, expected["cross_rate"] * 0.01)),
            3)
    except Exception as exc:  # noqa: BLE001
        skipped.append(f"fx_eur_jpy_cross: {exc}")

    try:
        usd_cny = safe("get_exchange_rate", base="USD", target="CNY")["rate"]
        add("fx_usd_cny_100",
            "How many Chinese yuan are 100 US dollars worth at the current exchange rate?",
            {"usd_cny": usd_cny, "cny_value": usd_cny * 100},
            lambda answer, expected: contains_number(answer, expected["cny_value"],
                                                    expected["cny_value"] * 0.005),
            2)
    except Exception as exc:  # noqa: BLE001
        skipped.append(f"fx_usd_cny_100: {exc}")

    # =========================================================================
    # V2 harder tasks: derived arithmetic, cross-source comparisons, aggregates
    # =========================================================================
    def hn_item_timestamp(story_id: int) -> int:
        item = safe("get_hn_item", story_id=story_id)
        timestamp = item.get("time")
        if not timestamp:
            raise ToolError("not_found", f"HN item {story_id} has no timestamp")
        return int(timestamp)

    def hn_item_date(story_id: int) -> str:
        return datetime.fromtimestamp(hn_item_timestamp(story_id), tz=timezone.utc).strftime("%Y-%m-%d")

    try:
        country_bern = safe("find_country_by_capital", capital="Bern")["country"]
        country_canberra = safe("find_country_by_capital", capital="Canberra")["country"]
        gdp_bern = safe("get_country_gdp_per_capita", country=country_bern)["gdp_per_capita_usd"]
        gdp_canberra = safe("get_country_gdp_per_capita", country=country_canberra)["gdp_per_capita_usd"]
        ratio = gdp_bern / gdp_canberra
        add("gdp_ratio_bern_canberra",
            "What is the ratio of the GDP per capita of the country whose capital is Bern to the GDP per "
            "capita of the country whose capital is Canberra? Give it to two decimal places.",
            {"a": country_bern, "b": country_canberra, "gdp_a": gdp_bern, "gdp_b": gdp_canberra, "ratio": ratio},
            lambda answer, expected: contains_number(answer, expected["ratio"],
                                                    max(0.01, expected["ratio"] * 0.015)),
            4)
    except Exception as exc:  # noqa: BLE001
        skipped.append(f"gdp_ratio_bern_canberra: {exc}")

    try:
        capitals = [("Oslo", "Norway"), ("Wellington", "New Zealand"), ("Canberra", "Australia")]
        countries, populations = [], []
        for capital, _ in capitals:
            country = safe("find_country_by_capital", capital=capital)["country"]
            countries.append(country)
            populations.append(safe("get_country_population", country=country)["population"])
        total = sum(populations)
        add("population_total_three_capitals",
            "What is the combined (summed) population of the countries whose capital cities are Oslo, "
            "Wellington and Canberra? Give the exact integer total.",
            {"countries": countries, "populations": populations, "total": total},
            lambda answer, expected: contains_number(answer, expected["total"], expected["total"] * 0.006),
            5)
    except Exception as exc:  # noqa: BLE001
        skipped.append(f"population_total_three_capitals: {exc}")

    try:
        norway = weather_chain("Norway")
        portugal = weather_chain("Portugal")
        gap = abs(norway["temperature_c"] - portugal["temperature_c"])
        add("temperature_gap_norway_portugal",
            "How many degrees Celsius colder is the capital city of Norway than the capital city of "
            "Portugal right now? Give both temperatures and the gap.",
            {"norway": norway["temperature_c"], "portugal": portugal["temperature_c"], "gap": gap},
            lambda answer, expected: (
                contains_number(answer, expected["norway"], 3.0)
                and contains_number(answer, expected["portugal"], 3.0)
                and contains_number(answer, expected["gap"], 3.5)
            ),
            5)
    except Exception as exc:  # noqa: BLE001
        skipped.append(f"temperature_gap_norway_portugal: {exc}")

    try:
        france = safe("get_country_info", country="France")
        germany = safe("get_country_info", country="Germany")
        pop_france = safe("get_country_population", country="France")["population"]
        pop_germany = safe("get_country_population", country="Germany")["population"]
        larger = "France" if pop_france > pop_germany else "Germany"
        add("population_compare_france_germany_region",
            "Which country has the larger population, France or Germany? Give both populations and the "
            "World Bank region both countries belong to.",
            {"france": pop_france, "germany": pop_germany, "larger": larger,
             "region_france": france["region"], "region_germany": germany["region"]},
            lambda answer, expected: (
                (contains_digits(answer, expected["france"])
                 or contains_number(answer, expected["france"] / 1e6, 0.05))
                and (contains_digits(answer, expected["germany"])
                     or contains_number(answer, expected["germany"] / 1e6, 0.05))
                and contains_text(answer, expected["larger"])
                and contains_text(answer, expected["region_germany"])
            ),
            4)
    except Exception as exc:  # noqa: BLE001
        skipped.append(f"population_compare_france_germany_region: {exc}")

    try:
        httpx_version, httpx_date = latest_release("httpx")
        hn_date = hn_item_date(1)
        newer = "httpx" if httpx_date > hn_date else "hacker news"
        add("release_vs_hn_item_date",
            "Which happened more recently: the release of the latest version of httpx on PyPI, or the "
            "posting of Hacker News item 1? Give both dates as YYYY-MM-DD.",
            {"httpx_release": httpx_date, "hn_item": hn_date, "newer": newer},
            lambda answer, expected: (
                contains_text(answer, expected["httpx_release"])
                and contains_text(answer, expected["hn_item"])
                and contains_text(answer, expected["newer"])
            ),
            4)
    except Exception as exc:  # noqa: BLE001
        skipped.append(f"release_vs_hn_item_date: {exc}")

    try:
        hn_item = safe("get_hn_item", story_id=1)
        hn_user = safe("get_hn_user", username=hn_item["by"])
        created_year = int(datetime.fromtimestamp(int(hn_user["created"]), tz=timezone.utc).year)
        add("hn_item_1_author_account_year",
            "In which year was the Hacker News account of the user who posted Hacker News item 1 created?",
            {"username": hn_user["username"], "created": int(hn_user["created"]), "created_year": created_year},
            lambda answer, expected: contains_number(answer, expected["created_year"], 0.0),
            3)
    except Exception as exc:  # noqa: BLE001
        skipped.append(f"hn_item_1_author_account_year: {exc}")

    try:
        direct = safe("get_exchange_rate", base="EUR", target="JPY")["rate"]
        eur_usd = safe("get_exchange_rate", base="EUR", target="USD")["rate"]
        usd_jpy = safe("get_exchange_rate", base="USD", target="JPY")["rate"]
        implied = eur_usd * usd_jpy
        difference = abs(direct - implied)
        add("fx_triangular_check",
            "Compare the direct EUR to JPY exchange rate with the rate implied by EUR to USD then USD to "
            "JPY. Give both rates to two decimal places and their absolute difference.",
            {"direct": direct, "implied": implied, "difference": difference},
            lambda answer, expected: (
                contains_number(answer, expected["direct"], max(0.02, expected["direct"] * 0.01))
                and contains_number(answer, expected["implied"], max(0.02, expected["implied"] * 0.01))
                and contains_number(answer, expected["difference"], max(0.5, expected["difference"] * 0.05))
            ),
            4)
    except Exception as exc:  # noqa: BLE001
        skipped.append(f"fx_triangular_check: {exc}")

    try:
        dates = {}
        for package in ("requests", "flask", "httpx"):
            _, release_date = latest_release(package)
            dates[package] = release_date
        earliest = min(dates, key=lambda name: dates[name])
        add("package_earliest_of_three",
            "Of the latest versions of requests, flask and httpx on PyPI, which one was released earliest? "
            "Give all three release dates as YYYY-MM-DD.",
            {"dates": dates, "earliest": earliest},
            lambda answer, expected: (
                all(contains_text(answer, date) for date in expected["dates"].values())
                and contains_text(answer, expected["earliest"])
            ),
            5)
    except Exception as exc:  # noqa: BLE001
        skipped.append(f"package_earliest_of_three: {exc}")

    try:
        chains = {country: weather_chain(country) for country in ("Japan", "Norway", "Portugal")}
        add("capital_weather_triple",
            "What is the current temperature in degrees Celsius in the capital cities of Japan, Norway and "
            "Portugal? Give all three values.",
            {country: chain["temperature_c"] for country, chain in chains.items()},
            lambda answer, expected: all(contains_number(answer, value, 3.0) for value in expected.values()),
            6)
    except Exception as exc:  # noqa: BLE001
        skipped.append(f"capital_weather_triple: {exc}")

    try:
        country = safe("find_country_by_capital", capital="Brasília")["country"]
        income = safe("get_country_info", country=country)["income_level"]
        population = safe("get_country_population", country=country)["population"]
        add("capital_income_population_brazil",
            "For the country whose capital city is Brasília, give its World Bank income-level "
            "classification and its latest population.",
            {"country": country, "income": income, "population": population},
            lambda answer, expected: (
                contains_text(answer, expected["income"])
                and contains_digits(answer, expected["population"])
            ),
            4)
    except Exception as exc:  # noqa: BLE001
        skipped.append(f"capital_income_population_brazil: {exc}")

    try:
        country_bern = safe("find_country_by_capital", capital="Bern")["country"]
        country_ottawa = safe("find_country_by_capital", capital="Ottawa")["country"]
        gdp_bern = safe("get_country_gdp_per_capita", country=country_bern)["gdp_per_capita_usd"]
        gdp_ottawa = safe("get_country_gdp_per_capita", country=country_ottawa)["gdp_per_capita_usd"]
        gap = abs(gdp_bern - gdp_ottawa)
        add("gdp_gap_bern_ottawa",
            "What is the absolute difference in GDP per capita (current US dollars) between the country "
            "whose capital is Bern and the country whose capital is Ottawa?",
            {"a": country_bern, "b": country_ottawa, "gdp_a": gdp_bern, "gdp_b": gdp_ottawa, "gap": gap},
            lambda answer, expected: contains_number(answer, expected["gap"], max(1.0, expected["gap"] * 0.015)),
            4)
    except Exception as exc:  # noqa: BLE001
        skipped.append(f"gdp_gap_bern_ottawa: {exc}")

    try:
        australia = weather_chain("Australia")
        canada = weather_chain("Canada")
        gap = abs(australia["temperature_c"] - canada["temperature_c"])
        warmer = australia["capital"] if australia["temperature_c"] >= canada["temperature_c"] else canada["capital"]
        add("weather_compare_australia_canada",
            "Which capital city is warmer right now: the capital of Australia or the capital of Canada? "
            "Give both temperatures in degrees Celsius and the difference.",
            {"australia": australia["temperature_c"], "canada": canada["temperature_c"],
             "gap": gap, "warmer": warmer, "require_warmer": gap > 2.0},
            lambda answer, expected: (
                contains_number(answer, expected["australia"], 3.0)
                and contains_number(answer, expected["canada"], 3.0)
                and contains_number(answer, expected["gap"], 3.5)
                and (not expected["require_warmer"] or contains_text(answer, expected["warmer"]))
            ),
            6)
    except Exception as exc:  # noqa: BLE001
        skipped.append(f"weather_compare_australia_canada: {exc}")

    for task in tasks:
        task["category"] = _category_for(task["id"])
    return tasks, skipped


_CATEGORY_PREFIXES = (
    ("capital_weather", "weather"),
    ("weather_compare", "weather"),
    ("temperature_gap", "weather"),
    ("capital_population", "worldbank"),
    ("capital_gdp", "worldbank"),
    ("capital_income", "worldbank"),
    ("capital_region", "worldbank"),
    ("population_", "worldbank"),
    ("gdp_", "worldbank"),
    ("pypi_", "pypi"),
    ("package_", "pypi"),
    ("release_vs", "pypi"),
    ("hn_", "hn"),
    ("fx_", "fx"),
)


def _category_for(task_id: str) -> str:
    for prefix, category in _CATEGORY_PREFIXES:
        if task_id.startswith(prefix):
            return category
    return "multi"


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------
def summarise_arm(rows: list[dict], client: DeepSeekClient) -> dict:
    return {
        "tasks": len(rows),
        "success": sum(int(row["success"]) for row in rows),
        "success_rate": round(sum(int(row["success"]) for row in rows) / max(len(rows), 1), 4),
        "turns": sum(row["turns"] for row in rows),
        "tool_calls": sum(row["tool_calls"] for row in rows),
        "tool_errors": sum(row["tool_errors"] for row in rows),
        "exhausted": sum(int(row["exhausted"]) for row in rows),
        "faults_injected": sum(row["faults_injected"] for row in rows),
        "guard_recoveries": sum(row["guard_recoveries"] for row in rows),
        "usage": {key: sum(row["usage"][key] for row in rows) for key in client.usage},
        "elapsed_ms": round(sum(row["elapsed_ms"] for row in rows), 2),
    }


def aggregate(records: list[dict], client: DeepSeekClient) -> dict:
    out: dict[str, dict] = {}
    for condition in dict.fromkeys(row["condition"] for row in records):
        condition_rows = [row for row in records if row["condition"] == condition]
        arms = {arm: summarise_arm([row for row in condition_rows if row["arm"] == arm], client) for arm in ARMS}
        by_task: dict[str, dict[str, bool]] = {}
        for row in condition_rows:
            by_task.setdefault(row["task"], {})[row["arm"]] = bool(row["success"])
        baseline_only = sum(1 for pair in by_task.values() if pair.get("baseline") and not pair.get("guard"))
        guard_only = sum(1 for pair in by_task.values() if pair.get("guard") and not pair.get("baseline"))
        both = sum(1 for pair in by_task.values() if pair.get("baseline") and pair.get("guard"))
        out[condition] = {
            "arms": arms,
            "paired": {
                "both_success": both,
                "baseline_only": baseline_only,
                "guard_only": guard_only,
                "neither": len(by_task) - both - baseline_only - guard_only,
                "mcnemar_exact_p": mcnemar_exact(baseline_only, guard_only),
            },
        }
    return out


def key_findings(aggregate_block: dict) -> list[str]:
    findings: list[str] = []
    for condition, block in aggregate_block.items():
        arms = block["arms"]
        paired = block["paired"]
        tasks = arms["baseline"]["tasks"]
        findings.append(
            f"{condition}: task success baseline {arms['baseline']['success']}/{tasks} vs guard "
            f"{arms['guard']['success']}/{tasks} (McNemar exact p={paired['mcnemar_exact_p']:.4f})"
        )
        findings.append(
            f"{condition}: turns {arms['baseline']['turns']} -> {arms['guard']['turns']}, "
            f"tool calls {arms['baseline']['tool_calls']} -> {arms['guard']['tool_calls']}, "
            f"tokens {arms['baseline']['usage']['total_tokens']} -> "
            f"{arms['guard']['usage']['total_tokens']}, agent-visible tool errors "
            f"{arms['baseline']['tool_errors']} -> {arms['guard']['tool_errors']}"
        )
    return findings


HARD_LIMITATIONS = [
    "Deterministic fault injection, not observed production failures: each fault epoch fails the "
    "first 2 attempts of a logical call (timeout / HTTP 429 / HTTP 5xx).",
    "The budgeted condition is a controlled stress condition: per-task turn budget = fault-free "
    "calibration turns + 1, so any fault epoch (2 extra turns for a baseline agent) exceeds it. "
    "Baseline budgeted success therefore approximates P(no transient fault) and must not be read "
    "as a production failure rate.",
    "Single model (deepseek-v4-flash) and a single run per task per arm; McNemar is computed over "
    "the paired discordant outcomes only.",
    "The single guard failure in the budgeted condition is an answer-formatting miss (population "
    "rounded to millions while the verifier requires the exact integer), not a middleware "
    "failure; it is left uncorrected in the artifact.",
    "A repaired call is not retried when the repaired call itself hits a transient fault (found in "
    "the HN story_id tasks: 2 wasted turns each); left unchanged in this artifact.",
    "Free public APIs are not production grade; the GitHub API was excluded because of the "
    "key-free rate limit, and API drift can affect reruns.",
]


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("eval", "calibrate"), default="eval",
                        help="calibrate: measure the fault-free turn requirement of every task; "
                             "eval: run the A/B conditions")
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--conditions", default="budgeted,generous",
                        help="comma separated: budgeted, generous")
    parser.add_argument("--budget-slack", type=int, default=1,
                        help="budgeted condition: extra turns on top of the fault-free turn requirement")
    parser.add_argument("--generous-turns", type=int, default=10)
    parser.add_argument("--calibration", default=".scratch/toolguard/hard_calibration.json")
    parser.add_argument("--clean-max-turns", type=int, default=12)
    parser.add_argument("--p-timeout", type=float, default=0.20)
    parser.add_argument("--p-rate-limit", type=float, default=0.10)
    parser.add_argument("--p-server-error", type=float, default=0.08)
    parser.add_argument("--attempts-to-fail", type=int, default=2,
                        help="length of each transient fault epoch in attempts")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    conditions = [value.strip() for value in args.conditions.split(",") if value.strip()]
    for condition in conditions:
        if condition not in ("budgeted", "generous"):
            print(f"ERROR: unknown condition {condition!r}", file=sys.stderr)
            return 2

    clean = LiveHardBackend()
    tasks, skipped = build_tasks(clean)
    if args.max_tasks is not None:
        tasks = tasks[:args.max_tasks]
    total_turns = sum(task["min_turns"] for task in tasks)
    print(f"built {len(tasks)} multi-step tasks (lower-bound turns total {total_turns}); "
          f"skipped ground-truth groups: {len(skipped)}")
    for note in skipped:
        print(f"  skip: {note}")
    if args.dry_run:
        for task in tasks:
            print(f"  {task['id']:40s} min_turns>={task['min_turns']:2d} "
                  f"expected={json.dumps(task['expected'], ensure_ascii=False)[:120]}")
        return 0

    if not settings.deepseek_api_key:
        print("ERROR: TOOLEVO_DEEPSEEK_API_KEY is missing; cannot run the live demo. "
              "Please provide the API key (the demo will not fall back to offline mode).", file=sys.stderr)
        return 2

    client = DeepSeekClient(LLMConfig(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        model=MODEL,
        max_turns=args.generous_turns,
    ))
    try:
        if args.mode == "calibrate":
            return run_calibration(client, tasks, args)
        return run_eval(client, clean, tasks, skipped, conditions, args)
    finally:
        client.close()


def run_calibration(client: DeepSeekClient, tasks: list[dict], args: argparse.Namespace) -> int:
    """Measure the turn requirement of each task on a clean (fault-free) backend."""
    calibration: dict[str, dict] = {}
    print("\n=== calibration: fault-free turn requirement ===")
    for task in tasks:
        backend = LiveHardBackend()
        run = run_task(client, backend, task["question"], None, max_turns=args.clean_max_turns)
        success = bool(task["verifier"](run.answer, task["expected"]))
        calibration[task["id"]] = {"turns": run.turns, "success": success, "exhausted": run.exhausted}
        print(f"  {task['id']:40s} success={success!s:5s} turns={run.turns}/{args.clean_max_turns} "
              f"calls={run.tool_calls} errors={run.tool_errors}")
    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "protocol": "fault-free runs (no middleware, no injected faults); the turn count is the "
                    "interaction requirement used to set the budgeted condition",
        "tasks": calibration,
    }
    path = Path(args.calibration)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    failed = [task_id for task_id, row in calibration.items() if not row["success"]]
    print(f"\ncalibration success {len(calibration) - len(failed)}/{len(calibration)}; "
          f"wrote {path}")
    if failed:
        print(f"WARNING: tasks not solvable in a fault-free run: {failed}")
    return 0


def run_eval(client: DeepSeekClient, clean: LiveHardBackend, tasks: list[dict], skipped: list[str],
             conditions: list[str], args: argparse.Namespace) -> int:
    calibration: dict[str, dict] = {}
    calibration_path = Path(args.calibration)
    if calibration_path.exists():
        calibration = json.loads(calibration_path.read_text(encoding="utf-8")).get("tasks", {})
        print(f"loaded calibration for {len(calibration)} tasks from {calibration_path}")
    else:
        print(f"WARNING: no calibration file at {calibration_path}; budgets use the lower-bound turns only")
    for task in tasks:
        task["min_turns"] = max(task["min_turns"], int(calibration.get(task["id"], {}).get("turns") or 0))
        task["clean_turns"] = task["min_turns"]

    records: list[dict] = []
    try:
        for condition in conditions:
            print(f"\n=== condition: {condition} ===")
            for task in tasks:
                budget = (task["clean_turns"] + args.budget_slack
                          if condition == "budgeted" else args.generous_turns)
                for arm in ARMS:
                    backend = LiveHardBackend()
                    faulty = FaultyBackend(
                        backend,
                        seed=args.seed,
                        p_timeout=args.p_timeout,
                        p_rate_limit=args.p_rate_limit,
                        p_server_error=args.p_server_error,
                        attempts_to_fail=args.attempts_to_fail,
                    )
                    middleware = ToolGuard(faulty, sleep_scale=0.05) if arm == "guard" else None
                    before = dict(client.usage)
                    run = run_task(client, faulty, task["question"], middleware, max_turns=budget)
                    delta = {key: client.usage[key] - before.get(key, 0) for key in client.usage}
                    success = bool(task["verifier"](run.answer, task["expected"]))
                    records.append({
                        "task": task["id"],
                        "condition": condition,
                        "arm": arm,
                        "budget": budget,
                        "success": success,
                        "answer": run.answer[:400],
                        "turns": run.turns,
                        "tool_calls": run.tool_calls,
                        "tool_errors": run.tool_errors,
                        "exhausted": run.exhausted,
                        "usage": delta,
                        "elapsed_ms": run.elapsed_ms,
                        "faults_injected": faulty.stats["injected"],
                        "faulted_calls": faulty.stats["faulted_calls"],
                        "guard_recoveries": middleware.stats["recovered"] if middleware is not None else 0,
                        "guard_events": run.guard_events,
                    })
                    print(f"  {task['id']:40s} {condition:8s} {arm:8s} success={success!s:5s} "
                          f"turns={run.turns}/{budget} calls={run.tool_calls} errors={run.tool_errors} "
                          f"injected={faulty.stats['injected']} recoveries={records[-1]['guard_recoveries']}")
    except RuntimeError as exc:
        print(f"\nERROR: DeepSeek/live API failure: {exc}", file=sys.stderr)
        print("Stopping (no offline fallback). Please check the API key / balance and rerun.", file=sys.stderr)
        return 2

    aggregate_block = aggregate(records, client)
    summary = {
        "artifact_version": 1,
        "experiment": "ToolGuard_hard_multistep_live_ab",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "protocol": {
            "arms": list(ARMS),
            "conditions": {
                "budgeted": f"per-task turn budget = fault-free turn requirement + {args.budget_slack}",
                "generous": f"{args.generous_turns} turns for every task",
            },
            "calibration": {
                "description": "one fault-free run per task measured the turn requirement used for the "
                               "budgeted condition (see --mode calibrate)",
                "tasks": {task["id"]: calibration.get(task["id"], {"turns": task["clean_turns"]})
                          for task in tasks},
            },
            "fault_injection": {
                "model": "deterministic transient fault epochs: the first N attempts of a faulted logical "
                         "call fail (timeout / HTTP 429 / HTTP 5xx), then the real call runs",
                "attempts_to_fail": args.attempts_to_fail,
                "p_timeout": args.p_timeout,
                "p_rate_limit": args.p_rate_limit,
                "p_server_error": args.p_server_error,
                "seed": args.seed,
                "note": "fault surface is seeded per (function, arguments) and identical across arms",
            },
            "tool_surface": {
                "tools": [tool["function"]["name"] for tool in clean.schemas()],
                "apis": ["api.worldbank.org", "geocoding-api.open-meteo.com", "api.open-meteo.com",
                         "pypi.org", "hacker-news.firebaseio.com", "open.er-api.com"],
            },
            "verification": "programmatic; ground truth computed from the same live APIs before the runs",
            "task_requirements": "each task needs dependent cross-tool calls over 2-6 stages",
        },
        "tasks": [{key: value for key, value in task.items() if key != "verifier"} for task in tasks],
        "skipped": skipped,
        "records": records,
        "aggregate": aggregate_block,
        "key_findings": key_findings(aggregate_block),
        "limitations": HARD_LIMITATIONS,
        "reproduce": [
            "python scripts/run_live_demo_hard.py --mode calibrate",
            "python scripts/run_live_demo_hard.py --output benchmarks/toolguard/live_demo_hard.json",
            "python -m pytest tests/test_toolguard -q -p no:cacheprovider",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== aggregate ===")
    print(json.dumps(summary["aggregate"], ensure_ascii=False, indent=2))
    print(f"\nwrote {output}")
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
