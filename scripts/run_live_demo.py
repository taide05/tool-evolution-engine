"""Live end-to-end demo: ToolGuard middleware vs baseline, real public APIs + DeepSeek.

The agent is a real DeepSeek tool-calling loop over six real public HTTP APIs.  The
only synthetic element is deterministic transient-fault injection (timeouts / rate
limits on the first attempt of each logical call), which models the failure class the
offline analysis found (96 server/transient errors) and gives the middleware something
to recover from.  Everything else -- LLM calls, HTTP requests, task verification -- is
live.  If the DeepSeek API is unavailable the demo stops and reports, it never falls
back to offline/mock data.

Usage:
    python scripts/run_live_demo.py --dry-run
    python scripts/run_live_demo.py --max-tasks 2
    python scripts/run_live_demo.py
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import unicodedata
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tool_evolution.toolguard.agent import DeepSeekClient, LLMConfig, run_task  # noqa: E402
from tool_evolution.toolguard.guard import ToolGuard  # noqa: E402
from tool_evolution.toolguard.live import FaultyBackend, LiveBackend, ToolError  # noqa: E402
from tool_evolution.utils.config import settings  # noqa: E402

DEFAULT_OUTPUT = ".scratch/toolguard/live_demo.json"
MODEL = "deepseek-v4-flash"


# ---------------------------------------------------------------------------
# verification helpers
# ---------------------------------------------------------------------------
def _norm(text: str) -> str:
    folded = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", " ", folded.lower()).strip()


def _digits(text: str) -> str:
    return re.sub(r"\D", "", str(text))


def _numbers(text: str) -> list[float]:
    return [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", str(text).replace(",", ""))]


def contains_text(answer: str, expected: str) -> bool:
    return _norm(expected)[:40] in _norm(answer)


def contains_number(answer: str, expected: float, tolerance: float = 0.0) -> bool:
    return any(abs(value - expected) <= tolerance for value in _numbers(answer))


def contains_digits(answer: str, expected: int | str) -> bool:
    return _digits(expected) in _digits(answer)


# ---------------------------------------------------------------------------
# task construction (ground truth computed live, before the agent runs)
# ---------------------------------------------------------------------------
def build_tasks(backend: LiveBackend, max_tasks: int | None) -> tuple[list[dict], list[str]]:
    tasks: list[dict] = []
    skipped: list[str] = []

    def add(task_id: str, question: str, expected: dict, verifier) -> None:
        tasks.append({"id": task_id, "question": question, "expected": expected, "verifier": verifier})

    def safe(tool: str, **arguments):
        last_error = None
        for _ in range(3):
            result = backend.call(tool, arguments)
            if result.ok:
                return result.response
            last_error = result.error
        raise ToolError("request_error", f"{tool} failed: {last_error}")

    try:
        japan = safe("get_country_info", country="Japan")
        add("country_capital_japan", "What is the capital city of Japan?",
            {"capital": japan["capital"]}, lambda a, e: contains_text(a, e["capital"]))
        australia = safe("get_country_info", country="Australia")
        add("country_capital_australia", "What is the capital city of Australia?",
            {"capital": australia["capital"]}, lambda a, e: contains_text(a, e["capital"]))
        add("country_capital_pair", "What are the capital cities of Japan and Australia?",
            {"capitals": [japan["capital"], australia["capital"]]},
            lambda a, e: all(contains_text(a, c) for c in e["capitals"]))
        germany = safe("get_country_info", country="Germany")
        add("country_capital_germany", "What is the capital city of Germany?",
            {"capital": germany["capital"]}, lambda a, e: contains_text(a, e["capital"]))
    except Exception as exc:  # noqa: BLE001
        skipped.append(f"country tasks: {exc}")

    try:
        numpy = safe("get_package_version", package="numpy")
        requests = safe("get_package_version", package="requests")
        add("pypi_numpy_version", "What is the latest released version of the Python package numpy?",
            {"version": numpy["version"]}, lambda a, e: contains_text(a, e["version"]))
        add("pypi_requests_version", "What is the latest released version of the Python package requests?",
            {"version": requests["version"]}, lambda a, e: contains_text(a, e["version"]))
        add("pypi_newer_package",
            "Compare the latest versions of numpy and requests on PyPI. Which package has the higher version number?",
            {"numpy": numpy["version"], "requests": requests["version"]},
            lambda a, e: contains_text(a, e["numpy"]) and contains_text(a, e["requests"]))
    except Exception as exc:  # noqa: BLE001
        skipped.append(f"pypi tasks: {exc}")

    try:
        berlin = safe("get_weather", latitude=52.52, longitude=13.41)
        tokyo = safe("get_weather", latitude=35.68, longitude=139.69)
        add("weather_berlin", "What is the current temperature in Berlin (latitude 52.52, longitude 13.41), in Celsius?",
            {"temperature": berlin["temperature_c"]}, lambda a, e: contains_number(a, e["temperature"], 1.5))
        add("weather_tokyo", "What is the current temperature in Tokyo (latitude 35.68, longitude 139.69), in Celsius?",
            {"temperature": tokyo["temperature_c"]}, lambda a, e: contains_number(a, e["temperature"], 1.5))
        add("weather_compare", "Compare the current temperatures in Berlin and Tokyo. Give both values in Celsius.",
            {"temperatures": [berlin["temperature_c"], tokyo["temperature_c"]]},
            lambda a, e: all(contains_number(a, t, 1.5) for t in e["temperatures"]))
    except Exception as exc:  # noqa: BLE001
        skipped.append(f"weather tasks: {exc}")

    try:
        stories = []
        for story_id in (8863, 1, 2, 3, 5, 42):
            result = backend.call("get_hn_item", {"story_id": story_id})
            if result.ok and result.response and result.response.get("title"):
                stories.append(result.response)
            if len(stories) >= 2:
                break
        if len(stories) < 2:
            raise ToolError("not_found", "could not find two Hacker News stories with titles")
        for index, story in enumerate(stories[:2], start=1):
            add(f"hn_story_{index}", f"What is the title of Hacker News item {story['id']}?",
                {"title": story["title"]}, lambda a, e: contains_text(a, e["title"]))
    except Exception as exc:  # noqa: BLE001
        skipped.append(f"hacker news tasks: {exc}")

    try:
        repo = safe("get_github_repo", owner="python", repo="cpython")
        add("github_stars", "How many stars does the GitHub repository python/cpython have?",
            {"stars": repo["stargazers_count"]}, lambda a, e: contains_digits(a, e["stars"]))
    except Exception as exc:  # noqa: BLE001
        skipped.append(f"github tasks: {exc}")

    try:
        fx = safe("get_exchange_rate", base="USD", target="CNY")
        add("fx_usd_cny", "What is the current exchange rate from US dollar to Chinese yuan (how many CNY per 1 USD)?",
            {"rate": fx["rate"]}, lambda a, e: contains_number(a, e["rate"], max(0.02, e["rate"] * 0.01)))
        for base, target in (("EUR", "JPY"), ("USD", "JPY"), ("GBP", "USD")):
            quote = safe("get_exchange_rate", base=base, target=target)
            add(f"fx_{base.lower()}_{target.lower()}",
                f"What is the current exchange rate from {base} to {target} (how many {target} per 1 {base})?",
                {"rate": quote["rate"]},
                lambda a, e: contains_number(a, e["rate"], max(0.02, e["rate"] * 0.01)))
    except Exception as exc:  # noqa: BLE001
        skipped.append(f"fx tasks: {exc}")

    for country in ("France", "Canada", "Brazil", "India"):
        try:
            info = safe("get_country_info", country=country)
            add(f"country_capital_{country.lower()}", f"What is the capital city of {country}?",
                {"capital": info["capital"]}, lambda a, e: contains_text(a, e["capital"]))
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"country {country}: {exc}")

    for package in ("pandas", "django", "flask"):
        try:
            info = safe("get_package_version", package=package)
            add(f"pypi_{package}_version", f"What is the latest released version of the Python package {package}?",
                {"version": info["version"]}, lambda a, e: contains_text(a, e["version"]))
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"pypi {package}: {exc}")

    for city, latitude, longitude in (("Paris", 48.85, 2.35), ("New York", 40.71, -74.01), ("Sydney", -33.87, 151.21)):
        try:
            weather = safe("get_weather", latitude=latitude, longitude=longitude)
            add(f"weather_{_norm(city).replace(' ', '_')}",
                f"What is the current temperature in {city} (latitude {latitude}, longitude {longitude}), in Celsius?",
                {"temperature": weather["temperature_c"]},
                lambda a, e: contains_number(a, e["temperature"], 1.5))
        except Exception as exc:  # noqa: BLE001
            skipped.append(f"weather {city}: {exc}")

    if max_tasks is not None:
        tasks = tasks[:max_tasks]
    return tasks, skipped


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------
def mcnemar_exact(baseline_only: int, guard_only: int) -> float:
    """Two-sided exact binomial p-value for discordant pairs."""
    n = baseline_only + guard_only
    if n == 0:
        return 1.0
    k = min(baseline_only, guard_only)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--p-timeout", type=float, default=0.22)
    parser.add_argument("--p-rate-limit", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    clean = LiveBackend()
    tasks, skipped = build_tasks(clean, args.max_tasks)
    print(f"built {len(tasks)} tasks (skipped ground-truth groups: {len(skipped)})")
    for note in skipped:
        print(f"  skip: {note}")
    if args.dry_run:
        for task in tasks:
            print(f"  {task['id']:26s} expected={task['expected']}")
        return 0

    if not settings.deepseek_api_key:
        print("ERROR: TOOLEVO_DEEPSEEK_API_KEY is missing; cannot run the live demo. "
              "Please provide the API key (the demo will not fall back to offline mode).", file=sys.stderr)
        return 2

    client = DeepSeekClient(LLMConfig(
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        model=MODEL,
        max_turns=args.max_turns,
    ))
    records: list[dict] = []
    try:
        for task in tasks:
            for arm in ("baseline", "guard"):
                faulty = FaultyBackend(
                    LiveBackend(), seed=args.seed,
                    p_timeout=args.p_timeout, p_rate_limit=args.p_rate_limit,
                )
                middleware = ToolGuard(faulty, sleep_scale=0.05) if arm == "guard" else None
                before = dict(client.usage)
                run = run_task(client, faulty, task["question"], middleware)
                delta = {key: client.usage[key] - before.get(key, 0) for key in client.usage}
                success = bool(task["verifier"](run.answer, task["expected"]))
                records.append({
                    "task": task["id"],
                    "arm": arm,
                    "success": success,
                    "answer": run.answer[:300],
                    "turns": run.turns,
                    "tool_calls": run.tool_calls,
                    "tool_errors": run.tool_errors,
                    "exhausted": run.exhausted,
                    "usage": delta,
                    "elapsed_ms": run.elapsed_ms,
                    "faults_injected": faulty.stats["injected"],
                    "guard_recoveries": middleware.stats["recovered"] if middleware is not None else 0,
                    "guard_events": run.guard_events,
                })
                print(f"  {task['id']:26s} {arm:8s} success={success!s:5s} turns={run.turns} "
                      f"tool_calls={run.tool_calls} errors={run.tool_errors} "
                      f"recoveries={records[-1]['guard_recoveries']}")
    except RuntimeError as exc:
        print(f"\nERROR: DeepSeek/live API failure: {exc}", file=sys.stderr)
        print("Stopping (no offline fallback). Please check the API key / balance and rerun.", file=sys.stderr)
        return 2
    finally:
        client.close()

    # aggregate
    per_arm: dict[str, dict] = {}
    for arm in ("baseline", "guard"):
        rows = [row for row in records if row["arm"] == arm]
        per_arm[arm] = {
            "tasks": len(rows),
            "success": sum(row["success"] for row in rows),
            "success_rate": round(sum(row["success"] for row in rows) / max(len(rows), 1), 4),
            "tool_calls": sum(row["tool_calls"] for row in rows),
            "tool_errors": sum(row["tool_errors"] for row in rows),
            "turns": sum(row["turns"] for row in rows),
            "usage": {key: sum(row["usage"][key] for row in rows) for key in client.usage},
            "elapsed_ms": round(sum(row["elapsed_ms"] for row in rows), 2),
            "exhausted": sum(row["exhausted"] for row in rows),
            "recoveries": sum(row["guard_recoveries"] for row in rows),
        }
    by_task: dict[str, dict[str, bool]] = {}
    for row in records:
        by_task.setdefault(row["task"], {})[row["arm"]] = row["success"]
    baseline_only = sum(1 for pair in by_task.values() if pair.get("baseline") and not pair.get("guard"))
    guard_only = sum(1 for pair in by_task.values() if pair.get("guard") and not pair.get("baseline"))
    both = sum(1 for pair in by_task.values() if pair.get("baseline") and pair.get("guard"))

    serializable_tasks = [{key: value for key, value in task.items() if key != "verifier"} for task in tasks]
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "protocol": {
            "arms": ["baseline", "guard"],
            "max_turns": args.max_turns,
            "fault_injection": {"p_timeout": args.p_timeout, "p_rate_limit": args.p_rate_limit, "seed": args.seed,
                                "note": "first attempt of each unique logical call may be faulted once"},
            "apis": ["api.worldbank.org", "pypi.org", "api.open-meteo.com",
                     "api.github.com", "hacker-news.firebaseio.com", "open.er-api.com"],
            "verification": "programmatic, expected values computed from the same live APIs before the runs",
        },
        "tasks": serializable_tasks,
        "skipped": skipped,
        "records": records,
        "aggregate": per_arm,
        "paired": {
            "both_success": both,
            "baseline_only": baseline_only,
            "guard_only": guard_only,
            "mcnemar_exact_p": mcnemar_exact(baseline_only, guard_only),
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== aggregate ===")
    print(json.dumps({"per_arm": per_arm, "paired": summary["paired"]}, ensure_ascii=False, indent=2))
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
