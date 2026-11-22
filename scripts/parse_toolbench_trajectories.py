"""Parse public StableToolBench baseline trajectories into ToolGuard replay records.

Input:  ``data_baselines/<method>/<group>/<query_id>.json`` (Apache-2.0 release,
        downloaded separately; not committed to this repo).
Output: JSONL tool-call records + per-query tool schemas + aggregate statistics.

Each baseline answer file contains ``answer_generation.train_messages``: a list of
full conversation attempts (one per try).  We flatten every attempt into tool-call
records so real failures (including failed retries) are preserved.

Usage:
    python scripts/parse_toolbench_trajectories.py \
        --input .scratch/stb/baselines/data_baselines \
        --output-dir data/toolguard
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

# Error-message taxonomy used by the middleware evaluation.
_ERROR_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("missing_required", re.compile(r"missing required param|required parameter|required argument", re.I)),
    ("unexpected_kwarg", re.compile(r"unexpected keyword|got an unexpected|invalid keyword", re.I)),
    ("no_such_function", re.compile(r"no such function|not a valid function|unknown function", re.I)),
    ("timeout", re.compile(r"timeout|timed out|time out", re.I)),
    ("rate_limit", re.compile(r"rate limit|too many requests|exceeded.*limit", re.I)),
    ("unauthorized", re.compile(r"unauthorized|not subscribed|blocked user|credentials|forbidden|401|403", re.I)),
    ("not_found", re.compile(r"not found|does not exist|no data", re.I)),
    ("parse_error", re.compile(r"parse|invalid json|json decode", re.I)),
    ("parameter", re.compile(r"parameter", re.I)),
]


def classify_error(message: str) -> str:
    for name, pattern in _ERROR_PATTERNS:
        if pattern.search(message):
            return name
    return "other_error"


def split_model(method: str) -> tuple[str, str]:
    """'gpt-4-0613_dfs' -> ('gpt-4-0613', 'dfs')."""
    if method.endswith("_cot"):
        return method[: -len("_cot")], "cot"
    if method.endswith("_dfs"):
        return method[: -len("_dfs")], "dfs"
    return method, "unknown"


def parse_observation(content: str) -> tuple[str | None, Any]:
    """Return (error_message, response) from a tool observation payload.

    Some released observations contain raw control characters, so fall back to
    non-strict JSON and finally to a regex extraction of the ``error`` field.
    Unparseable payloads are treated as successful responses (real errors are
    always carried in the ``error`` field of the virtual API server).
    """
    text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
    payload: Any = None
    for strict in (True, False):
        try:
            payload = json.loads(text, strict=strict)
            break
        except (json.JSONDecodeError, TypeError):
            continue
    if isinstance(payload, dict):
        error = payload.get("error")
        return (str(error) if error else None), payload.get("response")
    if payload is not None:
        return None, payload
    match = re.search(r'"error"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    if match:
        error = match.group(1).replace('\\"', '"').replace("\\n", " ").strip()
        return (error or None), text[:500]
    return None, text[:500]


def iter_answer_files(input_dir: Path, limit_files: int | None = None) -> Iterable[tuple[str, str, Path]]:
    methods = sorted(p for p in input_dir.iterdir() if p.is_dir())
    seen = 0
    for method_dir in methods:
        for group_dir in sorted(p for p in method_dir.iterdir() if p.is_dir()):
            for path in sorted(group_dir.glob("*.json")):
                yield method_dir.name, group_dir.name, path
                seen += 1
                if limit_files is not None and seen >= limit_files:
                    return


def load_json_loose(path: Path) -> dict:
    """Load a baseline answer file; a few releases contain raw control chars in strings."""
    text = path.read_text(encoding="utf-8", errors="replace")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(text, strict=False)


def parse_file(method: str, group: str, path: Path, schemas: dict[str, list[dict]]) -> tuple[list[dict], list[dict], dict]:
    data = load_json_loose(path)
    ag = data.get("answer_generation") or {}
    query_id = str(ag.get("query_id") or path.stem.split("_")[0])
    query = str(ag.get("query") or "")
    functions = ag.get("function") or []
    model, strategy = split_model(method)
    if functions and f"{group}/{query_id}" not in schemas:
        schemas[f"{group}/{query_id}"] = functions
    func_names = [f.get("name") for f in functions if isinstance(f, dict)]

    rows: list[dict] = []
    attempts = ag.get("train_messages") or []
    for try_index, messages in enumerate(attempts):
        if not isinstance(messages, list):
            continue
        step = 0
        for msg_index, msg in enumerate(messages):
            if not isinstance(msg, dict):
                continue
            call = msg.get("function_call")
            if not call:
                continue
            name = call.get("name") or ""
            raw_args = call.get("arguments")
            try:
                arguments = raw_args if isinstance(raw_args, dict) else json.loads(raw_args or "{}")
            except (json.JSONDecodeError, TypeError):
                arguments = {"__raw__": str(raw_args)[:500]}
            # Observation is the next 'function' message with the same name.
            observation = None
            for nxt in messages[msg_index + 1:]:
                if isinstance(nxt, dict) and nxt.get("role") == "function":
                    observation = nxt
                    break
                if isinstance(nxt, dict) and nxt.get("role") == "assistant":
                    break
            if name == "Finish" or observation is None:
                continue
            step += 1
            error, response = parse_observation(observation.get("content", ""))
            rows.append(
                {
                    "trace_id": f"{method}/{group}/{query_id}/try{try_index}",
                    "method": method,
                    "model": model,
                    "strategy": strategy,
                    "group": group,
                    "query_id": query_id,
                    "query": query,
                    "try_index": try_index,
                    "try_count": len(attempts),
                    "step": step,
                    "tool_name": name,
                    "arguments": arguments,
                    "observation_error": error,
                    "observation_response": response if isinstance(response, (dict, list)) else str(response)[:500] if response is not None else None,
                    "observation_ok": error is None,
                    "available_tools": func_names,
                    "valid_data": bool(ag.get("valid_data")),
                }
            )
    meta = {
        "trace_id": f"{method}/{group}/{query_id}",
        "method": method,
        "model": model,
        "strategy": strategy,
        "group": group,
        "query_id": query_id,
        "attempts": len(attempts),
        "valid_data": bool(ag.get("valid_data")),
        "final_answer": str(ag.get("final_answer") or "")[:1000],
    }
    return rows, [meta], schemas


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=".scratch/stb/baselines/data_baselines")
    parser.add_argument("--output-dir", default="data/toolguard")
    parser.add_argument("--limit-files", type=int, default=None)
    args = parser.parse_args()

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"input dir not found: {input_dir}", file=sys.stderr)
        return 2
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    calls_path = output_dir / "tool_calls.jsonl"
    traces_path = output_dir / "traces.jsonl"
    schemas_path = output_dir / "tool_schemas.json"
    schemas: dict[str, list[dict]] = {}
    stats: Counter[str] = Counter()
    by_method: dict[str, Counter[str]] = defaultdict(Counter)
    by_group: dict[str, Counter[str]] = defaultdict(Counter)
    errors_by_kind: Counter[str] = Counter()
    error_examples: dict[str, str] = {}
    tool_counts: Counter[str] = Counter()

    with calls_path.open("w", encoding="utf-8") as calls_fh, traces_path.open("w", encoding="utf-8") as traces_fh:
        for method, group, path in iter_answer_files(input_dir, args.limit_files):
            try:
                rows, metas, _ = parse_file(method, group, path, schemas)
            except Exception as exc:  # noqa: BLE001 - one corrupt release file must not abort the batch
                stats["skipped_files"] += 1
                print(f"skip {path}: {type(exc).__name__}: {exc}", file=sys.stderr)
                continue
            for meta in metas:
                stats["traces"] += 1
                stats["valid_traces"] += int(meta["valid_data"])
                by_method[method]["traces"] += 1
                by_group[group]["traces"] += 1
                traces_fh.write(json.dumps(meta, ensure_ascii=False) + "\n")
            for row in rows:
                calls_fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats["calls"] += 1
                tool_counts[row["tool_name"]] += 1
                if row["observation_ok"]:
                    stats["ok_calls"] += 1
                else:
                    stats["error_calls"] += 1
                    kind = classify_error(row["observation_error"] or "")
                    errors_by_kind[kind] += 1
                    error_examples.setdefault(kind, (row["observation_error"] or "")[:300])
                    by_method[method]["error_calls"] += 1
                    by_group[group]["error_calls"] += 1
                by_method[method]["calls"] += 1
                by_group[group]["calls"] += 1

    schemas_path.write_text(json.dumps(schemas, ensure_ascii=False), encoding="utf-8")
    summary = {
        "skipped_files": stats["skipped_files"],
        "traces": stats["traces"],
        "valid_traces": stats["valid_traces"],
        "calls": stats["calls"],
        "ok_calls": stats["ok_calls"],
        "error_calls": stats["error_calls"],
        "error_rate": round(stats["error_calls"] / max(stats["calls"], 1), 4),
        "error_kinds": dict(errors_by_kind.most_common()),
        "error_examples": error_examples,
        "top_tools": dict(tool_counts.most_common(20)),
        "by_method": {k: dict(v) for k, v in by_method.items()},
        "by_group": {k: dict(v) for k, v in by_group.items()},
    }
    (output_dir / "stats.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("error_examples", "top_tools", "by_method", "by_group")}, ensure_ascii=False, indent=2))
    print(f"wrote {summary['calls']} calls, {summary['traces']} traces, {len(schemas)} query schema sets -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
