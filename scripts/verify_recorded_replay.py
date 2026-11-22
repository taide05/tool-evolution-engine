"""Validate that the recorded backend reproduces the released real tool responses.

For a sample of real tool calls from the public trajectory release, we re-issue the
call against the recorded backend (deterministic cache) and compare with the
observation actually recorded in the trajectory.  This is the fidelity baseline
for every later repair experiment.

Usage:
    python scripts/verify_recorded_replay.py --sample 3000
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tool_evolution.toolguard.backend import RecordedBackend  # noqa: E402
from tool_evolution.toolguard.cache_index import CacheIndex  # noqa: E402


def load_calls(path: Path, sample: int | None, seed: int) -> list[dict]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8")]
    if sample is not None and sample < len(rows):
        rows = random.Random(seed).sample(rows, sample)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calls", default=".scratch/toolguard/tool_calls.jsonl")
    parser.add_argument("--cache", default=".scratch/stb/cache/tool_response_cache")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    index = CacheIndex.from_root(Path(args.cache))
    backend = RecordedBackend(Path(args.cache), index)
    rows = load_calls(Path(args.calls), args.sample, args.seed)
    print(f"cache API entries: {len(index)} | unique function names: {len(index.by_function)} | calls sampled: {len(rows)}")

    stats: Counter[str] = Counter()
    mismatches: list[dict] = []
    for row in rows:
        if "__raw__" in (row.get("arguments") or {}):
            stats["skipped_raw_arguments"] += 1
            continue
        if not index.candidates(row["tool_name"]):
            stats["unknown_function"] += 1
            continue
        stats["resolved"] += 1
        result = backend.call(row["tool_name"], row["arguments"])
        recorded_ok = bool(row["observation_ok"])
        if result.status == "miss":
            stats["replay_miss"] += 1
            if not recorded_ok:
                stats["replay_miss_on_recorded_error"] += 1
            continue
        stats["replay_hit"] += 1
        if result.ok == recorded_ok:
            stats["parity"] += 1
        else:
            stats["parity_mismatch"] += 1
            mismatches.append({"tool": row["tool_name"], "recorded_ok": recorded_ok,
                               "replay_ok": result.ok, "recorded_error": row["observation_error"],
                               "replay_error": result.error, "args": row["arguments"]})
        if not recorded_ok and not result.ok:
            if (row["observation_error"] or "").strip() == (result.error or "").strip():
                stats["error_text_exact"] += 1
            else:
                stats["error_text_diff"] += 1

    total = max(len(rows), 1)
    print("\n--- replay fidelity ---")
    for key in ("resolved", "unknown_function", "skipped_raw_arguments", "replay_hit", "replay_miss",
                "replay_miss_on_recorded_error", "parity", "parity_mismatch",
                "error_text_exact", "error_text_diff"):
        print(f"{key:28s} {stats[key]:6d}  ({100 * stats[key] / total:5.2f}% of sampled)")
    if stats["resolved"]:
        print(f"hit rate on resolved calls: {100 * stats['replay_hit'] / stats['resolved']:.2f}%")
    if stats["replay_hit"]:
        print(f"ok/error parity on hits: {100 * stats['parity'] / stats['replay_hit']:.2f}%")
    print(f"backend stats: {backend.stats}")
    if mismatches:
        print(f"\nexample mismatches ({min(5, len(mismatches))}):")
        for item in mismatches[:5]:
            print(" ", json.dumps(item, ensure_ascii=False)[:400])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
