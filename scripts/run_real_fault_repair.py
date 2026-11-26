"""Replay real online fault traces through ToolGuard and verify repairs online.

2026-11-25 evidence-chain follow-up: the live A/B experiments use deterministic *injected*
faults for reproducibility.  This runner instead feeds the middleware **genuine error
responses captured from the real public APIs** (``collect_real_fault_traces.py``) and asks
the guard to recover.  When the guard repairs a call, the repaired call is executed against
the real API, so a recovered result is verified online (not simulated).

No LLM and no paid API is used; only the public APIs that the live demo already uses.

Usage
-----
    python scripts/run_real_fault_repair.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tool_evolution.toolguard.backend import CallResult  # noqa: E402
from tool_evolution.toolguard.guard import ToolGuard  # noqa: E402
from tool_evolution.toolguard.live_hard import LiveHardBackend  # noqa: E402

DEFAULT_TRACES = "benchmarks/toolguard/real_traces/real_fault_traces.json"
DEFAULT_OUTPUT = "benchmarks/toolguard/real_traces/real_fault_repair_run.json"


class TraceReplayBackend:
    """Returns the captured real error once per matching call, then calls the real API."""

    def __init__(self, traces: list[dict]):
        self.real = LiveHardBackend(hn_schema="consistent")
        self._pending = {_key(trace["tool"], trace["arguments"]): trace for trace in traces}
        self.calls: list[dict] = []

    def schemas(self):
        return self.real.schemas()

    def available_tool_names(self):
        return [tool["function"]["name"] for tool in self.real.schemas()]

    def call(self, function_name: str, arguments: dict):
        key = _key(function_name, arguments)
        trace = self._pending.pop(key, None)
        self.calls.append({"tool": function_name, "arguments": dict(arguments),
                           "matched_trace": trace["trace_id"] if trace else None})
        if trace is not None:
            return CallResult("error", error=f"{trace['error_kind']}: {trace['error_message']}")
        return self.real.call(function_name, arguments)


def _key(tool: str, arguments: dict) -> str:
    return f"{tool}:{json.dumps(arguments, sort_keys=True, ensure_ascii=False, default=str)}"


def _event_summary(events) -> list[dict]:
    summary = []
    for event in events:
        summary.append({
            "rule": getattr(event, "rule", None),
            "schema_validated": getattr(event, "schema_validated", None),
            "schema_errors": list(getattr(event, "schema_errors", []) or []),
            "recovered": getattr(event, "recovered", None),
        })
    return summary


TRANSIENT_KINDS = ("timeout", "rate_limit", "server_error")


def _error_kind_prefix(error: str | None) -> str:
    text = (error or "").strip().lower()
    for kind in TRANSIENT_KINDS:
        if text.startswith(kind):
            return kind
    return ""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", default=DEFAULT_TRACES)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--verification-attempts", type=int, default=3,
                        help="retry the whole replay when the repaired call hits a transient "
                             "live-API failure (timeout/request_error/rate_limit)")
    args = parser.parse_args()

    traces_path = REPO / args.traces
    payload = json.loads(traces_path.read_text(encoding="utf-8"))
    traces = [trace for trace in payload["records"] if trace["outcome"] == "error"]
    records: list[dict] = []
    for trace in traces:
        result = None
        backend = None
        guard = None
        attempts_total = 0
        replay_attempts = 0
        for replay in range(1, args.verification_attempts + 1):
            replay_attempts = replay
            backend = TraceReplayBackend([trace])
            guard = ToolGuard(backend, sleep_scale=0.0)
            result = guard.call(trace["tool"], trace["arguments"])
            attempts_total += len(backend.calls)
            if result.ok:
                break
            if _error_kind_prefix(result.error) and replay < args.verification_attempts:
                time.sleep(2.0 * replay)  # live API unhealthy: retry the online verification
                continue
            break
        repaired_calls = [call for call in backend.calls if call["matched_trace"] is None]
        if result.ok:
            verification_status = "success"
        elif _error_kind_prefix(result.error):
            verification_status = "api_unavailable"
        else:
            verification_status = "still_error"
        record = {
            "trace_id": trace["trace_id"],
            "tool": trace["tool"],
            "bad_arguments": trace["arguments"],
            "real_error_kind": trace["error_kind"],
            "real_error_message": trace["error_message"],
            "guard_outcome": "recovered" if result.ok else "not_recovered",
            "verification_status": verification_status,
            "replay_attempts": replay_attempts,
            "guard_result_error": None if result.ok else result.error,
            "guard_attempts": backend.calls,
            "attempts_total": attempts_total,
            "repaired_calls": repaired_calls,
            "repaired_arguments": repaired_calls[-1]["arguments"] if repaired_calls else None,
            "online_verified": bool(result.ok),
            "guard_events": _event_summary(guard.events),
            "guard_stats": dict(guard.stats),
        }
        records.append(record)
        status = {"success": "RECOVERED+ONLINE-VERIFIED",
                  "api_unavailable": "verification unavailable (live API unhealthy)",
                  "still_error": "not recovered"}[verification_status]
        print(f"[{trace['trace_id']:32s}] {trace['error_kind']:16s} -> {status} "
              f"(replays={replay_attempts}, calls={attempts_total})")

    recovered = [record for record in records if record["online_verified"]]
    unavailable = [record for record in records if record["verification_status"] == "api_unavailable"]
    failed = [record for record in records
              if not record["online_verified"] and record["verification_status"] == "still_error"]
    summary = {
        "traces": len(records),
        "recovered_and_online_verified": len(recovered),
        "verification_unavailable": len(unavailable),
        "not_recovered": len(failed),
        "recovered_by_error_kind": _count(recovered, "real_error_kind"),
        "not_recovered_by_error_kind": _count(failed, "real_error_kind"),
        "recovered_by_tool": _count(recovered, "tool"),
    }
    output_payload = {
        "artifact_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "description": (
            "Real online fault traces replayed through ToolGuard. Each record feeds one "
            "genuine captured API error to the middleware; when the guard repairs the call "
            "the repaired call is executed against the real API, so online_verified means a "
            "real successful response. Not a full A/B: it isolates the repair layer without "
            "LLM/agent sampling, and complements (does not replace) the injected-fault A/B."),
        "runner": "scripts/run_real_fault_repair.py",
        "trace_source": str(traces_path.relative_to(REPO)),
        "model": None,
        "cost_usd": 0.0,
        "summary": summary,
        "records": records,
    }
    output = REPO / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{summary['recovered_and_online_verified']}/{summary['traces']} real faults "
          f"recovered and verified online -> {output}")
    return 0


def _count(records: list[dict], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in records:
        value = str(record.get(key))
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


if __name__ == "__main__":
    raise SystemExit(main())
