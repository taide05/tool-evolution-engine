"""Real online fault traces (2026-11-25).

The live A/B experiments inject deterministic faults for reproducibility; these files are
the complementary evidence: genuine errors captured from the real public APIs, replayed
through ToolGuard with the repaired call executed online.  The tests pin the trace schema,
the expected recoveries and the honest "not repairable" set, so the evidence cannot rot
silently.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRACES = ROOT / "benchmarks/toolguard/real_traces/real_fault_traces.json"
REPAIR_RUN = ROOT / "benchmarks/toolguard/real_traces/real_fault_repair_run.json"

RECOVERED_IDS = {
    "hn_story_id_as_string",
    "weather_numeric_strings",
    "country_population_year_as_string",
}
# invalid_argument traces that are NOT derivable: a non-numeric string, an empty required
# string, and an integral float (the guard does not widen float->integer; concrete finding).
NON_DERIVABLE_INVALID_ARGUMENT = {
    "weather_non_numeric_latitude",
    "hn_story_id_float",
    "geocode_empty_city",
}
NON_REPAIRABLE_KINDS = {"not_found", "request_error"}


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_real_fault_traces_are_captured_from_real_calls() -> None:
    payload = _load(TRACES)
    assert payload["summary"]["cases"] == 16
    assert payload["summary"]["real_errors_captured"] == 16
    assert payload["summary"]["unexpected_success"] == 0
    assert payload["cost_usd"] == 0.0
    for record in payload["records"]:
        assert record["outcome"] == "error"
        assert record["source"].startswith("real public API call")
        assert record["error_kind"] and record["error_message"]
        assert record["captured_at"]
    kinds = payload["summary"]["by_error_kind"]
    assert kinds.get("invalid_argument", 0) >= 5
    assert kinds.get("not_found", 0) >= 8


def test_real_fault_replay_recovers_only_derivable_contract_errors() -> None:
    payload = _load(REPAIR_RUN)
    records = {record["trace_id"]: record for record in payload["records"]}
    assert payload["summary"]["traces"] == len(records) == 16
    recovered = {trace_id for trace_id, record in records.items()
                 if record["online_verified"]}
    assert recovered == RECOVERED_IDS
    assert payload["summary"]["recovered_and_online_verified"] == len(RECOVERED_IDS)
    for trace_id in RECOVERED_IDS:
        record = records[trace_id]
        assert record["guard_outcome"] == "recovered"
        assert record["repaired_arguments"] is not None
        assert record["repaired_arguments"] != record["bad_arguments"]
        # the repaired call was executed against the real API (online verification)
        assert any(call["matched_trace"] is None for call in record["guard_attempts"])
    non_derivable_invalid = {
        trace_id for trace_id, record in records.items()
        if trace_id not in RECOVERED_IDS and record["real_error_kind"] == "invalid_argument"
    }
    assert non_derivable_invalid == NON_DERIVABLE_INVALID_ARGUMENT
    for trace_id, record in records.items():
        if trace_id in RECOVERED_IDS or record["real_error_kind"] == "invalid_argument":
            continue
        assert record["online_verified"] is False
        # everything else the guard did not recover must be semantically non-repairable
        assert record["real_error_kind"] in NON_REPAIRABLE_KINDS, (trace_id, record)
    assert payload["cost_usd"] == 0.0
