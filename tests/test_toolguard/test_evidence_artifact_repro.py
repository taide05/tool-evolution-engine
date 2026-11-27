"""Reproduce-metadata tests for the evidence JSONs and the historical drift artifacts.

The 2026-11-25 review found that the reproduce block of a *subset* artifact
(``real_traces/real_trace_ab_smoke.json``, ``real_traces/hard_pilot_v5.json``) still named
the V4 headline output path and omitted the protocol flags that produced it, so replaying
the documented command would have overwritten the headline artifact.  These tests pin the
artifact-specific reproduce metadata and the protocol flags.

``benchmarks/toolguard/real_traces/*.json`` are evidence data, not contract artifacts, so
they are covered here instead of by ``artifact_contracts.validate_artifacts``.
"""

from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

BENCHMARKS = ROOT / "benchmarks" / "toolguard"
EVIDENCE = BENCHMARKS / "real_traces"
SHARED_CALIBRATION = ".scratch/toolguard/multimodel_calibration.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _outputs(commands: list[str]) -> list[str]:
    outputs: list[str] = []
    for command in commands:
        try:
            parts = shlex.split(command, posix=False)
        except ValueError:
            parts = command.split()
        for index, part in enumerate(parts):
            if part == "--output" and index + 1 < len(parts):
                outputs.append(parts[index + 1].strip('"').strip("'").replace("\\", "/"))
    return outputs


def _assert_outputs_are_self(commands: list[str], name: str) -> None:
    for output in _outputs(commands):
        if "benchmarks/toolguard" in output:
            assert Path(output).name == name, (
                f"reproduce command writes {output}, expected {name}")


def _run_multimodel_runner(*arguments: str) -> subprocess.CompletedProcess:
    """Run the offline multimodel metadata paths (no live calls, no API keys required)."""
    return subprocess.run(
        [sys.executable, "scripts/run_multimodel_live.py", *arguments],
        cwd=ROOT, capture_output=True, text=True, timeout=300)


def _temp_artifact_copy(name: str) -> Path:
    workdir = ROOT / ".scratch" / "test-toolguard" / uuid.uuid4().hex
    workdir.mkdir(parents=True, exist_ok=True)
    target = workdir / name
    shutil.copyfile(BENCHMARKS / name, target)
    return target


def test_real_trace_smoke_metadata_is_artifact_specific() -> None:
    data = _load(EVIDENCE / "real_trace_ab_smoke.json")
    commands = data["reproduce"]
    joined = " ".join(commands)
    assert "--fault-source real-trace" in joined
    assert "--task-ids" in joined
    assert "--output benchmarks/toolguard/real_traces/real_trace_ab_smoke.json" in joined
    assert data["experiment"].endswith("real_trace_fault_source_smoke")
    assert "not the V4" in data["protocol"]["task_pool"]
    assert data["protocol"]["fault_injection"]["source"] == "real-trace"
    limitations = " ".join(data["limitations"])
    lowered = limitations.lower()
    assert "robustness" in lowered and "task subset" in lowered
    # the generic synthetic-fault wording must be REPLACED, not merely appended to
    assert "deterministic transient-fault" not in lowered
    assert "deterministic transient fault" not in lowered
    assert "no synthetic transient-fault injection" in lowered
    assert "6-task subset" in limitations
    assert "not the v4 headline protocol" in lowered
    # a 6-task / 12-cell smoke, not a headline protocol
    assert len(data["tasks"]) == 6
    assert len(data["records"]) == 12
    _assert_outputs_are_self(commands, "real_trace_ab_smoke.json")


def test_hard_pilot_v5_metadata_is_artifact_specific() -> None:
    data = _load(EVIDENCE / "hard_pilot_v5.json")
    commands = data["reproduce"]
    joined = " ".join(commands)
    hard_ids = ["hard_gdp_triple_gap", "hard_weather_spread_three",
                "hard_pypi_release_span", "hard_hn_author_age_gap"]
    assert "--include-hard-tasks" in joined
    for task_id in hard_ids:
        assert task_id in joined
    assert "--output benchmarks/toolguard/real_traces/hard_pilot_v5.json" in joined
    assert data["experiment"] == "ToolGuard_hard_multistep_live_ab_v5_hard_task_pilot"
    assert "V5 hard-task pilot" in data["protocol"]["task_pool"]
    assert sorted(task["id"] for task in data["tasks"]) == sorted(hard_ids)
    limitations = " ".join(data["limitations"])
    lowered = limitations.lower()
    assert "pilot" in lowered and "ceiling" in lowered
    assert "not the V4 main 31-task protocol" in limitations
    # the generic V2 task-pool limitation must be replaced, not appended to
    assert "V3 task-pool" not in limitations
    assert "This runner now produces the V3 task-pool artifact" not in limitations
    assert len(data["records"]) == 24
    _assert_outputs_are_self(commands, "hard_pilot_v5.json")


def test_v3_artifacts_live_reproduce_uses_drifting_hn_schema() -> None:
    for name in ("live_demo_hard_v3.json", "live_demo_multimodel_v3.json"):
        data = _load(BENCHMARKS / name)
        commands = data["reproduce"]
        live_commands = [command for command in commands if "--mode eval" in command]
        assert live_commands, f"{name}: no live eval reproduce command"
        for command in live_commands:
            assert "--hn-schema drifting" in command, f"{name}: {command}"
        from_commands = [command for command in commands if "--from-records" in command]
        assert from_commands, f"{name}: no from-records reproduce command"
        for command in from_commands:
            assert f"--output benchmarks/toolguard/{name}" in command, f"{name}: {command}"
        _assert_outputs_are_self(commands, name)


def test_hn_attribution_control_reproduce_stays_on_itself() -> None:
    name = "live_demo_hard_v3_hnconsistent.json"
    data = _load(BENCHMARKS / name)
    joined = " ".join(data["reproduce"])
    assert "--hn-schema consistent" in joined
    assert "--budgets clean+1" in joined
    assert "--repeats clean+1:3" in joined
    assert "live_demo_hard_v3.json" not in joined.replace(name, "")
    _assert_outputs_are_self(data["reproduce"], name)


def test_all_evidence_reproduce_outputs_stay_inside_their_own_file() -> None:
    for path in sorted(EVIDENCE.glob("*.json")):
        data = _load(path)
        commands = data.get("reproduce") or []
        _assert_outputs_are_self(commands, path.name)


# ---------------------------------------------------------------------------
# multi-model calibration provenance (2026-11-25 review follow-up)
# ---------------------------------------------------------------------------
def test_multimodel_v4_reproduce_does_not_recalibrate() -> None:
    """V4 reuses the V3 per-model anchor: reproduction replays eval/from-records only."""
    data = _load(BENCHMARKS / "live_demo_multimodel_v4.json")
    commands = data["reproduce"]
    joined = " ".join(commands)
    assert "--mode calibrate" not in joined
    eval_commands = [command for command in commands if "--mode eval" in command]
    from_commands = [command for command in commands if "--from-records" in command]
    assert len(eval_commands) == 1 and len(from_commands) == 1
    assert "--calibration " in eval_commands[0]
    assert "--calibration-mode reused" in eval_commands[0]
    assert "--calibration-reused-from V3" in eval_commands[0]
    calibration = data["protocol"]["calibration"]
    assert calibration["mode"] == "reused"
    assert calibration["reused_from"] == "V3"
    assert calibration["required_input"] is True
    assert calibration["reproduce_regenerates_calibration"] is False
    fresh = calibration["fresh_calibration"]
    assert "not the exact" in fresh["note"]
    assert "multimodel_calibration_v4_fresh.json" in fresh["command"]
    assert "--mode calibrate" in fresh["command"]
    limitations = " ".join(data["limitations"])
    assert "read-only" in limitations and "never regenerate" in limitations


def test_multimodel_v3_reproduce_does_not_overwrite_shared_calibration() -> None:
    v3 = _load(BENCHMARKS / "live_demo_multimodel_v3.json")
    v4 = _load(BENCHMARKS / "live_demo_multimodel_v4.json")
    v3_calibration = v3["protocol"]["calibration"]
    v4_calibration = v4["protocol"]["calibration"]
    # both artifacts take the same committed anchor as a read-only input ...
    assert v3_calibration["source"] == v4_calibration["source"] == SHARED_CALIBRATION
    for data in (v3, v4):
        assert "--mode calibrate" not in " ".join(data["reproduce"])
        assert data["protocol"]["calibration"]["required_input"] is True
        assert data["protocol"]["calibration"]["reproduce_regenerates_calibration"] is False
    # ... and a fresh calibration goes to a separate per-artifact path, never the anchor
    v3_fresh = v3_calibration["fresh_calibration"]["command"]
    v4_fresh = v4_calibration["fresh_calibration"]["command"]
    assert "multimodel_calibration_v3_fresh.json" in v3_fresh
    assert "multimodel_calibration_v4_fresh.json" in v4_fresh
    assert v3_fresh != v4_fresh
    for command in (v3_fresh, v4_fresh):
        assert SHARED_CALIBRATION not in command
        assert "fresh calibration, not the exact" in v3_calibration["fresh_calibration"]["note"]
    assert v3_calibration["mode"] == "fresh"
    assert v3_calibration["reused_from"] is None


def test_multimodel_from_records_preserves_calibration_provenance() -> None:
    copy = _temp_artifact_copy("live_demo_multimodel_v4.json")
    try:
        result = _run_multimodel_runner("--from-records", "--output", str(copy))
        assert result.returncode == 0, result.stderr
        rebuilt = _load(copy)
        calibration = rebuilt["protocol"]["calibration"]
        assert calibration["mode"] == "reused"
        assert calibration["reused_from"] == "V3"
        assert calibration["reproduce_regenerates_calibration"] is False
        assert "--mode calibrate" not in " ".join(rebuilt["reproduce"])
        assert rebuilt["calibration"].get("provenance", {}).get("mode") == "reused"
    finally:
        shutil.rmtree(copy.parent, ignore_errors=True)


def test_multimodel_update_metadata_preserves_calibration_provenance() -> None:
    for name, expected_mode in (("live_demo_multimodel_v3.json", "fresh"),
                                ("live_demo_multimodel_v4.json", "reused")):
        copy = _temp_artifact_copy(name)
        try:
            original = _load(copy)
            result = _run_multimodel_runner("--update-metadata", "--output", str(copy))
            assert result.returncode == 0, result.stderr
            refreshed = _load(copy)
            calibration = refreshed["protocol"]["calibration"]
            assert calibration["mode"] == expected_mode, name
            assert refreshed["calibration"].get("provenance", {}).get("mode") == expected_mode
            # metadata-only: records/aggregate data blocks are byte-identical
            for key in ("records", "aggregate"):
                assert json.dumps(original[key], sort_keys=True) == \
                    json.dumps(refreshed[key], sort_keys=True), f"{name}: {key} changed"
        finally:
            shutil.rmtree(copy.parent, ignore_errors=True)
