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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

BENCHMARKS = ROOT / "benchmarks" / "toolguard"
EVIDENCE = BENCHMARKS / "real_traces"


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
    limitations = " ".join(data["limitations"]).lower()
    assert "robustness" in limitations and "task subset" in limitations
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
    limitations = " ".join(data["limitations"]).lower()
    assert "pilot" in limitations and "ceiling" in limitations
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
