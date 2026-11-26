"""Contract tests for the committed ToolGuard benchmark artifacts.

The artifacts are the evidence behind TOOLGUARD.md; these tests make sure their internal
aggregates stay consistent with the raw records and that the reproduce/limitation blocks
are present. No network; no benchmark data needed.

Note: this machine's pytest temp-dir ACL is restricted, so the negative test writes into
``.scratch/test-toolguard/<uuid>`` instead of using the ``tmp_path`` fixture.
"""

from __future__ import annotations

import json
import shutil
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from tool_evolution.toolguard.artifact_contracts import validate_artifacts  # noqa: E402


def test_committed_artifacts_satisfy_contracts() -> None:
    problems = validate_artifacts(ROOT / "benchmarks" / "toolguard")
    assert problems == [], "\n".join(problems)


def test_contract_detects_broken_aggregate() -> None:
    workdir = ROOT / ".scratch" / "test-toolguard" / uuid.uuid4().hex
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        artifact = {
            "artifact_version": 2,
            "generated_at": "2026-01-01T00:00:00+00:00",
            "limitations": ["x"],
            "reproduce": ["python -m pytest"],
            "tasks": [{"id": "t1"}],
            "records": [
                {"task": "t1", "condition": "clean+1", "arm": "baseline", "success": True,
                 "turns": 3, "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                                       "total_tokens": 2}},
                {"task": "t1", "condition": "clean+1", "arm": "guard", "success": True,
                 "turns": 3, "usage": {"prompt_tokens": 1, "completion_tokens": 1,
                                       "total_tokens": 2}},
            ],
            "aggregate": {"conditions": {"clean+1": {"arms": {
                "baseline": {"cells": 1, "success": 0, "usage": {"total_tokens": 2}},
                "guard": {"cells": 1, "success": 1, "usage": {"total_tokens": 2}},
            }, "paired_cells": {"both_success": 1, "baseline_only": 0, "guard_only": 0,
                                "neither": 0, "mcnemar_exact_p": 1.0}}}},
        }
        (workdir / "broken.json").write_text(json.dumps(artifact), encoding="utf-8")
        problems = validate_artifacts(workdir)
        assert any("success" in problem for problem in problems), problems
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _copy_artifact(workdir: Path, name: str) -> dict:
    data = json.loads((ROOT / "benchmarks" / "toolguard" / name).read_text(encoding="utf-8"))
    (workdir / name).write_text(json.dumps(data), encoding="utf-8")
    return data


def test_contract_detects_reproduce_writing_another_artifact() -> None:
    """A reproduce command must point at the artifact it documents, never at the V4 headline."""
    workdir = ROOT / ".scratch" / "test-toolguard" / uuid.uuid4().hex
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        data = _copy_artifact(workdir, "live_demo_hard_v3.json")
        data["reproduce"] = [
            "python scripts/run_live_demo_hard_v2.py --from-records "
            "--output benchmarks/toolguard/live_demo_hard_v4.json"]
        (workdir / "live_demo_hard_v3.json").write_text(json.dumps(data), encoding="utf-8")
        problems = validate_artifacts(workdir)
        assert any("reproduce command writes" in problem for problem in problems), problems
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_contract_detects_protocol_calibration_task_mismatch() -> None:
    """The per-run protocol.calibration.tasks block must cover exactly the artifact tasks."""
    workdir = ROOT / ".scratch" / "test-toolguard" / uuid.uuid4().hex
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        data = _copy_artifact(workdir, "live_demo_hard_v4.json")
        calibration_tasks = data["protocol"]["calibration"]["tasks"]
        calibration_tasks.pop(sorted(calibration_tasks)[0])
        (workdir / "live_demo_hard_v4.json").write_text(json.dumps(data), encoding="utf-8")
        problems = validate_artifacts(workdir)
        assert any("protocol.calibration.tasks" in problem for problem in problems), problems
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_contract_detects_multimodel_calibration_task_mismatch() -> None:
    """Each per-model calibration task block must cover exactly the artifact tasks."""
    workdir = ROOT / ".scratch" / "test-toolguard" / uuid.uuid4().hex
    workdir.mkdir(parents=True, exist_ok=True)
    try:
        data = _copy_artifact(workdir, "live_demo_multimodel_v4.json")
        model_block = next(iter(data["calibration"]["models"].values()))
        model_block["tasks"].pop(sorted(model_block["tasks"])[0])
        (workdir / "live_demo_multimodel_v4.json").write_text(json.dumps(data), encoding="utf-8")
        problems = validate_artifacts(workdir)
        assert any("calibration.models[" in problem for problem in problems), problems
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
