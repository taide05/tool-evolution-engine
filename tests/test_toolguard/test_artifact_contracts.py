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
