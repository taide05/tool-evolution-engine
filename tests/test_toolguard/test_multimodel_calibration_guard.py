"""Unit tests for the multimodel calibration overwrite guard.

``run_multimodel_live.run_calibration`` must refuse to replace an existing calibration file
unless ``--overwrite-calibration`` is passed: the committed V3/V4 artifacts use the shared
per-model calibration as a read-only budget input.  These tests exercise the pure guard
function directly - no ``--mode calibrate`` run and no API call.
"""

from __future__ import annotations

import shutil
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from run_multimodel_live import calibration_write_error  # noqa: E402


def _temp_workdir() -> Path:
    """Unique scratch directory (avoids the Windows tmp_path ACL problems)."""
    workdir = ROOT / ".scratch" / "test-toolguard" / uuid.uuid4().hex
    workdir.mkdir(parents=True, exist_ok=True)
    return workdir


def test_calibration_write_allowed_when_target_missing() -> None:
    workdir = _temp_workdir()
    try:
        target = workdir / "multimodel_calibration.json"
        assert not target.exists()
        assert calibration_write_error(target, overwrite=False) is None
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_calibration_write_refused_when_target_exists_without_overwrite() -> None:
    workdir = _temp_workdir()
    try:
        target = workdir / "multimodel_calibration.json"
        target.write_text('{"models": {}}\n', encoding="utf-8")
        error = calibration_write_error(target, overwrite=False)
        assert error is not None
        assert "refusing to overwrite" in error
        assert "--overwrite-calibration" in error
        assert str(target) in error
        # the guard only reports; it does not touch the existing file
        assert target.read_text(encoding="utf-8") == '{"models": {}}\n'
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def test_calibration_write_allowed_when_target_exists_with_overwrite() -> None:
    workdir = _temp_workdir()
    try:
        target = workdir / "multimodel_calibration.json"
        target.write_text('{"models": {}}\n', encoding="utf-8")
        assert calibration_write_error(target, overwrite=True) is None
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
