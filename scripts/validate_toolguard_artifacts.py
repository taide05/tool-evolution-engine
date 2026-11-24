"""CLI wrapper: check the committed ToolGuard artifacts against their contracts.

Usage:
    python scripts/validate_toolguard_artifacts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tool_evolution.toolguard.artifact_contracts import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
