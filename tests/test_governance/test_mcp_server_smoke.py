import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_mcp_stdio_server_starts(tmp_path, monkeypatch):
    env = dict(os.environ)
    env["TOOLEVO_DB_PATH"] = str(tmp_path / "engine.db")
    proc = subprocess.Popen(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "run_mcp_server.py")],
        cwd=str(PROJECT_ROOT), env=env,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    try:
        time.sleep(3)
        assert proc.poll() is None, (
            f"MCP stdio server exited early (code {proc.poll()}): "
            f"{proc.stdout.read().decode(errors='replace')[:500]}"
        )
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
