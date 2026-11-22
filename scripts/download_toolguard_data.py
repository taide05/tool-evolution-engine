"""Download and prepare the public data ToolGuard replays.

Sources (all public, no API key):
  * StableToolBench baseline trajectories (Apache-2.0)   -> data_baselines.zip
  * ToolBench real API environment, Apr-2024 snapshot     -> toolenv2404_filtered.tar.gz
  * ToolBench recorded real API responses (cache)         -> server_cache.zip
  * StableToolBench solvable queries (shipped in repo; we only need the schema source)

The HuggingFace mirror is configurable because huggingface.co may be unreachable
from some networks.

Usage:
    python scripts/download_toolguard_data.py --data-dir .scratch/toolguard
    python scripts/download_toolguard_data.py --only baselines toolenv
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

HF = "https://hf-mirror.com/datasets/stabletoolbench"
FILES = {
    "baselines": (f"{HF}/baselines/resolve/main/data_baselines.zip", "data_baselines.zip"),
    "toolenv": (f"{HF}/ToolEnv2404/resolve/main/toolenv2404_filtered.tar.gz", "toolenv2404_filtered.tar.gz"),
    "cache": (f"{HF}/Cache/resolve/main/server_cache.zip", "server_cache.zip"),
}


def download(url: str, dest: Path, timeout: int = 600) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"exists: {dest} ({dest.stat().st_size / 1048576:.1f} MB)")
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "toolguard"})
    with urllib.request.urlopen(req, timeout=timeout) as response, dest.open("wb") as handle:
        total = 0
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
            total += len(chunk)
            if total % (50 << 20) < (1 << 20):
                print(f"  {total / 1048576:.0f} MB", flush=True)
    print(f"saved {dest} ({dest.stat().st_size / 1048576:.1f} MB)")


def extract_zip(path: Path, target: Path) -> None:
    marker = target / ".extracted"
    if marker.exists():
        print(f"already extracted: {target}")
        return
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        archive.extractall(target)
    marker.write_text("ok", encoding="utf-8")
    print(f"extracted {path.name} -> {target}")


def extract_tar(path: Path, target: Path) -> None:
    marker = target / ".extracted"
    if marker.exists():
        print(f"already extracted: {target}")
        return
    target.mkdir(parents=True, exist_ok=True)
    with tarfile.open(path, "r:gz") as archive:
        archive.extractall(target, filter="data")
    marker.write_text("ok", encoding="utf-8")
    print(f"extracted {path.name} -> {target}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=".scratch/toolguard")
    parser.add_argument("--raw-dir", default=".scratch/toolguard/raw")
    parser.add_argument("--only", nargs="*", choices=sorted(FILES), default=None)
    args = parser.parse_args()

    raw = Path(args.raw_dir)
    data = Path(args.data_dir)
    names = args.only or list(FILES)
    for name in names:
        url, filename = FILES[name]
        dest = raw / filename
        try:
            download(url, dest)
        except Exception as exc:  # noqa: BLE001 - report and continue to next artifact
            print(f"ERROR downloading {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        if name == "baselines":
            extract_zip(dest, data / "baselines")
        elif name == "toolenv":
            extract_tar(dest, data / "toolenv")
        elif name == "cache":
            extract_zip(dest, data / "cache")

    print("\nnext:")
    print(f"  python scripts/parse_toolbench_trajectories.py --input {data / 'baselines'} --output-dir {data}")
    print("  python scripts/eval_toolguard_rules.py   # requires solvable_queries/ in the StableToolBench checkout")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
