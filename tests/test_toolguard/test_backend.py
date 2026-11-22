"""Tests for schema index, cache index and the recorded backend.

Uses an explicit workspace scratch directory instead of pytest ``tmp_path`` because
the local Windows sandbox denies pytest's default temp-dir management.
"""

from __future__ import annotations

import json
import shutil
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tool_evolution.toolguard.backend import RecordedBackend  # noqa: E402
from tool_evolution.toolguard.cache_index import CacheIndex  # noqa: E402
from tool_evolution.toolguard.schemas import build_schema_index  # noqa: E402

_SCRATCH = Path(__file__).resolve().parents[2] / ".scratch" / "test-toolguard"


def _fresh_dir() -> Path:
    path = _SCRATCH / uuid.uuid4().hex
    path.mkdir(parents=True)
    return path


def test_build_schema_index_from_queries():
    root = _fresh_dir()
    queries = [
        {
            "query_id": 1,
            "api_list": [
                {
                    "category_name": "Business",
                    "tool_name": "My Tool",
                    "api_name": "Search Things",
                    "required_parameters": [
                        {"name": "query", "type": "STRING", "description": "", "default": "foo"}
                    ],
                    "optional_parameters": [
                        {"name": "limit", "type": "NUMBER", "description": ""}
                    ],
                }
            ],
        }
    ]
    (root / "G1_instruction.json").write_text(json.dumps(queries), encoding="utf-8")
    index, stats = build_schema_index(root)
    assert stats["api_entries"] == 1
    schema = index["search_things_for_my_tool"]
    assert schema.required == ("query",)
    assert schema.properties["query"]["example_value"] == "foo"
    assert schema.properties["limit"]["type"] == "number"
    shutil.rmtree(root, ignore_errors=True)


def test_cache_index_and_recorded_backend():
    root = _fresh_dir()
    api_dir = root / "Business" / "my_tool_for_Business"
    api_dir.mkdir(parents=True)
    payload = {
        str({"query": "foo"}): {"error": "", "response": {"ok": True}},
        str({"query": "bar"}): {"error": "Missing input parameters.", "response": ""},
    }
    (api_dir / "search_things.json").write_text(json.dumps(payload), encoding="utf-8")

    index = CacheIndex.from_root(root)
    assert len(index) == 1
    fn = index.entries[0].function_name
    assert fn == "search_things_for_my_tool"

    backend = RecordedBackend(root, index)
    ok = backend.call(fn, {"query": "foo"})
    assert ok.ok and ok.response == {"ok": True}
    error = backend.call(fn, {"query": "bar"})
    assert error.status == "error"
    miss = backend.call(fn, {"query": "missing"})
    assert miss.status == "miss"
    assert backend.call(fn, {"query": "foo"}).ok
    shutil.rmtree(root, ignore_errors=True)
