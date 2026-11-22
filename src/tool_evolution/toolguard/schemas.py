"""Authoritative function schemas for the evaluation queries.

The per-query ``api_list`` in the released solvable-query files contains the full
agent-visible API specification (parameter names/types/descriptions/defaults).  We
rebuild the function schema index from that source because the answer files only
store a partial copy for some queries.
"""

from __future__ import annotations

import json
from pathlib import Path

from .naming import change_name, function_name, standardize
from .records import FunctionSchema

_TYPE_MAP = {"NUMBER": "number", "INTEGER": "integer", "STRING": "string", "BOOLEAN": "boolean"}


def build_schema_index(queries_dir: Path) -> tuple[dict[str, FunctionSchema], dict[str, int]]:
    """Return (function_name -> schema, stats) from solvable query files."""
    index: dict[str, FunctionSchema] = {}
    stats = {"queries": 0, "api_entries": 0, "conflicts": 0, "queries_seen": 0}
    for path in sorted(Path(queries_dir).glob("*.json")):
        queries = json.loads(path.read_text(encoding="utf-8"))
        for query in queries:
            stats["queries"] += 1
            for api in query.get("api_list") or []:
                if not isinstance(api, dict) or not api.get("api_name"):
                    continue
                stats["api_entries"] += 1
                fn = function_name(api["api_name"], api.get("tool_name") or "")
                properties: dict[str, dict[str, object]] = {}
                required: list[str] = []
                for param in api.get("required_parameters") or []:
                    name, prop = _param(param)
                    if name:
                        properties[name] = prop
                        required.append(name)
                for param in api.get("optional_parameters") or []:
                    name, prop = _param(param)
                    if name:
                        properties[name] = prop
                schema = FunctionSchema(name=fn, properties=properties, required=tuple(required))
                existing = index.get(fn)
                if existing is not None and existing != schema:
                    stats["conflicts"] += 1
                    continue
                index[fn] = schema
    return index, stats


def _param(param: dict) -> tuple[str | None, dict[str, object]]:
    if not isinstance(param, dict) or not param.get("name"):
        return None, {}
    name = change_name(standardize(str(param["name"])))
    raw_type = str(param.get("type") or "string").upper()
    prop: dict[str, object] = {
        "type": _TYPE_MAP.get(raw_type, "string"),
        "description": str(param.get("description") or ""),
    }
    default = param.get("default")
    if default not in (None, ""):
        prop["example_value"] = default
    return name, prop
