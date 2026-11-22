"""Tool backends: the middleware talks to a backend, never to a benchmark directly.

``RecordedBackend`` replays the released real-API response cache deterministically
(cache hit -> the real recorded response, miss -> explicit MISS).  The same
interface is implemented by the live-API demo backend.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from .cache_index import CacheEntry, CacheIndex


@dataclass
class CallResult:
    status: str  # "ok" | "error" | "miss"
    error: str | None = None
    response: Any = None
    source: str = "recorded"
    entry: CacheEntry | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"


class ToolBackend(Protocol):
    def call(self, function_name: str, arguments: dict[str, Any]) -> CallResult: ...


class RecordedBackend:
    """Deterministic backend over the released ToolBench response cache.

    Cache values are keyed by ``str(arguments_dict)`` (ToolBench convention); we
    also tolerate key-order differences via a canonicalized secondary index.
    """

    def __init__(self, cache_root: Path, index: CacheIndex | None = None):
        self.cache_root = Path(cache_root)
        self.index = index or CacheIndex.from_root(self.cache_root)
        self._payloads: dict[Path, dict[str, Any]] = {}
        self._canonical: dict[Path, dict[str, Any]] = {}
        self.stats = {"calls": 0, "hits": 0, "misses": 0, "error_payloads": 0, "files_loaded": 0}

    # -- cache files --------------------------------------------------------
    def _payload_map(self, path: Path) -> dict[str, Any]:
        if path not in self._payloads:
            try:
                data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
                if not isinstance(data, dict):
                    data = {}
            except (json.JSONDecodeError, OSError):
                data = {}
            self._payloads[path] = data
            self.stats["files_loaded"] += 1
        return self._payloads[path]

    def _canonical_map(self, path: Path) -> dict[str, Any]:
        if path not in self._canonical:
            mapping: dict[str, Any] = {}
            for raw_key, payload in self._payload_map(path).items():
                parsed = _parse_cache_key(raw_key)
                if parsed is not None:
                    mapping[_canonical_key(parsed)] = payload
            self._canonical[path] = mapping
        return self._canonical[path]

    # -- public API ---------------------------------------------------------
    def candidates(self, function_name: str) -> list[CacheEntry]:
        return self.index.candidates(function_name)

    def call(self, function_name: str, arguments: dict[str, Any]) -> CallResult:
        self.stats["calls"] += 1
        exact_key = str(arguments)
        canonical = _canonical_key(arguments)
        for entry in self.index.candidates(function_name):
            payload = self._payload_map(entry.path).get(exact_key)
            if payload is None:
                payload = self._canonical_map(entry.path).get(canonical)
            if payload is None:
                continue
            self.stats["hits"] += 1
            if not isinstance(payload, dict):
                return CallResult("ok", response=payload, entry=entry)
            error = payload.get("error")
            if error:
                self.stats["error_payloads"] += 1
                return CallResult("error", error=str(error), response=payload.get("response"), entry=entry)
            return CallResult("ok", response=payload.get("response"), entry=entry)
        self.stats["misses"] += 1
        return CallResult("miss", error="CACHE_MISS")

    def has_entry(self, function_name: str, arguments: dict[str, Any]) -> bool:
        exact_key = str(arguments)
        canonical = _canonical_key(arguments)
        for entry in self.index.candidates(function_name):
            if exact_key in self._payload_map(entry.path):
                return True
            if canonical in self._canonical_map(entry.path):
                return True
        return False


def _parse_cache_key(raw: Any) -> dict[str, Any] | None:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return None
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _canonical_key(arguments: dict[str, Any]) -> str:
    return json.dumps(_normalize(arguments), sort_keys=True, ensure_ascii=False, default=str)


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize(v) for v in value]
    return value
