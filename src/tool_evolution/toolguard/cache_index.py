"""Index of the released response cache, keyed by the agent-visible function name.

The ToolEnv2404 snapshot shipped with StableToolBench is a *filtered* tool set, so
some APIs used in the released trajectories are missing from it.  The response cache
itself is the authoritative view of what the environment could execute, so we derive
the function-name -> cache-path mapping from the cache tree:

    <root>/<Category>/<standard_tool>_for_<standard_category>/<api_stem>.json
    function name = (api_stem + "_for_" + standard_tool)[-64:]
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .naming import function_name, standardize


@dataclass(frozen=True)
class CacheEntry:
    category: str
    tool_dir: str
    api_stem: str
    function_name: str
    path: Path

    @property
    def standard_tool(self) -> str:
        return self.tool_dir.rsplit("_for_", 1)[0]

    @property
    def standard_category(self) -> str:
        return self.tool_dir.rsplit("_for_", 1)[-1]


class CacheIndex:
    def __init__(self, entries: list[CacheEntry]):
        self.entries = entries
        self.by_function: dict[str, list[CacheEntry]] = {}
        for entry in entries:
            self.by_function.setdefault(entry.function_name, []).append(entry)

    def candidates(self, function_name: str) -> list[CacheEntry]:
        return self.by_function.get(function_name, [])

    def __len__(self) -> int:
        return len(self.entries)

    @classmethod
    def from_root(cls, root: Path) -> "CacheIndex":
        root = Path(root)
        entries: list[CacheEntry] = []
        for category_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            category = category_dir.name
            for tool_dir in sorted(p for p in category_dir.iterdir() if p.is_dir()):
                if "_for_" not in tool_dir.name:
                    continue
                standard_tool = tool_dir.name.rsplit("_for_", 1)[0]
                for api_file in sorted(tool_dir.glob("*.json")):
                    api_stem = api_file.stem
                    fn = function_name(api_stem, standard_tool)
                    entries.append(
                        CacheEntry(
                            category=category,
                            tool_dir=tool_dir.name,
                            api_stem=api_stem,
                            function_name=fn,
                            path=api_file,
                        )
                    )
        # Deduplicate identical paths (should not happen, defensive).
        seen: set[Path] = set()
        unique = [e for e in entries if not (e.path in seen or seen.add(e.path))]
        return cls(unique)
