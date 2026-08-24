import json
from collections import Counter
import aiosqlite
from ..utils.config import settings


class PreferenceLearner:
    """Learns per-agent parameter preferences from success trajectories.

    Histogram-based deviation detection: personal sample count must reach
    min_pref_samples, the top value share must exceed pref_share_threshold,
    and the top value must differ from the global KDE mode.
    """

    def __init__(self, conn: aiosqlite.Connection,
                 min_samples: int | None = None,
                 share_threshold: float | None = None):
        self.conn = conn
        self.min_samples = min_samples if min_samples is not None else settings.min_pref_samples
        self.share_threshold = (share_threshold if share_threshold is not None
                                else settings.pref_share_threshold)

    async def _global_modes(self, tool_name: str, tool_version: str) -> dict:
        cursor = await self.conn.execute(
            """SELECT param_name, default_value FROM param_distributions
               WHERE tool_name=? AND tool_version=?""",
            (tool_name, tool_version)
        )
        modes = {}
        for row in await cursor.fetchall():
            r = dict(row)
            val = json.loads(r["default_value"]) if r["default_value"] else None
            if val is not None:
                modes[r["param_name"]] = val
        return modes

    async def _agent_success_rows(self, agent_id: str) -> list[dict]:
        cursor = await self.conn.execute(
            """SELECT tool_name, tool_version, params FROM trajectories
               WHERE agent_id=? AND success=1""",
            (agent_id,)
        )
        return [dict(r) for r in await cursor.fetchall()]

    async def learn(self) -> dict:
        """Returns nested preferences {tool_name: {param_name: value}}."""
        cursor = await self.conn.execute(
            """SELECT DISTINCT agent_id FROM trajectories
               WHERE success=1 AND agent_id NOT LIKE 'executor:%'"""
        )
        agents = [r[0] for r in await cursor.fetchall()]
        prefs: dict[str, dict] = {}
        for agent_id in agents:
            rows = await self._agent_success_rows(agent_id)
            grouped: dict[tuple[str, str], list[dict]] = {}
            for r in rows:
                params = json.loads(r["params"]) if r["params"] else {}
                grouped.setdefault((r["tool_name"], r["tool_version"]), []).append(params)
            for (tool_name, tool_version), params_list in grouped.items():
                modes = await self._global_modes(tool_name, tool_version)
                if not modes:
                    continue
                all_keys: set[str] = set()
                for p in params_list:
                    all_keys.update(p.keys())
                for param in all_keys:
                    values = [p[param] for p in params_list if param in p]
                    if len(values) < self.min_samples:
                        continue
                    counts = Counter(json.dumps(v, sort_keys=True) for v in values)
                    top_key, top_count = counts.most_common(1)[0]
                    if top_count / len(values) <= self.share_threshold:
                        continue
                    if param in modes and json.dumps(modes[param], sort_keys=True) == top_key:
                        continue
                    prefs.setdefault(tool_name, {})[param] = json.loads(top_key)
        return prefs

    async def save_to_cache(self, prefs: dict) -> None:
        await self.conn.execute(
            """INSERT OR REPLACE INTO memory_cache (key, value, updated_at)
               VALUES ('user_preferences', ?, datetime('now'))""",
            (json.dumps(prefs, ensure_ascii=False),)
        )
        await self.conn.commit()
