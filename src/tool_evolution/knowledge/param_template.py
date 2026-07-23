import json
import aiosqlite
from ..collection.store import TraceStore
from ..analysis.kde_analyzer import KDEAnalyzer
from ..utils.config import settings


class ParamTemplateManager:
    def __init__(self, conn: aiosqlite.Connection):
        self.conn = conn
        self.store = TraceStore(conn)
        self.kde = KDEAnalyzer(min_samples=settings.min_samples)

    async def save(self, tool_name: str, tool_version: str, distributions: dict[str, dict]) -> None:
        for param_name, dist in distributions.items():
            await self.conn.execute(
                """INSERT OR REPLACE INTO param_distributions
                   (tool_name, tool_version, param_name, param_type, kde_params,
                    default_value, lower_bound, upper_bound, sample_count, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
                (tool_name, tool_version, param_name, dist["param_type"],
                 json.dumps(dist.get("kde_params", {})),
                 json.dumps(dist.get("default_value")),
                 json.dumps(dist.get("lower_bound")),
                 json.dumps(dist.get("upper_bound")),
                 dist.get("sample_count", 0))
            )
        await self.conn.commit()

    async def get_template(self, tool_name: str, tool_version: str) -> dict | None:
        cursor = await self.conn.execute(
            "SELECT * FROM param_distributions WHERE tool_name=? AND tool_version=?",
            (tool_name, tool_version)
        )
        rows = await cursor.fetchall()
        if not rows:
            return None
        result = {}
        for row in rows:
            r = dict(row)
            result[r["param_name"]] = {
                "param_type": r["param_type"],
                "default_value": json.loads(r["default_value"]) if r["default_value"] else None,
                "lower_bound": json.loads(r["lower_bound"]) if r["lower_bound"] else None,
                "upper_bound": json.loads(r["upper_bound"]) if r["upper_bound"] else None,
                "sample_count": r["sample_count"],
            }
        return result

    async def generate(self, tool_name: str, tool_version: str) -> dict | None:
        params_list = await self.store.get_success_params(tool_name, tool_version, limit=500)
        if len(params_list) < settings.min_samples:
            return None
        all_keys = set()
        for p in params_list:
            all_keys.update(p.keys())
        filtered = [{k: v for k, v in p.items() if k in all_keys} for p in params_list]
        dists = self.kde.analyze(tool_name, tool_version, filtered)
        if dists:
            await self.save(tool_name, tool_version, dists)
        return await self.get_template(tool_name, tool_version)
