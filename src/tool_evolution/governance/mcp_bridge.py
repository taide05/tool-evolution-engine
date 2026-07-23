import json
import aiosqlite
from mcp.server.fastmcp import FastMCP


class MCPBridge:
    def __init__(self, conn: aiosqlite.Connection):
        self.conn = conn
        self.mcp = FastMCP("Tool Evolution Engine - Memory Bridge")

    async def search_memory(self, query: str) -> list[dict]:
        return await self._search_entities(query)

    async def update_memory(self, entity: str, relations: list[str]) -> None:
        await self._set_cache(f"entity:{entity}", {"entity": entity, "relations": relations})

    async def get_user_preferences(self) -> dict:
        prefs = await self._get_cache("user_preferences")
        return prefs or {}

    async def _set_cache(self, key: str, value: dict) -> None:
        await self.conn.execute(
            "INSERT OR REPLACE INTO memory_cache (key, value, updated_at) VALUES (?, ?, datetime('now'))",
            (key, json.dumps(value))
        )
        await self.conn.commit()

    async def _get_cache(self, key: str) -> dict | None:
        cursor = await self.conn.execute(
            "SELECT value FROM memory_cache WHERE key=?", (key,)
        )
        row = await cursor.fetchone()
        return json.loads(row[0]) if row else None

    async def _search_entities(self, query: str) -> list[dict]:
        cursor = await self.conn.execute(
            "SELECT key, value FROM memory_cache WHERE key LIKE ?",
            (f"%{query}%",)
        )
        rows = await cursor.fetchall()
        return [json.loads(row["value"]) for row in rows if row["key"].startswith("entity:")]
