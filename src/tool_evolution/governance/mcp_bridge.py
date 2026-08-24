import json
import aiosqlite
from mcp.server.fastmcp import FastMCP
from .relation_store import RelationStore, extract_entities

mcp = FastMCP("Tool Evolution Engine - Memory Bridge")


class MCPBridge:
    """Bidirectional bridge between Tool Evolution Engine and MCP memory.

    Exposes memory operations as both async Python methods (for REST routes)
    and MCP tools (for direct agent consumption via stdio transport).
    """

    def __init__(self, conn: aiosqlite.Connection):
        self.conn = conn
        self.relations = RelationStore(conn)

    # ── public API (callable from REST or programmatically) ──────────

    async def search_memory(self, query: str) -> list[dict]:
        return await self._search_entities(query)

    async def update_memory(self, entity: str, relations: list[str]) -> None:
        await self._set_cache(f"entity:{entity}", {"entity": entity, "relations": relations})

    async def get_user_preferences(self) -> dict:
        prefs = await self._get_cache("user_preferences")
        return prefs or {}

    async def extract_and_update(self, trace_result: dict) -> None:
        """Auto-extract entities from a successful trace result and persist to memory."""
        if not trace_result or not trace_result.get("success"):
            return
        result = trace_result.get("result")
        if not result or not isinstance(result, dict):
            return
        # Extract entity names from result keys and known fields
        for entity in extract_entities(result):
            await self._set_cache(
                f"entity:{entity}",
                {"entity": entity, "source": trace_result.get("tool_name", "unknown"), "relations": []},
            )

    # ── internal helpers ─────────────────────────────────────────────

    async def _set_cache(self, key: str, value: dict) -> None:
        await self.conn.execute(
            "INSERT OR REPLACE INTO memory_cache (key, value, updated_at) VALUES (?, ?, datetime('now'))",
            (key, json.dumps(value, ensure_ascii=False)),
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
            (f"%{query}%",),
        )
        rows = await cursor.fetchall()
        return [json.loads(row["value"]) for row in rows if row["key"].startswith("entity:")]

    async def extract_relations(self, root_id: str) -> int:
        """Build co-occurrence relations for all successful traces of a task."""
        return await self.relations.build_for_task(root_id)

    async def search_relations(self, entity: str) -> list[dict]:
        return await self.relations.search_relations(entity)


# ── MCP tool registration ────────────────────────────────────────────
# These tools are what AI agents see when they connect via MCP protocol.
# The actual DB connection is injected per-call via FastAPI dependency (REST)
# or by the standalone stdio entry point (run_mcp_server.py).

_bridge_instance: MCPBridge | None = None


def set_bridge(bridge: MCPBridge) -> None:
    global _bridge_instance
    _bridge_instance = bridge


def _get_bridge() -> MCPBridge:
    if _bridge_instance is None:
        raise RuntimeError("MCPBridge not initialized. Call set_bridge() first.")
    return _bridge_instance


@mcp.tool()
async def search_memory(query: str) -> list[dict]:
    """Search the memory cache for entities matching the given query.

    Args:
        query: Search term to look up in stored entities.
    Returns:
        List of matching entity records.
    """
    return await _get_bridge().search_memory(query)


@mcp.tool()
async def update_memory(entity: str, relations: list[str]) -> dict:
    """Store or update an entity in the memory cache.

    Args:
        entity: Entity name to store.
        relations: List of related entities or tags.
    Returns:
        Confirmation dict with the stored entity name.
    """
    await _get_bridge().update_memory(entity, relations)
    return {"status": "ok", "entity": entity}


@mcp.tool()
async def get_user_preferences() -> dict:
    """Retrieve stored user preferences for parameter injection.

    Returns:
        User preferences dict, or empty dict if none set.
    """
    return await _get_bridge().get_user_preferences()


@mcp.tool()
async def search_relations(entity: str) -> list[dict]:
    """Search co-occurrence relations of an entity in the memory graph.

    Args:
        entity: Entity name to look up relations for.
    Returns:
        List of relation records with source_entity/target_entity/relation_type/strength.
    """
    return await _get_bridge().search_relations(entity)
