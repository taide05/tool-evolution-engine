import json
import aiosqlite
from .schemas import TraceReport, ErrorType
from ..utils.database import transaction


class TraceStore:
    def __init__(self, conn: aiosqlite.Connection):
        self.conn = conn

    async def insert(self, report: TraceReport) -> None:
        async with transaction(self.conn):
            cursor = await self.conn.execute(
                """INSERT INTO trajectories
                   (trace_id, parent_trace_id, agent_id, tool_name, tool_version,
                    trace_type, params, success, result, error_type, error_message,
                    latency_ms, token_count, source)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (report.trace_id, report.parent_trace_id, report.agent_id,
                 report.tool_name, report.tool_version, report.trace_type.value,
                 json.dumps(report.params), int(report.success),
                 json.dumps(report.result) if report.result else None,
                 report.error_type.value if report.error_type else None,
                 report.error_message, report.latency_ms, report.token_count,
                 report.source)
            )
            await self.conn.execute(
                "INSERT INTO trajectories_fts(rowid, tool_name, error_message) VALUES (?, ?, ?)",
                (cursor.lastrowid, report.tool_name, report.error_message or "")
            )

    async def get_by_tool(self, tool_name: str, limit: int = 100) -> list[dict]:
        cursor = await self.conn.execute(
            "SELECT * FROM trajectories WHERE tool_name=? ORDER BY created_at DESC LIMIT ?",
            (tool_name, limit)
        )
        return [dict(row) for row in await cursor.fetchall()]

    async def get_task_tree(self, root_id: str) -> list[dict]:
        cursor = await self.conn.execute(
            """WITH RECURSIVE tree AS (
                SELECT *, 0 AS depth FROM trajectories WHERE trace_id=?
                UNION ALL
                SELECT t.*, tree.depth+1 FROM trajectories t
                JOIN tree ON t.parent_trace_id=tree.trace_id
            ) SELECT * FROM tree ORDER BY depth, created_at""",
            (root_id,)
        )
        return [dict(row) for row in await cursor.fetchall()]

    async def search(self, fts_query: str) -> list[dict]:
        escaped = fts_query.replace('"', '""')
        cursor = await self.conn.execute(
            "SELECT * FROM trajectories_fts WHERE trajectories_fts MATCH ?",
            (f'"{escaped}"',)
        )
        return [dict(row) for row in await cursor.fetchall()]

    async def count_failures(self, error_type: ErrorType | None = None) -> int:
        if error_type:
            cursor = await self.conn.execute(
                "SELECT COUNT(*) FROM trajectories WHERE success=0 AND error_type=?",
                (error_type.value,)
            )
        else:
            cursor = await self.conn.execute(
                "SELECT COUNT(*) FROM trajectories WHERE success=0"
            )
        row = await cursor.fetchone()
        return row[0]

    async def get_success_params(self, tool_name: str, tool_version: str, limit: int = 200,
                                 exclude_agent_prefix: str | None = None) -> list[dict]:
        query = ("SELECT params FROM trajectories WHERE tool_name=? AND tool_version=? "
                 "AND success=1")
        args: list = [tool_name, tool_version]
        if exclude_agent_prefix:
            query += " AND agent_id NOT LIKE ?"
            args.append(f"{exclude_agent_prefix}%")
        query += " ORDER BY created_at DESC LIMIT ?"
        args.append(limit)
        cursor = await self.conn.execute(query, args)
        rows = await cursor.fetchall()
        return [json.loads(row["params"]) for row in rows if row["params"]]

    async def get_all_traces(self, limit: int = 100, offset: int = 0,
                             exclude_agent_prefix: str | None = None) -> list[dict]:
        query = "SELECT rowid, * FROM trajectories"
        args: list = []
        if exclude_agent_prefix:
            query += " WHERE agent_id NOT LIKE ?"
            args.append(f"{exclude_agent_prefix}%")
        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        args.extend([limit, offset])
        cursor = await self.conn.execute(query, args)
        return [dict(row) for row in await cursor.fetchall()]

    async def get_recent_traces(self, days: int = 30, limit: int = 10000) -> list[dict]:
        cursor = await self.conn.execute(
            "SELECT rowid, * FROM trajectories WHERE created_at >= datetime('now', ?) ORDER BY created_at DESC LIMIT ?",
            (f"-{days} days", limit)
        )
        return [dict(row) for row in await cursor.fetchall()]
