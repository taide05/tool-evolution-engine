"""执行审计——execution_tasks/execution_steps 读写。"""

import json

import aiosqlite


class ExecutionAudit:
    def __init__(self, conn: aiosqlite.Connection):
        self.conn = conn

    async def create_task(self, task_id: str, task_description: str, mode: str,
                          plan: dict | None, skill_name: str | None = None) -> bool:
        """INSERT status='pending'——ON CONFLICT DO NOTHING；返回是否拿到插入权（幂等判定）。"""
        cursor = await self.conn.execute(
            """INSERT INTO execution_tasks (task_id, task_description, skill_name,
               mode, plan, status)
               VALUES (?, ?, ?, ?, ?, 'pending')
               ON CONFLICT(task_id) DO NOTHING""",
            (task_id, task_description, skill_name, mode,
             json.dumps(plan) if plan is not None else None),
        )
        await self.conn.commit()
        return cursor.rowcount > 0

    async def update_status(self, task_id: str, status: str,
                            summary: str | None = None) -> None:
        if status in ("success", "failed", "cancelled"):
            await self.conn.execute(
                """UPDATE execution_tasks SET status=?, summary=?,
                   updated_at=datetime('now'), finished_at=datetime('now')
                   WHERE task_id=?""",
                (status, summary, task_id))
        else:
            await self.conn.execute(
                """UPDATE execution_tasks SET status=?, summary=?,
                   updated_at=datetime('now') WHERE task_id=?""",
                (status, summary, task_id))
        await self.conn.commit()

    async def add_step(self, task_id: str, step_index: int, tool_name: str,
                       params: dict, result: dict | None, status: str,
                       latency_ms: int | None, tokens: int,
                       rules_triggered: list[str] | None,
                       repair_hint_applied: dict | None,
                       adapter: str) -> None:
        await self.conn.execute(
            """INSERT INTO execution_steps
               (task_id, step_index, tool_name, params, result, status,
                latency_ms, tokens, rules_triggered, repair_hint_applied, adapter)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (task_id, step_index, tool_name,
             json.dumps(params),
             json.dumps(result) if result else None,
             status, latency_ms, tokens,
             json.dumps(rules_triggered) if rules_triggered else None,
             json.dumps(repair_hint_applied) if repair_hint_applied else None,
             adapter),
        )
        await self.conn.commit()

    async def get_task(self, task_id: str) -> dict | None:
        cursor = await self.conn.execute(
            "SELECT * FROM execution_tasks WHERE task_id=?", (task_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        task = dict(row)
        task["steps"] = await self.get_task_steps(task_id)
        return task

    async def get_task_steps(self, task_id: str) -> list[dict]:
        cursor = await self.conn.execute(
            "SELECT * FROM execution_steps WHERE task_id=? ORDER BY step_index",
            (task_id,))
        steps = []
        for row in await cursor.fetchall():
            step = dict(row)
            step["params"] = json.loads(step["params"]) if step["params"] else {}
            step["result"] = json.loads(step["result"]) if step["result"] else None
            step["rules_triggered"] = (
                json.loads(step["rules_triggered"]) if step["rules_triggered"] else []
            )
            step["repair_hint_applied"] = (
                json.loads(step["repair_hint_applied"])
                if step["repair_hint_applied"] else None
            )
            steps.append(step)
        return steps
