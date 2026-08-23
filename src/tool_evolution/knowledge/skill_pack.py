import aiosqlite


class SkillPackManager:
    """Manages discovered_skills (read-only, from DAG mining) and
    deployed_skills (writable, managed by SkillGovernor) tables."""

    def __init__(self, conn: aiosqlite.Connection):
        self.conn = conn

    async def add_discovery(self, skill: dict) -> int:
        cursor = await self.conn.execute(
            """INSERT INTO discovered_skills (name, dag_definition, param_template, frequency, status)
               VALUES (?, ?, ?, ?, 'canary')""",
            (skill["name"], skill["dag_definition"],
             skill.get("param_template", "{}"), skill.get("frequency", 0.0))
        )
        await self.conn.commit()
        return cursor.lastrowid

    async def promote_to_deployed(self, discovery_id: int) -> int:
        cursor = await self.conn.execute(
            "SELECT * FROM discovered_skills WHERE id=?", (discovery_id,)
        )
        row = await cursor.fetchone()
        if not row:
            raise ValueError(f"Discovery {discovery_id} not found")
        disc = dict(row)

        await self.conn.execute(
            "UPDATE discovered_skills SET status='promoted' WHERE id=?", (discovery_id,)
        )

        cursor = await self.conn.execute(
            """INSERT INTO deployed_skills
               (discovery_id, name, dag_definition, param_template, status)
               VALUES (?, ?, ?, ?, 'canary_5')""",
            (discovery_id, disc["name"], disc["dag_definition"],
             disc.get("param_template", "{}"))
        )
        await self.conn.commit()
        return cursor.lastrowid

    async def get_deployed(self, name: str) -> dict | None:
        cursor = await self.conn.execute(
            "SELECT * FROM deployed_skills WHERE name=?", (name,)
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def list_deployed(self, status_filter: str | None = None) -> list[dict]:
        if status_filter:
            cursor = await self.conn.execute(
                "SELECT * FROM deployed_skills WHERE status=?", (status_filter,)
            )
        else:
            cursor = await self.conn.execute("SELECT * FROM deployed_skills")
        return [dict(row) for row in await cursor.fetchall()]

    async def list_discoveries(self) -> list[dict]:
        cursor = await self.conn.execute("SELECT * FROM discovered_skills")
        return [dict(row) for row in await cursor.fetchall()]
