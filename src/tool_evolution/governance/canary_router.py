"""Canary traffic router with consistent hashing.

Routes requests between stable (default behavior) and canary (optimized
skill) variants based on the skill's canary status. Uses hash-based
consistent routing so the same caller always hits the same variant.
"""

import hashlib
import aiosqlite

CANARY_PCT: dict[str, int] = {
    "canary_5": 5,
    "canary_15": 15,
    "canary_50": 50,
    "active": 100,
    "deprecated": 0,
    "offline": 0,
}


class CanaryRouter:
    def __init__(self, conn: aiosqlite.Connection):
        self.conn = conn

    @staticmethod
    def canary_pct(status: str) -> int:
        return CANARY_PCT.get(status, 0)

    def decide(self, request_hash: str, status: str) -> str:
        """Return 'canary' or 'stable' for a single request.

        Uses MD5 of the request_hash modulo 100 for deterministic
        routing — same request_hash always maps to the same bucket.
        """
        pct = CANARY_PCT.get(status, 0)
        if pct == 0:
            return "stable"
        if pct == 100:
            return "canary"

        bucket = int(hashlib.md5(request_hash.encode()).hexdigest(), 16) % 100
        return "canary" if bucket < pct else "stable"

    async def get_skill(self, name: str) -> dict | None:
        cursor = await self.conn.execute(
            "SELECT * FROM deployed_skills WHERE name=?",
            (name,),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def record_invocation(
        self,
        skill_id: int,
        variant: str,
        success: bool,
        latency_ms: int,
        tokens: int,
    ) -> None:
        await self.conn.execute(
            """INSERT INTO canary_invocations
               (skill_id, variant, success, latency_ms, tokens)
               VALUES (?, ?, ?, ?, ?)""",
            (skill_id, variant, int(success), latency_ms, tokens),
        )
        await self.conn.commit()

    async def compare_variants(self, skill_id: int, min_samples: int = 30) -> dict | None:
        """Compare canary vs stable metrics from recorded invocations.

        Returns None if insufficient samples for either variant.
        Returns {"canary_rate": float, "stable_rate": float, "rollback": bool, "promote": bool}.
        """
        cursor = await self.conn.execute(
            """SELECT variant, COUNT(*) as total,
                      SUM(CASE WHEN success THEN 1 ELSE 0 END) as successes
               FROM canary_invocations
               WHERE skill_id = ?
               GROUP BY variant""",
            (skill_id,),
        )
        rows = await cursor.fetchall()
        stats = {row["variant"]: {"total": row["total"], "successes": row["successes"]} for row in rows}

        canary = stats.get("canary", {"total": 0, "successes": 0})
        stable = stats.get("stable", {"total": 0, "successes": 0})

        if canary["total"] < min_samples or stable["total"] < min_samples:
            return None

        canary_rate = canary["successes"] / canary["total"]
        stable_rate = stable["successes"] / stable["total"]

        return {
            "canary_rate": round(canary_rate, 4),
            "stable_rate": round(stable_rate, 4),
            "canary_samples": canary["total"],
            "stable_samples": stable["total"],
            "rollback": canary_rate < stable_rate - 0.10,
            "promote": canary_rate >= stable_rate,
        }
