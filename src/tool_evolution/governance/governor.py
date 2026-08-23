"""SkillGovernor -- credit scoring + canary promotion + A/B rollback + lifecycle management.

Governs deployed_skills through:
- 3D weighted credit scoring (success_rate * 0.4 + latency * 0.3 + token * 0.3)
- Canary promotion ladder: canary_5 -> canary_15 -> canary_50 -> active
- A/B rollback: B < A - 10pp triggers demotion to deprecated
- Idle decay: 0.95^days after idle_decay_days (7d default)
- record_call: single write point for all skill state changes
"""

import aiosqlite
from ..utils.config import settings


class SkillGovernor:
    """Governs deployed skill lifecycle with credit scoring and canary promotion.

    Single write point principle: all deployed_skills state mutations go through
    record_call, promote, demote, or update_all_scores -- never raw SQL in callers.
    """

    def __init__(self, conn: aiosqlite.Connection):
        self.conn = conn

    async def score_skill(self, skill_id: int) -> float:
        """Compute 3D weighted credit score for a deployed skill.

        Components (weights):
          - success_rate (0.4): success_count / total_calls * 100
          - latency_score (0.3): min(100, 1000 / avg_latency * 50), normalized
          - token_score (0.3): min(100, 500 / avg_tokens * 50), normalized

        Applies idle decay if last_used_at exceeds idle_decay_days.
        Returns 50.0 for skills with no calls (neutral baseline).
        Returns 0.0 for nonexistent skills.
        """
        cursor = await self.conn.execute(
            "SELECT * FROM deployed_skills WHERE id=?", (skill_id,)
        )
        row = await cursor.fetchone()
        if not row:
            return 0.0
        skill = dict(row)
        if skill["total_calls"] == 0:
            return 50.0

        success_rate = skill["success_count"] / skill["total_calls"] * 100

        avg_latency = skill["total_latency_ms"] / skill["total_calls"]
        avg_tokens = skill["total_tokens"] / skill["total_calls"]

        latency_score = min(100, 1000 / max(avg_latency, 1) * 50)
        token_score = min(100, 500 / max(avg_tokens, 1) * 50)

        base_score = success_rate * 0.4 + latency_score * 0.3 + token_score * 0.3

        # Idle decay: after idle_decay_days, apply 0.95^excess_days
        if skill.get("last_used_at"):
            cursor = await self.conn.execute(
                "SELECT (julianday('now') - julianday(?)) AS days_idle",
                (skill["last_used_at"],),
            )
            days_row = await cursor.fetchone()
            days_idle = days_row[0] if days_row else 0
            if days_idle > settings.idle_decay_days:
                decay = 0.95 ** (days_idle - settings.idle_decay_days)
                base_score *= decay

        return round(base_score, 2)

    async def update_all_scores(self) -> None:
        """Batch-update credit_score and status for all non-offline skills.

        Status transitions:
          score < 20  -> offline
          score < 40  -> deprecated
          score >= 80 -> promote to next canary/active tier
        """
        cursor = await self.conn.execute(
            "SELECT id FROM deployed_skills WHERE status NOT IN ('offline')"
        )
        rows = await cursor.fetchall()
        for row in rows:
            skill_id = row["id"]
            score = await self.score_skill(skill_id)
            new_status = None
            if score < 20:
                new_status = "offline"
            elif score < 40:
                new_status = "deprecated"
            elif score >= 80:
                cursor2 = await self.conn.execute(
                    "SELECT status FROM deployed_skills WHERE id=?", (skill_id,)
                )
                current = dict(await cursor2.fetchone())
                new_status = await self._promote_status(current["status"])

            await self.conn.execute(
                "UPDATE deployed_skills SET credit_score=? WHERE id=?", (score, skill_id)
            )
            if new_status:
                await self.conn.execute(
                    "UPDATE deployed_skills SET status=? WHERE id=?", (new_status, skill_id)
                )
        await self.conn.commit()

    async def record_call(
        self, skill_id: int, success: bool, latency_ms: int, tokens: int
    ) -> None:
        """Record a single invocation -- the canonical write point for skill stats.

        Atomically increments success_count (+1 if success), total_calls (+1),
        total_latency_ms, total_tokens, and sets last_used_at to now.
        """
        await self.conn.execute(
            """UPDATE deployed_skills SET
               success_count = success_count + ?,
               total_calls = total_calls + 1,
               total_latency_ms = total_latency_ms + ?,
               total_tokens = total_tokens + ?,
               last_used_at = datetime('now')
               WHERE id=?""",
            (1 if success else 0, latency_ms, tokens, skill_id),
        )
        await self.conn.commit()

    async def promote(self, skill_id: int) -> str:
        """Advance skill to next canary/active tier.

        Promotion ladder: canary_5 -> canary_15 -> canary_50 -> active.
        Raises ValueError if skill not found.
        """
        cursor = await self.conn.execute(
            "SELECT status FROM deployed_skills WHERE id=?", (skill_id,)
        )
        row = await cursor.fetchone()
        if not row:
            raise ValueError(f"Skill {skill_id} not found")
        current = dict(row)["status"]
        new_status = await self._promote_status(current)
        await self.conn.execute(
            "UPDATE deployed_skills SET status=? WHERE id=?", (new_status, skill_id)
        )
        await self.conn.commit()
        return new_status

    async def _promote_status(self, current: str) -> str:
        """Return the next status in the canary ladder, or current if at top."""
        order = ["canary_5", "canary_15", "canary_50", "active"]
        if current in order:
            idx = order.index(current)
            if idx < len(order) - 1:
                return order[idx + 1]
        return current

    async def demote(self, skill_id: int, reason: str) -> str:
        """Demote skill to deprecated status.

        The reason parameter is reserved for audit trail (deprecation_reason
        column is a two-week-plan item).
        """
        _ = reason
        await self.conn.execute(
            "UPDATE deployed_skills SET status='deprecated' WHERE id=?", (skill_id,)
        )
        await self.conn.commit()
        return "deprecated"
