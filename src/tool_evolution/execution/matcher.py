"""确定性技能匹配器——任务描述 vs deployed_skills（active），纯工具名命中打分。"""

import re

import aiosqlite

from ..utils.config import settings


class SkillMatcher:
    def __init__(self, conn: aiosqlite.Connection, threshold: float | None = None):
        self.conn = conn
        self.threshold = (
            threshold if threshold is not None else settings.skill_match_threshold
        )

    async def match(self, task_description: str) -> dict | None:
        """返回 {'skill': dict, 'score': float} 或 None。

        打分 = 描述命中的技能工具数 / 技能工具数（技能 name "a → b → c" 拆工具名）。
        只认 status='active'；score >= threshold 命中；并列取 credit_score 高者
        （再并列取 id 小者）。
        """
        cursor = await self.conn.execute(
            "SELECT * FROM deployed_skills WHERE status='active'"
        )
        rows = [dict(r) for r in await cursor.fetchall()]
        if not rows:
            return None

        description = task_description or ""
        best: dict | None = None
        for skill in rows:
            tools = _skill_tools(skill["name"])
            if not tools:
                continue
            hits = sum(1 for t in tools if t in description)
            score = hits / len(tools)
            if score < self.threshold:
                continue
            if best is None or (
                score,
                skill["credit_score"],
                -skill["id"],
            ) > (
                best["score"],
                best["skill"]["credit_score"],
                -best["skill"]["id"],
            ):
                best = {"skill": skill, "score": score}
        return best


def _skill_tools(name: str) -> list[str]:
    """技能 name 的 'a → b → c' 拆为工具名列表（剔除空段）。"""
    return [seg.strip() for seg in re.split(r"[→\-\s>]+", name) if seg.strip()]
