import json

from tool_evolution.execution.matcher import SkillMatcher


async def _seed_skill(conn, name, credit_score=50.0, status="active"):
    cursor = await conn.execute(
        """INSERT INTO deployed_skills (name, dag_definition, param_template,
           credit_score, status)
           VALUES (?, ?, ?, ?, ?)""",
        (name, json.dumps({"nodes": [], "edges": []}), "{}",
         credit_score, status),
    )
    await conn.commit()
    return cursor.lastrowid


class TestSkillMatcher:
    async def test_full_hit_scores_1(self, db_conn):
        await _seed_skill(db_conn, "search_api → analyze_api")
        matcher = SkillMatcher(db_conn)
        result = await matcher.match("调用 search_api 和 analyze_api 完成任务")
        assert result is not None
        assert result["score"] == 1.0
        assert result["skill"]["name"] == "search_api → analyze_api"

    async def test_partial_hit_exact_threshold(self, db_conn):
        await _seed_skill(db_conn, "search_api → analyze_api")
        matcher = SkillMatcher(db_conn)
        result = await matcher.match("调用 search_api 完成任务")
        assert result is not None
        assert result["score"] == 0.5

    async def test_below_threshold_no_match(self, db_conn):
        await _seed_skill(db_conn, "search_api → detail_api → analyze_api")
        matcher = SkillMatcher(db_conn)
        result = await matcher.match("调用 search_api 完成任务")
        assert result is None

    async def test_no_tool_mention_no_match(self, db_conn):
        await _seed_skill(db_conn, "search_api → analyze_api")
        matcher = SkillMatcher(db_conn)
        assert await matcher.match("写一份季度报告") is None

    async def test_no_active_skills_none(self, db_conn):
        await _seed_skill(db_conn, "search_api → analyze_api", status="canary_5")
        matcher = SkillMatcher(db_conn)
        assert await matcher.match("调用 search_api") is None

    async def test_tie_breaks_by_credit_score(self, db_conn):
        await _seed_skill(db_conn, "search_api → analyze_api", credit_score=30.0)
        await _seed_skill(db_conn, "search_api → detail_api", credit_score=80.0)
        matcher = SkillMatcher(db_conn)
        # 两技能均 2/2 全命中并列 1.0 → 取 credit_score 高者
        result = await matcher.match("调用 search_api analyze_api detail_api")
        assert result is not None
        assert result["skill"]["name"] == "search_api → detail_api"

    async def test_tool_name_substring_in_running_text(self, db_conn):
        await _seed_skill(db_conn, "search_api → detail_api")
        matcher = SkillMatcher(db_conn)
        result = await matcher.match("请用search_api查询然后detail_api展开")
        assert result is not None
        assert result["score"] == 1.0
