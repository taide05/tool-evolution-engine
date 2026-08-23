import pytest
from tool_evolution.governance.governor import SkillGovernor


@pytest.fixture
async def governor(db_conn):
    return SkillGovernor(db_conn)


async def _insert_discovery(governor, discovery_id: int) -> None:
    """Insert a minimal discovered_skills row to satisfy FK constraint."""
    await governor.conn.execute(
        """INSERT OR IGNORE INTO discovered_skills (id, name, dag_definition, param_template, frequency, status)
           VALUES (?, ?, '{}', '{}', 1.0, 'promoted')""",
        (discovery_id, f"discovery_{discovery_id}"),
    )
    await governor.conn.commit()


class TestSkillGovernor:
    async def test_score_skill_perfect(self, governor):
        await _insert_discovery(governor, 1)
        await governor.conn.execute(
            """INSERT INTO deployed_skills (id, discovery_id, name, dag_definition, param_template,
               success_count, total_calls, total_latency_ms, total_tokens, status)
               VALUES (1, 1, 'PerfectSkill', '{}', '{}', 100, 100, 5000, 25000, 'active')"""
        )
        await governor.conn.commit()
        score = await governor.score_skill(1)
        assert score == 100.0

    async def test_score_skill_failing(self, governor):
        await _insert_discovery(governor, 2)
        await governor.conn.execute(
            """INSERT INTO deployed_skills (id, discovery_id, name, dag_definition, param_template,
               success_count, total_calls, total_latency_ms, total_tokens, status)
               VALUES (2, 2, 'BadSkill', '{}', '{}', 0, 100, 100000, 200000, 'active')"""
        )
        await governor.conn.commit()
        score = await governor.score_skill(2)
        assert score < 40.0

    async def test_record_call_updates_stats(self, governor):
        await _insert_discovery(governor, 3)
        await governor.conn.execute(
            """INSERT INTO deployed_skills (id, discovery_id, name, dag_definition, param_template,
               success_count, total_calls, total_latency_ms, total_tokens, status)
               VALUES (3, 3, 'TestSkill', '{}', '{}', 0, 0, 0, 0, 'active')"""
        )
        await governor.conn.commit()
        await governor.record_call(3, success=True, latency_ms=100, tokens=50)
        cursor = await governor.conn.execute("SELECT * FROM deployed_skills WHERE id=3")
        row = await cursor.fetchone()
        skill = dict(row)
        assert skill["success_count"] == 1
        assert skill["total_calls"] == 1
        assert skill["total_latency_ms"] == 100

    async def test_promote_canary_to_next_level(self, governor):
        await _insert_discovery(governor, 4)
        await governor.conn.execute(
            """INSERT INTO deployed_skills (id, discovery_id, name, dag_definition, param_template,
               success_count, total_calls, total_latency_ms, total_tokens, status)
               VALUES (4, 4, 'CanarySkill', '{}', '{}', 50, 50, 5000, 25000, 'canary_5')"""
        )
        await governor.conn.commit()
        new_status = await governor.promote(4)
        assert new_status == "canary_15"

    async def test_demote_to_deprecated(self, governor):
        await _insert_discovery(governor, 5)
        await governor.conn.execute(
            """INSERT INTO deployed_skills (id, discovery_id, name, dag_definition, param_template,
               success_count, total_calls, total_latency_ms, total_tokens, status)
               VALUES (5, 5, 'DoomedSkill', '{}', '{}', 0, 100, 100000, 200000, 'canary_5')"""
        )
        await governor.conn.commit()
        new_status = await governor.demote(5, "low score")
        assert new_status == "deprecated"

    async def test_ab_compare_removed(self, governor):
        assert not hasattr(governor, "ab_compare")

    async def test_idle_decay(self, governor):
        await _insert_discovery(governor, 7)
        await governor.conn.execute(
            """INSERT INTO deployed_skills (id, discovery_id, name, dag_definition, param_template,
               success_count, total_calls, total_latency_ms, total_tokens, status, last_used_at)
               VALUES (7, 7, 'IdleSkill', '{}', '{}', 80, 100, 10000, 50000, 'active',
               datetime('now', '-30 days'))"""
        )
        await governor.conn.commit()
        score = await governor.score_skill(7)
        assert score < 100.0  # decayed from 100
