import pytest
from tool_evolution.knowledge.rule_engine import RuleEngine


@pytest.fixture
async def engine(db_conn):
    return RuleEngine(db_conn)


class TestRuleEngine:
    async def test_add_and_retrieve_rules(self, engine):
        rule = {
            "tool_name": "search", "tool_version": "1.0.0",
            "rule_type": "range_rule",
            "condition": {"param_names": ["max_results"]},
            "action": {"validate_before_call": True},
            "status": "active"
        }
        rid = await engine.add_rule(rule)
        assert rid is not None

    async def test_check_triggers_rule(self, engine):
        await engine.add_rule({
            "tool_name": "search", "tool_version": "1.0.0",
            "rule_type": "range_rule",
            "condition": {"param_names": ["max_results"]},
            "action": {"validate_before_call": True},
            "status": "active"
        })
        triggered = await engine.check("search", "1.0.0", {"max_results": 999})
        assert len(triggered) >= 1
        assert triggered[0]["rule_type"] == "range_rule"

    async def test_check_no_match(self, engine):
        triggered = await engine.check("nonexistent", "1.0.0", {})
        assert triggered == []

    async def test_mark_hit_and_miss(self, engine):
        rid = await engine.add_rule({
            "tool_name": "t", "tool_version": "1.0.0",
            "rule_type": "auth_rule",
            "condition": {}, "action": {}, "status": "active"
        })
        await engine.mark_hit(rid)
        await engine.mark_miss(rid)
        rules = await engine.check("t", "1.0.0", {})
        assert rules[0]["hit_count"] == 1
        assert rules[0]["miss_count"] == 1

    async def test_deprecate_version(self, engine):
        await engine.add_rule({
            "tool_name": "search", "tool_version": "1.0.0",
            "rule_type": "range_rule",
            "condition": {}, "action": {}, "status": "active"
        })
        await engine.deprecate_version("search", "1.0.0")
        triggered = await engine.check("search", "1.0.0", {})
        for r in triggered:
            assert r["status"] == "deprecated"
