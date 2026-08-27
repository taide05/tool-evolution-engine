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


class TestRuntimeRules:
    async def test_only_runtime_rule_types_returned(self, engine):
        for rtype in ("range_rule", "auth_rule", "retry_rule",
                      "timeout_rule", "circuit_breaker_rule"):
            await engine.add_rule({
                "tool_name": "runtime_tool", "tool_version": "1.0.0",
                "rule_type": rtype, "condition": {"x": 1},
                "action": {"y": 2}, "status": "active",
            })
        rules = await engine.get_runtime_rules("runtime_tool", "1.0.0")
        assert {r["rule_type"] for r in rules} == {
            "retry_rule", "timeout_rule", "circuit_breaker_rule"
        }

    async def test_deprecated_excluded(self, engine):
        await engine.add_rule({
            "tool_name": "runtime_tool", "tool_version": "1.0.0",
            "rule_type": "timeout_rule", "condition": {},
            "action": {}, "status": "deprecated",
        })
        rules = await engine.get_runtime_rules("runtime_tool", "1.0.0")
        assert rules == []

    async def test_condition_action_parsed_to_dict(self, engine):
        await engine.add_rule({
            "tool_name": "runtime_tool", "tool_version": "1.0.0",
            "rule_type": "retry_rule",
            "condition": {"on_error": "quota_exhausted"},
            "action": {"delay_seconds": 60, "max_retries": 3},
            "status": "active",
        })
        rules = await engine.get_runtime_rules("runtime_tool", "1.0.0")
        assert rules[0]["condition"] == {"on_error": "quota_exhausted"}
        assert rules[0]["action"] == {"delay_seconds": 60, "max_retries": 3}


class TestRangeRuleBounds:
    async def test_check_range_rule_respects_template_bounds(self, engine, db_conn):
        from tool_evolution.knowledge.param_template import ParamTemplateManager
        await ParamTemplateManager(db_conn).save("search", "1.0.0", {
            "max_results": {"param_type": "int", "default_value": 10,
                            "lower_bound": 0, "upper_bound": 100,
                            "sample_count": 50}})
        await engine.add_rule({
            "tool_name": "search", "tool_version": "1.0.0",
            "rule_type": "range_rule",
            "condition": {"param_names": ["max_results"]},
            "action": {"validate_before_call": True},
            "status": "active"})
        assert await engine.check("search", "1.0.0", {"max_results": 5}) == []
        assert len(await engine.check("search", "1.0.0", {"max_results": 999})) >= 1
