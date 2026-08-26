import json

from tool_evolution.execution.assembler import PlanAssembler
from tool_evolution.knowledge.rule_engine import RuleEngine
from tool_evolution.governance.mcp_bridge import MCPBridge


def _skill_row(name="search_api → detail_api", dag=None, param_template=None,
               skill_id=1):
    dag = dag or {"nodes": [{"tool_name": "search_api"},
                            {"tool_name": "detail_api"}],
                  "edges": [{"from": "search_api", "to": "detail_api"}]}
    return {
        "id": skill_id,
        "name": name,
        "dag_definition": json.dumps(dag),
        "param_template": json.dumps(param_template) if param_template else None,
    }


async def _seed_distribution(conn, tool_name, param_name, default_value):
    await conn.execute(
        """INSERT OR REPLACE INTO param_distributions
           (tool_name, tool_version, param_name, param_type, kde_params,
            default_value, lower_bound, upper_bound, sample_count)
           VALUES (?, '1.0.0', ?, 'int', '{}', ?, NULL, NULL, 30)""",
        (tool_name, param_name, json.dumps(default_value)),
    )
    await conn.commit()


class TestPlanAssembler:
    async def test_task_params_used_when_no_template(self, db_conn):
        assembler = PlanAssembler(db_conn)
        plan = await assembler.assemble(_skill_row(), task_params={"query": "hello"})
        assert plan["blocked"] is False
        for node in plan["nodes"]:
            assert node["params"].get("query") == "hello"

    async def test_kde_default_applied(self, db_conn):
        await _seed_distribution(db_conn, "search_api", "max_results", 15)
        assembler = PlanAssembler(db_conn)
        plan = await assembler.assemble(_skill_row())
        search_node = plan["nodes"][0]
        assert search_node["params"].get("max_results") == 15

    async def test_none_default_skipped(self, db_conn):
        await _seed_distribution(db_conn, "search_api", "max_results", None)
        assembler = PlanAssembler(db_conn)
        plan = await assembler.assemble(_skill_row())
        assert "max_results" not in plan["nodes"][0]["params"]

    async def test_preference_overrides_kde(self, db_conn):
        await _seed_distribution(db_conn, "search_api", "max_results", 15)
        await MCPBridge(db_conn)._set_cache(
            "user_preferences", {"search_api": {"max_results": 40}}
        )
        assembler = PlanAssembler(db_conn)
        plan = await assembler.assemble(_skill_row())
        assert plan["nodes"][0]["params"]["max_results"] == 40

    async def test_task_params_highest_priority(self, db_conn):
        await _seed_distribution(db_conn, "search_api", "max_results", 15)
        await MCPBridge(db_conn)._set_cache(
            "user_preferences", {"search_api": {"max_results": 40}}
        )
        assembler = PlanAssembler(db_conn)
        plan = await assembler.assemble(_skill_row(), task_params={"max_results": 99})
        assert plan["nodes"][0]["params"]["max_results"] == 99

    async def test_precheck_blocks(self, db_conn):
        engine = RuleEngine(db_conn)
        await engine.add_rule({
            "tool_name": "search_api", "tool_version": "1.0.0",
            "rule_type": "range_rule",
            "condition": {"param_names": ["max_results"]},
            "action": {"validate_before_call": True},
        })
        assembler = PlanAssembler(db_conn)
        plan = await assembler.assemble(_skill_row(), task_params={"max_results": 5})
        assert plan["blocked"] is True
        assert plan["precheck_rules"]
        assert plan["block_reason"]

    async def test_missing_param_template_ok(self, db_conn):
        assembler = PlanAssembler(db_conn)
        plan = await assembler.assemble(_skill_row(param_template=None))
        assert plan["blocked"] is False
        assert len(plan["nodes"]) == 2

    async def test_unresolvable_edge_blocked(self, db_conn):
        dag = {"nodes": [{"tool_name": "search_api"}],
               "edges": [{"from": "search_api", "to": "ghost_api"}]}
        assembler = PlanAssembler(db_conn)
        plan = await assembler.assemble(_skill_row(dag=dag))
        assert plan["blocked"] is True
        assert "ghost_api" in (plan["block_reason"] or "")

    async def test_edges_resolved_to_indices(self, db_conn):
        assembler = PlanAssembler(db_conn)
        plan = await assembler.assemble(_skill_row())
        assert plan["edges"] == [{"from": 0, "to": 1}]

    async def test_duplicate_tool_suffix_resolution(self, db_conn):
        dag = {"nodes": [{"tool_name": "search_api"}, {"tool_name": "search_api"}],
               "edges": [{"from": "search_api", "to": "search_api_1"}]}
        assembler = PlanAssembler(db_conn)
        plan = await assembler.assemble(_skill_row(dag=dag))
        assert plan["blocked"] is False
        assert plan["edges"] == [{"from": 0, "to": 1}]
