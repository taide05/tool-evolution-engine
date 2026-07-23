import json
import pytest
from tool_evolution.knowledge.skill_pack import SkillPackManager


@pytest.fixture
async def manager(db_conn):
    return SkillPackManager(db_conn)


class TestSkillPackManager:
    async def test_add_discovery(self, manager):
        did = await manager.add_discovery({
            "name": "search → analyze",
            "dag_definition": json.dumps({"nodes": [{"tool_name": "search"}, {"tool_name": "analyze"}], "edges": [{"from": "search", "to": "analyze"}]}),
            "param_template": json.dumps({}),
            "frequency": 0.15,
        })
        assert did is not None
        discoveries = await manager.list_discoveries()
        assert len(discoveries) == 1
        assert discoveries[0]["status"] == "canary"

    async def test_promote_to_deployed(self, manager):
        did = await manager.add_discovery({
            "name": "ComplianceCheck",
            "dag_definition": json.dumps({"nodes": [{"tool_name": "t"}], "edges": []}),
            "param_template": json.dumps({"lang": "zh"}),
            "frequency": 0.20,
        })
        dep_id = await manager.promote_to_deployed(did)
        assert dep_id is not None
        deployed = await manager.get_deployed("ComplianceCheck")
        assert deployed is not None
        assert deployed["status"] == "canary_5"

    async def test_list_deployed_by_status(self, manager):
        did1 = await manager.add_discovery({
            "name": "SkillA", "dag_definition": json.dumps({"nodes": [], "edges": []}),
            "param_template": json.dumps({}), "frequency": 0.1,
        })
        did2 = await manager.add_discovery({
            "name": "SkillB", "dag_definition": json.dumps({"nodes": [], "edges": []}),
            "param_template": json.dumps({}), "frequency": 0.1,
        })
        await manager.promote_to_deployed(did1)
        await manager.promote_to_deployed(did2)
        skills = await manager.list_deployed("canary_5")
        assert len(skills) == 2
