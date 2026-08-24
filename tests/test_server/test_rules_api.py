import json
import pytest


@pytest.fixture
async def seeded(setup_db):
    # 审阅修正：必须用 setup_db 的连接播种（app 走 dependency_overrides 的同一连接），
    # 不能用 tests/conftest.py 的 db_conn（那是另一个 :memory: 库，app 看不到）
    conn = setup_db
    cursor = await conn.execute(
        """INSERT INTO rules (tool_name, tool_version, rule_type, condition, action)
           VALUES ('repair_api', '1.0.0', 'range_rule', ?, ?)""",
        (json.dumps({"param_names": ["max_results"]}),
         json.dumps({"validate_before_call": True,
                     "error_msg_template": "max_results must be between 1 and 20, got 30"})))
    rule_id = cursor.lastrowid
    await conn.execute(
        """INSERT INTO repair_hints (rule_id, content_hash, suggestion, fix, model)
           VALUES (?, 'abc123', '检查 max_results 取值范围',
                   ?, 'deepseek-chat')""",
        (rule_id, json.dumps({"param": "max_results", "suggested_value": 10})))
    await conn.commit()
    return rule_id


class TestRulesHintApi:
    async def test_get_hint_200(self, client, seeded):
        resp = await client.get(f"/api/rules/{seeded}/hint")
        assert resp.status_code == 200
        body = resp.json()
        assert body["hint"]["rule_id"] == seeded
        assert body["hint"]["fix"] == {"param": "max_results", "suggested_value": 10}

    async def test_get_hint_404_unknown_rule(self, client):
        resp = await client.get("/api/rules/9999/hint")
        assert resp.status_code == 404

    async def test_get_hint_404_rule_without_hint(self, client, setup_db):
        cursor = await setup_db.execute(
            """INSERT INTO rules (tool_name, tool_version, rule_type, condition, action)
               VALUES ('repair_fetch', '1.0.0', 'timeout_rule', '{}', '{}')""")
        rule_id = cursor.lastrowid
        await setup_db.commit()
        resp = await client.get(f"/api/rules/{rule_id}/hint")
        assert resp.status_code == 404
