class TestSkillsAPI:
    async def test_list_discoveries_empty(self, client):
        r = await client.get("/api/skills/discoveries")
        assert r.status_code == 200
        assert r.json()["discoveries"] == []

    async def test_list_deployed_empty(self, client):
        r = await client.get("/api/skills/deployed")
        assert r.status_code == 200
        assert r.json()["skills"] == []

    async def test_list_rules(self, client):
        r = await client.get("/api/rules")
        assert r.status_code == 200
        assert "rules" in r.json()


class TestInvokeRouting:
    async def test_invoke_routes_without_writes(self, client, setup_db):
        conn = setup_db
        await conn.execute(
            "INSERT INTO discovered_skills (id, name, dag_definition, param_template, frequency, status) "
            "VALUES (1, 'inv', '{}', '{}', 1.0, 'promoted')"
        )
        await conn.execute(
            "INSERT INTO deployed_skills (id, discovery_id, name, dag_definition, param_template, status) "
            "VALUES (1, 1, 'inv-skill', '{}', '{}', 'canary_50')"
        )
        await conn.commit()

        r1 = await client.post("/api/skills/inv-skill/invoke", json={"params": {"query": "a", "n": 1}})
        r2 = await client.post("/api/skills/inv-skill/invoke", json={"params": {"n": 1, "query": "a"}})
        assert r1.status_code == 200
        body = r1.json()
        assert body["variant"] in ("stable", "canary")
        assert body["result"]["status"] == "routed"
        assert body["variant"] == r2.json()["variant"]  # 参数顺序无关（md5 sort_keys）

        cursor = await conn.execute("SELECT COUNT(*) FROM canary_invocations")
        assert (await cursor.fetchone())[0] == 0
        cursor = await conn.execute("SELECT total_calls FROM deployed_skills WHERE id=1")
        assert (await cursor.fetchone())[0] == 0

    async def test_invoke_missing_skill_404(self, client):
        r = await client.post("/api/skills/nope/invoke", json={})
        assert r.status_code == 404
