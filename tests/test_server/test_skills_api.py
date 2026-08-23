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
