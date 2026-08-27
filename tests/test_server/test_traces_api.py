class TestTracesAPI:
    async def test_health(self, client):
        r = await client.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    async def test_report_trace(self, client):
        import uuid
        r = await client.post("/api/traces/report", json={
            "trace_id": f"api-{uuid.uuid4().hex[:8]}", "agent_id": "test", "tool_name": "search",
            "success": True, "latency_ms": 42
        })
        assert r.status_code == 200

    async def test_seed_traces(self, client):
        import uuid
        sid1, sid2 = f"seed-{uuid.uuid4().hex[:8]}", f"seed-{uuid.uuid4().hex[:8]}"
        r = await client.post("/api/traces/seed", json=[
            {"trace_id": sid1, "agent_id": "test", "tool_name": "search", "success": True, "latency_ms": 10},
            {"trace_id": sid2, "agent_id": "test", "tool_name": "search", "success": False, "latency_ms": 500,
             "error_type": "timeout", "error_message": "timeout"},
        ])
        assert r.status_code == 200
        assert r.json()["count"] == 2

    async def test_analytics_summary(self, client):
        r = await client.get("/api/analytics/summary")
        assert r.status_code == 200
        data = r.json()
        assert "total_traces" in data


class TestPreferenceLearnEntry:
    async def test_report_triggers_learn_every_20th(self, setup_db, client, monkeypatch):
        from tool_evolution.server.routes import traces as traces_module

        learn_calls = {"n": 0}

        async def fake_learn_bg():
            learn_calls["n"] += 1

        # 测节流调度接线；真实学习函数体由 test_preference_learner 覆盖
        monkeypatch.setattr(traces_module, "_learn_prefs_bg", fake_learn_bg)
        traces_module._learn_counter = 0
        for i in range(19):
            resp = await client.post("/api/traces/report", json={
                "trace_id": f"p{i}", "agent_id": "a", "tool_name": "search_api",
                "success": True, "latency_ms": 10})
            assert resp.status_code == 200
        assert learn_calls["n"] == 0
        resp = await client.post("/api/traces/report", json={
            "trace_id": "p20", "agent_id": "a", "tool_name": "search_api",
            "success": True, "latency_ms": 10})
        assert resp.status_code == 200
        for _ in range(100):
            if learn_calls["n"] == 1:
                break
            await asyncio.sleep(0.01)
        assert learn_calls["n"] == 1
