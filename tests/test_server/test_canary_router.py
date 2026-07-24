import pytest
from tool_evolution.governance.canary_router import CanaryRouter


class TestCanaryRouterDecision:
    def test_active_always_canary(self):
        router = CanaryRouter(None)
        results = [router.decide(f"req-{i}", "active") for i in range(100)]
        assert all(v == "canary" for v in results)

    def test_offline_always_stable(self):
        router = CanaryRouter(None)
        results = [router.decide(f"req-{i}", "offline") for i in range(100)]
        assert all(v == "stable" for v in results)

    def test_deprecated_always_stable(self):
        router = CanaryRouter(None)
        results = [router.decide(f"req-{i}", "deprecated") for i in range(100)]
        assert all(v == "stable" for v in results)

    def test_consistent_hashing_same_input_same_output(self):
        router = CanaryRouter(None)
        h = "user-abc-session-123"
        decisions = [router.decide(h, "canary_50") for _ in range(20)]
        assert len(set(decisions)) == 1

    def test_canary_5_approx_rate(self):
        router = CanaryRouter(None)
        results = [router.decide(f"req-{i}", "canary_5") for i in range(2000)]
        canary_count = sum(1 for v in results if v == "canary")
        # With 2000 samples, expect roughly 100 canary (5%). Allow wide tolerance.
        assert 40 <= canary_count <= 160

    def test_canary_15_approx_rate(self):
        router = CanaryRouter(None)
        results = [router.decide(f"req-{i}", "canary_15") for i in range(2000)]
        canary_count = sum(1 for v in results if v == "canary")
        assert 200 <= canary_count <= 400

    def test_canary_50_approx_rate(self):
        router = CanaryRouter(None)
        results = [router.decide(f"req-{i}", "canary_50") for i in range(2000)]
        canary_count = sum(1 for v in results if v == "canary")
        assert 850 <= canary_count <= 1150

    def test_unknown_status_returns_stable(self):
        router = CanaryRouter(None)
        assert router.decide("req-1", "nonexistent") == "stable"

    def test_canary_pct_mapping(self):
        assert CanaryRouter.canary_pct("canary_5") == 5
        assert CanaryRouter.canary_pct("canary_15") == 15
        assert CanaryRouter.canary_pct("canary_50") == 50
        assert CanaryRouter.canary_pct("active") == 100
        assert CanaryRouter.canary_pct("offline") == 0


class TestCanaryRouterDB:
    async def test_get_skill_returns_none_for_missing(self, db_conn):
        router = CanaryRouter(db_conn)
        skill = await router.get_skill("nonexistent-skill")
        assert skill is None

    async def test_get_skill_returns_deployed(self, db_conn):
        await db_conn.execute(
            "INSERT INTO discovered_skills (id, name, dag_definition, param_template, frequency, status) "
            "VALUES (1, 'test', '{}', '{}', 1.0, 'promoted')"
        )
        await db_conn.execute(
            "INSERT INTO deployed_skills (id, discovery_id, name, dag_definition, param_template, status) "
            "VALUES (1, 1, 'test-skill', '{}', '{}', 'canary_5')"
        )
        await db_conn.commit()
        router = CanaryRouter(db_conn)
        skill = await router.get_skill("test-skill")
        assert skill is not None
        assert skill["status"] == "canary_5"

    async def test_record_and_compare(self, db_conn):
        await db_conn.execute(
            "INSERT INTO discovered_skills (id, name, dag_definition, param_template, frequency, status) "
            "VALUES (2, 'comp', '{}', '{}', 1.0, 'promoted')"
        )
        await db_conn.execute(
            "INSERT INTO deployed_skills (id, discovery_id, name, dag_definition, param_template, status) "
            "VALUES (2, 2, 'compare-skill', '{}', '{}', 'canary_5')"
        )
        await db_conn.commit()
        router = CanaryRouter(db_conn)

        # Record 40 canary (90% success) + 40 stable (80% success)
        for _ in range(40):
            await router.record_invocation(2, "canary", success=True, latency_ms=100, tokens=50)
        for _ in range(4):
            await router.record_invocation(2, "canary", success=False, latency_ms=100, tokens=50)
        for _ in range(40):
            await router.record_invocation(2, "stable", success=True, latency_ms=100, tokens=50)
        for _ in range(10):
            await router.record_invocation(2, "stable", success=False, latency_ms=100, tokens=50)

        result = await router.compare_variants(2, min_samples=30)
        assert result is not None
        assert result["canary_rate"] == pytest.approx(0.9, abs=0.1)
        assert result["stable_rate"] == pytest.approx(0.8, abs=0.1)
        assert result["promote"] is True
        assert result["rollback"] is False

    async def test_compare_insufficient_samples(self, db_conn):
        await db_conn.execute(
            "INSERT INTO discovered_skills (id, name, dag_definition, param_template, frequency, status) "
            "VALUES (3, 'few', '{}', '{}', 1.0, 'promoted')"
        )
        await db_conn.execute(
            "INSERT INTO deployed_skills (id, discovery_id, name, dag_definition, param_template, status) "
            "VALUES (3, 3, 'few-skill', '{}', '{}', 'canary_5')"
        )
        await db_conn.commit()
        router = CanaryRouter(db_conn)
        await router.record_invocation(3, "canary", success=True, latency_ms=100, tokens=50)
        result = await router.compare_variants(3, min_samples=30)
        assert result is None
