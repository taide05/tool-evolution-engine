from httpx import AsyncClient, ASGITransport
from tool_evolution.server.app import app
from tool_evolution.server.deps import get_db


class TestHealthProbe:
    async def test_health_ok_with_db(self, setup_db):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            r = await ac.get("/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}

    async def test_health_degraded_when_db_down(self, setup_db):
        # 依赖解析阶段抛异常会走 exception handler（500）而非端点内 try/except——
        # 必须 override 一个"已关闭连接"，让 SELECT 1 在端点体内失败
        import aiosqlite
        async def _bad_db():
            conn = await aiosqlite.connect(":memory:")
            await conn.close()
            yield conn
        app.dependency_overrides[get_db] = _bad_db
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            r = await ac.get("/health")
        assert r.status_code == 503
        assert r.json() == {"status": "degraded"}
        app.dependency_overrides.pop(get_db, None)
