import pytest
from httpx import AsyncClient, ASGITransport
from tool_evolution.server.app import app
from tool_evolution.server.auth import require_api_key
from tool_evolution.utils.config import settings


class TestRequireApiKey:
    def test_missing_key_raises(self, monkeypatch):
        monkeypatch.setattr(settings, "api_key", None)
        with pytest.raises(RuntimeError, match="TOOLEVO_API_KEY"):
            require_api_key()

    def test_empty_key_raises(self, monkeypatch):
        monkeypatch.setattr(settings, "api_key", "")
        with pytest.raises(RuntimeError, match="TOOLEVO_API_KEY"):
            require_api_key()

    def test_present_key_passes(self, monkeypatch):
        monkeypatch.setattr(settings, "api_key", "k")
        require_api_key()


class TestApiKeyMiddleware:
    async def test_no_key_returns_401(self, setup_db, monkeypatch):
        monkeypatch.setattr(settings, "api_key", "test-key")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            r = await ac.get("/api/rules")
        assert r.status_code == 401

    async def test_wrong_key_returns_401(self, setup_db, monkeypatch):
        monkeypatch.setattr(settings, "api_key", "test-key")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            r = await ac.get("/api/rules", headers={"X-API-Key": "wrong"})
        assert r.status_code == 401

    async def test_correct_key_passes(self, setup_db, monkeypatch):
        monkeypatch.setattr(settings, "api_key", "test-key")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            r = await ac.get("/api/rules", headers={"X-API-Key": "test-key"})
        assert r.status_code == 200

    async def test_health_exempt_even_without_key_set(self, setup_db, monkeypatch):
        # 真"无 key"场景：api_key 未配置时 /health 仍豁免，非 health 端点仍 401
        monkeypatch.setattr(settings, "api_key", None)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            r_health = await ac.get("/health")
            r_rules = await ac.get("/api/rules")
        assert r_health.status_code == 200
        assert r_rules.status_code == 401
