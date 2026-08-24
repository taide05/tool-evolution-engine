from tool_evolution.utils.config import Settings, settings


class TestRepairSettings:
    def test_repair_defaults(self):
        assert settings.repair_llm_model == "deepseek-chat"
        assert settings.repair_concurrency == 4
        assert settings.repair_timeout_s == 30.0
        assert settings.repair_retries == 2
        assert settings.deepseek_base_url == "https://api.deepseek.com"

    def test_repair_env_prefix(self, monkeypatch):
        monkeypatch.setenv("TOOLEVO_DEEPSEEK_API_KEY", "sk-test")
        s = Settings()
        assert s.deepseek_api_key == "sk-test"
