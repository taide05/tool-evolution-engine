from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="TOOLEVO_")

    db_path: Path = Path("data/engine.db")
    batch_size: int = 100
    flush_interval_s: int = 5
    min_samples: int = 30
    scan_window_days: int = 30
    max_dag_nodes: int = 10
    min_support: float = 0.05
    credit_update_interval_s: int = 3600
    idle_decay_days: int = 7
    canary_threshold: int = 50
    ab_rollback_margin: float = 0.10
    canary_check_interval_s: int = 300
    canary_min_samples: int = 30
    min_pref_samples: int = 20
    pref_share_threshold: float = 0.6
    deepseek_api_key: str | None = None
    deepseek_base_url: str = "https://api.deepseek.com"
    repair_llm_model: str = "deepseek-v4-flash"  # deepseek-chat 已于 2026-07-24 停用（官方迁移：chat→v4-flash）
    repair_concurrency: int = 4
    repair_timeout_s: float = 30.0
    repair_retries: int = 2
    api_key: str | None = None
    log_level: str = "INFO"


settings = Settings()
