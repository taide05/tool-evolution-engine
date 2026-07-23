from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
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

    class Config:
        env_prefix = "TOOLEVO_"


settings = Settings()
