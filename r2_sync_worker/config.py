import os
from functools import lru_cache
from typing import Optional


class Settings:
    r2_endpoint: str
    r2_region: Optional[str]
    r2_access_key_id: str
    r2_secret_access_key: str
    dataset_root: str
    bind_host: str
    bind_port: int

    def __init__(self) -> None:
        self.r2_endpoint = os.environ.get("R2_ENDPOINT", "").rstrip("/")
        self.r2_region = os.environ.get("R2_REGION")
        self.r2_access_key_id = os.environ.get("R2_ACCESS_KEY_ID", "")
        self.r2_secret_access_key = os.environ.get("R2_SECRET_ACCESS_KEY", "")
        self.dataset_root = os.environ.get(
            "AITK_R2_DATASETS_ROOT", "/app/ai-toolkit/datasets/r2"
        )
        self.bind_host = os.environ.get("R2_SYNC_BIND_HOST", "0.0.0.0")
        self.bind_port = int(os.environ.get("R2_SYNC_BIND_PORT", "8080"))

        if not self.r2_endpoint:
            raise RuntimeError("R2_ENDPOINT is required for the sync worker")
        if not self.r2_access_key_id or not self.r2_secret_access_key:
            raise RuntimeError("R2 access key/secret must be provided")


@lru_cache
def get_settings() -> Settings:
    return Settings()

