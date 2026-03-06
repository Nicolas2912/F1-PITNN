from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_name: str = "pitnn"
    seed: int = 42
    deterministic: bool = True
    device: str = "cpu"


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    telemetry_hz: int = 10
    horizon_seconds: int = 5


class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    runtime: RuntimeConfig = RuntimeConfig()
    data: DataConfig = DataConfig()


def load_app_config(path: Path) -> AppConfig:
    raw = yaml.safe_load(path.read_text())
    return AppConfig.model_validate(raw)


def load_runtime_config(path: Path) -> RuntimeConfig:
    return load_app_config(path).runtime
