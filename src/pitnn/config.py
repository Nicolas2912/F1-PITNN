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


def load_runtime_config(path: Path) -> RuntimeConfig:
    raw = yaml.safe_load(path.read_text())
    runtime_raw = raw.get("runtime", raw)
    return RuntimeConfig.model_validate(runtime_raw)
