from __future__ import annotations

from pathlib import Path

from pitnn.config import RuntimeConfig, load_runtime_config
from pitnn.reproducibility import set_global_seed


def main() -> None:
    config_path = Path("configs/base.yaml")
    config: RuntimeConfig = load_runtime_config(config_path)
    set_global_seed(config.seed, deterministic=config.deterministic)
    print(
        "PITNN high-fidelity harness ready "
        f"(seed={config.seed}, deterministic={config.deterministic}, device={config.device})"
    )


__all__ = ["main"]
