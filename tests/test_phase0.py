from pathlib import Path

from pitnn.config import load_runtime_config
from pitnn.reproducibility import reproducibility_snapshot, set_global_seed


def test_runtime_config_loads() -> None:
    cfg = load_runtime_config(Path("configs/base.yaml"))
    assert cfg.seed == 42
    assert cfg.deterministic is True


def test_reproducibility_snapshot_has_seed() -> None:
    set_global_seed(7, deterministic=True)
    snap = reproducibility_snapshot()
    assert snap["PYTHONHASHSEED"] == "7"
