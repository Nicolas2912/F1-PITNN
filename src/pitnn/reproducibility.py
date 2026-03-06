from __future__ import annotations

import os
import random

import numpy as np


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Set process-wide seeds and deterministic hints for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def reproducibility_snapshot() -> dict[str, str]:
    """Expose environment reproducibility values for logs/metadata."""
    return {
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", ""),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ""),
    }
