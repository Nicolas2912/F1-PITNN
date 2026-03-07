from __future__ import annotations

import json
import platform
from statistics import median
import sys
import sysconfig
import time
from typing import Any

import numpy as np

ROOT = __import__("pathlib").Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.high_fidelity import HighFidelityTireModelParameters, ThermalFieldSolver2D  # noqa: E402
from models.high_fidelity.native_diffusion import (  # noqa: E402
    native_diffusion_available,
    run_native_diffuse_vectorized_implicit,
)


CASES: dict[str, dict[str, Any]] = {
    "smoke": {"radial_cells": 4, "theta_cells": 8, "width_zones": 3},
    "dev": {"radial_cells": 8, "theta_cells": 16, "width_zones": 3},
    "full": {"radial_cells": 24, "theta_cells": 72, "width_zones": 3},
}


def _kernel_inputs(
    *,
    radial_cells: int,
    theta_cells: int,
    width_zones: int,
    seed: int,
) -> tuple[ThermalFieldSolver2D, dict[str, Any]]:
    params = HighFidelityTireModelParameters(
        radial_cells=radial_cells,
        theta_cells=theta_cells,
        width_zones=width_zones,
        diffusion_max_iterations=10,
        diffusion_tolerance_k=1e-6,
    )
    solver = ThermalFieldSolver2D(params)
    rng = np.random.default_rng(seed)
    field = rng.normal(loc=343.15, scale=7.5, size=(radial_cells, theta_cells, width_zones)).astype(float)
    source = rng.normal(loc=1.2e4, scale=1.5e3, size=field.shape).astype(float)
    rho_cp, k_r, k_theta, k_w, _ = solver.layer_property_maps(
        wear=0.08,
        grain_index_w=np.array([0.05, 0.02, 0.04], dtype=float),
        blister_index_w=np.array([0.01, 0.03, 0.02], dtype=float),
        age_index=0.15,
    )
    return solver, {
        "field": field,
        "source_w_per_m3": source,
        "rho_cp": rho_cp,
        "k_r": k_r,
        "k_theta": k_theta,
        "k_w": k_w,
        "dt_s": 0.01,
    }


def _python_kernel(solver: ThermalFieldSolver2D, payload: dict[str, Any]) -> tuple[np.ndarray, int]:
    return solver._diffuse_vectorized_implicit_python(
        np.array(payload["field"], dtype=float, copy=True),
        source_w_per_m3=np.array(payload["source_w_per_m3"], dtype=float, copy=True),
        rho_cp=np.array(payload["rho_cp"], dtype=float, copy=True),
        k_r=np.array(payload["k_r"], dtype=float, copy=True),
        k_theta=np.array(payload["k_theta"], dtype=float, copy=True),
        k_w=np.array(payload["k_w"], dtype=float, copy=True),
        dt_s=float(payload["dt_s"]),
    )


def _native_kernel(solver: ThermalFieldSolver2D, payload: dict[str, Any]) -> tuple[np.ndarray, int]:
    return run_native_diffuse_vectorized_implicit(
        field=np.array(payload["field"], dtype=float, copy=True),
        source_w_per_m3=np.array(payload["source_w_per_m3"], dtype=float, copy=True),
        rho_cp=np.array(payload["rho_cp"], dtype=float, copy=True),
        k_r=np.array(payload["k_r"], dtype=float, copy=True),
        k_theta=np.array(payload["k_theta"], dtype=float, copy=True),
        k_w=np.array(payload["k_w"], dtype=float, copy=True),
        dt_s=float(payload["dt_s"]),
        radial_coeff_minus=np.array(solver._radial_coeff_minus, dtype=float, copy=True),
        radial_coeff_plus=np.array(solver._radial_coeff_plus, dtype=float, copy=True),
        theta_coeff=np.array(solver._theta_coeff, dtype=float, copy=True),
        width_coeff_minus=np.array(solver._width_coeff_minus, dtype=float, copy=True),
        width_coeff_plus=np.array(solver._width_coeff_plus, dtype=float, copy=True),
        diffusion_max_iterations=int(solver.parameters.diffusion_max_iterations),
        diffusion_tolerance_k=float(solver.parameters.diffusion_tolerance_k),
    )


def _time_kernel(label: str, solver: ThermalFieldSolver2D, payload: dict[str, Any], repeats: int) -> dict[str, Any]:
    runner = _native_kernel if label == "native" else _python_kernel
    times: list[float] = []
    last_iterations = 0
    for _ in range(repeats):
        start = time.perf_counter()
        _field, last_iterations = runner(solver, payload)
        times.append(time.perf_counter() - start)
    return {
        "median_runtime_s": median(times),
        "best_runtime_s": min(times),
        "iterations": last_iterations,
    }


def main() -> None:
    payload: dict[str, Any] = {
        "environment": {
            "python_version": platform.python_version(),
            "python_compiler": platform.python_compiler(),
            "numpy_version": np.__version__,
            "compiler": sysconfig.get_config_var("CC"),
            "cpu_architecture": platform.machine(),
            "native_diffusion_available": native_diffusion_available(),
        },
        "cases": {},
    }

    for idx, (name, config) in enumerate(CASES.items()):
        solver, kernel_payload = _kernel_inputs(seed=20260307 + idx, **config)
        case_result = {
            "shape": list(kernel_payload["field"].shape),
            "python": _time_kernel("python", solver, kernel_payload, repeats=7),
        }
        if native_diffusion_available():
            native_result = _time_kernel("native", solver, kernel_payload, repeats=7)
            case_result["native"] = native_result
            case_result["speedup_pct"] = (
                (case_result["python"]["median_runtime_s"] - native_result["median_runtime_s"])
                / case_result["python"]["median_runtime_s"]
                * 100.0
            )
        payload["cases"][name] = case_result

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
