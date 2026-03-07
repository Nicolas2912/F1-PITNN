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
    run_native_build_source_and_diffuse_implicit,
    run_native_build_source_field,
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


def _source_inputs(
    solver: ThermalFieldSolver2D,
    *,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    _, _, _, _, layer_index = solver.layer_property_maps(
        wear=0.08,
        grain_index_w=np.array([0.05, 0.02, 0.04], dtype=float),
        blister_index_w=np.array([0.01, 0.03, 0.02], dtype=float),
        age_index=0.15,
    )
    return {
        "wheel_angular_speed_radps": 187.0,
        "time_s": 0.37,
        "volumetric_source_w_per_m3": 18_500.0,
        "extra_source_w_per_m3": rng.normal(loc=420.0, scale=35.0, size=solver._scratch_shape).astype(float),
        "zone_weights": np.array([0.22, 0.51, 0.27], dtype=float),
        "layer_index": layer_index,
        "layer_source_weights": np.asarray([12_000.0, 8_500.0, 7_200.0, 5_600.0, 4_300.0], dtype=float),
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


def _python_source_and_diffuse(
    solver: ThermalFieldSolver2D,
    payload: dict[str, Any],
    source_payload: dict[str, Any],
) -> tuple[np.ndarray, int]:
    source = solver.source_field_w_per_m3(
        volumetric_source_w_per_m3=float(source_payload["volumetric_source_w_per_m3"]),
        wheel_angular_speed_radps=float(source_payload["wheel_angular_speed_radps"]),
        time_s=float(source_payload["time_s"]),
        zone_weights=np.array(source_payload["zone_weights"], dtype=float, copy=True),
        layer_source_weights={
            "tread": float(source_payload["layer_source_weights"][0]),
            "belt": float(source_payload["layer_source_weights"][1]),
            "carcass": float(source_payload["layer_source_weights"][2]),
            "sidewall": float(source_payload["layer_source_weights"][3]),
            "inner_liner": float(source_payload["layer_source_weights"][4]),
        },
    )
    source = source + np.array(source_payload["extra_source_w_per_m3"], dtype=float, copy=True)
    return solver._diffuse_vectorized_implicit_python(
        np.array(payload["field"], dtype=float, copy=True),
        source_w_per_m3=source,
        rho_cp=np.array(payload["rho_cp"], dtype=float, copy=True),
        k_r=np.array(payload["k_r"], dtype=float, copy=True),
        k_theta=np.array(payload["k_theta"], dtype=float, copy=True),
        k_w=np.array(payload["k_w"], dtype=float, copy=True),
        dt_s=float(payload["dt_s"]),
    )


def _native_source_and_diffuse(
    solver: ThermalFieldSolver2D,
    payload: dict[str, Any],
    source_payload: dict[str, Any],
) -> tuple[np.ndarray, int]:
    result_field, iterations, _source = run_native_build_source_and_diffuse_implicit(
        field=np.array(payload["field"], dtype=float, copy=True),
        extra_source_w_per_m3=np.array(source_payload["extra_source_w_per_m3"], dtype=float, copy=True),
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
        source_volumetric_fraction=float(solver.parameters.source_volumetric_fraction),
        volumetric_source_w_per_m3=float(source_payload["volumetric_source_w_per_m3"]),
        wheel_angular_speed_radps=float(source_payload["wheel_angular_speed_radps"]),
        time_s=float(source_payload["time_s"]),
        theta_delta_rad=float(solver._theta_delta_rad),
        patch_radial_indices=np.array(solver._patch_radial_indices, dtype=int, copy=True),
        theta_offsets=np.array(solver._theta_offsets, dtype=int, copy=True),
        width_indices=np.array(solver._width_indices, dtype=int, copy=True),
        layer_index=np.array(source_payload["layer_index"], dtype=int, copy=True),
        zone_weights=np.array(source_payload["zone_weights"], dtype=float, copy=True),
        layer_source_weights=np.array(source_payload["layer_source_weights"], dtype=float, copy=True),
    )
    return result_field, iterations


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
        source_payload = _source_inputs(solver, seed=20260407 + idx)
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
            python_source_field = solver.source_field_w_per_m3(
                volumetric_source_w_per_m3=float(source_payload["volumetric_source_w_per_m3"]),
                wheel_angular_speed_radps=float(source_payload["wheel_angular_speed_radps"]),
                time_s=float(source_payload["time_s"]),
                zone_weights=np.array(source_payload["zone_weights"], dtype=float, copy=True),
                layer_source_weights={
                    "tread": float(source_payload["layer_source_weights"][0]),
                    "belt": float(source_payload["layer_source_weights"][1]),
                    "carcass": float(source_payload["layer_source_weights"][2]),
                    "sidewall": float(source_payload["layer_source_weights"][3]),
                    "inner_liner": float(source_payload["layer_source_weights"][4]),
                },
            )
            native_source_field = run_native_build_source_field(
                radial_cells=solver.parameters.radial_cells,
                theta_cells=solver.parameters.theta_cells,
                width_zones=solver.parameters.width_zones,
                source_volumetric_fraction=float(solver.parameters.source_volumetric_fraction),
                volumetric_source_w_per_m3=float(source_payload["volumetric_source_w_per_m3"]),
                wheel_angular_speed_radps=float(source_payload["wheel_angular_speed_radps"]),
                time_s=float(source_payload["time_s"]),
                theta_delta_rad=float(solver._theta_delta_rad),
                patch_radial_indices=np.array(solver._patch_radial_indices, dtype=int, copy=True),
                theta_offsets=np.array(solver._theta_offsets, dtype=int, copy=True),
                width_indices=np.array(solver._width_indices, dtype=int, copy=True),
                layer_index=np.array(source_payload["layer_index"], dtype=int, copy=True),
                zone_weights=np.array(source_payload["zone_weights"], dtype=float, copy=True),
                layer_source_weights=np.array(source_payload["layer_source_weights"], dtype=float, copy=True),
            )
            if not np.array_equal(python_source_field, native_source_field):
                raise RuntimeError(f"Native source field differs for case {name}")
            python_times = []
            native_times = []
            for _ in range(7):
                start = time.perf_counter()
                _python_source_and_diffuse(solver, kernel_payload, source_payload)
                python_times.append(time.perf_counter() - start)
                start = time.perf_counter()
                _native_source_and_diffuse(solver, kernel_payload, source_payload)
                native_times.append(time.perf_counter() - start)
            case_result["source_plus_diffuse"] = {
                "python_median_runtime_s": median(python_times),
                "native_median_runtime_s": median(native_times),
                "improvement_s": median(python_times) - median(native_times),
                "improvement_pct": (
                    (median(python_times) - median(native_times)) / median(python_times) * 100.0
                    if median(python_times) > 0.0
                    else 0.0
                ),
            }
        payload["cases"][name] = case_result

    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
