from __future__ import annotations

import os

import numpy as np

try:
    from ._hf_simulator_native import step_flash_field as _native_step_flash_field
    from ._hf_simulator_native import step_sidewall_field as _native_step_sidewall_field
except ImportError:
    _native_step_flash_field = None
    _native_step_sidewall_field = None


def native_simulator_kernels_available() -> bool:
    return _native_step_flash_field is not None and _native_step_sidewall_field is not None


def native_simulator_kernels_enabled() -> bool:
    value = os.getenv("PITNN_USE_NATIVE_SIMULATOR_KERNELS", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def run_native_step_flash_field(
    *,
    flash_field_tw_k: np.ndarray,
    surface_cell_areas_tw: np.ndarray,
    bulk_surface_w_k: np.ndarray,
    road_surface_temp_w_k: np.ndarray,
    zone_weights: np.ndarray,
    theta_indices: np.ndarray,
    width_indices: np.ndarray,
    ambient_temp_k: float,
    friction_to_tire_w: float,
    friction_fraction: float,
    bulk_coupling_time_s: float,
    ambient_cooling_time_s: float,
    patch_relaxation_time_s: float,
    road_cooling_time_s: float,
    areal_heat_capacity_j_per_m2k: float,
    max_delta_above_bulk_k: float,
    dt_s: float,
) -> np.ndarray:
    if _native_step_flash_field is None:
        msg = "Native simulator kernel extension is not available"
        raise RuntimeError(msg)
    result = _native_step_flash_field(
        flash_field_tw_k,
        surface_cell_areas_tw,
        bulk_surface_w_k,
        road_surface_temp_w_k,
        zone_weights,
        theta_indices,
        width_indices,
        ambient_temp_k,
        friction_to_tire_w,
        friction_fraction,
        bulk_coupling_time_s,
        ambient_cooling_time_s,
        patch_relaxation_time_s,
        road_cooling_time_s,
        areal_heat_capacity_j_per_m2k,
        max_delta_above_bulk_k,
        dt_s,
    )
    return np.asarray(result, dtype=float)


def run_native_step_sidewall_field(
    *,
    sidewall_field_tw_k: np.ndarray,
    shoulder_temp_by_zone_k: np.ndarray,
    gas_temp_k: float,
    ambient_temp_k: float,
    rim_temp_k: float,
    solar_w_m2: float,
    wind_mps: float,
    wind_yaw_rad: float,
    wheel_wake_factor: float,
    brake_heat_to_sidewall_w: float,
    dt_s: float,
) -> tuple[np.ndarray, float]:
    if _native_step_sidewall_field is None:
        msg = "Native simulator kernel extension is not available"
        raise RuntimeError(msg)
    result_field, total_heat_w = _native_step_sidewall_field(
        sidewall_field_tw_k,
        shoulder_temp_by_zone_k,
        gas_temp_k,
        ambient_temp_k,
        rim_temp_k,
        solar_w_m2,
        wind_mps,
        wind_yaw_rad,
        wheel_wake_factor,
        brake_heat_to_sidewall_w,
        dt_s,
    )
    return np.asarray(result_field, dtype=float), float(total_heat_w)
