from __future__ import annotations

import os
import numpy as np

try:
    from ._hf_diffusion_native import diffuse_vectorized_implicit as _native_diffuse_vectorized_implicit
    from ._hf_diffusion_native import build_source_field as _native_build_source_field
    from ._hf_diffusion_native import build_source_and_diffuse_implicit as _native_build_source_and_diffuse_implicit
    from ._hf_diffusion_native import thermal_step_multi_substep as _native_thermal_step_multi_substep
except ImportError:
    _native_diffuse_vectorized_implicit = None
    _native_build_source_field = None
    _native_build_source_and_diffuse_implicit = None
    _native_thermal_step_multi_substep = None


def native_diffusion_available() -> bool:
    return (
        _native_diffuse_vectorized_implicit is not None
        and _native_build_source_field is not None
        and _native_build_source_and_diffuse_implicit is not None
    )


def native_multi_substep_available() -> bool:
    return _native_thermal_step_multi_substep is not None


def native_diffusion_enabled() -> bool:
    value = os.getenv("PITNN_USE_NATIVE_DIFFUSION", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _ensure_float_array(value: np.ndarray) -> np.ndarray:
    if isinstance(value, np.ndarray) and value.dtype == float:
        return value
    return np.asarray(value, dtype=float)


def run_native_diffuse_vectorized_implicit(
    *,
    field: np.ndarray,
    source_w_per_m3: np.ndarray,
    rho_cp: np.ndarray,
    k_r: np.ndarray,
    k_theta: np.ndarray,
    k_w: np.ndarray,
    dt_s: float,
    radial_coeff_minus: np.ndarray,
    radial_coeff_plus: np.ndarray,
    theta_coeff: np.ndarray,
    width_coeff_minus: np.ndarray,
    width_coeff_plus: np.ndarray,
    diffusion_max_iterations: int,
    diffusion_tolerance_k: float,
) -> tuple[np.ndarray, int]:
    if _native_diffuse_vectorized_implicit is None:
        msg = "Native diffusion extension is not available"
        raise RuntimeError(msg)
    result_field, iterations = _native_diffuse_vectorized_implicit(
        field,
        source_w_per_m3,
        rho_cp,
        k_r,
        k_theta,
        k_w,
        dt_s,
        radial_coeff_minus,
        radial_coeff_plus,
        theta_coeff,
        width_coeff_minus,
        width_coeff_plus,
        diffusion_max_iterations,
        diffusion_tolerance_k,
    )
    return _ensure_float_array(result_field), int(iterations)


def run_native_build_source_field(
    *,
    radial_cells: int,
    theta_cells: int,
    width_zones: int,
    source_volumetric_fraction: float,
    volumetric_source_w_per_m3: float,
    wheel_angular_speed_radps: float,
    time_s: float,
    theta_delta_rad: float,
    patch_radial_indices: np.ndarray,
    theta_offsets: np.ndarray,
    width_indices: np.ndarray,
    layer_index: np.ndarray,
    zone_weights: np.ndarray | None,
    layer_source_weights: np.ndarray | None,
) -> np.ndarray:
    if _native_build_source_field is None:
        msg = "Native diffusion extension is not available"
        raise RuntimeError(msg)
    result = _native_build_source_field(
        radial_cells,
        theta_cells,
        width_zones,
        source_volumetric_fraction,
        volumetric_source_w_per_m3,
        wheel_angular_speed_radps,
        time_s,
        theta_delta_rad,
        patch_radial_indices,
        theta_offsets,
        width_indices,
        layer_index,
        zone_weights,
        layer_source_weights,
    )
    return _ensure_float_array(result)


def run_native_build_source_and_diffuse_implicit(
    *,
    field: np.ndarray,
    extra_source_w_per_m3: np.ndarray | None,
    rho_cp: np.ndarray,
    k_r: np.ndarray,
    k_theta: np.ndarray,
    k_w: np.ndarray,
    dt_s: float,
    radial_coeff_minus: np.ndarray,
    radial_coeff_plus: np.ndarray,
    theta_coeff: np.ndarray,
    width_coeff_minus: np.ndarray,
    width_coeff_plus: np.ndarray,
    diffusion_max_iterations: int,
    diffusion_tolerance_k: float,
    source_volumetric_fraction: float,
    volumetric_source_w_per_m3: float,
    wheel_angular_speed_radps: float,
    time_s: float,
    theta_delta_rad: float,
    patch_radial_indices: np.ndarray,
    theta_offsets: np.ndarray,
    width_indices: np.ndarray,
    layer_index: np.ndarray,
    zone_weights: np.ndarray | None,
    layer_source_weights: np.ndarray | None,
) -> tuple[np.ndarray, int, np.ndarray]:
    if _native_build_source_and_diffuse_implicit is None:
        msg = "Native diffusion extension is not available"
        raise RuntimeError(msg)
    result_field, iterations, source_field = _native_build_source_and_diffuse_implicit(
        field,
        extra_source_w_per_m3,
        rho_cp,
        k_r,
        k_theta,
        k_w,
        dt_s,
        radial_coeff_minus,
        radial_coeff_plus,
        theta_coeff,
        width_coeff_minus,
        width_coeff_plus,
        diffusion_max_iterations,
        diffusion_tolerance_k,
        source_volumetric_fraction,
        volumetric_source_w_per_m3,
        wheel_angular_speed_radps,
        time_s,
        theta_delta_rad,
        patch_radial_indices,
        theta_offsets,
        width_indices,
        layer_index,
        zone_weights,
        layer_source_weights,
    )
    return (
        _ensure_float_array(result_field),
        int(iterations),
        _ensure_float_array(source_field),
    )


def run_native_thermal_step_multi_substep(
    *,
    field: np.ndarray,
    extra_source_w_per_m3: np.ndarray | None,
    cell_volumes_m3: np.ndarray,
    dt_s: float,
    substeps: int,
    radial_coeff_minus: np.ndarray,
    radial_coeff_plus: np.ndarray,
    theta_coeff: np.ndarray,
    width_coeff_minus: np.ndarray,
    width_coeff_plus: np.ndarray,
    diffusion_max_iterations: int,
    diffusion_tolerance_k: float,
    source_volumetric_fraction: float,
    volumetric_source_w_per_m3: float,
    wheel_angular_speed_radps: float,
    time_s: float,
    theta_delta_rad: float,
    patch_radial_indices: np.ndarray,
    theta_offsets: np.ndarray,
    width_indices: np.ndarray,
    layer_slices: np.ndarray,
    volumetric_heat_capacity_by_layer: np.ndarray,
    k_r_base_by_layer: np.ndarray,
    k_theta_base_by_layer: np.ndarray,
    k_w_base_by_layer: np.ndarray,
    shoulder_bias_by_layer: np.ndarray,
    center_bias_by_layer: np.ndarray,
    bead_bias_by_layer: np.ndarray,
    temp_sensitivity_by_layer: np.ndarray,
    wear_sensitivity_by_layer: np.ndarray,
    reinforcement_density_by_layer: np.ndarray,
    cord_angle_deg_by_layer: np.ndarray,
    grain_index_w: np.ndarray,
    blister_index_w: np.ndarray,
    wear: float,
    age_index: float,
    construction_enabled: bool,
    construction_bead_width_fraction: float,
    construction_temp_reference_k: float,
    tread_blister_conductivity_penalty: float,
    zone_weights: np.ndarray | None,
    layer_source_weights: np.ndarray | None,
    minimum_temperature_k: float,
    maximum_temperature_k: float,
    enable_profiling: bool,
) -> tuple[np.ndarray, int, float, float, float, float, float]:
    if _native_thermal_step_multi_substep is None:
        msg = "Native multi-substep thermal extension is not available"
        raise RuntimeError(msg)
    result_field, iterations, source_energy_j, initial_energy_j, final_energy_j, advection_time_s, diffusion_time_s = _native_thermal_step_multi_substep(
        field,
        extra_source_w_per_m3,
        cell_volumes_m3,
        dt_s,
        substeps,
        radial_coeff_minus,
        radial_coeff_plus,
        theta_coeff,
        width_coeff_minus,
        width_coeff_plus,
        diffusion_max_iterations,
        diffusion_tolerance_k,
        source_volumetric_fraction,
        volumetric_source_w_per_m3,
        wheel_angular_speed_radps,
        time_s,
        theta_delta_rad,
        patch_radial_indices,
        theta_offsets,
        width_indices,
        layer_slices,
        volumetric_heat_capacity_by_layer,
        k_r_base_by_layer,
        k_theta_base_by_layer,
        k_w_base_by_layer,
        shoulder_bias_by_layer,
        center_bias_by_layer,
        bead_bias_by_layer,
        temp_sensitivity_by_layer,
        wear_sensitivity_by_layer,
        reinforcement_density_by_layer,
        cord_angle_deg_by_layer,
        grain_index_w,
        blister_index_w,
        wear,
        age_index,
        construction_enabled,
        construction_bead_width_fraction,
        construction_temp_reference_k,
        tread_blister_conductivity_penalty,
        zone_weights,
        layer_source_weights,
        minimum_temperature_k,
        maximum_temperature_k,
        enable_profiling,
    )
    return (
        _ensure_float_array(result_field),
        int(iterations),
        float(source_energy_j),
        float(initial_energy_j),
        float(final_energy_j),
        float(advection_time_s),
        float(diffusion_time_s),
    )
