from __future__ import annotations

import math
import time

import numpy as np

from models.high_fidelity import (
    HighFidelityTireInputs,
    HighFidelityTireModelParameters,
    HighFidelityTireSimulator,
    ThermalFieldSolver2D,
)
from models.physics import celsius_to_kelvin


def _inputs(**overrides: float) -> HighFidelityTireInputs:
    payload: dict[str, float] = {
        "speed_mps": 58.0,
        "wheel_angular_speed_radps": 185.0,
        "normal_load_n": 3_800.0,
        "slip_ratio_cmd": 0.08,
        "slip_angle_cmd_rad": 0.05,
        "ambient_temp_k": celsius_to_kelvin(30.0),
        "track_temp_k": celsius_to_kelvin(44.0),
    }
    payload.update(overrides)
    return HighFidelityTireInputs(**payload)


def _legacy_advect_theta_upwind(
    solver: ThermalFieldSolver2D,
    field: np.ndarray,
    *,
    omega: float,
    dt_s: float,
) -> np.ndarray:
    theta_delta = (2.0 * math.pi) / max(solver.parameters.theta_cells, 1)
    if abs(omega) <= 1e-12:
        return field
    cfl = abs(omega) * dt_s / max(theta_delta, 1e-12)
    cfl = min(cfl, solver.parameters.advection_cfl_limit)
    if omega >= 0.0:
        return field - cfl * (field - np.roll(field, 1, axis=1))
    return field - cfl * (np.roll(field, -1, axis=1) - field)


def _legacy_diffuse_implicit(
    solver: ThermalFieldSolver2D,
    field: np.ndarray,
    *,
    source_w_per_m3: np.ndarray,
    rho_cp: np.ndarray,
    k_r: np.ndarray,
    k_theta: np.ndarray,
    k_w: np.ndarray,
    dt_s: float,
) -> np.ndarray:
    rhs = field + dt_s * source_w_per_m3 / np.maximum(rho_cp, 1e-12)
    estimate = rhs.copy()
    radial_cells, theta_cells, width_zones = estimate.shape

    for _ in range(max(solver.parameters.diffusion_max_iterations, 1)):
        max_delta = 0.0
        for r in range(radial_cells):
            for t in range(theta_cells):
                t_minus = (t - 1) % theta_cells
                t_plus = (t + 1) % theta_cells
                for w in range(width_zones):
                    alpha_r = k_r[r, t, w] / max(rho_cp[r, t, w], 1e-12)
                    alpha_theta = k_theta[r, t, w] / max(rho_cp[r, t, w], 1e-12)
                    alpha_w = k_w[r, t, w] / max(rho_cp[r, t, w], 1e-12)

                    coeff_r_minus = alpha_r * solver._radial_coeff_minus[r]
                    coeff_r_plus = alpha_r * solver._radial_coeff_plus[r]
                    coeff_theta = alpha_theta * solver._theta_coeff[r]
                    coeff_w_minus = alpha_w * solver._width_coeff_minus[w]
                    coeff_w_plus = alpha_w * solver._width_coeff_plus[w]

                    neighbor_sum = 0.0
                    diagonal = 1.0
                    if r > 0:
                        neighbor_sum += coeff_r_minus * estimate[r - 1, t, w]
                        diagonal += dt_s * coeff_r_minus
                    if r < radial_cells - 1:
                        neighbor_sum += coeff_r_plus * estimate[r + 1, t, w]
                        diagonal += dt_s * coeff_r_plus

                    neighbor_sum += coeff_theta * estimate[r, t_minus, w]
                    neighbor_sum += coeff_theta * estimate[r, t_plus, w]
                    diagonal += 2.0 * dt_s * coeff_theta

                    if w > 0:
                        neighbor_sum += coeff_w_minus * estimate[r, t, w - 1]
                        diagonal += dt_s * coeff_w_minus
                    if w < width_zones - 1:
                        neighbor_sum += coeff_w_plus * estimate[r, t, w + 1]
                        diagonal += dt_s * coeff_w_plus

                    updated = (rhs[r, t, w] + dt_s * neighbor_sum) / max(diagonal, 1e-12)
                    max_delta = max(max_delta, abs(updated - estimate[r, t, w]))
                    estimate[r, t, w] = updated
        if max_delta < solver.parameters.diffusion_tolerance_k:
            break
    return estimate


def _legacy_solver_step(
    solver: ThermalFieldSolver2D,
    field: np.ndarray,
    *,
    inputs: HighFidelityTireInputs,
    time_s: float,
    dt_s: float,
    volumetric_source_w_per_m3: float,
) -> np.ndarray:
    rho_cp, k_r, k_theta, k_w, _ = solver.layer_property_maps(wear=0.0)
    omega = inputs.wheel_angular_speed_radps
    max_cfl = abs(omega) * dt_s / max((2.0 * math.pi) / max(solver.parameters.theta_cells, 1), 1e-9)
    substeps_dt = max(1, math.ceil(dt_s / max(solver.parameters.internal_solver_dt_s, 1e-6)))
    substeps_cfl = max(1, math.ceil(max_cfl / max(solver.parameters.advection_cfl_limit, 1e-6)))
    substeps = min(max(substeps_dt, substeps_cfl), max(solver.parameters.max_solver_substeps, 1))
    dt_sub = dt_s / substeps
    out = field.copy()
    for sub_idx in range(substeps):
        t_sub = time_s + sub_idx * dt_sub
        source = solver.source_field_w_per_m3(
            volumetric_source_w_per_m3=volumetric_source_w_per_m3,
            wheel_angular_speed_radps=omega,
            time_s=t_sub,
        )
        out = _legacy_advect_theta_upwind(solver, out, omega=omega, dt_s=dt_sub)
        out = _legacy_diffuse_implicit(
            solver,
            out,
            source_w_per_m3=source,
            rho_cp=rho_cp,
            k_r=k_r,
            k_theta=k_theta,
            k_w=k_w,
            dt_s=dt_sub,
        )
    return out


def _legacy_boundary_source_field_w_per_m3(
    simulator: HighFidelityTireSimulator,
    *,
    thermal_field_rtw_k: np.ndarray,
    zone_friction_to_tire_w: np.ndarray,
    road_conduction_w_by_zone: np.ndarray,
    rim_conduction_w: float,
    brake_heat_to_tire_w: float,
    zone_weights: np.ndarray,
    wheel_angular_speed_radps: float,
    time_s: float,
) -> np.ndarray:
    del zone_weights
    source = np.zeros_like(thermal_field_rtw_k)
    flash_fraction = 0.0
    if simulator.parameters.flash_layer.enabled:
        flash_fraction = float(np.clip(simulator.parameters.flash_layer.friction_fraction, 0.0, 0.95))
    bulk_fraction = 1.0 - flash_fraction
    cell_volumes = simulator._thermal_solver.cell_volumes_m3
    _, theta_indices, width_indices = simulator._thermal_solver.contact_patch_indices(
        wheel_angular_speed_radps=wheel_angular_speed_radps,
        time_s=time_s,
    )
    radial_cells = thermal_field_rtw_k.shape[0]
    deep_start = max(int(round(0.48 * (radial_cells - 1))), 0)
    bulk_radial_indices = np.arange(deep_start, radial_cells, dtype=int)
    radial_positions = np.linspace(0.0, 1.0, bulk_radial_indices.shape[0], dtype=float)
    radial_weights = 0.15 + 0.85 * radial_positions**1.4
    radial_weights /= max(float(np.sum(radial_weights)), 1e-12)
    for zone_idx, width_idx in enumerate(width_indices):
        patch_power_w = bulk_fraction * zone_friction_to_tire_w[zone_idx] - road_conduction_w_by_zone[zone_idx]
        for local_idx, radial_idx in enumerate(bulk_radial_indices):
            patch_index = np.ix_(np.array([radial_idx], dtype=int), theta_indices, np.array([width_idx], dtype=int))
            patch_volume = float(np.sum(cell_volumes[patch_index]))
            source[patch_index] += patch_power_w * radial_weights[local_idx] / max(patch_volume, 1e-12)
    inner_ring_volume = float(np.sum(cell_volumes[0, :, :]))
    source[0, :, :] += (brake_heat_to_tire_w - rim_conduction_w) / max(inner_ring_volume, 1e-12)
    return source


def test_p3_radial_grid_is_nonuniform_and_biased_toward_outer_radius() -> None:
    params = HighFidelityTireModelParameters(
        radial_cells=16,
        theta_cells=32,
        radial_spacing_bias=2.2,
    )
    solver = ThermalFieldSolver2D(params)
    centers = solver.radial_centers_m
    dr_inner = centers[1] - centers[0]
    dr_outer = centers[-1] - centers[-2]

    assert centers.shape == (16,)
    assert dr_outer < dr_inner


def test_p3_solver_source_window_moves_with_rotation_phase() -> None:
    params = HighFidelityTireModelParameters(
        radial_cells=12,
        theta_cells=36,
    )
    solver = ThermalFieldSolver2D(params)
    omega = 210.0
    source_0 = solver.source_field_w_per_m3(
        volumetric_source_w_per_m3=12_000.0,
        wheel_angular_speed_radps=omega,
        time_s=0.0,
    )
    source_later = solver.source_field_w_per_m3(
        volumetric_source_w_per_m3=12_000.0,
        wheel_angular_speed_radps=omega,
        time_s=0.01,
    )

    assert not np.array_equal(source_0, source_later)
    assert np.any(np.abs(source_0[-1, :] - source_later[-1, :]) > 1e-12)


def test_p3_boundary_source_matches_legacy_distribution() -> None:
    params = HighFidelityTireModelParameters(
        use_2d_thermal_solver=True,
        no_op_thermal_step=False,
        radial_cells=36,
        theta_cells=72,
        width_zones=3,
    )
    simulator = HighFidelityTireSimulator(params)
    thermal_field = simulator._thermal_solver.initial_temperature_field(celsius_to_kelvin(37.0))
    zone_friction_to_tire_w = np.array([1800.0, 2250.0, 1675.0], dtype=float)
    road_conduction_w_by_zone = np.array([240.0, 275.0, 210.0], dtype=float)
    zone_weights = np.array([0.25, 0.45, 0.30], dtype=float)

    for time_s in np.linspace(0.0, 0.19, 9, dtype=float):
        legacy = _legacy_boundary_source_field_w_per_m3(
            simulator,
            thermal_field_rtw_k=thermal_field,
            zone_friction_to_tire_w=zone_friction_to_tire_w,
            road_conduction_w_by_zone=road_conduction_w_by_zone,
            rim_conduction_w=380.0,
            brake_heat_to_tire_w=1250.0,
            zone_weights=zone_weights,
            wheel_angular_speed_radps=185.0,
            time_s=float(time_s),
        )
        optimized = simulator._boundary_source_field_w_per_m3(
            thermal_field_rtw_k=thermal_field,
            zone_friction_to_tire_w=zone_friction_to_tire_w,
            road_conduction_w_by_zone=road_conduction_w_by_zone,
            rim_conduction_w=380.0,
            brake_heat_to_tire_w=1250.0,
            zone_weights=zone_weights,
            wheel_angular_speed_radps=185.0,
            time_s=float(time_s),
        )

        assert np.allclose(optimized, legacy, rtol=0.0, atol=1e-7)


def test_p3_boundary_source_fast_path_is_not_slower_than_legacy() -> None:
    params = HighFidelityTireModelParameters(
        use_2d_thermal_solver=True,
        no_op_thermal_step=False,
        radial_cells=36,
        theta_cells=72,
        width_zones=3,
    )
    simulator = HighFidelityTireSimulator(params)
    thermal_field = simulator._thermal_solver.initial_temperature_field(celsius_to_kelvin(37.0))
    zone_friction_to_tire_w = np.array([1800.0, 2250.0, 1675.0], dtype=float)
    road_conduction_w_by_zone = np.array([240.0, 275.0, 210.0], dtype=float)
    zone_weights = np.array([0.25, 0.45, 0.30], dtype=float)
    time_samples = np.linspace(0.0, 0.45, 120, dtype=float)

    start = time.perf_counter()
    legacy_checksum = 0.0
    for time_s in time_samples:
        legacy_checksum += float(
            np.sum(
                _legacy_boundary_source_field_w_per_m3(
                    simulator,
                    thermal_field_rtw_k=thermal_field,
                    zone_friction_to_tire_w=zone_friction_to_tire_w,
                    road_conduction_w_by_zone=road_conduction_w_by_zone,
                    rim_conduction_w=380.0,
                    brake_heat_to_tire_w=1250.0,
                    zone_weights=zone_weights,
                    wheel_angular_speed_radps=185.0,
                    time_s=float(time_s),
                )
            )
        )
    legacy_elapsed_s = time.perf_counter() - start

    start = time.perf_counter()
    optimized_checksum = 0.0
    for time_s in time_samples:
        optimized_checksum += float(
            np.sum(
                simulator._boundary_source_field_w_per_m3(
                    thermal_field_rtw_k=thermal_field,
                    zone_friction_to_tire_w=zone_friction_to_tire_w,
                    road_conduction_w_by_zone=road_conduction_w_by_zone,
                    rim_conduction_w=380.0,
                    brake_heat_to_tire_w=1250.0,
                    zone_weights=zone_weights,
                    wheel_angular_speed_radps=185.0,
                    time_s=float(time_s),
                )
            )
        )
    optimized_elapsed_s = time.perf_counter() - start

    assert math.isclose(optimized_checksum, legacy_checksum, rel_tol=0.0, abs_tol=1e-6)
    assert optimized_elapsed_s <= legacy_elapsed_s * 0.99


def test_p3_solver_energy_residual_is_small_in_nominal_step() -> None:
    params = HighFidelityTireModelParameters(
        radial_cells=10,
        theta_cells=24,
        internal_solver_dt_s=0.01,
        use_2d_thermal_solver=True,
    )
    solver = ThermalFieldSolver2D(params)
    field = solver.initial_temperature_field(celsius_to_kelvin(30.0))
    result = solver.step(
        temperature_field_rt_k=field,
        inputs=_inputs(wheel_angular_speed_radps=170.0),
        time_s=0.0,
        dt_s=0.02,
        volumetric_source_w_per_m3=8_500.0,
    )

    assert np.isfinite(result.temperature_field_rt_k).all()
    assert result.substeps >= 1
    assert result.energy_residual_pct < 1.0


def test_p3_solver_adi_path_stays_close_to_legacy_reference_on_small_grid() -> None:
    params = HighFidelityTireModelParameters(
        radial_cells=4,
        theta_cells=8,
        internal_solver_dt_s=0.05,
        use_2d_thermal_solver=True,
    )
    solver = ThermalFieldSolver2D(params)
    field = solver.initial_temperature_field(celsius_to_kelvin(30.0))
    inputs = _inputs(wheel_angular_speed_radps=140.0)

    legacy = _legacy_solver_step(
        solver,
        field,
        inputs=inputs,
        time_s=0.0,
        dt_s=0.05,
        volumetric_source_w_per_m3=8_500.0,
    )
    modern = solver.step(
        temperature_field_rtw_k=field,
        inputs=inputs,
        time_s=0.0,
        dt_s=0.05,
        volumetric_source_w_per_m3=8_500.0,
    ).temperature_field_rtw_k

    assert np.max(np.abs(modern - legacy)) < 3.0
    assert abs(float(np.mean(modern)) - float(np.mean(legacy))) < 0.5


def test_p3_solver_exposes_profiling_fields_when_enabled() -> None:
    params = HighFidelityTireModelParameters(
        radial_cells=6,
        theta_cells=12,
        internal_solver_dt_s=0.02,
        use_2d_thermal_solver=True,
        no_op_thermal_step=False,
        enable_profiling=True,
    )
    simulator = HighFidelityTireSimulator(params)
    state = simulator.initial_state(ambient_temp_k=celsius_to_kelvin(30.0))
    state = simulator.step(state, _inputs(), dt_s=0.05)
    diag = simulator.diagnostics(state, _inputs())

    assert state.last_solver_advection_time_s is not None
    assert state.last_solver_diffusion_time_s is not None
    assert state.last_solver_diffusion_iterations is not None
    assert state.last_solver_diffusion_iterations >= 3
    assert state.last_wheel_coupling_time_s is not None
    assert diag.solver_advection_time_s is not None
    assert diag.solver_diffusion_time_s is not None
    assert diag.wheel_coupling_time_s is not None


def test_p3_simulator_2d_mode_is_finite_and_bounded() -> None:
    params = HighFidelityTireModelParameters(
        use_2d_thermal_solver=True,
        no_op_thermal_step=False,
        radial_cells=10,
        theta_cells=20,
        internal_solver_dt_s=0.01,
    )
    simulator = HighFidelityTireSimulator(params)
    state = simulator.initial_state(ambient_temp_k=celsius_to_kelvin(30.0))
    control = _inputs(slip_ratio_cmd=0.11, slip_angle_cmd_rad=0.07)

    for _ in range(30):
        state = simulator.step(state, control, dt_s=0.02)

    assert state.thermal_field_rt_k is not None
    assert state.thermal_field_rtw_k is not None
    assert np.isfinite(state.thermal_field_rt_k).all()
    assert np.isfinite(state.thermal_field_rtw_k).all()
    assert np.isfinite(state.temperature_nodes_k).all()
    assert np.min(state.thermal_field_rt_k) >= params.minimum_temperature_k
    assert np.max(state.thermal_field_rt_k) <= params.maximum_temperature_k
    assert state.last_solver_substeps >= 1
    assert state.last_energy_residual_pct < 1.0

    diag = simulator.diagnostics(state, control)
    assert diag.thermal_grid_shape == (10, 20, 3)
    assert diag.solver_substeps >= 1
    assert math.isfinite(diag.energy_residual_pct)
    assert math.isclose(
        diag.surface_temperature_k,
        float(np.mean(state.thermal_field_rtw_k[-1, :, :])),
        rel_tol=0.0,
        abs_tol=1e-9,
    )
    assert len(diag.per_width_surface_temp_k) == 3
    assert len(diag.per_width_flash_surface_temp_k) == 3
    assert len(diag.per_width_bulk_temp_k) == 3


def test_p3_brake_heat_path_warms_inner_tire_and_rim() -> None:
    params = HighFidelityTireModelParameters(
        use_2d_thermal_solver=True,
        no_op_thermal_step=False,
        radial_cells=10,
        theta_cells=20,
        internal_solver_dt_s=0.01,
    )
    simulator = HighFidelityTireSimulator(params)
    baseline_state = simulator.initial_state(ambient_temp_k=celsius_to_kelvin(30.0))
    brake_state = simulator.initial_state(ambient_temp_k=celsius_to_kelvin(30.0))
    baseline_control = _inputs(brake_power_w=0.0)
    brake_control = _inputs(brake_power_w=40_000.0)

    for _ in range(20):
        baseline_state = simulator.step(baseline_state, baseline_control, dt_s=0.02)
        brake_state = simulator.step(brake_state, brake_control, dt_s=0.02)

    baseline_diag = simulator.diagnostics(baseline_state, baseline_control)
    brake_diag = simulator.diagnostics(brake_state, brake_control)

    assert brake_diag.brake_heat_to_tire_w > 0.0
    assert brake_diag.brake_heat_to_rim_w > 0.0
    assert brake_state.temperature_nodes_k[6] > baseline_state.temperature_nodes_k[6]
    assert np.mean(brake_state.thermal_field_rtw_k[0, :, :]) > np.mean(baseline_state.thermal_field_rtw_k[0, :, :])


def test_p3_core_proxy_responds_on_longer_horizon_but_lags_surface() -> None:
    params = HighFidelityTireModelParameters(
        use_2d_thermal_solver=True,
        no_op_thermal_step=False,
        radial_cells=10,
        theta_cells=20,
        internal_solver_dt_s=0.01,
    )
    simulator = HighFidelityTireSimulator(params)
    state = simulator.initial_state(ambient_temp_k=celsius_to_kelvin(30.0))
    control = _inputs(
        speed_mps=54.0,
        slip_ratio_cmd=0.09,
        slip_angle_cmd_rad=0.06,
        brake_power_w=18_000.0,
        track_temp_k=celsius_to_kelvin(46.0),
    )

    initial_core_k = state.core_temperature_k
    for _ in range(60):
        state = simulator.step(state, control, dt_s=0.05)

    diag = simulator.diagnostics(state, control)
    assert state.core_temperature_k >= initial_core_k
    assert diag.surface_temperature_k > state.core_temperature_k
    assert diag.flash_surface_temperature_k is not None
    assert diag.bulk_core_temperature_k is not None
    assert diag.cavity_gas_temperature_k is not None


def test_p3_width_resolved_field_preserves_three_zone_information() -> None:
    params = HighFidelityTireModelParameters(
        use_2d_thermal_solver=True,
        no_op_thermal_step=False,
        radial_cells=10,
        theta_cells=20,
        internal_solver_dt_s=0.01,
    )
    simulator = HighFidelityTireSimulator(params)
    state = simulator.initial_state(ambient_temp_k=celsius_to_kelvin(30.0))
    control = _inputs(
        speed_mps=52.0,
        slip_ratio_cmd=0.07,
        slip_angle_cmd_rad=0.04,
        track_temp_k=celsius_to_kelvin(42.0),
    )

    for _ in range(20):
        state = simulator.step(state, control, dt_s=0.05)

    assert state.thermal_field_rtw_k is not None
    assert state.thermal_field_rtw_k.shape == (10, 20, 3)
    diag = simulator.diagnostics(state, control)
    assert len(diag.per_width_surface_temp_k) == 3
    assert len(diag.per_width_flash_surface_temp_k) == 3
    assert len(diag.layer_mean_temp_k) >= 4


def test_p3_flash_layer_runs_hotter_than_bulk_surface_under_slip_heating() -> None:
    params = HighFidelityTireModelParameters(
        use_2d_thermal_solver=True,
        no_op_thermal_step=False,
        radial_cells=10,
        theta_cells=20,
        internal_solver_dt_s=0.01,
    )
    simulator = HighFidelityTireSimulator(params)
    state = simulator.initial_state(ambient_temp_k=celsius_to_kelvin(30.0))
    control = _inputs(
        speed_mps=60.0,
        slip_ratio_cmd=0.14,
        slip_angle_cmd_rad=0.09,
        track_temp_k=celsius_to_kelvin(47.0),
    )

    for _ in range(25):
        state = simulator.step(state, control, dt_s=0.03)

    diag = simulator.diagnostics(state, control)
    assert state.flash_temperature_field_tw_k is not None
    assert diag.flash_surface_temperature_k is not None
    assert np.max(state.flash_temperature_field_tw_k) > float(np.mean(state.thermal_field_rtw_k[-1, :, :]))
    assert diag.flash_surface_temperature_k - diag.surface_temperature_k > 2.0


def test_p3_uniform_full_resolution_field_remains_uniform_without_sources() -> None:
    params = HighFidelityTireModelParameters(
        use_2d_thermal_solver=True,
        no_op_thermal_step=False,
        radial_cells=24,
        theta_cells=72,
        internal_solver_dt_s=0.01,
    )
    solver = ThermalFieldSolver2D(params)
    field = solver.initial_temperature_field(celsius_to_kelvin(31.0))
    control = _inputs(
        speed_mps=60.0,
        wheel_angular_speed_radps=181.8,
        normal_load_n=3_900.0,
        slip_ratio_cmd=0.0,
        slip_angle_cmd_rad=0.0,
        ambient_temp_k=celsius_to_kelvin(31.0),
        track_temp_k=celsius_to_kelvin(31.0),
        road_surface_temp_k=celsius_to_kelvin(31.0),
        road_bulk_temp_k=celsius_to_kelvin(31.0),
        wind_mps=0.0,
    )

    result = solver.step(
        temperature_field_rtw_k=field,
        inputs=control,
        time_s=0.0,
        dt_s=0.2,
        volumetric_source_w_per_m3=0.0,
    )

    assert np.allclose(result.temperature_field_rtw_k, field, atol=1e-9, rtol=0.0)
    assert result.energy_residual_pct < 1e-9


def test_p3_structural_hysteresis_source_matches_solver_deposition_volume() -> None:
    params = HighFidelityTireModelParameters(
        use_2d_thermal_solver=True,
        no_op_thermal_step=False,
        use_local_temp_friction_partition=True,
        use_reduced_patch_mechanics=True,
        use_structural_hysteresis_model=True,
        radial_cells=20,
        theta_cells=60,
        internal_solver_dt_s=0.01,
    )
    simulator = HighFidelityTireSimulator(params)
    state = simulator.initial_state(ambient_temp_k=celsius_to_kelvin(31.0))
    control = _inputs(
        speed_mps=60.0,
        wheel_angular_speed_radps=181.8,
        normal_load_n=3_900.0,
        slip_ratio_cmd=0.10,
        slip_angle_cmd_rad=0.085,
        ambient_temp_k=celsius_to_kelvin(31.0),
        track_temp_k=celsius_to_kelvin(47.0),
        road_surface_temp_k=celsius_to_kelvin(47.0),
        road_bulk_temp_k=celsius_to_kelvin(42.0),
        wind_mps=7.0,
        solar_w_m2=260.0,
        rubbering_level=0.72,
        asphalt_roughness=1.05,
        asphalt_effusivity=1.08,
    )

    for _ in range(4):
        state = simulator.step(state, control, dt_s=0.2)

    diag = simulator.diagnostics(state, control)
    assert diag.surface_temperature_k - celsius_to_kelvin(47.0) < 120.0
    assert diag.flash_surface_temperature_k is not None
    assert diag.flash_surface_temperature_k - diag.surface_temperature_k < 80.0
