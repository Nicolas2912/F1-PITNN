from __future__ import annotations

import math

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
