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

    for _ in range(120):
        state = simulator.step(state, control, dt_s=0.02)

    assert state.thermal_field_rt_k is not None
    assert np.isfinite(state.thermal_field_rt_k).all()
    assert np.isfinite(state.temperature_nodes_k).all()
    assert np.min(state.thermal_field_rt_k) >= params.minimum_temperature_k
    assert np.max(state.thermal_field_rt_k) <= params.maximum_temperature_k
    assert state.last_solver_substeps >= 1
    assert state.last_energy_residual_pct < 1.0

    diag = simulator.diagnostics(state, control)
    assert diag.thermal_grid_shape == (10, 20)
    assert diag.solver_substeps >= 1
    assert math.isfinite(diag.energy_residual_pct)
    assert math.isclose(
        diag.surface_temperature_k,
        float(np.mean(state.thermal_field_rt_k[-1, :])),
        rel_tol=0.0,
        abs_tol=1e-9,
    )


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

    for _ in range(60):
        baseline_state = simulator.step(baseline_state, baseline_control, dt_s=0.02)
        brake_state = simulator.step(brake_state, brake_control, dt_s=0.02)

    baseline_diag = simulator.diagnostics(baseline_state, baseline_control)
    brake_diag = simulator.diagnostics(brake_state, brake_control)

    assert brake_diag.brake_heat_to_tire_w > 0.0
    assert brake_diag.brake_heat_to_rim_w > 0.0
    assert brake_state.temperature_nodes_k[6] > baseline_state.temperature_nodes_k[6]
    assert np.mean(brake_state.thermal_field_rt_k[0, :]) > np.mean(baseline_state.thermal_field_rt_k[0, :])


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
    for _ in range(900):
        state = simulator.step(state, control, dt_s=0.02)

    diag = simulator.diagnostics(state, control)
    assert state.core_temperature_k > initial_core_k + 0.05
    assert diag.surface_temperature_k > state.core_temperature_k
