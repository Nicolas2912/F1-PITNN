from __future__ import annotations

import math

import numpy as np
import pytest

from models.high_fidelity import (
    HighFidelityTireInputs,
    HighFidelityTireModelParameters,
    HighFidelityTireSimulator,
    WheelForceCouplingModel,
)
from models.physics import celsius_to_kelvin


def _inputs(**overrides: float | None) -> HighFidelityTireInputs:
    payload: dict[str, float | None] = {
        "speed_mps": 60.0,
        "wheel_angular_speed_radps": 185.0,
        "normal_load_n": 3_800.0,
        "slip_ratio_cmd": 0.09,
        "slip_angle_cmd_rad": 0.055,
        "ambient_temp_k": celsius_to_kelvin(30.0),
        "track_temp_k": celsius_to_kelvin(42.0),
    }
    payload.update(overrides)
    return HighFidelityTireInputs(**payload)


def test_p5_wheel_coupling_converges_within_iteration_budget_for_nominal_case() -> None:
    params = HighFidelityTireModelParameters(
        use_reduced_patch_mechanics=True,
        use_local_temp_friction_partition=True,
    )
    model = WheelForceCouplingModel(params)

    baseline = model.solve(
        inputs=_inputs(),
        surface_temp_k=celsius_to_kelvin(70.0),
    )
    driven = model.solve(
        inputs=_inputs(
            drive_torque_nm=0.92 * baseline.net_wheel_torque_nm,
            lateral_force_target_n=0.88 * baseline.lateral_force_n,
        ),
        surface_temp_k=celsius_to_kelvin(70.0),
    )

    assert driven.converged
    assert 1 <= driven.iterations <= params.max_coupling_iterations
    assert abs(driven.effective_slip_ratio) <= params.max_effective_slip_ratio
    assert abs(driven.effective_slip_angle_rad) <= params.max_effective_slip_angle_rad
    assert abs(driven.torque_residual_nm) <= params.coupling_torque_tolerance_nm
    assert abs(driven.lateral_force_residual_n) <= params.coupling_force_tolerance_n


def test_p5_wheel_coupling_fallback_is_deterministic_and_bounded_when_not_converged() -> None:
    params = HighFidelityTireModelParameters(
        use_reduced_patch_mechanics=True,
        use_local_temp_friction_partition=True,
        max_coupling_iterations=1,
        max_effective_slip_ratio=0.18,
        max_effective_slip_angle_rad=0.12,
    )
    model = WheelForceCouplingModel(params)
    stressed_inputs = _inputs(
        drive_torque_nm=8_000.0,
        lateral_force_target_n=9_000.0,
    )

    first = model.solve(
        inputs=stressed_inputs,
        surface_temp_k=celsius_to_kelvin(75.0),
    )
    second = model.solve(
        inputs=stressed_inputs,
        surface_temp_k=celsius_to_kelvin(75.0),
    )

    assert not first.converged
    assert first.iterations == 1
    assert math.isclose(first.effective_slip_ratio, second.effective_slip_ratio, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(
        first.effective_slip_angle_rad,
        second.effective_slip_angle_rad,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert abs(first.effective_slip_ratio) <= params.max_effective_slip_ratio
    assert abs(first.effective_slip_angle_rad) <= params.max_effective_slip_angle_rad


def test_p5_simulator_reports_coupled_effective_slips_and_force_balance() -> None:
    params = HighFidelityTireModelParameters(
        use_2d_thermal_solver=True,
        no_op_thermal_step=False,
        use_reduced_patch_mechanics=True,
        use_local_temp_friction_partition=True,
        use_structural_hysteresis_model=True,
        radial_cells=10,
        theta_cells=20,
        internal_solver_dt_s=0.01,
    )
    simulator = HighFidelityTireSimulator(params)
    state = simulator.initial_state(ambient_temp_k=celsius_to_kelvin(30.0))

    pre_model = WheelForceCouplingModel(params)
    baseline = pre_model.solve(
        inputs=_inputs(),
        surface_temp_k=celsius_to_kelvin(65.0),
    )
    control = _inputs(
        drive_torque_nm=0.85 * baseline.net_wheel_torque_nm,
        lateral_force_target_n=0.90 * baseline.lateral_force_n,
    )

    next_state = simulator.step(state, control, dt_s=0.05)
    diag = simulator.diagnostics(next_state, control)

    assert diag.coupling_iterations >= 1
    assert diag.coupling_iterations <= params.max_coupling_iterations
    assert abs(diag.effective_slip_ratio) <= params.max_effective_slip_ratio
    assert abs(diag.effective_slip_angle_rad) <= params.max_effective_slip_angle_rad
    assert math.isfinite(diag.longitudinal_force_n)
    assert math.isfinite(diag.lateral_force_n)
    assert math.isfinite(diag.torque_residual_nm)
    assert math.isfinite(diag.lateral_force_residual_n)
    assert diag.friction_power_total_w >= 0.0
    assert diag.contact_patch_length_m > 0.0
    assert diag.contact_patch_width_m > 0.0
    assert 0.0 <= diag.sliding_fraction <= 1.0
    assert diag.effective_mu > 0.0
    assert sum(diag.per_zone_friction_power_w) == pytest.approx(diag.friction_power_total_w, rel=1e-6)
    assert sum(diag.per_zone_friction_power_tire_w) == pytest.approx(diag.friction_power_tire_w, rel=1e-6)
    assert sum(diag.per_zone_friction_power_road_w) == pytest.approx(diag.friction_power_road_w, rel=1e-6)
    assert abs(diag.torque_residual_nm) <= params.coupling_torque_tolerance_nm
    assert abs(diag.lateral_force_residual_n) <= params.coupling_force_tolerance_n


def test_p5_hot_flash_temperature_reduces_available_grip() -> None:
    params = HighFidelityTireModelParameters(
        use_reduced_patch_mechanics=True,
        use_local_temp_friction_partition=True,
    )
    model = WheelForceCouplingModel(params)
    inputs = _inputs(slip_ratio_cmd=0.08, slip_angle_cmd_rad=0.04)

    cooler = model.solve(
        inputs=inputs,
        surface_temp_k=celsius_to_kelvin(85.0),
        bulk_surface_temp_k=celsius_to_kelvin(80.0),
        flash_surface_temp_k=celsius_to_kelvin(85.0),
    )
    overheated = model.solve(
        inputs=inputs,
        surface_temp_k=celsius_to_kelvin(120.0),
        bulk_surface_temp_k=celsius_to_kelvin(90.0),
        flash_surface_temp_k=celsius_to_kelvin(125.0),
    )

    assert overheated.effective_mu < cooler.effective_mu
    assert abs(overheated.longitudinal_force_n) < abs(cooler.longitudinal_force_n)


def test_p5_local_temperature_partition_increases_tire_heat_with_hotter_flash_layer() -> None:
    params = HighFidelityTireModelParameters(
        use_reduced_patch_mechanics=True,
        use_local_temp_friction_partition=True,
    )
    model = WheelForceCouplingModel(params)
    inputs = _inputs(slip_ratio_cmd=0.10, slip_angle_cmd_rad=0.05)

    cooler = model.solve(
        inputs=inputs,
        surface_temp_k=celsius_to_kelvin(88.0),
        bulk_surface_temp_k=celsius_to_kelvin(85.0),
        flash_surface_temp_k=celsius_to_kelvin(90.0),
    )
    hotter = model.solve(
        inputs=inputs,
        surface_temp_k=celsius_to_kelvin(98.0),
        bulk_surface_temp_k=celsius_to_kelvin(90.0),
        flash_surface_temp_k=celsius_to_kelvin(118.0),
    )

    assert np.mean(hotter.zone_effective_mu) < np.mean(cooler.zone_effective_mu)
    assert np.mean(hotter.zone_sliding_fraction) >= np.mean(cooler.zone_sliding_fraction)
