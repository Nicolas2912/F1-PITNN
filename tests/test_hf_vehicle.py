from __future__ import annotations

import math

from models.high_fidelity import (
    HighFidelityTireModelParameters,
    HighFidelityVehicleSimulator,
)
from models.physics import celsius_to_kelvin
from models.vehicle_thermal import VehicleInputs, VehicleParameters


def _input(**overrides: float) -> VehicleInputs:
    payload: dict[str, float] = {
        "speed_mps": 58.0,
        "ax_mps2": 0.0,
        "ay_mps2": 0.0,
        "steering_angle_rad": 0.0,
        "yaw_rate_radps": 0.0,
        "brake_power_w": 0.0,
        "drive_power_w": 0.0,
        "ambient_temp_k": celsius_to_kelvin(30.0),
        "track_temp_k": celsius_to_kelvin(44.0),
    }
    payload.update(overrides)
    return VehicleInputs(**payload)


def _sim() -> HighFidelityVehicleSimulator:
    tire_params = HighFidelityTireModelParameters(
        use_2d_thermal_solver=True,
        no_op_thermal_step=False,
        radial_cells=10,
        theta_cells=20,
        internal_solver_dt_s=0.01,
    )
    return HighFidelityVehicleSimulator(
        tire_parameters_by_wheel={wheel: tire_params for wheel in ("FL", "FR", "RL", "RR")}
    )


def test_p6_load_conservation_and_transfer_signs_hold() -> None:
    sim = _sim()
    state = sim.initial_state(ambient_temp_k=celsius_to_kelvin(30.0))
    diag = sim.diagnostics(state, _input(ax_mps2=-4.0, ay_mps2=5.5))

    assert diag.load_conservation_error_pct < 0.5

    params = VehicleParameters()
    downforce = params.aero_downforce_coeff_n_per_mps2 * (58.0**2)
    front_static = params.mass_kg * params.gravity_mps2 * params.front_static_weight_fraction
    front_static += downforce * params.aero_front_fraction
    assert diag.front_axle_load_n > front_static
    assert diag.right_minus_left_load_n > 0.0


def test_p6_turning_produces_right_left_thermal_asymmetry() -> None:
    sim = _sim()
    left_state = sim.initial_state(ambient_temp_k=celsius_to_kelvin(30.0))
    right_state = sim.initial_state(ambient_temp_k=celsius_to_kelvin(30.0))

    left_turn = _input(speed_mps=56.0, ay_mps2=6.0, steering_angle_rad=0.07, yaw_rate_radps=0.11)
    right_turn = _input(speed_mps=56.0, ay_mps2=-6.0, steering_angle_rad=-0.07, yaw_rate_radps=-0.11)

    for _ in range(120):
        left_state = sim.step(left_state, left_turn, dt_s=0.05)
        right_state = sim.step(right_state, right_turn, dt_s=0.05)

    left_diag = sim.diagnostics(left_state, left_turn)
    right_diag = sim.diagnostics(right_state, right_turn)

    left_turn_right_avg = 0.5 * (
        left_diag.wheel_core_temp_c["FR"] + left_diag.wheel_core_temp_c["RR"]
    )
    left_turn_left_avg = 0.5 * (
        left_diag.wheel_core_temp_c["FL"] + left_diag.wheel_core_temp_c["RL"]
    )
    right_turn_right_avg = 0.5 * (
        right_diag.wheel_core_temp_c["FR"] + right_diag.wheel_core_temp_c["RR"]
    )
    right_turn_left_avg = 0.5 * (
        right_diag.wheel_core_temp_c["FL"] + right_diag.wheel_core_temp_c["RL"]
    )

    assert left_turn_right_avg > left_turn_left_avg
    assert right_turn_left_avg > right_turn_right_avg


def test_p6_braking_shifts_front_load_and_front_torque_bias() -> None:
    params = VehicleParameters(brake_bias_front=0.60)
    tire_params = HighFidelityTireModelParameters(
        use_2d_thermal_solver=True,
        no_op_thermal_step=False,
        radial_cells=10,
        theta_cells=20,
        internal_solver_dt_s=0.01,
    )
    sim = HighFidelityVehicleSimulator(
        parameters=params,
        tire_parameters_by_wheel={wheel: tire_params for wheel in ("FL", "FR", "RL", "RR")},
    )
    state = sim.initial_state(ambient_temp_k=celsius_to_kelvin(30.0))
    diag = sim.diagnostics(state, _input(speed_mps=60.0, ax_mps2=-6.0, brake_power_w=80_000.0))

    front_total = diag.wheel_brake_torque_nm["FL"] + diag.wheel_brake_torque_nm["FR"]
    rear_total = diag.wheel_brake_torque_nm["RL"] + diag.wheel_brake_torque_nm["RR"]
    assert diag.front_axle_load_n > diag.rear_axle_load_n
    assert math.isclose(front_total / (front_total + rear_total), 0.60, rel_tol=0.03, abs_tol=0.03)


def test_p6_front_axle_turning_targets_stay_within_coupling_envelope() -> None:
    sim = _sim()
    state = sim.initial_state(ambient_temp_k=celsius_to_kelvin(30.0))
    control = _input(
        speed_mps=56.0,
        ay_mps2=6.5,
        steering_angle_rad=0.075,
        yaw_rate_radps=0.115,
        drive_power_w=18_000.0,
    )

    state = sim.step(state, control, dt_s=0.2)
    diag = sim.diagnostics(state, control)

    assert diag.wheel_coupling_converged["FL"]
    assert diag.wheel_coupling_converged["FR"]
    assert abs(diag.wheel_effective_slip_angle_rad["FL"]) <= 0.22
    assert abs(diag.wheel_effective_slip_angle_rad["FR"]) <= 0.22


def test_p6_long_run_remains_finite_and_coupling_stays_bounded() -> None:
    sim = _sim()
    state = sim.initial_state(ambient_temp_k=celsius_to_kelvin(30.0))
    control = _input(
        speed_mps=64.0,
        ax_mps2=-2.0,
        ay_mps2=4.0,
        steering_angle_rad=0.08,
        yaw_rate_radps=0.12,
        brake_power_w=40_000.0,
        drive_power_w=15_000.0,
    )

    for _ in range(300):
        state = sim.step(state, control, dt_s=0.05)

    diag = sim.diagnostics(state, control)
    for wheel, wheel_state in state.wheel_states.items():
        assert all(math.isfinite(value) for value in wheel_state.temperature_nodes_k)
        assert 0.0 <= wheel_state.wear <= 1.0
        assert abs(diag.wheel_effective_slip_ratio[wheel]) <= 0.25
        assert abs(diag.wheel_effective_slip_angle_rad[wheel]) <= 0.22
        assert math.isfinite(diag.wheel_core_temp_c[wheel])
