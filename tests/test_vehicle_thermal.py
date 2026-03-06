from __future__ import annotations

import math

from models.physics import celsius_to_kelvin
from models.vehicle_thermal import VehicleInputs, VehicleParameters, VehicleThermalSimulator


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


def test_vehicle_phase5_load_sum_consistency() -> None:
    sim = VehicleThermalSimulator()
    state = sim.initial_state(ambient_temp_k=celsius_to_kelvin(30.0))
    diag = sim.diagnostics(state, _input(ax_mps2=-4.0, ay_mps2=5.5))

    assert diag.load_conservation_error_pct < 0.5


def test_vehicle_phase5_front_rear_transfer_sign() -> None:
    params = VehicleParameters()
    sim = VehicleThermalSimulator(parameters=params)
    state = sim.initial_state(ambient_temp_k=celsius_to_kelvin(30.0))

    brake_input = _input(speed_mps=60.0, ax_mps2=-6.0, brake_power_w=90_000.0)
    accel_input = _input(speed_mps=60.0, ax_mps2=3.5, drive_power_w=70_000.0)

    brake_diag = sim.diagnostics(state, brake_input)
    accel_diag = sim.diagnostics(state, accel_input)

    downforce = params.aero_downforce_coeff_n_per_mps2 * (60.0**2)
    front_static = params.mass_kg * params.gravity_mps2 * params.front_static_weight_fraction
    front_static += downforce * params.aero_front_fraction

    assert brake_diag.front_axle_load_n > front_static
    assert accel_diag.front_axle_load_n < front_static


def test_vehicle_phase5_left_right_transfer_sign() -> None:
    sim = VehicleThermalSimulator()
    state = sim.initial_state(ambient_temp_k=celsius_to_kelvin(30.0))

    left_turn_diag = sim.diagnostics(state, _input(speed_mps=52.0, ay_mps2=6.5))
    right_turn_diag = sim.diagnostics(state, _input(speed_mps=52.0, ay_mps2=-6.5))

    assert left_turn_diag.right_minus_left_load_n > 0.0
    assert right_turn_diag.right_minus_left_load_n < 0.0


def test_vehicle_phase5_brake_bias_distribution() -> None:
    params = VehicleParameters(brake_bias_front=0.60)
    sim = VehicleThermalSimulator(parameters=params)
    state = sim.initial_state(ambient_temp_k=celsius_to_kelvin(30.0))

    diag = sim.diagnostics(state, _input(brake_power_w=80_000.0))
    front_total = diag.wheel_brake_power_w["FL"] + diag.wheel_brake_power_w["FR"]
    rear_total = diag.wheel_brake_power_w["RL"] + diag.wheel_brake_power_w["RR"]

    assert math.isclose(front_total / (front_total + rear_total), 0.60, rel_tol=1e-9, abs_tol=1e-9)


def test_vehicle_phase5_long_run_finite_and_bounded() -> None:
    sim = VehicleThermalSimulator()
    state = sim.initial_state(ambient_temp_k=celsius_to_kelvin(30.0))
    control = _input(
        speed_mps=64.0,
        ax_mps2=-2.0,
        ay_mps2=4.0,
        steering_angle_rad=0.08,
        yaw_rate_radps=0.12,
        brake_power_w=40_000.0,
    )

    for _ in range(900):
        state = sim.step(state, control, dt_s=0.05)

    for wheel_state in state.wheel_states.values():
        assert all(math.isfinite(value) for value in wheel_state.as_vector())
        assert 0.0 <= wheel_state.wear <= 1.0
