from __future__ import annotations

import math

import numpy as np

from models.high_fidelity import (
    HighFidelityTireInputs,
    HighFidelityTireSimulator,
    inputs_from_legacy,
    state_from_legacy,
    state_to_legacy,
)
from models.physics import TireInputs, TireThermalSimulator, celsius_to_kelvin


def test_high_fidelity_no_op_step_preserves_temperatures_and_advances_time() -> None:
    simulator = HighFidelityTireSimulator()
    state = simulator.initial_state(ambient_temp_k=celsius_to_kelvin(30.0), wear=0.12)
    inputs = HighFidelityTireInputs(
        speed_mps=60.0,
        wheel_angular_speed_radps=190.0,
        normal_load_n=3_700.0,
        slip_ratio_cmd=0.08,
        slip_angle_cmd_rad=0.05,
        ambient_temp_k=celsius_to_kelvin(30.0),
        track_temp_k=celsius_to_kelvin(44.0),
    )

    state_next = simulator.step(state, inputs, dt_s=0.1)

    assert np.allclose(state_next.temperature_nodes_k, state.temperature_nodes_k)
    assert math.isclose(state_next.time_s, state.time_s + 0.1, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(state_next.wear, state.wear, rel_tol=0.0, abs_tol=1e-12)

    diag = simulator.diagnostics(state_next, inputs)
    assert diag.no_op_mode
    assert math.isclose(diag.core_temperature_k, state_next.core_temperature_k, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(diag.mean_temperature_k, celsius_to_kelvin(30.0), rel_tol=0.0, abs_tol=1e-12)
    assert diag.excitation_frequency_hz > 0.0
    assert diag.loss_modulus_pa >= 0.0
    assert diag.hysteresis_power_density_w_per_m3 >= 0.0
    assert diag.energy_source_total_w >= 0.0


def test_high_fidelity_legacy_adapters_roundtrip_state_and_inputs() -> None:
    legacy_inputs = TireInputs(
        speed_mps=64.0,
        wheel_angular_speed_radps=202.0,
        normal_load_n=3_900.0,
        slip_ratio=0.09,
        slip_angle_rad=0.06,
        brake_power_w=3_500.0,
        ambient_temp_k=celsius_to_kelvin(29.0),
        track_temp_k=celsius_to_kelvin(43.0),
    )
    hf_inputs = inputs_from_legacy(legacy_inputs)

    assert math.isclose(hf_inputs.speed_mps, legacy_inputs.speed_mps, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(hf_inputs.slip_ratio_cmd, legacy_inputs.slip_ratio, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(
        hf_inputs.slip_angle_cmd_rad,
        legacy_inputs.slip_angle_rad,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        hf_inputs.road_surface_temp_k or 0.0,
        legacy_inputs.track_temp_k,
        rel_tol=0.0,
        abs_tol=1e-12,
    )

    legacy_sim = TireThermalSimulator()
    legacy_state = legacy_sim.initial_state(ambient_temp_k=celsius_to_kelvin(31.0), wear=0.08)
    hf_state = state_from_legacy(legacy_state)
    roundtrip = state_to_legacy(hf_state)

    assert hf_state.temperature_nodes_k.shape == (9,)
    assert math.isclose(roundtrip.t_carcass_k, legacy_state.t_carcass_k, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(roundtrip.t_sidewall_k, legacy_state.t_sidewall_k, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(roundtrip.wear, legacy_state.wear, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(roundtrip.time_s, legacy_state.time_s, rel_tol=0.0, abs_tol=1e-12)


def test_high_fidelity_simulate_stream_length_and_time_progression() -> None:
    simulator = HighFidelityTireSimulator()
    initial = simulator.initial_state(ambient_temp_k=celsius_to_kelvin(28.0))
    inputs_stream = [
        HighFidelityTireInputs(
            speed_mps=55.0,
            wheel_angular_speed_radps=175.0,
            normal_load_n=3_500.0,
            slip_ratio_cmd=0.05,
            slip_angle_cmd_rad=0.04,
        )
        for _ in range(5)
    ]

    states = simulator.simulate(initial, inputs_stream, dt_s=0.2)

    assert len(states) == 6
    assert math.isclose(states[-1].time_s, 1.0, rel_tol=0.0, abs_tol=1e-12)
