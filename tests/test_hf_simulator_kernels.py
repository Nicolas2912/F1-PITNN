from __future__ import annotations

import os

import numpy as np
import pytest

from models.high_fidelity import HighFidelityTireInputs, HighFidelityTireModelParameters, HighFidelityTireSimulator
from models.high_fidelity.native_simulator_kernels import native_simulator_kernels_available
from models.physics import celsius_to_kelvin


def _inputs() -> HighFidelityTireInputs:
    return HighFidelityTireInputs(
        speed_mps=61.0,
        wheel_angular_speed_radps=192.0,
        normal_load_n=3_900.0,
        slip_ratio_cmd=0.09,
        slip_angle_cmd_rad=0.04,
        ambient_temp_k=celsius_to_kelvin(29.0),
        track_temp_k=celsius_to_kelvin(45.0),
        road_surface_temp_k=celsius_to_kelvin(43.0),
        road_bulk_temp_k=celsius_to_kelvin(41.0),
        brake_power_w=18_000.0,
        solar_w_m2=220.0,
        wind_mps=5.0,
        wind_yaw_rad=0.13,
        wheel_wake_factor=1.18,
        road_moisture=0.06,
        rubbering_level=0.72,
        asphalt_effusivity=1.08,
    )


def _params() -> HighFidelityTireModelParameters:
    return HighFidelityTireModelParameters(
        use_2d_thermal_solver=True,
        no_op_thermal_step=False,
        use_reduced_patch_mechanics=True,
        use_local_temp_friction_partition=True,
        use_structural_hysteresis_model=True,
        radial_cells=8,
        theta_cells=18,
        internal_solver_dt_s=0.01,
    )


class _EnvVar:
    def __init__(self, key: str, value: str | None) -> None:
        self._key = key
        self._value = value
        self._old = os.environ.get(key)

    def __enter__(self) -> None:
        if self._value is None:
            os.environ.pop(self._key, None)
        else:
            os.environ[self._key] = self._value

    def __exit__(self, *_args: object) -> None:
        if self._old is None:
            os.environ.pop(self._key, None)
        else:
            os.environ[self._key] = self._old


def _seeded_state(sim: HighFidelityTireSimulator) -> tuple:
    state = sim.initial_state(ambient_temp_k=celsius_to_kelvin(29.0))
    assert state.thermal_field_rtw_k is not None
    assert state.flash_temperature_field_tw_k is not None
    assert state.sidewall_field_tw_k is not None

    thermal = np.array(state.thermal_field_rtw_k, dtype=float, copy=True)
    flash = np.array(state.flash_temperature_field_tw_k, dtype=float, copy=True)
    sidewall = np.array(state.sidewall_field_tw_k, dtype=float, copy=True)

    radial_gradient = np.linspace(0.0, 18.0, thermal.shape[0], dtype=float)[:, None, None]
    theta_gradient = np.linspace(-2.5, 3.0, thermal.shape[1], dtype=float)[None, :, None]
    width_gradient = np.linspace(-1.0, 1.5, thermal.shape[2], dtype=float)[None, None, :]
    thermal += radial_gradient + theta_gradient + width_gradient
    flash += np.linspace(0.0, 6.0, flash.shape[0], dtype=float)[:, None]
    flash += np.linspace(-0.8, 1.4, flash.shape[1], dtype=float)[None, :]
    sidewall += np.linspace(0.0, 4.0, sidewall.shape[0], dtype=float)[:, None]
    sidewall += np.linspace(-0.5, 0.9, sidewall.shape[1], dtype=float)[None, :]
    return state, thermal, flash, sidewall


@pytest.mark.skipif(not native_simulator_kernels_available(), reason="native simulator kernels extension not built")
def test_native_flash_kernel_matches_python_reference() -> None:
    sim = HighFidelityTireSimulator(_params())
    state, thermal, flash, _sidewall = _seeded_state(sim)
    inputs = _inputs()
    road_state = sim._resolve_road_state(state=state, inputs=inputs)
    zone_weights = sim._zone_weights(inputs=inputs)

    with _EnvVar("PITNN_USE_NATIVE_SIMULATOR_KERNELS", None):
        python_result = sim._step_flash_field(
            flash_field_tw_k=flash,
            thermal_field_rtw_k=thermal,
            road_state=road_state,
            ambient_temp_k=inputs.ambient_temp_k,
            friction_to_tire_w=4_800.0,
            zone_weights=zone_weights,
            wheel_angular_speed_radps=inputs.wheel_angular_speed_radps,
            time_s=0.37,
            dt_s=0.05,
        )

    with _EnvVar("PITNN_USE_NATIVE_SIMULATOR_KERNELS", "1"):
        native_result = sim._step_flash_field(
            flash_field_tw_k=flash,
            thermal_field_rtw_k=thermal,
            road_state=road_state,
            ambient_temp_k=inputs.ambient_temp_k,
            friction_to_tire_w=4_800.0,
            zone_weights=zone_weights,
            wheel_angular_speed_radps=inputs.wheel_angular_speed_radps,
            time_s=0.37,
            dt_s=0.05,
        )

    assert np.array_equal(native_result, python_result)


@pytest.mark.skipif(not native_simulator_kernels_available(), reason="native simulator kernels extension not built")
def test_native_sidewall_kernel_matches_python_reference() -> None:
    sim = HighFidelityTireSimulator(_params())
    _state, thermal, _flash, sidewall = _seeded_state(sim)
    inputs = _inputs()

    with _EnvVar("PITNN_USE_NATIVE_SIMULATOR_KERNELS", None):
        python_field, python_heat = sim._step_sidewall_field(
            sidewall_field_tw_k=sidewall,
            thermal_field_rtw_k=thermal,
            inputs=inputs,
            rim_temp_k=celsius_to_kelvin(120.0),
            brake_heat_to_sidewall_w=1_700.0,
            dt_s=0.05,
        )

    with _EnvVar("PITNN_USE_NATIVE_SIMULATOR_KERNELS", "1"):
        native_field, native_heat = sim._step_sidewall_field(
            sidewall_field_tw_k=sidewall,
            thermal_field_rtw_k=thermal,
            inputs=inputs,
            rim_temp_k=celsius_to_kelvin(120.0),
            brake_heat_to_sidewall_w=1_700.0,
            dt_s=0.05,
        )

    assert np.array_equal(native_field, python_field)
    assert native_heat == python_heat


@pytest.mark.skipif(not native_simulator_kernels_available(), reason="native simulator kernels extension not built")
def test_native_simulator_kernel_dispatch_matches_python_for_reduced_step() -> None:
    sim = HighFidelityTireSimulator(_params())
    inputs = _inputs()
    state = sim.initial_state(ambient_temp_k=inputs.ambient_temp_k)

    with _EnvVar("PITNN_USE_NATIVE_SIMULATOR_KERNELS", None):
        python_state = sim.step(state, inputs, dt_s=0.05)

    with _EnvVar("PITNN_USE_NATIVE_SIMULATOR_KERNELS", "1"):
        native_state = sim.step(state, inputs, dt_s=0.05)

    assert np.array_equal(native_state.temperature_nodes_k, python_state.temperature_nodes_k)
    assert np.array_equal(native_state.thermal_field_rt_k, python_state.thermal_field_rt_k)
    assert np.array_equal(native_state.thermal_field_rtw_k, python_state.thermal_field_rtw_k)
    assert np.array_equal(native_state.flash_temperature_field_tw_k, python_state.flash_temperature_field_tw_k)
    assert np.array_equal(native_state.sidewall_field_tw_k, python_state.sidewall_field_tw_k)
