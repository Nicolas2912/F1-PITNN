from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import statistics
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.high_fidelity import HighFidelityTireInputs, HighFidelityTireModelParameters, HighFidelityTireSimulator  # noqa: E402
from models.high_fidelity.native_simulator_kernels import native_simulator_kernels_available  # noqa: E402
from models.physics import celsius_to_kelvin  # noqa: E402


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


def _inputs() -> HighFidelityTireInputs:
    return HighFidelityTireInputs(
        speed_mps=63.0,
        wheel_angular_speed_radps=198.0,
        normal_load_n=3_950.0,
        slip_ratio_cmd=0.10,
        slip_angle_cmd_rad=0.05,
        ambient_temp_k=celsius_to_kelvin(29.0),
        track_temp_k=celsius_to_kelvin(46.0),
        road_surface_temp_k=celsius_to_kelvin(43.0),
        road_bulk_temp_k=celsius_to_kelvin(41.0),
        brake_power_w=20_000.0,
        solar_w_m2=240.0,
        wind_mps=5.5,
        wind_yaw_rad=0.14,
        wheel_wake_factor=1.2,
        road_moisture=0.08,
        rubbering_level=0.75,
        asphalt_effusivity=1.09,
    )


def _params(*, profiling: bool = False) -> HighFidelityTireModelParameters:
    return HighFidelityTireModelParameters(
        use_2d_thermal_solver=True,
        no_op_thermal_step=False,
        use_reduced_patch_mechanics=True,
        use_local_temp_friction_partition=True,
        use_structural_hysteresis_model=True,
        radial_cells=12,
        theta_cells=36,
        internal_solver_dt_s=0.01,
        enable_profiling=profiling,
    )


def _seeded_payload(sim: HighFidelityTireSimulator) -> tuple:
    state = sim.initial_state(ambient_temp_k=celsius_to_kelvin(29.0))
    assert state.thermal_field_rtw_k is not None
    assert state.flash_temperature_field_tw_k is not None
    assert state.sidewall_field_tw_k is not None
    thermal = np.array(state.thermal_field_rtw_k, dtype=float, copy=True)
    flash = np.array(state.flash_temperature_field_tw_k, dtype=float, copy=True)
    sidewall = np.array(state.sidewall_field_tw_k, dtype=float, copy=True)

    radial_gradient = np.linspace(0.0, 24.0, thermal.shape[0], dtype=float)[:, None, None]
    theta_gradient = np.linspace(-3.5, 4.0, thermal.shape[1], dtype=float)[None, :, None]
    width_gradient = np.linspace(-1.0, 1.6, thermal.shape[2], dtype=float)[None, None, :]
    thermal += radial_gradient + theta_gradient + width_gradient
    flash += np.linspace(0.0, 9.0, flash.shape[0], dtype=float)[:, None]
    flash += np.linspace(-1.0, 1.5, flash.shape[1], dtype=float)[None, :]
    sidewall += np.linspace(0.0, 6.0, sidewall.shape[0], dtype=float)[:, None]
    sidewall += np.linspace(-0.7, 1.1, sidewall.shape[1], dtype=float)[None, :]
    return state, thermal, flash, sidewall


def _summary_stats(before: float, after: float) -> dict[str, float]:
    delta = before - after
    pct = (delta / before) * 100.0 if before > 0.0 else 0.0
    return {"before_s": before, "after_s": after, "improvement_s": delta, "improvement_pct": pct}


def _bench_flash_and_sidewall() -> dict[str, dict[str, float]]:
    sim = HighFidelityTireSimulator(_params())
    inputs = _inputs()
    state, thermal, flash, sidewall = _seeded_payload(sim)
    road_state = sim._resolve_road_state(state=state, inputs=inputs)
    zone_weights = sim._zone_weights(inputs=inputs)

    def run_flash(native: bool) -> tuple[np.ndarray, float]:
        with _EnvVar("PITNN_USE_NATIVE_SIMULATOR_KERNELS", "1" if native else None):
            start = time.perf_counter()
            result = sim._step_flash_field(
                flash_field_tw_k=flash,
                thermal_field_rtw_k=thermal,
                road_state=road_state,
                ambient_temp_k=inputs.ambient_temp_k,
                friction_to_tire_w=5_200.0,
                zone_weights=zone_weights,
                wheel_angular_speed_radps=inputs.wheel_angular_speed_radps,
                time_s=0.42,
                dt_s=0.05,
            )
            return result, time.perf_counter() - start

    def run_sidewall(native: bool) -> tuple[tuple[np.ndarray, float], float]:
        with _EnvVar("PITNN_USE_NATIVE_SIMULATOR_KERNELS", "1" if native else None):
            start = time.perf_counter()
            result = sim._step_sidewall_field(
                sidewall_field_tw_k=sidewall,
                thermal_field_rtw_k=thermal,
                inputs=inputs,
                rim_temp_k=celsius_to_kelvin(125.0),
                brake_heat_to_sidewall_w=1_900.0,
                dt_s=0.05,
            )
            return result, time.perf_counter() - start

    python_flash, _ = run_flash(native=False)
    native_flash, _ = run_flash(native=True)
    python_sidewall, _ = run_sidewall(native=False)
    native_sidewall, _ = run_sidewall(native=True)

    if not np.array_equal(python_flash, native_flash):
        raise RuntimeError("Flash field outputs differ between Python and native implementations")
    if not np.array_equal(python_sidewall[0], native_sidewall[0]) or python_sidewall[1] != native_sidewall[1]:
        raise RuntimeError("Sidewall field outputs differ between Python and native implementations")

    flash_python_runs = [run_flash(native=False)[1] for _ in range(200)]
    flash_native_runs = [run_flash(native=True)[1] for _ in range(200)]
    sidewall_python_runs = [run_sidewall(native=False)[1] for _ in range(300)]
    sidewall_native_runs = [run_sidewall(native=True)[1] for _ in range(300)]

    return {
        "flash_kernel": _summary_stats(
            statistics.median(flash_python_runs),
            statistics.median(flash_native_runs),
        ),
        "sidewall_kernel": _summary_stats(
            statistics.median(sidewall_python_runs),
            statistics.median(sidewall_native_runs),
        ),
    }


def _bench_reduced_step() -> dict[str, float]:
    sim = HighFidelityTireSimulator(_params(profiling=True))
    inputs = _inputs()
    initial_state = sim.initial_state(ambient_temp_k=inputs.ambient_temp_k)

    def run_once(native: bool) -> tuple:
        with _EnvVar("PITNN_USE_NATIVE_SIMULATOR_KERNELS", "1" if native else None):
            state = replace(initial_state)
            start = time.perf_counter()
            for _ in range(4):
                state = sim.step(state, inputs, dt_s=0.05)
            return state, time.perf_counter() - start

    python_state, _ = run_once(native=False)
    native_state, _ = run_once(native=True)

    for attr in (
        "temperature_nodes_k",
        "thermal_field_rt_k",
        "thermal_field_rtw_k",
        "flash_temperature_field_tw_k",
        "sidewall_field_tw_k",
    ):
        left = getattr(python_state, attr)
        right = getattr(native_state, attr)
        if isinstance(left, np.ndarray):
            if not np.array_equal(left, right):
                raise RuntimeError(f"Reduced step output differs for {attr}")
        elif left != right:
            raise RuntimeError(f"Reduced step scalar output differs for {attr}")

    python_runs = [run_once(native=False)[1] for _ in range(10)]
    native_runs = [run_once(native=True)[1] for _ in range(10)]
    return _summary_stats(statistics.median(python_runs), statistics.median(native_runs))


def _profile_wheel_coupling_and_boundaries() -> dict[str, float]:
    sim = HighFidelityTireSimulator(_params(profiling=True))
    inputs = _inputs()
    state = sim.initial_state(ambient_temp_k=inputs.ambient_temp_k)

    coupling_times = []
    total_step_times = []
    for _ in range(6):
        start = time.perf_counter()
        state = sim.step(state, inputs, dt_s=0.05)
        total_step_times.append(time.perf_counter() - start)
        coupling_times.append(float(state.last_wheel_coupling_time_s or 0.0))

    boundary = sim._boundary_model
    zone_friction = np.array([1200.0, 1600.0, 900.0], dtype=float)
    flash_temp = np.array([356.0, 362.0, 351.0], dtype=float)
    bulk_temp = np.array([341.0, 344.0, 339.0], dtype=float)
    road_temp = np.array([316.0, 317.0, 315.0], dtype=float)
    moisture = np.array([0.08, 0.05, 0.09], dtype=float)
    sliding = np.array([0.22, 0.34, 0.18], dtype=float)
    contact_temp = np.array([349.0, 351.0, 347.0], dtype=float)
    pressure = np.array([1.04, 1.07, 1.02], dtype=float)
    zone_area = np.array([0.0045, 0.0050, 0.0042], dtype=float)

    partition_runs = []
    conduction_runs = []
    for _ in range(2000):
        start = time.perf_counter()
        boundary.partition_friction_power_by_zone(
            zone_friction_power_w=zone_friction,
            flash_temp_w_k=flash_temp,
            bulk_temp_w_k=bulk_temp,
            road_surface_temp_w_k=road_temp,
            road_moisture_w=moisture,
            asphalt_effusivity=1.08,
            rubbering_level=0.75,
            zone_sliding_fraction=sliding,
            zone_contact_temp_w_k=contact_temp,
            zone_contact_pressure_factor=pressure,
        )
        partition_runs.append(time.perf_counter() - start)

        start = time.perf_counter()
        boundary.road_conduction_power_w_by_zone(
            tire_surface_temp_w_k=bulk_temp,
            road_surface_temp_w_k=road_temp,
            zone_contact_patch_area_m2=zone_area,
            road_moisture_w=moisture,
            asphalt_effusivity=1.08,
            wind_mps=5.5,
        )
        conduction_runs.append(time.perf_counter() - start)

    return {
        "median_total_step_s": statistics.median(total_step_times),
        "median_wheel_coupling_time_s": statistics.median(coupling_times),
        "wheel_coupling_share_pct": (
            statistics.median(coupling_times) / statistics.median(total_step_times) * 100.0
            if statistics.median(total_step_times) > 0.0
            else 0.0
        ),
        "median_boundary_partition_s": statistics.median(partition_runs),
        "median_boundary_conduction_s": statistics.median(conduction_runs),
    }


def main() -> None:
    if not native_simulator_kernels_available():
        raise SystemExit("Native simulator kernels extension is not available. Build it first.")

    payload = {
        "native_simulator_kernels_available": True,
        "kernels": _bench_flash_and_sidewall(),
        "reduced_step": _bench_reduced_step(),
        "profiling": _profile_wheel_coupling_and_boundaries(),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
