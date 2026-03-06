from __future__ import annotations

import json
import importlib.util
from pathlib import Path
import statistics
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.high_fidelity import HighFidelityTireInputs, HighFidelityTireModelParameters, HighFidelityTireSimulator, HighFidelityVehicleSimulator  # noqa: E402
from models.physics import celsius_to_kelvin  # noqa: E402
from models.vehicle_thermal import VehicleInputs  # noqa: E402
def _load_run_module():
    module_path = ROOT / "scripts" / "run_high_fidelity_no_data.py"
    spec = importlib.util.spec_from_file_location("run_high_fidelity_no_data", module_path)
    if spec is None or spec.loader is None:
        msg = f"Unable to load module from {module_path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_high_fidelity_no_data"] = module
    spec.loader.exec_module(module)
    return module


def _bench_tire_step(repeats: int = 5) -> dict[str, float]:
    params = HighFidelityTireModelParameters(
        use_2d_thermal_solver=True,
        no_op_thermal_step=False,
        radial_cells=10,
        theta_cells=20,
        internal_solver_dt_s=0.01,
        enable_profiling=True,
    )
    simulator = HighFidelityTireSimulator(params)
    state = simulator.initial_state(ambient_temp_k=celsius_to_kelvin(30.0))
    inputs = HighFidelityTireInputs(
        speed_mps=54.0,
        wheel_angular_speed_radps=185.0,
        normal_load_n=3_800.0,
        slip_ratio_cmd=0.09,
        slip_angle_cmd_rad=0.06,
        brake_power_w=18_000.0,
        ambient_temp_k=celsius_to_kelvin(30.0),
        track_temp_k=celsius_to_kelvin(46.0),
    )
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        state = simulator.step(state, inputs, dt_s=0.05)
        durations.append(time.perf_counter() - start)
    return {
        "mean_step_s": statistics.mean(durations),
        "p95_step_s": max(durations),
        "solver_substeps": float(state.last_solver_substeps),
        "solver_diffusion_iterations": float(state.last_solver_diffusion_iterations or 0),
    }


def _bench_vehicle_step(repeats: int = 3) -> dict[str, float]:
    params = HighFidelityTireModelParameters(
        use_2d_thermal_solver=True,
        no_op_thermal_step=False,
        radial_cells=8,
        theta_cells=12,
        internal_solver_dt_s=0.01,
        enable_profiling=True,
    )
    simulator = HighFidelityVehicleSimulator(
        tire_parameters_by_wheel={wheel: params for wheel in ("FL", "FR", "RL", "RR")}
    )
    state = simulator.initial_state(ambient_temp_k=celsius_to_kelvin(30.0))
    inputs = VehicleInputs(
        speed_mps=56.0,
        ay_mps2=6.0,
        steering_angle_rad=0.07,
        yaw_rate_radps=0.11,
        ambient_temp_k=celsius_to_kelvin(30.0),
        track_temp_k=celsius_to_kelvin(44.0),
    )
    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        state = simulator.step(state, inputs, dt_s=0.05)
        durations.append(time.perf_counter() - start)
    return {
        "mean_step_s": statistics.mean(durations),
        "p95_step_s": max(durations),
    }


def _bench_harness() -> dict[str, float]:
    run_module = _load_run_module()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        start = time.perf_counter()
        run_module.run_high_fidelity_no_data(
            preset="smoke",
            output_path=tmp / "hf_bench.json",
            summary_path=tmp / "hf_bench.md",
            dt_s=0.2,
        )
        duration = time.perf_counter() - start
    return {"smoke_runtime_s": duration}


def main() -> None:
    payload = {
        "tire_step": _bench_tire_step(),
        "vehicle_step": _bench_vehicle_step(),
        "harness": _bench_harness(),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
