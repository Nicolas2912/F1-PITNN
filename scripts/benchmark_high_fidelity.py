from __future__ import annotations

import json
import importlib.util
import os
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


BENCHMARK_CONFIG = {
    "preset": "smoke",
    "dt_s": 0.2,
    "duration_scale": 0.03,
    "lhs_samples": 4,
    "sobol_samples": 6,
    "seed": 2026,
}


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
        deterministic_runs = [
            _run_harness_once(run_module, tmp, workers=1, suffix=f"determinism_{idx}")
            for idx in range(2)
        ]
        _assert_equivalent_outputs(deterministic_runs[0], deterministic_runs[1])

        serial_runs = [
            _run_harness_once(run_module, tmp, workers=1, suffix=f"serial_{idx}")
            for idx in range(3)
        ]
        parallel_runs = [
            _run_harness_once(run_module, tmp, workers=2, suffix=f"parallel_{idx}")
            for idx in range(3)
        ]
        for run in serial_runs[1:] + parallel_runs:
            _assert_equivalent_outputs(serial_runs[0], run)

    serial_median = statistics.median(run["runtime_s"] for run in serial_runs)
    parallel_median = statistics.median(run["runtime_s"] for run in parallel_runs)
    improvement_s = serial_median - parallel_median
    improvement_pct = (improvement_s / serial_median) * 100.0 if serial_median > 0.0 else 0.0
    if improvement_pct < 1.0:
        msg = f"Expected at least 1% speedup, got {improvement_pct:.2f}%"
        raise RuntimeError(msg)
    return {
        "serial_runtime_median_s": serial_median,
        "parallel_runtime_median_s": parallel_median,
        "improvement_s": improvement_s,
        "improvement_pct": improvement_pct,
    }


def _bench_native_harness() -> dict[str, float]:
    run_module = _load_run_module()
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        python_runs = []
        native_runs = []
        for idx in range(3):
            with _EnvVar("PITNN_USE_NATIVE_DIFFUSION", None), _EnvVar("PITNN_USE_NATIVE_SIMULATOR_KERNELS", None):
                python_runs.append(_run_harness_once(run_module, tmp, workers=1, suffix=f"python_{idx}"))
            with _EnvVar("PITNN_USE_NATIVE_DIFFUSION", "1"), _EnvVar("PITNN_USE_NATIVE_SIMULATOR_KERNELS", "1"):
                native_runs.append(_run_harness_once(run_module, tmp, workers=1, suffix=f"native_{idx}"))
        for run in native_runs:
            _assert_equivalent_outputs(python_runs[0], run)
        for run in python_runs[1:]:
            _assert_equivalent_outputs(python_runs[0], run)

    python_median = statistics.median(run["runtime_s"] for run in python_runs)
    native_median = statistics.median(run["runtime_s"] for run in native_runs)
    improvement_s = python_median - native_median
    improvement_pct = (improvement_s / python_median) * 100.0 if python_median > 0.0 else 0.0
    if improvement_pct < 1.0:
        msg = f"Expected at least 1% native harness speedup, got {improvement_pct:.2f}%"
        raise RuntimeError(msg)
    return {
        "python_runtime_median_s": python_median,
        "native_runtime_median_s": native_median,
        "improvement_s": improvement_s,
        "improvement_pct": improvement_pct,
    }


def _normalize_artifact(payload: dict) -> dict:
    normalized = json.loads(json.dumps(payload))
    metadata = normalized.get("metadata", {})
    for key in ("created_at_utc", "results_path", "summary_path"):
        metadata.pop(key, None)
    normalized.pop("timing", None)
    return normalized


def _normalize_summary(text: str) -> str:
    return "\n".join(
        line
        for line in text.splitlines()
        if not line.startswith("- created_at_utc:")
        and not line.startswith("- results_json:")
        and not line.startswith("- summary_md:")
        and not line.startswith("- total_elapsed_s:")
        and not line.startswith("| baseline |")
        and not line.startswith("| lhs |")
        and not line.startswith("| sobol |")
        and not line.startswith("| done |")
    )


def _run_harness_once(run_module, tmp: Path, *, workers: int, suffix: str) -> dict:
    output_json = tmp / f"{suffix}.json"
    output_summary = tmp / f"{suffix}.md"
    start = time.perf_counter()
    run_module.run_high_fidelity_no_data(
        output_path=output_json,
        summary_path=output_summary,
        workers=workers,
        **BENCHMARK_CONFIG,
    )
    runtime = time.perf_counter() - start
    artifact = json.loads(output_json.read_text(encoding="utf-8"))
    summary = output_summary.read_text(encoding="utf-8")
    return {
        "runtime_s": runtime,
        "artifact": _normalize_artifact(artifact),
        "summary": _normalize_summary(summary),
    }


def _assert_equivalent_outputs(left: dict, right: dict) -> None:
    if left["artifact"] != right["artifact"]:
        raise RuntimeError("Normalized artifacts differ between benchmark runs")
    if left["summary"] != right["summary"]:
        raise RuntimeError("Summary output differs between benchmark runs")


def main() -> None:
    payload = {
        "tire_step": _bench_tire_step(),
        "vehicle_step": _bench_vehicle_step(),
        "harness": _bench_harness(),
        "native_harness": _bench_native_harness(),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
