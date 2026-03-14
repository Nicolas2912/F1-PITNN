from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
import json
import multiprocessing as mp
from pathlib import Path
from queue import Empty
import sys
import threading
import time

import numpy as np
from tqdm import tqdm

try:
    from sklearn.ensemble import ExtraTreesRegressor
except ImportError:  # pragma: no cover - optional runtime dependency
    ExtraTreesRegressor = None

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.high_fidelity import (  # noqa: E402
    HighFidelityTireModelParameters,
    HighFidelityUQ,
    HighFidelityVehicleSimulator,
)
from models.high_fidelity.types import ConstructionParameters, InternalCouplingParameters, LocalContactParameters  # noqa: E402
from models.high_fidelity.reporting import render_high_fidelity_no_data_summary  # noqa: E402
from models.vehicle_thermal import VehicleInputs, VehicleParameters  # noqa: E402

OUTPUT_DIR = ROOT / "reports" / "results"
RESULTS_FILE = OUTPUT_DIR / "high_fidelity_no_data_results.json"
SUMMARY_FILE = OUTPUT_DIR / "high_fidelity_no_data_summary.md"
_WORKER_PROGRESS_QUEUE = None


@dataclass(frozen=True)
class UQSurrogateConfig:
    enabled: bool = False
    kind: str = "extra_trees"
    sobol_train_samples: int | None = None
    sobol_validation_samples: int | None = None
    ridge_alpha: float = 1e-6
    max_rmse_c: float = 0.75
    max_abs_error_c: float = 2.0
    min_prediction_samples: int = 32
    extra_trees_estimators: int = 600


class QuadraticRidgeSurrogate:
    def __init__(self, *, lower: np.ndarray, upper: np.ndarray, ridge_alpha: float) -> None:
        self._lower = np.asarray(lower, dtype=float)
        self._upper = np.asarray(upper, dtype=float)
        self._ridge_alpha = float(ridge_alpha)
        self._weights: np.ndarray | None = None

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        scale = np.maximum(self._upper - self._lower, 1e-12)
        return 2.0 * (np.asarray(x, dtype=float) - self._lower) / scale - 1.0

    def _features(self, x: np.ndarray) -> np.ndarray:
        x_norm = self._normalize(x)
        return np.hstack([np.ones((x_norm.shape[0], 1), dtype=float), x_norm, x_norm * x_norm])

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        features = self._features(x)
        ridge = np.sqrt(self._ridge_alpha) * np.eye(features.shape[1], dtype=float)
        augmented_features = np.vstack([features, ridge])
        target_matrix = np.asarray(y, dtype=float)
        augmented_targets = np.vstack([target_matrix, np.zeros((features.shape[1], target_matrix.shape[1]), dtype=float)])
        self._weights, *_ = np.linalg.lstsq(augmented_features, augmented_targets, rcond=None)

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._weights is None:
            msg = "Surrogate must be fit before predict"
            raise RuntimeError(msg)
        return self._features(x) @ self._weights


class ExtraTreesSurrogate:
    def __init__(self, *, estimators: int, random_state: int) -> None:
        if ExtraTreesRegressor is None:
            msg = "scikit-learn is required for the extra_trees surrogate"
            raise RuntimeError(msg)
        self._model = ExtraTreesRegressor(
            n_estimators=max(int(estimators), 100),
            random_state=int(random_state),
            n_jobs=-1,
            min_samples_leaf=1,
        )

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self._model.fit(np.asarray(x, dtype=float), np.asarray(y, dtype=float))

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(self._model.predict(np.asarray(x, dtype=float)), dtype=float)


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    duration_s: float
    speed_mps: float
    ax_mps2: float
    ay_mps2: float
    steering_angle_rad: float
    yaw_rate_radps: float
    brake_power_w: float
    drive_power_w: float
    ambient_temp_k: float
    track_temp_k: float
    road_bulk_temp_k: float | None = None
    wind_mps: float = 0.0
    wind_yaw_rad: float = 0.0
    humidity_rel: float = 0.50
    solar_w_m2: float = 0.0
    road_moisture: float = 0.0
    rubbering_level: float = 0.0
    asphalt_roughness: float = 1.0
    asphalt_effusivity: float = 1.0
    include_in_uq: bool = True


@dataclass(frozen=True)
class FidelityPreset:
    name: str
    radial_cells: int
    theta_cells: int
    internal_solver_dt_s: float
    lhs_samples: int
    sobol_samples: int
    diagnostics_stride: int


@dataclass
class ProgressTracker:
    bar: tqdm | None = None
    phase_totals: dict[str, int] = field(default_factory=dict)
    _run_started_at: float = field(default_factory=time.perf_counter)
    _phase_started_at: dict[str, float] = field(default_factory=dict)
    _phase_elapsed_s: dict[str, float] = field(default_factory=dict)
    _phase_completed: dict[str, int] = field(default_factory=dict)
    _phase_order: list[str] = field(default_factory=list)
    _active_phase: str | None = None

    def set_phase(self, label: str) -> None:
        now = time.perf_counter()
        if self._active_phase is not None:
            self._phase_elapsed_s[self._active_phase] = (
                self._phase_elapsed_s.get(self._active_phase, 0.0)
                + max(now - self._phase_started_at.get(self._active_phase, now), 0.0)
            )
        self._active_phase = label
        self._phase_started_at[label] = now
        self._phase_completed.setdefault(label, 0)
        if label not in self._phase_order:
            self._phase_order.append(label)
        if self.bar is not None:
            self.bar.set_description(label)
            self._refresh_bar(now)
            self.bar.refresh()

    def advance(self, amount: int = 1) -> None:
        if amount <= 0:
            return
        if self._active_phase is not None:
            self._phase_completed[self._active_phase] = self._phase_completed.get(self._active_phase, 0) + amount
        if self.bar is not None:
            self.bar.update(amount)
            self._refresh_bar(time.perf_counter())

    def close(self) -> None:
        if self._active_phase is not None:
            now = time.perf_counter()
            self._phase_elapsed_s[self._active_phase] = (
                self._phase_elapsed_s.get(self._active_phase, 0.0)
                + max(now - self._phase_started_at.get(self._active_phase, now), 0.0)
            )
            self._active_phase = None
        if self.bar is not None:
            self.bar.close()

    def timings(self) -> dict[str, object]:
        phases: dict[str, dict[str, float | int | None]] = {}
        now = time.perf_counter()
        for label in self._phase_order:
            elapsed_s = self._phase_elapsed_s.get(label, 0.0)
            if label == self._active_phase:
                elapsed_s += max(now - self._phase_started_at.get(label, now), 0.0)
            completed_units = int(self._phase_completed.get(label, 0))
            total_units = int(self.phase_totals.get(label, completed_units))
            avg_seconds_per_unit = elapsed_s / completed_units if completed_units > 0 else None
            throughput_units_per_s = completed_units / elapsed_s if elapsed_s > 0.0 else None
            phases[label] = {
                "elapsed_s": float(elapsed_s),
                "completed_units": completed_units,
                "total_units": total_units,
                "avg_seconds_per_unit": (
                    float(avg_seconds_per_unit) if avg_seconds_per_unit is not None else None
                ),
                "throughput_units_per_s": (
                    float(throughput_units_per_s) if throughput_units_per_s is not None else None
                ),
            }
        return {
            "total_elapsed_s": float(now - self._run_started_at),
            "phases": phases,
        }

    def _refresh_bar(self, now: float) -> None:
        if self.bar is None or self._active_phase is None:
            return
        phase = self._active_phase
        completed = int(self._phase_completed.get(phase, 0))
        total = int(self.phase_totals.get(phase, completed))
        phase_elapsed_s = max(now - self._phase_started_at.get(phase, now), 0.0)
        total_elapsed_s = max(now - self._run_started_at, 0.0)
        eta_s = None
        if completed > 0 and total > completed:
            eta_s = phase_elapsed_s * (total - completed) / completed
        eta_label = _format_seconds(eta_s)
        self.bar.set_postfix_str(
            f"{completed}/{total} phase_elapsed={_format_seconds(phase_elapsed_s)} "
            f"phase_eta={eta_label} total_elapsed={_format_seconds(total_elapsed_s)}",
            refresh=False,
        )


class ProcessPoolRunner:
    def __init__(self, workers: int) -> None:
        self._workers = max(int(workers), 1)
        start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        self._context = mp.get_context(start_method)
        self._pool: mp.pool.Pool | None = None

    def iter_map_unordered(self, worker_fn, tasks: list[dict]):
        if len(tasks) == 0:
            return
        if self._pool is None:
            self._pool = self._context.Pool(processes=self._workers)
        yield from self._pool.imap_unordered(worker_fn, tasks, chunksize=1)

    def close(self) -> None:
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None

    def restart(self) -> None:
        self.close()

    def terminate(self) -> None:
        if self._pool is not None:
            self._pool.terminate()
            self._pool.join()
            self._pool = None

    def __enter__(self) -> ProcessPoolRunner:
        return self

    def __exit__(self, exc_type, exc, _tb) -> None:
        if exc_type is None:
            self.close()
        else:
            self.terminate()


class _QueueProgressDrainer:
    def __init__(self, queue, progress_tracker: ProgressTracker) -> None:
        self._queue = queue
        self._progress_tracker = progress_tracker
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="uq-progress-drainer", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join()
        self.drain()

    def drain(self) -> None:
        while True:
            try:
                amount = self._queue.get_nowait()
            except Empty:
                break
            self._progress_tracker.advance(int(amount))

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                amount = self._queue.get(timeout=0.1)
            except Empty:
                continue
            self._progress_tracker.advance(int(amount))


PRESETS: dict[str, FidelityPreset] = {
    "smoke": FidelityPreset(
        name="smoke",
        radial_cells=4,
        theta_cells=8,
        internal_solver_dt_s=0.05,
        lhs_samples=2,
        sobol_samples=2,
        diagnostics_stride=1,
    ),
    "dev": FidelityPreset(
        name="dev",
        radial_cells=8,
        theta_cells=16,
        internal_solver_dt_s=0.03,
        lhs_samples=8,
        sobol_samples=8,
        diagnostics_stride=2,
    ),
    "full": FidelityPreset(
        name="full",
        radial_cells=24,
        theta_cells=72,
        internal_solver_dt_s=0.01,
        lhs_samples=128,
        sobol_samples=512,
        diagnostics_stride=1,
    ),
}


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    if value < 60.0:
        return f"{value:.1f}s"
    minutes, seconds = divmod(value, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m{seconds:04.1f}s"
    hours, minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h{int(minutes):02d}m{seconds:04.1f}s"


def fidelity_preset(name: str) -> FidelityPreset:
    preset = PRESETS.get(name)
    if preset is None:
        msg = f"Unknown preset '{name}'. Expected one of {sorted(PRESETS)}"
        raise ValueError(msg)
    return preset


def default_tire_parameters(
    *,
    radial_cells: int = 24,
    theta_cells: int = 72,
    internal_solver_dt_s: float = 0.01,
) -> HighFidelityTireModelParameters:
    return HighFidelityTireModelParameters(
        use_2d_thermal_solver=True,
        no_op_thermal_step=False,
        use_local_temp_friction_partition=True,
        use_reduced_patch_mechanics=True,
        use_structural_hysteresis_model=True,
        internal_coupling=InternalCouplingParameters(enabled=True),
        local_contact=LocalContactParameters(enabled=True),
        construction=ConstructionParameters(enabled=True),
        radial_cells=radial_cells,
        theta_cells=theta_cells,
        internal_solver_dt_s=internal_solver_dt_s,
    )


def default_scenarios(*, duration_scale: float = 1.0) -> tuple[ScenarioConfig, ...]:
    return (
        ScenarioConfig(
            name="steady_corner",
            duration_s=10.0 * duration_scale,
            speed_mps=60.0,
            ax_mps2=0.0,
            ay_mps2=7.8,
            steering_angle_rad=0.085,
            yaw_rate_radps=0.132,
            brake_power_w=0.0,
            drive_power_w=30_000.0,
            ambient_temp_k=304.15,
            track_temp_k=320.15,
            road_bulk_temp_k=315.15,
            wind_mps=7.0,
            wind_yaw_rad=0.10,
            solar_w_m2=260.0,
            rubbering_level=0.72,
            asphalt_roughness=1.05,
            asphalt_effusivity=1.08,
        ),
        ScenarioConfig(
            name="straight_braking",
            duration_s=7.0 * duration_scale,
            speed_mps=72.0,
            ax_mps2=-8.4,
            ay_mps2=0.0,
            steering_angle_rad=0.0,
            yaw_rate_radps=0.0,
            brake_power_w=135_000.0,
            drive_power_w=0.0,
            ambient_temp_k=304.15,
            track_temp_k=321.15,
            road_bulk_temp_k=316.15,
            wind_mps=5.0,
            solar_w_m2=200.0,
            rubbering_level=0.68,
            asphalt_roughness=1.02,
            asphalt_effusivity=1.05,
        ),
        ScenarioConfig(
            name="straight_acceleration",
            duration_s=7.0 * duration_scale,
            speed_mps=42.0,
            ax_mps2=5.6,
            ay_mps2=0.0,
            steering_angle_rad=0.0,
            yaw_rate_radps=0.0,
            brake_power_w=0.0,
            drive_power_w=115_000.0,
            ambient_temp_k=304.15,
            track_temp_k=320.15,
            road_bulk_temp_k=315.15,
            wind_mps=4.5,
            solar_w_m2=240.0,
            rubbering_level=0.74,
            asphalt_roughness=1.04,
            asphalt_effusivity=1.06,
        ),
        ScenarioConfig(
            name="combined_brake_corner",
            duration_s=8.0 * duration_scale,
            speed_mps=58.0,
            ax_mps2=-5.6,
            ay_mps2=6.4,
            steering_angle_rad=0.078,
            yaw_rate_radps=0.118,
            brake_power_w=92_000.0,
            drive_power_w=0.0,
            ambient_temp_k=304.15,
            track_temp_k=321.15,
            road_bulk_temp_k=316.15,
            wind_mps=6.0,
            wind_yaw_rad=0.08,
            humidity_rel=0.56,
            solar_w_m2=190.0,
            rubbering_level=0.70,
            asphalt_roughness=1.03,
            asphalt_effusivity=1.07,
        ),
        ScenarioConfig(
            name="cooldown",
            duration_s=7.0 * duration_scale,
            speed_mps=34.0,
            ax_mps2=-1.4,
            ay_mps2=0.0,
            steering_angle_rad=0.0,
            yaw_rate_radps=0.0,
            brake_power_w=8_000.0,
            drive_power_w=0.0,
            ambient_temp_k=301.15,
            track_temp_k=312.15,
            road_bulk_temp_k=308.15,
            wind_mps=9.0,
            humidity_rel=0.68,
            solar_w_m2=120.0,
            road_moisture=0.20,
            rubbering_level=0.50,
            asphalt_roughness=0.96,
            asphalt_effusivity=1.10,
        ),
        ScenarioConfig(
            name="long_stint",
            duration_s=55.0 * duration_scale,
            speed_mps=56.0,
            ax_mps2=0.3,
            ay_mps2=5.0,
            steering_angle_rad=0.058,
            yaw_rate_radps=0.094,
            brake_power_w=7_000.0,
            drive_power_w=22_000.0,
            ambient_temp_k=304.15,
            track_temp_k=322.15,
            road_bulk_temp_k=317.15,
            wind_mps=5.5,
            wind_yaw_rad=0.05,
            humidity_rel=0.48,
            solar_w_m2=280.0,
            rubbering_level=0.82,
            asphalt_roughness=1.06,
            asphalt_effusivity=1.09,
            include_in_uq=False,
        ),
    )


def _vehicle_inputs_for_scenario(scenario: ScenarioConfig) -> VehicleInputs:
    return VehicleInputs(
        speed_mps=scenario.speed_mps,
        ax_mps2=scenario.ax_mps2,
        ay_mps2=scenario.ay_mps2,
        steering_angle_rad=scenario.steering_angle_rad,
        yaw_rate_radps=scenario.yaw_rate_radps,
        brake_power_w=scenario.brake_power_w,
        drive_power_w=scenario.drive_power_w,
        ambient_temp_k=scenario.ambient_temp_k,
        track_temp_k=scenario.track_temp_k,
        road_bulk_temp_k=scenario.track_temp_k if scenario.road_bulk_temp_k is None else scenario.road_bulk_temp_k,
        wind_mps=scenario.wind_mps,
        wind_yaw_rad=scenario.wind_yaw_rad,
        humidity_rel=scenario.humidity_rel,
        solar_w_m2=scenario.solar_w_m2,
        road_moisture=scenario.road_moisture,
        rubbering_level=scenario.rubbering_level,
        asphalt_roughness=scenario.asphalt_roughness,
        asphalt_effusivity=scenario.asphalt_effusivity,
    )


def _vehicle_simulator(
    *,
    tire_parameters: HighFidelityTireModelParameters,
    vehicle_parameters: VehicleParameters,
) -> HighFidelityVehicleSimulator:
    return HighFidelityVehicleSimulator(
        parameters=vehicle_parameters,
        tire_parameters_by_wheel={wheel: tire_parameters for wheel in ("FL", "FR", "RL", "RR")},
    )


def _batched(items: list[object], batch_size: int) -> list[list[object]]:
    if len(items) == 0:
        return []
    return [items[idx : idx + batch_size] for idx in range(0, len(items), batch_size)]


def _batch_size(item_count: int, workers: int) -> int:
    return max(1, (item_count + max(workers, 1) - 1) // max(workers, 1))


def _iter_process_pool_map(worker_fn, tasks: list[dict], workers: int):
    if len(tasks) == 0:
        return
    start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
    with mp.get_context(start_method).Pool(processes=workers) as pool:
        yield from pool.imap_unordered(worker_fn, tasks, chunksize=1)


def _iter_parallel_map(
    *,
    worker_fn,
    tasks: list[dict],
    workers: int,
    pool_runner: ProcessPoolRunner | None,
):
    if pool_runner is not None:
        yield from pool_runner.iter_map_unordered(worker_fn, tasks)
        return
    yield from _iter_process_pool_map(worker_fn, tasks, workers)


def _parallel_progress_context(
    *,
    progress_tracker: ProgressTracker | None,
    pool_runner: ProcessPoolRunner | None,
):
    if progress_tracker is None:
        return None, None
    if pool_runner is not None and getattr(pool_runner, "_context", None) is not None:
        context = pool_runner._context
    else:
        start_method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        context = mp.get_context(start_method)
    if context.get_start_method() != "fork":
        return None, None
    queue = context.Queue()
    drainer = _QueueProgressDrainer(queue=queue, progress_tracker=progress_tracker)
    drainer.start()
    return queue, drainer


def _run_scenario_task(task: dict) -> tuple[str, dict]:
    scenario: ScenarioConfig = task["scenario"]
    result = run_single_scenario(
        scenario=scenario,
        tire_parameters=task["tire_parameters"],
        vehicle_parameters=task["vehicle_parameters"],
        dt_s=task["dt_s"],
        diagnostics_stride=task["diagnostics_stride"],
        inputs=task["inputs"],
        prepared_inputs=task["prepared_inputs"],
    )
    return scenario.name, result


def _evaluate_lhs_sample(task: dict) -> tuple[int, dict[str, tuple[list[float], list[float]]]]:
    batch_result = _evaluate_lhs_batch(
        {
            **task,
            "batch": [(task["sample_idx"], task["sample"])],
        }
    )
    return batch_result[0]


def _evaluate_sobol_sample(task: dict) -> tuple[int, float]:
    batch_result = _evaluate_sobol_batch(
        {
            **task,
            "batch": [(task["eval_idx"], task["sample"])],
        }
    )
    return batch_result[0]


def _evaluate_lhs_batch(task: dict) -> list[tuple[int, dict[str, tuple[list[float], list[float]]]]]:
    uq = HighFidelityUQ()
    scenarios: tuple[ScenarioConfig, ...] = task["scenarios"]
    scenario_inputs: dict[str, VehicleInputs] = task["scenario_inputs"]
    outputs: list[tuple[int, dict[str, tuple[list[float], list[float]]]]] = []
    for sample_idx, sample in task["batch"]:
        sampled_tire_parameters = uq.apply_sample(base=task["tire_parameters"], sample=sample)
        simulator = _vehicle_simulator(
            tire_parameters=sampled_tire_parameters,
            vehicle_parameters=task["vehicle_parameters"],
        )
        prepared_inputs = {
            scenario.name: simulator.prepare_inputs(scenario_inputs[scenario.name]) for scenario in scenarios
        }
        scenario_results: dict[str, tuple[list[float], list[float]]] = {}
        for scenario in scenarios:
            result = run_single_scenario_temperature_trace(
                scenario=scenario,
                tire_parameters=sampled_tire_parameters,
                vehicle_parameters=task["vehicle_parameters"],
                dt_s=task["dt_s"],
                diagnostics_stride=task["diagnostics_stride"],
                simulator=simulator,
                inputs=scenario_inputs[scenario.name],
                prepared_inputs=prepared_inputs[scenario.name],
            )
            scenario_results[scenario.name] = (
                result["mean_core_temp_c"],
                result["mean_surface_temp_c"],
            )
            if _WORKER_PROGRESS_QUEUE is not None:
                _WORKER_PROGRESS_QUEUE.put(1)
        outputs.append((sample_idx, scenario_results))
    return outputs


def _evaluate_sobol_batch(task: dict) -> list[tuple[int, dict[str, float]]]:
    uq = HighFidelityUQ()
    scenario: ScenarioConfig = task["scenario"]
    scenario_inputs: VehicleInputs = task["scenario_inputs"]
    outputs: list[tuple[int, dict[str, float]]] = []
    for eval_idx, sample in task["batch"]:
        sampled_tire_parameters = uq.apply_sample(base=task["tire_parameters"], sample=sample)
        simulator = _vehicle_simulator(
            tire_parameters=sampled_tire_parameters,
            vehicle_parameters=task["vehicle_parameters"],
        )
        result = run_single_scenario_temperature_trace(
            scenario=scenario,
            tire_parameters=sampled_tire_parameters,
            vehicle_parameters=task["vehicle_parameters"],
            dt_s=task["dt_s"],
            diagnostics_stride=task["diagnostics_stride"],
            simulator=simulator,
            inputs=scenario_inputs,
            prepared_inputs=simulator.prepare_inputs(scenario_inputs),
        )
        outputs.append((
            eval_idx,
            {
                "peak_mean_surface_temp_c": float(result["peak_mean_surface_temp_c"]),
                "peak_mean_core_temp_c": float(result["peak_mean_core_temp_c"]),
                "end_mean_surface_temp_c": float(result["end_mean_surface_temp_c"]),
                "end_mean_core_temp_c": float(result["end_mean_core_temp_c"]),
            },
        ))
        if _WORKER_PROGRESS_QUEUE is not None:
            _WORKER_PROGRESS_QUEUE.put(1)
    return outputs


def _batched_uq_tasks(
    *,
    items: list[tuple[int, dict[str, float]]],
    workers: int,
    base_task: dict,
    batch_size: int | None = None,
) -> list[dict]:
    resolved_batch_size = _batch_size(len(items), workers) if batch_size is None else max(int(batch_size), 1)
    return [
        {
            **base_task,
            "batch": [(item_idx, dict(sample)) for item_idx, sample in batch],
        }
        for batch in _batched(items, resolved_batch_size)
    ]


def _sample_matrix(
    *,
    payloads: list[tuple[int, dict[str, float]]],
    prior_names: tuple[str, ...],
) -> np.ndarray:
    return np.asarray(
        [[float(sample[name]) for name in prior_names] for _idx, sample in payloads],
        dtype=float,
    )


def _fit_multi_output_surrogate(
    *,
    kind: str,
    priors,
    train_payloads: list[tuple[int, dict[str, float]]],
    target_matrix: np.ndarray,
    ridge_alpha: float,
    random_state: int,
    extra_trees_estimators: int,
):
    x_train = _sample_matrix(payloads=train_payloads, prior_names=tuple(prior.name for prior in priors))
    if kind == "extra_trees":
        surrogate = ExtraTreesSurrogate(
            estimators=extra_trees_estimators,
            random_state=random_state,
        )
        surrogate.fit(x_train, target_matrix)
        return surrogate
    surrogate = QuadraticRidgeSurrogate(
        lower=np.asarray([float(prior.lower) for prior in priors], dtype=float),
        upper=np.asarray([float(prior.upper) for prior in priors], dtype=float),
        ridge_alpha=ridge_alpha,
    )
    surrogate.fit(x_train, target_matrix)
    return surrogate


def _evaluate_sobol_payloads(
    *,
    eval_payloads: list[tuple[int, dict[str, float]]],
    scenario: ScenarioConfig,
    scenario_inputs: VehicleInputs,
    tire_parameters: HighFidelityTireModelParameters,
    vehicle_parameters: VehicleParameters,
    dt_s: float,
    diagnostics_stride: int,
    workers: int,
    progress_tracker: ProgressTracker | None,
    pool_runner: ProcessPoolRunner | None,
) -> list[tuple[int, dict[str, float]]]:
    sobol_results: list[tuple[int, dict[str, float]]] = []
    if workers > 1 and len(eval_payloads) > 1:
        progress_queue, drainer = _parallel_progress_context(
            progress_tracker=progress_tracker,
            pool_runner=pool_runner,
        )
        try:
            global _WORKER_PROGRESS_QUEUE
            _WORKER_PROGRESS_QUEUE = progress_queue
            tasks = _batched_uq_tasks(
                items=eval_payloads,
                workers=workers,
                base_task={
                    "scenario": scenario,
                    "scenario_inputs": scenario_inputs,
                    "tire_parameters": tire_parameters,
                    "vehicle_parameters": vehicle_parameters,
                    "dt_s": dt_s,
                    "diagnostics_stride": diagnostics_stride,
                },
            )
            if drainer is not None and hasattr(pool_runner, "restart"):
                pool_runner.restart()
            for batch_result in _iter_parallel_map(
                worker_fn=_evaluate_sobol_batch,
                tasks=tasks,
                workers=workers,
                pool_runner=pool_runner,
            ):
                sobol_results.extend(batch_result)
                if drainer is None and progress_tracker is not None:
                    progress_tracker.advance(len(batch_result))
        finally:
            _WORKER_PROGRESS_QUEUE = None
            if drainer is not None:
                drainer.stop()
            if progress_queue is not None:
                progress_queue.close()
                progress_queue.join_thread()
    else:
        sobol_results = _evaluate_sobol_batch(
            {
                "batch": eval_payloads,
                "scenario": scenario,
                "scenario_inputs": scenario_inputs,
                "tire_parameters": tire_parameters,
                "vehicle_parameters": vehicle_parameters,
                "dt_s": dt_s,
                "diagnostics_stride": diagnostics_stride,
            }
        )
        if progress_tracker is not None:
            progress_tracker.advance(len(sobol_results))
    sobol_results.sort(key=lambda item: item[0])
    return sobol_results


def _sobol_surrogate_splits(
    *,
    eval_payloads: list[tuple[int, dict[str, float]]],
    train_count: int,
    validation_count: int,
    seed: int,
) -> tuple[list[tuple[int, dict[str, float]]], list[tuple[int, dict[str, float]]], list[tuple[int, dict[str, float]]]]:
    rng = np.random.default_rng(seed)
    order = np.arange(len(eval_payloads), dtype=int)
    rng.shuffle(order)
    train_idx = np.sort(order[:train_count])
    validation_idx = np.sort(order[train_count : train_count + validation_count])
    predicted_idx = np.sort(order[train_count + validation_count :])
    return (
        [eval_payloads[int(idx)] for idx in train_idx],
        [eval_payloads[int(idx)] for idx in validation_idx],
        [eval_payloads[int(idx)] for idx in predicted_idx],
    )


def _sobol_surrogate_error_summary(
    *,
    predicted: np.ndarray,
    exact: np.ndarray,
) -> tuple[float, float]:
    delta = np.asarray(predicted, dtype=float) - np.asarray(exact, dtype=float)
    return float(np.sqrt(np.mean(delta * delta))), float(np.max(np.abs(delta)))


def _should_enable_progress(enabled: bool | None) -> bool:
    if enabled is not None:
        return enabled
    return sys.stderr.isatty()


def _build_progress_tracker(
    *,
    scenarios: tuple[ScenarioConfig, ...],
    uq_scenario_count: int,
    lhs_samples: int,
    sobol_eval_count: int,
    enabled: bool | None,
) -> ProgressTracker:
    phase_totals = {
        "baseline": len(scenarios),
        "lhs": int(lhs_samples) * max(int(uq_scenario_count), 1),
        "sobol": int(sobol_eval_count),
        "done": 0,
    }
    if not _should_enable_progress(enabled):
        return ProgressTracker(phase_totals=phase_totals)
    total = len(scenarios) + phase_totals["lhs"] + sobol_eval_count
    bar = tqdm(
        total=total,
        desc="baseline",
        unit="run",
        dynamic_ncols=True,
        leave=True,
        file=sys.stderr,
    )
    return ProgressTracker(bar=bar, phase_totals=phase_totals)


def run_high_fidelity_no_data(
    *,
    output_path: Path = RESULTS_FILE,
    summary_path: Path = SUMMARY_FILE,
    preset: str = "full",
    lhs_samples: int | None = None,
    sobol_samples: int | None = None,
    seed: int = 2026,
    dt_s: float = 0.2,
    duration_scale: float = 1.0,
    diagnostics_stride: int | None = None,
    workers: int = 1,
    progress: bool | None = None,
    tire_parameters: HighFidelityTireModelParameters | None = None,
    vehicle_parameters: VehicleParameters | None = None,
    uq_surrogate: UQSurrogateConfig | None = None,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    active_preset = fidelity_preset(preset)
    lhs_samples = active_preset.lhs_samples if lhs_samples is None else lhs_samples
    sobol_samples = active_preset.sobol_samples if sobol_samples is None else sobol_samples
    diagnostics_stride = active_preset.diagnostics_stride if diagnostics_stride is None else diagnostics_stride
    tire_params = tire_parameters if tire_parameters is not None else default_tire_parameters(
        radial_cells=active_preset.radial_cells,
        theta_cells=active_preset.theta_cells,
        internal_solver_dt_s=active_preset.internal_solver_dt_s,
    )
    vehicle_params = vehicle_parameters if vehicle_parameters is not None else VehicleParameters()
    surrogate_config = uq_surrogate if uq_surrogate is not None else UQSurrogateConfig()
    scenarios = default_scenarios(duration_scale=duration_scale)
    progress_tracker = _build_progress_tracker(
        scenarios=scenarios,
        uq_scenario_count=sum(1 for scenario in scenarios if scenario.include_in_uq),
        lhs_samples=lhs_samples,
        sobol_eval_count=sobol_samples * (2 + len(HighFidelityUQ().default_tire_priors(parameters=tire_params))),
        enabled=progress,
    )
    pool_runner = ProcessPoolRunner(workers=workers) if workers > 1 else None
    try:
        baseline = run_scenario_pack(
            scenarios=scenarios,
            tire_parameters=tire_params,
            vehicle_parameters=vehicle_params,
            dt_s=dt_s,
            diagnostics_stride=diagnostics_stride,
            workers=workers,
            progress_tracker=progress_tracker,
            pool_runner=pool_runner,
        )

        uq = HighFidelityUQ()
        priors = uq.default_tire_priors(parameters=tire_params)
        lhs_result = run_lhs_uq(
            scenarios=tuple(s for s in scenarios if s.include_in_uq),
            tire_parameters=tire_params,
            vehicle_parameters=vehicle_params,
            dt_s=dt_s,
            uq=uq,
            lhs_samples=lhs_samples,
            seed=seed,
            diagnostics_stride=diagnostics_stride,
            workers=workers,
            progress_tracker=progress_tracker,
            pool_runner=pool_runner,
        )
        sobol_result = run_sobol_uq(
            scenario=next(s for s in scenarios if s.name == "combined_brake_corner"),
            tire_parameters=tire_params,
            vehicle_parameters=vehicle_params,
            dt_s=dt_s,
            uq=uq,
            sobol_samples=sobol_samples,
            seed=seed,
            diagnostics_stride=diagnostics_stride,
            workers=workers,
            progress_tracker=progress_tracker,
            pool_runner=pool_runner,
            surrogate_config=surrogate_config,
        )
    finally:
        if pool_runner is not None:
            pool_runner.close()

    progress_tracker.set_phase("done")
    timing_summary = progress_tracker.timings()
    artifact = {
        "metadata": {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "dt_s": float(dt_s),
            "duration_scale": float(duration_scale),
            "lhs_samples": int(lhs_samples),
            "sobol_samples": int(sobol_samples),
            "seed": int(seed),
            "preset": active_preset.name,
            "diagnostics_stride": int(diagnostics_stride),
            "radial_cells": int(tire_params.radial_cells),
            "theta_cells": int(tire_params.theta_cells),
            "internal_solver_dt_s": float(tire_params.internal_solver_dt_s),
            "results_path": str(output_path),
            "summary_path": str(summary_path),
            "default_output_mode": "bands_plus_baseline",
            "scenario_names": [scenario.name for scenario in scenarios],
            "uq_scenario_names": [scenario.name for scenario in scenarios if scenario.include_in_uq],
            "uq_surrogate": asdict(surrogate_config),
        },
        "timing": timing_summary,
        "priors": [asdict(prior) for prior in priors],
        "baseline": baseline,
        "uq": {
            "lhs": lhs_result,
            "sobol": sobol_result,
        },
        "plausibility_checks": build_plausibility_checks(baseline=baseline, sobol=sobol_result),
    }

    output_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    summary_text = render_high_fidelity_no_data_summary(artifact)
    summary_path.write_text(summary_text, encoding="utf-8")
    progress_tracker.close()
    return artifact


def run_scenario_pack(
    *,
    scenarios: tuple[ScenarioConfig, ...],
    tire_parameters: HighFidelityTireModelParameters,
    vehicle_parameters: VehicleParameters,
    dt_s: float,
    diagnostics_stride: int = 1,
    workers: int = 1,
    progress_tracker: ProgressTracker | None = None,
    pool_runner: ProcessPoolRunner | None = None,
) -> dict:
    simulator = _vehicle_simulator(
        tire_parameters=tire_parameters,
        vehicle_parameters=vehicle_parameters,
    )
    scenario_inputs = {scenario.name: _vehicle_inputs_for_scenario(scenario) for scenario in scenarios}
    scenario_prepared_inputs = {
        scenario.name: simulator.prepare_inputs(scenario_inputs[scenario.name]) for scenario in scenarios
    }
    scenario_traces: dict[str, dict] = {}
    scenario_summaries: dict[str, dict] = {}
    if progress_tracker is not None:
        progress_tracker.set_phase("baseline")
    if workers > 1 and len(scenarios) > 1:
        tasks = [
            {
                "scenario": scenario,
                "tire_parameters": tire_parameters,
                "vehicle_parameters": vehicle_parameters,
                "dt_s": dt_s,
                "diagnostics_stride": diagnostics_stride,
                "inputs": scenario_inputs[scenario.name],
                "prepared_inputs": scenario_prepared_inputs[scenario.name],
            }
            for scenario in scenarios
        ]
        scenario_results_by_name: dict[str, dict] = {}
        for scenario_name, result in _iter_parallel_map(
            worker_fn=_run_scenario_task,
            tasks=tasks,
            workers=min(workers, len(tasks)),
            pool_runner=pool_runner,
        ):
            scenario_results_by_name[scenario_name] = result
            if progress_tracker is not None:
                progress_tracker.advance(1)
        for scenario in scenarios:
            result = scenario_results_by_name[scenario.name]
            scenario_traces[scenario.name] = result["trace"]
            scenario_summaries[scenario.name] = result["summary"]
    else:
        for scenario in scenarios:
            result = run_single_scenario(
                scenario=scenario,
                tire_parameters=tire_parameters,
                vehicle_parameters=vehicle_parameters,
                dt_s=dt_s,
                diagnostics_stride=diagnostics_stride,
                simulator=simulator,
                inputs=scenario_inputs[scenario.name],
                prepared_inputs=scenario_prepared_inputs[scenario.name],
            )
            scenario_traces[scenario.name] = result["trace"]
            scenario_summaries[scenario.name] = result["summary"]
            if progress_tracker is not None:
                progress_tracker.advance(1)
    return {
        "scenario_traces": scenario_traces,
        "scenario_summaries": scenario_summaries,
    }


def run_single_scenario(
    *,
    scenario: ScenarioConfig,
    tire_parameters: HighFidelityTireModelParameters,
    vehicle_parameters: VehicleParameters,
    dt_s: float,
    diagnostics_stride: int = 1,
    simulator: HighFidelityVehicleSimulator | None = None,
    inputs: VehicleInputs | None = None,
    prepared_inputs=None,
) -> dict:
    active_simulator = simulator if simulator is not None else _vehicle_simulator(
        tire_parameters=tire_parameters,
        vehicle_parameters=vehicle_parameters,
    )
    active_inputs = inputs if inputs is not None else _vehicle_inputs_for_scenario(scenario)
    active_prepared_inputs = (
        active_simulator.prepare_inputs(active_inputs) if prepared_inputs is None else prepared_inputs
    )
    state = active_simulator.initial_state(ambient_temp_k=scenario.ambient_temp_k)

    steps = max(int(round(scenario.duration_s / dt_s)), 1)
    time_s = [0.0]
    mean_core_temp_c = []
    mean_surface_temp_c = []
    load_error_pct = []
    max_energy_residual_pct = []
    coupling_converged_fraction = []
    any_non_finite = False

    initial_diag = active_simulator.diagnostics(state, active_inputs, prepared_inputs=active_prepared_inputs)
    mean_core_temp_c.append(_mean_dict_value(initial_diag.wheel_core_temp_c))
    mean_surface_temp_c.append(_mean_dict_value(initial_diag.wheel_surface_temp_c))
    load_error_pct.append(initial_diag.load_conservation_error_pct)
    max_energy_residual_pct.append(_max_energy_residual(initial_diag))
    coupling_converged_fraction.append(_coupling_converged_fraction(initial_diag))
    any_non_finite = any_non_finite or _diag_has_non_finite(initial_diag)

    converged_wheel_steps = 0
    total_wheel_steps = 0

    diagnostics_stride = max(int(diagnostics_stride), 1)
    for step in range(1, steps + 1):
        state = active_simulator.step(state, active_inputs, dt_s=dt_s, prepared_inputs=active_prepared_inputs)
        if step % diagnostics_stride == 0 or step == steps:
            diag = active_simulator.diagnostics(state, active_inputs, prepared_inputs=active_prepared_inputs)
            time_s.append(step * dt_s)
            mean_core_temp_c.append(_mean_dict_value(diag.wheel_core_temp_c))
            mean_surface_temp_c.append(_mean_dict_value(diag.wheel_surface_temp_c))
            load_error_pct.append(diag.load_conservation_error_pct)
            max_energy_residual_pct.append(_max_energy_residual(diag))
            coupling_converged_fraction.append(_coupling_converged_fraction(diag))
            any_non_finite = any_non_finite or _diag_has_non_finite(diag)
            converged_wheel_steps += sum(1 for ok in diag.wheel_coupling_converged.values() if ok)
            total_wheel_steps += len(diag.wheel_coupling_converged)

    trace = {
        "time_s": time_s,
        "mean_core_temp_c": mean_core_temp_c,
        "mean_surface_temp_c": mean_surface_temp_c,
        "load_error_pct": load_error_pct,
        "max_energy_residual_pct": max_energy_residual_pct,
        "coupling_converged_fraction": coupling_converged_fraction,
    }
    summary = {
        "end_mean_core_temp_c": float(mean_core_temp_c[-1]),
        "peak_mean_core_temp_c": float(max(mean_core_temp_c)),
        "end_mean_surface_temp_c": float(mean_surface_temp_c[-1]),
        "peak_mean_surface_temp_c": float(max(mean_surface_temp_c)),
        "max_load_error_pct": float(max(load_error_pct)),
        "max_energy_residual_pct": float(max(max_energy_residual_pct)),
        "coupling_convergence_rate": float(
            converged_wheel_steps / max(total_wheel_steps, 1)
        ),
        "all_outputs_finite": not any_non_finite,
    }
    return {"trace": trace, "summary": summary}


def _wheel_surface_temperature_c(state) -> float:
    if state.thermal_field_rtw_k is not None:
        return float(np.mean(state.thermal_field_rtw_k[-1, :, :]) - 273.15)
    return float(np.mean(state.temperature_nodes_k[:3]) - 273.15)


def _state_mean_core_surface_temperatures_c(state) -> tuple[float, float]:
    core_temps_c = [wheel_state.core_temperature_c for wheel_state in state.wheel_states.values()]
    surface_temps_c = [_wheel_surface_temperature_c(wheel_state) for wheel_state in state.wheel_states.values()]
    return float(np.mean(core_temps_c)), float(np.mean(surface_temps_c))


def run_single_scenario_temperature_trace(
    *,
    scenario: ScenarioConfig,
    tire_parameters: HighFidelityTireModelParameters,
    vehicle_parameters: VehicleParameters,
    dt_s: float,
    diagnostics_stride: int = 1,
    simulator: HighFidelityVehicleSimulator | None = None,
    inputs: VehicleInputs | None = None,
    prepared_inputs=None,
) -> dict[str, list[float] | float]:
    active_simulator = simulator if simulator is not None else _vehicle_simulator(
        tire_parameters=tire_parameters,
        vehicle_parameters=vehicle_parameters,
    )
    active_inputs = inputs if inputs is not None else _vehicle_inputs_for_scenario(scenario)
    active_prepared_inputs = (
        active_simulator.prepare_inputs(active_inputs) if prepared_inputs is None else prepared_inputs
    )
    state = active_simulator.initial_state(ambient_temp_k=scenario.ambient_temp_k)

    steps = max(int(round(scenario.duration_s / dt_s)), 1)
    mean_core_temp_c: list[float] = []
    mean_surface_temp_c: list[float] = []

    initial_core_c, initial_surface_c = _state_mean_core_surface_temperatures_c(state)
    mean_core_temp_c.append(initial_core_c)
    mean_surface_temp_c.append(initial_surface_c)

    diagnostics_stride = max(int(diagnostics_stride), 1)
    for step in range(1, steps + 1):
        state = active_simulator.step(state, active_inputs, dt_s=dt_s, prepared_inputs=active_prepared_inputs)
        if step % diagnostics_stride == 0 or step == steps:
            mean_core_c, mean_surface_c = _state_mean_core_surface_temperatures_c(state)
            mean_core_temp_c.append(mean_core_c)
            mean_surface_temp_c.append(mean_surface_c)

    return {
        "mean_core_temp_c": mean_core_temp_c,
        "mean_surface_temp_c": mean_surface_temp_c,
        "end_mean_core_temp_c": float(mean_core_temp_c[-1]),
        "peak_mean_core_temp_c": float(max(mean_core_temp_c)),
        "end_mean_surface_temp_c": float(mean_surface_temp_c[-1]),
        "peak_mean_surface_temp_c": float(max(mean_surface_temp_c)),
    }


def run_lhs_uq(
    *,
    scenarios: tuple[ScenarioConfig, ...],
    tire_parameters: HighFidelityTireModelParameters,
    vehicle_parameters: VehicleParameters,
    dt_s: float,
    uq: HighFidelityUQ,
    lhs_samples: int,
    seed: int,
    diagnostics_stride: int = 1,
    workers: int = 1,
    progress_tracker: ProgressTracker | None = None,
    pool_runner: ProcessPoolRunner | None = None,
) -> dict:
    priors = uq.default_tire_priors(parameters=tire_parameters)
    unit_samples = uq.latin_hypercube(priors=priors, sample_count=lhs_samples, seed=seed)
    mapped = uq.map_priors(unit_samples=unit_samples, priors=priors)
    scenario_inputs = {scenario.name: _vehicle_inputs_for_scenario(scenario) for scenario in scenarios}

    scenario_core_traces: dict[str, list[np.ndarray]] = {scenario.name: [] for scenario in scenarios}
    scenario_surface_traces: dict[str, list[np.ndarray]] = {scenario.name: [] for scenario in scenarios}
    scenario_end_metrics: dict[str, list[float]] = {scenario.name: [] for scenario in scenarios}
    scenario_peak_metrics: dict[str, list[float]] = {scenario.name: [] for scenario in scenarios}
    scenario_end_surface_metrics: dict[str, list[float]] = {scenario.name: [] for scenario in scenarios}
    scenario_peak_surface_metrics: dict[str, list[float]] = {scenario.name: [] for scenario in scenarios}

    sample_payloads = [
        (sample_idx, {prior.name: float(mapped[prior.name][sample_idx]) for prior in priors})
        for sample_idx in range(lhs_samples)
    ]
    lhs_results: list[tuple[int, dict[str, tuple[list[float], list[float]]]]] = []
    if progress_tracker is not None:
        progress_tracker.set_phase("lhs")
    if workers > 1 and lhs_samples > 1:
        progress_queue, drainer = _parallel_progress_context(
            progress_tracker=progress_tracker,
            pool_runner=pool_runner,
        )
        try:
            global _WORKER_PROGRESS_QUEUE
            _WORKER_PROGRESS_QUEUE = progress_queue
            tasks = _batched_uq_tasks(
                items=sample_payloads,
                workers=workers,
                base_task={
                    "scenarios": scenarios,
                    "scenario_inputs": scenario_inputs,
                    "tire_parameters": tire_parameters,
                "vehicle_parameters": vehicle_parameters,
                "dt_s": dt_s,
                "diagnostics_stride": diagnostics_stride,
                },
            )
            if drainer is not None and hasattr(pool_runner, "restart"):
                pool_runner.restart()
            for batch_result in _iter_parallel_map(
                worker_fn=_evaluate_lhs_batch,
                tasks=tasks,
                workers=workers,
                pool_runner=pool_runner,
            ):
                lhs_results.extend(batch_result)
                if drainer is None and progress_tracker is not None:
                    progress_tracker.advance(len(batch_result))
        finally:
            _WORKER_PROGRESS_QUEUE = None
            if drainer is not None:
                drainer.stop()
            if progress_queue is not None:
                progress_queue.close()
                progress_queue.join_thread()
    else:
        for sample_idx, sample in sample_payloads:
            lhs_results.append(
                _evaluate_lhs_sample(
                    {
                        "sample_idx": sample_idx,
                        "sample": sample,
                        "scenarios": scenarios,
                        "scenario_inputs": scenario_inputs,
                        "tire_parameters": tire_parameters,
                        "vehicle_parameters": vehicle_parameters,
                        "dt_s": dt_s,
                        "diagnostics_stride": diagnostics_stride,
                    }
                )
            )
            if progress_tracker is not None:
                progress_tracker.advance(len(scenarios))

    lhs_results.sort(key=lambda item: item[0])
    for _sample_idx, sample_result in lhs_results:
        for scenario in scenarios:
            core_trace = np.asarray(sample_result[scenario.name][0], dtype=float)
            surface_trace = np.asarray(sample_result[scenario.name][1], dtype=float)
            scenario_core_traces[scenario.name].append(core_trace)
            scenario_surface_traces[scenario.name].append(surface_trace)
            scenario_end_metrics[scenario.name].append(float(core_trace[-1]))
            scenario_peak_metrics[scenario.name].append(float(np.max(core_trace)))
            scenario_end_surface_metrics[scenario.name].append(float(surface_trace[-1]))
            scenario_peak_surface_metrics[scenario.name].append(float(np.max(surface_trace)))

    scenario_envelopes: dict[str, dict] = {}
    for scenario in scenarios:
        core_outputs = np.stack(scenario_core_traces[scenario.name], axis=0)
        surface_outputs = np.stack(scenario_surface_traces[scenario.name], axis=0)
        core_trace_envelope = uq.quantile_envelope(core_outputs)
        surface_trace_envelope = uq.quantile_envelope(surface_outputs)
        end_values = np.asarray(scenario_end_metrics[scenario.name], dtype=float)
        peak_values = np.asarray(scenario_peak_metrics[scenario.name], dtype=float)
        end_surface_values = np.asarray(scenario_end_surface_metrics[scenario.name], dtype=float)
        peak_surface_values = np.asarray(scenario_peak_surface_metrics[scenario.name], dtype=float)
        scenario_envelopes[scenario.name] = {
            "mean_core_temp_c_trace": {
                "q05": core_trace_envelope.q05.tolist(),
                "q50": core_trace_envelope.q50.tolist(),
                "q95": core_trace_envelope.q95.tolist(),
            },
            "mean_surface_temp_c_trace": {
                "q05": surface_trace_envelope.q05.tolist(),
                "q50": surface_trace_envelope.q50.tolist(),
                "q95": surface_trace_envelope.q95.tolist(),
            },
            "end_mean_core_temp_c": _scalar_quantiles(end_values),
            "peak_mean_core_temp_c": _scalar_quantiles(peak_values),
            "end_mean_surface_temp_c": _scalar_quantiles(end_surface_values),
            "peak_mean_surface_temp_c": _scalar_quantiles(peak_surface_values),
        }

    return {
        "scenario_envelopes": scenario_envelopes,
    }


def run_sobol_uq(
    *,
    scenario: ScenarioConfig,
    tire_parameters: HighFidelityTireModelParameters,
    vehicle_parameters: VehicleParameters,
    dt_s: float,
    uq: HighFidelityUQ,
    sobol_samples: int,
    seed: int,
    diagnostics_stride: int = 1,
    workers: int = 1,
    progress_tracker: ProgressTracker | None = None,
    pool_runner: ProcessPoolRunner | None = None,
    surrogate_config: UQSurrogateConfig | None = None,
) -> dict:
    metric_candidates = (
        "peak_mean_surface_temp_c",
        "peak_mean_core_temp_c",
        "end_mean_surface_temp_c",
        "end_mean_core_temp_c",
    )
    priors = uq.default_tire_priors(parameters=tire_parameters)
    scenario_inputs = _vehicle_inputs_for_scenario(scenario)
    prior_names = tuple(prior.name for prior in priors)
    unit_a = uq.random_unit_samples(priors=priors, sample_count=sobol_samples, seed=seed)
    unit_b = uq.random_unit_samples(priors=priors, sample_count=sobol_samples, seed=seed + 1)
    matrix_a = uq._map_prior_matrix(unit_samples=unit_a, priors=priors)
    matrix_b = uq._map_prior_matrix(unit_samples=unit_b, priors=priors)

    eval_payloads: list[tuple[int, dict[str, float]]] = []
    y_ab_offsets: list[int] = []
    for row_idx in range(sobol_samples):
        eval_payloads.append((row_idx, {name: float(value) for name, value in zip(prior_names, matrix_a[row_idx], strict=True)}))
    for row_idx in range(sobol_samples):
        eval_payloads.append((sobol_samples + row_idx, {name: float(value) for name, value in zip(prior_names, matrix_b[row_idx], strict=True)}))
    for dim in range(len(priors)):
        y_ab_offsets.append(len(eval_payloads))
        matrix_ab = matrix_a.copy()
        matrix_ab[:, dim] = matrix_b[:, dim]
        for row_idx in range(sobol_samples):
            eval_payloads.append((
                y_ab_offsets[-1] + row_idx,
                {name: float(value) for name, value in zip(prior_names, matrix_ab[row_idx], strict=True)},
            ))

    sobol_results: list[tuple[int, dict[str, float]]] = []
    active_surrogate = surrogate_config if surrogate_config is not None else UQSurrogateConfig()
    if progress_tracker is not None:
        progress_tracker.set_phase("sobol")
    if active_surrogate.enabled and len(eval_payloads) > 0:
        requested_train = active_surrogate.sobol_train_samples
        train_count = min(
            max(requested_train if requested_train is not None else max(4 * len(priors), 128), 1),
            len(eval_payloads),
        )
        validation_count = min(
            max(
                active_surrogate.sobol_validation_samples
                if active_surrogate.sobol_validation_samples is not None
                else max(len(priors), 32),
                0,
            ),
            max(len(eval_payloads) - train_count, 0),
        )
        predicted_count = len(eval_payloads) - train_count - validation_count
        if (
            predicted_count >= max(int(active_surrogate.min_prediction_samples), 1)
            and active_surrogate.kind in {"quadratic_ridge", "extra_trees"}
        ):
            train_payloads, validation_payloads, predicted_payloads = _sobol_surrogate_splits(
                eval_payloads=eval_payloads,
                train_count=train_count,
                validation_count=validation_count,
                seed=seed + 303,
            )
            exact_payloads = train_payloads + validation_payloads
            sobol_results = _evaluate_sobol_payloads(
                eval_payloads=exact_payloads,
                scenario=scenario,
                scenario_inputs=scenario_inputs,
                tire_parameters=tire_parameters,
                vehicle_parameters=vehicle_parameters,
                dt_s=dt_s,
                diagnostics_stride=diagnostics_stride,
                workers=workers,
                progress_tracker=None,
                pool_runner=pool_runner,
            )
            result_by_eval_idx = {eval_idx: values for eval_idx, values in sobol_results}
            surrogate = _fit_multi_output_surrogate(
                kind=active_surrogate.kind,
                priors=priors,
                train_payloads=train_payloads,
                target_matrix=np.asarray(
                    [
                        [float(result_by_eval_idx[eval_idx][metric_name]) for metric_name in metric_candidates]
                        for eval_idx, _sample in train_payloads
                    ],
                    dtype=float,
                ),
                ridge_alpha=active_surrogate.ridge_alpha,
                random_state=seed + 404,
                extra_trees_estimators=active_surrogate.extra_trees_estimators,
            )
            prior_names = tuple(prior.name for prior in priors)
            if validation_payloads:
                validation_pred = surrogate.predict(
                    _sample_matrix(payloads=validation_payloads, prior_names=prior_names)
                )
                validation_exact = np.asarray(
                    [
                        [float(result_by_eval_idx[eval_idx][metric_name]) for metric_name in metric_candidates]
                        for eval_idx, _sample in validation_payloads
                    ],
                    dtype=float,
                )
                rmse_c, max_abs_error_c = _sobol_surrogate_error_summary(
                    predicted=validation_pred,
                    exact=validation_exact,
                )
            else:
                rmse_c, max_abs_error_c = 0.0, 0.0
            if (
                np.isfinite(rmse_c)
                and np.isfinite(max_abs_error_c)
                and rmse_c <= float(active_surrogate.max_rmse_c)
                and max_abs_error_c <= float(active_surrogate.max_abs_error_c)
            ):
                if predicted_payloads:
                    prediction_matrix = surrogate.predict(
                        _sample_matrix(payloads=predicted_payloads, prior_names=prior_names)
                    )
                    for row_idx, (eval_idx, _sample) in enumerate(predicted_payloads):
                        result_by_eval_idx[eval_idx] = {
                            metric_name: float(prediction_matrix[row_idx, metric_idx])
                            for metric_idx, metric_name in enumerate(metric_candidates)
                        }
                if progress_tracker is not None:
                    progress_tracker.advance(len(eval_payloads))
                sobol_results = sorted(result_by_eval_idx.items(), key=lambda item: item[0])
            else:
                sobol_results = _evaluate_sobol_payloads(
                    eval_payloads=eval_payloads,
                    scenario=scenario,
                    scenario_inputs=scenario_inputs,
                    tire_parameters=tire_parameters,
                    vehicle_parameters=vehicle_parameters,
                    dt_s=dt_s,
                    diagnostics_stride=diagnostics_stride,
                    workers=workers,
                    progress_tracker=progress_tracker,
                    pool_runner=pool_runner,
                )
        else:
            sobol_results = _evaluate_sobol_payloads(
                eval_payloads=eval_payloads,
                scenario=scenario,
                scenario_inputs=scenario_inputs,
                tire_parameters=tire_parameters,
                vehicle_parameters=vehicle_parameters,
                dt_s=dt_s,
                diagnostics_stride=diagnostics_stride,
                workers=workers,
                progress_tracker=progress_tracker,
                pool_runner=pool_runner,
            )
    else:
        sobol_results = _evaluate_sobol_payloads(
            eval_payloads=eval_payloads,
            scenario=scenario,
            scenario_inputs=scenario_inputs,
            tire_parameters=tire_parameters,
            vehicle_parameters=vehicle_parameters,
            dt_s=dt_s,
            diagnostics_stride=diagnostics_stride,
            workers=workers,
            progress_tracker=progress_tracker,
            pool_runner=pool_runner,
        )

    ordered_metrics = {
        metric_name: np.zeros(len(eval_payloads), dtype=float)
        for metric_name in metric_candidates
    }
    for eval_idx, values in sobol_results:
        for metric_name in metric_candidates:
            ordered_metrics[metric_name][eval_idx] = float(values[metric_name])

    selected_metric = metric_candidates[0]
    selected_sobol = None
    fallback_metric = None
    for metric_name in metric_candidates:
        ordered_results = ordered_metrics[metric_name]
        y_a = ordered_results[:sobol_samples]
        y_b = ordered_results[sobol_samples : 2 * sobol_samples]
        y_ab_by_dim = [
            ordered_results[offset : offset + sobol_samples]
            for offset in y_ab_offsets
        ]
        sobol = uq.sobol_indices_from_evaluations(
            priors=priors,
            y_a=y_a,
            y_b=y_b,
            y_ab_by_dim=y_ab_by_dim,
        )
        if fallback_metric is None:
            fallback_metric = metric_name
            selected_sobol = sobol
        if sobol.variance > 1e-12:
            selected_metric = metric_name
            selected_sobol = sobol
            break

    assert selected_sobol is not None
    if selected_sobol.variance <= 1e-12:
        selected_metric = fallback_metric if fallback_metric is not None else metric_candidates[0]
    return {
        "objective_metric": f"{scenario.name}.{selected_metric}",
        "variance": selected_sobol.variance,
        "indices": [asdict(index) for index in selected_sobol.indices],
    }


def build_plausibility_checks(*, baseline: dict, sobol: dict) -> dict[str, bool]:
    summaries = baseline["scenario_summaries"]
    return {
        "all_peak_core_finite": bool(
            all(np.isfinite(summary["peak_mean_core_temp_c"]) for summary in summaries.values())
        ),
        "all_outputs_finite": bool(
            all(summary["all_outputs_finite"] for summary in summaries.values())
        ),
        "all_load_errors_below_0.5pct": bool(
            all(summary["max_load_error_pct"] < 0.5 for summary in summaries.values())
        ),
        "all_energy_residuals_below_1pct": bool(
            all(summary["max_energy_residual_pct"] < 1.0 for summary in summaries.values())
        ),
        "all_coupling_rates_above_0.99": bool(
            all(summary["coupling_convergence_rate"] >= 0.99 for summary in summaries.values())
        ),
        "sobol_indices_present": bool(len(sobol["indices"]) > 0),
    }


def _mean_dict_value(values: dict[str, float]) -> float:
    return float(np.mean(list(values.values())))


def _max_energy_residual(diag) -> float:
    return float(np.max([item.energy_residual_pct for item in diag.tire_diagnostics.values()]))


def _coupling_converged_fraction(diag) -> float:
    return float(np.mean(list(diag.wheel_coupling_converged.values())))


def _diag_has_non_finite(diag) -> bool:
    arrays = [
        np.asarray(list(diag.wheel_core_temp_c.values()), dtype=float),
        np.asarray(list(diag.wheel_surface_temp_c.values()), dtype=float),
        np.asarray(list(diag.wheel_effective_slip_ratio.values()), dtype=float),
        np.asarray(list(diag.wheel_effective_slip_angle_rad.values()), dtype=float),
        np.asarray(list(diag.wheel_load_n.values()), dtype=float),
    ]
    diag_values = []
    for tire_diag in diag.tire_diagnostics.values():
        diag_values.extend(
            [
                tire_diag.core_temperature_k,
                tire_diag.surface_temperature_k,
                tire_diag.mean_temperature_k,
                tire_diag.energy_residual_pct,
                tire_diag.friction_power_total_w,
                tire_diag.friction_power_tire_w,
                tire_diag.friction_power_road_w,
                tire_diag.road_conduction_w,
                tire_diag.rim_conduction_w,
                tire_diag.brake_heat_to_tire_w,
                tire_diag.brake_heat_to_rim_w,
                tire_diag.effective_slip_ratio,
                tire_diag.effective_slip_angle_rad,
                tire_diag.longitudinal_force_n,
                tire_diag.lateral_force_n,
                tire_diag.torque_residual_nm,
                tire_diag.lateral_force_residual_n,
            ]
        )
    arrays.append(np.asarray(diag_values, dtype=float))
    return not all(np.isfinite(values).all() for values in arrays)


def _scalar_quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "q05": float(np.quantile(values, 0.05)),
        "q50": float(np.quantile(values, 0.50)),
        "q95": float(np.quantile(values, 0.95)),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run high-fidelity no-data scenario harness.")
    parser.add_argument("--preset", choices=sorted(PRESETS), default="full")
    parser.add_argument("--output-json", type=Path, default=RESULTS_FILE)
    parser.add_argument("--output-summary", type=Path, default=SUMMARY_FILE)
    parser.add_argument("--lhs-samples", type=int, default=None)
    parser.add_argument("--sobol-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dt-s", type=float, default=0.2)
    parser.add_argument("--duration-scale", type=float, default=1.0)
    parser.add_argument("--radial-cells", type=int, default=None)
    parser.add_argument("--theta-cells", type=int, default=None)
    parser.add_argument("--internal-dt-s", type=float, default=None)
    parser.add_argument("--diagnostics-stride", type=int, default=None)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--uq-surrogate", choices=["none", "quadratic_ridge", "extra_trees"], default="none")
    parser.add_argument("--uq-surrogate-sobol-train-samples", type=int, default=None)
    parser.add_argument("--uq-surrogate-sobol-validation-samples", type=int, default=None)
    parser.add_argument("--uq-surrogate-ridge-alpha", type=float, default=1e-6)
    parser.add_argument("--uq-surrogate-max-rmse-c", type=float, default=0.75)
    parser.add_argument("--uq-surrogate-max-abs-error-c", type=float, default=2.0)
    parser.add_argument("--uq-surrogate-min-prediction-samples", type=int, default=32)
    parser.add_argument("--uq-surrogate-extra-trees-estimators", type=int, default=600)
    parser.add_argument("--progress", action="store_true", default=None)
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    args = parser.parse_args()

    preset = fidelity_preset(args.preset)
    tire_parameters = default_tire_parameters(
        radial_cells=preset.radial_cells if args.radial_cells is None else args.radial_cells,
        theta_cells=preset.theta_cells if args.theta_cells is None else args.theta_cells,
        internal_solver_dt_s=preset.internal_solver_dt_s if args.internal_dt_s is None else args.internal_dt_s,
    )
    surrogate_config = UQSurrogateConfig(
        enabled=args.uq_surrogate != "none",
        kind=args.uq_surrogate,
        sobol_train_samples=args.uq_surrogate_sobol_train_samples,
        sobol_validation_samples=args.uq_surrogate_sobol_validation_samples,
        ridge_alpha=float(args.uq_surrogate_ridge_alpha),
        max_rmse_c=float(args.uq_surrogate_max_rmse_c),
        max_abs_error_c=float(args.uq_surrogate_max_abs_error_c),
        min_prediction_samples=int(args.uq_surrogate_min_prediction_samples),
        extra_trees_estimators=int(args.uq_surrogate_extra_trees_estimators),
    )
    run_high_fidelity_no_data(
        preset=args.preset,
        output_path=args.output_json,
        summary_path=args.output_summary,
        lhs_samples=preset.lhs_samples if args.lhs_samples is None else args.lhs_samples,
        sobol_samples=preset.sobol_samples if args.sobol_samples is None else args.sobol_samples,
        seed=args.seed,
        dt_s=args.dt_s,
        duration_scale=args.duration_scale,
        diagnostics_stride=preset.diagnostics_stride if args.diagnostics_stride is None else args.diagnostics_stride,
        workers=max(int(args.workers), 1),
        progress=args.progress,
        tire_parameters=tire_parameters,
        uq_surrogate=surrogate_config,
    )


if __name__ == "__main__":
    main()
