from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest

from models.high_fidelity import HighFidelityTireModelParameters


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        msg = f"Unable to load module from {path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_modules():
    root = Path(__file__).resolve().parents[1]
    run_module = _load_module(
        "run_high_fidelity_no_data",
        root / "scripts" / "run_high_fidelity_no_data.py",
    )
    report_module = _load_module(
        "report_high_fidelity_no_data",
        root / "scripts" / "report_high_fidelity_no_data.py",
    )
    return root, run_module, report_module


def _normalized_artifact(payload: dict) -> dict:
    normalized = json.loads(json.dumps(payload))
    metadata = normalized.get("metadata", {})
    for key in ("created_at_utc", "results_path", "summary_path"):
        metadata.pop(key, None)
    normalized.pop("timing", None)
    return normalized


def _normalized_summary(text: str) -> str:
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


class _RecordingProgressTracker:
    def __init__(self) -> None:
        self.phases: list[str] = []
        self.advances: list[int] = []

    def set_phase(self, label: str) -> None:
        self.phases.append(label)

    def advance(self, amount: int = 1) -> None:
        self.advances.append(amount)

    def close(self) -> None:
        return None

    def timings(self) -> dict:
        return {"total_elapsed_s": 0.0, "phases": {}}


def test_p8_end_to_end_harness_generates_results_and_report_smoke(tmp_path: Path) -> None:
    _root, run_module, report_module = _load_modules()
    default_params = run_module.default_tire_parameters(radial_cells=4, theta_cells=8, internal_solver_dt_s=0.05)
    assert default_params.use_local_temp_friction_partition
    assert default_params.use_reduced_patch_mechanics
    assert default_params.use_structural_hysteresis_model
    assert default_params.internal_coupling.enabled
    assert default_params.local_contact.enabled
    assert default_params.construction.enabled
    results_path = tmp_path / "hf_results.json"
    summary_path = tmp_path / "hf_summary.md"
    rerendered_summary_path = tmp_path / "hf_summary_rerendered.md"

    artifact = run_module.run_high_fidelity_no_data(
        preset="smoke",
        output_path=results_path,
        summary_path=summary_path,
        seed=77,
        dt_s=0.2,
        duration_scale=0.15,
    )

    assert results_path.exists()
    assert summary_path.exists()

    payload = json.loads(results_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["preset"] == "smoke"
    assert payload["metadata"]["lhs_samples"] == 2
    assert payload["metadata"]["sobol_samples"] == 2
    assert payload["metadata"]["diagnostics_stride"] == 1
    assert payload["metadata"]["radial_cells"] == 4
    assert payload["metadata"]["theta_cells"] == 8
    assert payload["timing"]["total_elapsed_s"] >= 0.0
    assert set(payload["timing"]["phases"]) >= {"baseline", "lhs", "sobol", "done"}
    assert "scenario_envelopes" in payload["uq"]["lhs"]
    assert len(payload["uq"]["sobol"]["indices"]) > 0
    assert "all_outputs_finite" in payload["plausibility_checks"]
    assert artifact["metadata"]["seed"] == 77

    summary_text = report_module.write_high_fidelity_no_data_report(
        results_path=results_path,
        output_path=rerendered_summary_path,
    )
    assert rerendered_summary_path.exists()
    assert summary_text.startswith("# High-Fidelity No-Data Summary")
    assert "## UQ-First Scenario Metrics" in summary_text
    assert "## Sobol Ranking" in summary_text
    assert "## Runtime Breakdown" in summary_text


def test_p8_end_to_end_parallel_workers_match_serial_outputs(tmp_path: Path) -> None:
    _root, run_module, report_module = _load_modules()
    serial_results_path = tmp_path / "hf_serial.json"
    serial_summary_path = tmp_path / "hf_serial.md"
    parallel_results_path = tmp_path / "hf_parallel.json"
    parallel_summary_path = tmp_path / "hf_parallel.md"

    run_module.run_high_fidelity_no_data(
        preset="smoke",
        output_path=serial_results_path,
        summary_path=serial_summary_path,
        seed=77,
        dt_s=0.2,
        duration_scale=0.05,
        lhs_samples=2,
        sobol_samples=2,
        workers=1,
    )
    run_module.run_high_fidelity_no_data(
        preset="smoke",
        output_path=parallel_results_path,
        summary_path=parallel_summary_path,
        seed=77,
        dt_s=0.2,
        duration_scale=0.05,
        lhs_samples=2,
        sobol_samples=2,
        workers=2,
    )

    serial_payload = json.loads(serial_results_path.read_text(encoding="utf-8"))
    parallel_payload = json.loads(parallel_results_path.read_text(encoding="utf-8"))
    assert _normalized_artifact(serial_payload) == _normalized_artifact(parallel_payload)
    assert _normalized_summary(serial_summary_path.read_text(encoding="utf-8")) == _normalized_summary(
        parallel_summary_path.read_text(encoding="utf-8")
    )

    serial_rerendered = report_module.write_high_fidelity_no_data_report(
        results_path=serial_results_path,
        output_path=tmp_path / "hf_serial_rerendered.md",
    )
    parallel_rerendered = report_module.write_high_fidelity_no_data_report(
        results_path=parallel_results_path,
        output_path=tmp_path / "hf_parallel_rerendered.md",
    )
    assert _normalized_summary(serial_rerendered) == _normalized_summary(parallel_rerendered)


def test_p8_sobol_surrogate_keeps_baseline_artifact_identical(tmp_path: Path) -> None:
    _root, run_module, _report_module = _load_modules()
    exact_results_path = tmp_path / "hf_exact.json"
    exact_summary_path = tmp_path / "hf_exact.md"
    surrogate_results_path = tmp_path / "hf_surrogate.json"
    surrogate_summary_path = tmp_path / "hf_surrogate.md"

    run_module.run_high_fidelity_no_data(
        preset="smoke",
        output_path=exact_results_path,
        summary_path=exact_summary_path,
        seed=77,
        dt_s=0.2,
        duration_scale=0.05,
        lhs_samples=2,
        sobol_samples=2,
        workers=1,
        progress=False,
    )
    run_module.run_high_fidelity_no_data(
        preset="smoke",
        output_path=surrogate_results_path,
        summary_path=surrogate_summary_path,
        seed=77,
        dt_s=0.2,
        duration_scale=0.05,
        lhs_samples=2,
        sobol_samples=2,
        workers=1,
        progress=False,
        uq_surrogate=run_module.UQSurrogateConfig(enabled=True, sobol_train_samples=2, sobol_validation_samples=0),
    )

    exact_payload = json.loads(exact_results_path.read_text(encoding="utf-8"))
    surrogate_payload = json.loads(surrogate_results_path.read_text(encoding="utf-8"))
    assert exact_payload["baseline"] == surrogate_payload["baseline"]
    assert surrogate_payload["metadata"]["uq_surrogate"]["enabled"] is True


def test_p8_sobol_surrogate_reduces_exact_eval_count(monkeypatch: pytest.MonkeyPatch) -> None:
    _root, run_module, _report_module = _load_modules()
    scenarios = tuple(s for s in run_module.default_scenarios(duration_scale=0.05) if s.include_in_uq)
    tire_parameters = run_module.default_tire_parameters(radial_cells=4, theta_cells=8, internal_solver_dt_s=0.05)
    exact_eval_counts: list[int] = []
    original = run_module._evaluate_sobol_payloads

    def counted(*args, **kwargs):
        exact_eval_counts.append(len(kwargs["eval_payloads"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(run_module, "_evaluate_sobol_payloads", counted)

    result = run_module.run_sobol_uq(
        scenario=next(s for s in scenarios if s.name == "combined_brake_corner"),
        tire_parameters=tire_parameters,
        vehicle_parameters=run_module.VehicleParameters(),
        dt_s=0.2,
        uq=run_module.HighFidelityUQ(),
        sobol_samples=3,
        seed=77,
        diagnostics_stride=1,
        workers=1,
        progress_tracker=None,
        surrogate_config=run_module.UQSurrogateConfig(
            enabled=True,
            sobol_train_samples=6,
            sobol_validation_samples=3,
            min_prediction_samples=1,
            max_rmse_c=1e9,
            max_abs_error_c=1e9,
        ),
    )

    assert exact_eval_counts == [9]
    assert result["objective_metric"].endswith(".peak_mean_surface_temp_c")


def test_p8_sobol_surrogate_falls_back_when_validation_error_exceeds_thresholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _root, run_module, _report_module = _load_modules()
    scenarios = tuple(s for s in run_module.default_scenarios(duration_scale=0.05) if s.include_in_uq)
    tire_parameters = run_module.default_tire_parameters(radial_cells=4, theta_cells=8, internal_solver_dt_s=0.05)
    baseline_result = run_module.run_sobol_uq(
        scenario=next(s for s in scenarios if s.name == "combined_brake_corner"),
        tire_parameters=tire_parameters,
        vehicle_parameters=run_module.VehicleParameters(),
        dt_s=0.2,
        uq=run_module.HighFidelityUQ(),
        sobol_samples=3,
        seed=77,
        diagnostics_stride=1,
        workers=1,
        progress_tracker=None,
        surrogate_config=run_module.UQSurrogateConfig(enabled=False),
    )
    eval_counts: list[int] = []
    original = run_module._evaluate_sobol_payloads

    def counted(*args, **kwargs):
        eval_counts.append(len(kwargs["eval_payloads"]))
        return original(*args, **kwargs)

    class _BadSurrogate:
        def predict(self, x):
            return 500.0 * __import__("numpy").ones((x.shape[0], 4), dtype=float)

    monkeypatch.setattr(run_module, "_evaluate_sobol_payloads", counted)
    monkeypatch.setattr(run_module, "_fit_multi_output_surrogate", lambda **kwargs: _BadSurrogate())

    fallback_result = run_module.run_sobol_uq(
        scenario=next(s for s in scenarios if s.name == "combined_brake_corner"),
        tire_parameters=tire_parameters,
        vehicle_parameters=run_module.VehicleParameters(),
        dt_s=0.2,
        uq=run_module.HighFidelityUQ(),
        sobol_samples=3,
        seed=77,
        diagnostics_stride=1,
        workers=1,
        progress_tracker=None,
        surrogate_config=run_module.UQSurrogateConfig(
            enabled=True,
            sobol_train_samples=6,
            sobol_validation_samples=3,
            min_prediction_samples=1,
            max_rmse_c=0.01,
            max_abs_error_c=0.01,
        ),
    )

    assert eval_counts == [9, 66]
    assert fallback_result == baseline_result


def test_p8_lhs_progress_advances_per_sample_in_serial_mode() -> None:
    _root, run_module, _report_module = _load_modules()
    progress_tracker = _RecordingProgressTracker()
    scenarios = tuple(s for s in run_module.default_scenarios(duration_scale=0.05) if s.include_in_uq)
    tire_parameters = run_module.default_tire_parameters(radial_cells=4, theta_cells=8, internal_solver_dt_s=0.05)
    uq = run_module.HighFidelityUQ()

    result = run_module.run_lhs_uq(
        scenarios=scenarios,
        tire_parameters=tire_parameters,
        vehicle_parameters=run_module.VehicleParameters(),
        dt_s=0.2,
        uq=uq,
        lhs_samples=3,
        seed=77,
        diagnostics_stride=1,
        workers=1,
        progress_tracker=progress_tracker,
    )

    assert progress_tracker.phases == ["lhs"]
    assert progress_tracker.advances == [5, 5, 5]
    assert len(result["scenario_envelopes"]) == len(scenarios)


def test_p8_lhs_parallel_progress_advances_per_sample_before_batch_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    _root, run_module, _report_module = _load_modules()
    progress_tracker = _RecordingProgressTracker()
    scenarios = tuple(s for s in run_module.default_scenarios(duration_scale=0.05) if s.include_in_uq)
    tire_parameters = run_module.default_tire_parameters(radial_cells=4, theta_cells=8, internal_solver_dt_s=0.05)
    observed_advances_during_iteration: list[int] = []

    def fake_iter_parallel_map(*, worker_fn, tasks, workers, pool_runner):
        del workers, pool_runner
        assert [len(task["batch"]) for task in tasks] == [2, 2]
        for task in tasks:
            batch_result = worker_fn(task)
            observed_advances_during_iteration.append(sum(progress_tracker.advances))
            yield batch_result

    monkeypatch.setattr(run_module, "_iter_parallel_map", fake_iter_parallel_map)

    result = run_module.run_lhs_uq(
        scenarios=scenarios,
        tire_parameters=tire_parameters,
        vehicle_parameters=run_module.VehicleParameters(),
        dt_s=0.2,
        uq=run_module.HighFidelityUQ(),
        lhs_samples=4,
        seed=77,
        diagnostics_stride=1,
        workers=2,
        progress_tracker=progress_tracker,
        pool_runner=object(),
    )

    assert progress_tracker.phases == ["lhs"]
    assert sum(progress_tracker.advances) == 20
    assert observed_advances_during_iteration[0] > 0
    assert len(result["scenario_envelopes"]) == len(scenarios)


def test_p8_lhs_parallel_progress_restarts_reused_pool_before_worker_queue_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _root, run_module, _report_module = _load_modules()
    scenarios = tuple(s for s in run_module.default_scenarios(duration_scale=0.05) if s.include_in_uq)
    tire_parameters = run_module.default_tire_parameters(radial_cells=4, theta_cells=8, internal_solver_dt_s=0.05)
    restart_calls: list[str] = []

    class FakeQueue:
        def put(self, _value: int) -> None:
            return None

        def close(self) -> None:
            return None

        def join_thread(self) -> None:
            return None

    class FakeDrainer:
        def stop(self) -> None:
            return None

    class FakePoolRunner:
        def restart(self) -> None:
            restart_calls.append("restart")

    def fake_parallel_progress_context(*, progress_tracker, pool_runner):
        del progress_tracker, pool_runner
        return FakeQueue(), FakeDrainer()

    def fake_iter_parallel_map(*, worker_fn, tasks, workers, pool_runner):
        del workers, pool_runner
        for task in tasks:
            yield worker_fn(task)

    monkeypatch.setattr(run_module, "_parallel_progress_context", fake_parallel_progress_context)
    monkeypatch.setattr(run_module, "_iter_parallel_map", fake_iter_parallel_map)

    run_module.run_lhs_uq(
        scenarios=scenarios,
        tire_parameters=tire_parameters,
        vehicle_parameters=run_module.VehicleParameters(),
        dt_s=0.2,
        uq=run_module.HighFidelityUQ(),
        lhs_samples=4,
        seed=77,
        diagnostics_stride=1,
        workers=2,
        progress_tracker=_RecordingProgressTracker(),
        pool_runner=FakePoolRunner(),
    )

    assert restart_calls == ["restart"]


def test_p8_sobol_falls_back_from_degenerate_surface_metric(monkeypatch: pytest.MonkeyPatch) -> None:
    _root, run_module, _report_module = _load_modules()

    def fake_evaluate_sobol_batch(task: dict) -> list[tuple[int, dict[str, float]]]:
        outputs: list[tuple[int, dict[str, float]]] = []
        for eval_idx, _sample in task["batch"]:
            outputs.append((
                eval_idx,
                {
                    "peak_mean_surface_temp_c": 1926.85,
                    "peak_mean_core_temp_c": float(eval_idx),
                    "end_mean_surface_temp_c": 1926.85,
                    "end_mean_core_temp_c": float(eval_idx) * 0.5,
                },
            ))
        return outputs

    monkeypatch.setattr(run_module, "_evaluate_sobol_batch", fake_evaluate_sobol_batch)

    result = run_module.run_sobol_uq(
        scenario=run_module.default_scenarios(duration_scale=0.05)[0],
        tire_parameters=run_module.default_tire_parameters(radial_cells=4, theta_cells=8, internal_solver_dt_s=0.05),
        vehicle_parameters=run_module.VehicleParameters(),
        dt_s=0.2,
        uq=run_module.HighFidelityUQ(),
        sobol_samples=2,
        seed=77,
        diagnostics_stride=1,
        workers=1,
    )

    assert result["objective_metric"].endswith(".peak_mean_core_temp_c")
    assert result["variance"] > 0.0
    assert any(index["total_order"] > 0.0 for index in result["indices"])


def test_p8_baseline_parallel_progress_updates_before_all_results_are_sorted(monkeypatch: pytest.MonkeyPatch) -> None:
    _root, run_module, _report_module = _load_modules()
    scenarios = run_module.default_scenarios(duration_scale=0.05)[:3]
    tire_parameters = run_module.default_tire_parameters(radial_cells=4, theta_cells=8, internal_solver_dt_s=0.05)
    progress_tracker = _RecordingProgressTracker()
    completion_order: list[str] = []

    def fake_iter_process_pool_map(worker_fn, tasks, workers):
        del worker_fn, workers
        reversed_tasks = list(reversed(tasks))
        for task in reversed_tasks:
            scenario_name = task["scenario"].name
            completion_order.append(scenario_name)
            yield (
                scenario_name,
                {
                    "trace": {"time_s": [0.0], "mean_core_temp_c": [0.0], "mean_surface_temp_c": [0.0], "load_error_pct": [0.0], "max_energy_residual_pct": [0.0], "coupling_converged_fraction": [1.0]},
                    "summary": {
                        "end_mean_core_temp_c": float(len(completion_order)),
                        "peak_mean_core_temp_c": float(len(completion_order)),
                        "end_mean_surface_temp_c": float(len(completion_order)),
                        "peak_mean_surface_temp_c": float(len(completion_order)),
                        "max_load_error_pct": 0.0,
                        "max_energy_residual_pct": 0.0,
                        "coupling_convergence_rate": 1.0,
                        "all_outputs_finite": True,
                    },
                },
            )
            assert len(progress_tracker.advances) == len(completion_order)

    monkeypatch.setattr(run_module, "_iter_process_pool_map", fake_iter_process_pool_map)

    result = run_module.run_scenario_pack(
        scenarios=scenarios,
        tire_parameters=tire_parameters,
        vehicle_parameters=run_module.VehicleParameters(),
        dt_s=0.2,
        diagnostics_stride=1,
        workers=2,
        progress_tracker=progress_tracker,
    )

    assert progress_tracker.phases == ["baseline"]
    assert progress_tracker.advances == [1, 1, 1]
    assert completion_order == [scenario.name for scenario in reversed(scenarios)]
    assert list(result["scenario_summaries"]) == [scenario.name for scenario in scenarios]


def test_p8_end_to_end_parallel_reuses_single_process_pool(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _root, run_module, _report_module = _load_modules()
    runner_count = 0
    iter_calls: list[tuple[str, int]] = []

    class FakeProcessPoolRunner:
        def __init__(self, workers: int) -> None:
            nonlocal runner_count
            runner_count += 1
            self.workers = workers
            self.closed = False

        def iter_map_unordered(self, worker_fn, tasks):
            iter_calls.append((worker_fn.__name__, len(tasks)))
            for task in tasks:
                yield worker_fn(task)

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(run_module, "ProcessPoolRunner", FakeProcessPoolRunner)

    run_module.run_high_fidelity_no_data(
        preset="smoke",
        output_path=tmp_path / "results.json",
        summary_path=tmp_path / "summary.md",
        seed=77,
        dt_s=0.2,
        duration_scale=0.05,
        lhs_samples=2,
        sobol_samples=2,
        workers=2,
        progress=False,
    )

    assert runner_count == 1
    assert [name for name, _count in iter_calls] == [
        "_run_scenario_task",
        "_evaluate_lhs_batch",
        "_evaluate_sobol_batch",
    ]


def test_p8_parallel_uq_batches_reduce_task_count(monkeypatch: pytest.MonkeyPatch) -> None:
    _root, run_module, _report_module = _load_modules()
    scenarios = tuple(s for s in run_module.default_scenarios(duration_scale=0.05) if s.include_in_uq)
    tire_parameters = run_module.default_tire_parameters(radial_cells=4, theta_cells=8, internal_solver_dt_s=0.05)
    lhs_task_batches: list[int] = []
    sobol_task_batches: list[int] = []
    progress_tracker = _RecordingProgressTracker()

    def fake_iter_parallel_map(*, worker_fn, tasks, workers, pool_runner):
        del workers, pool_runner
        batch_sizes = [len(task["batch"]) for task in tasks]
        if worker_fn.__name__ == "_evaluate_lhs_batch":
            lhs_task_batches.extend(batch_sizes)
            for task in tasks:
                yield worker_fn(task)
            return
        if worker_fn.__name__ == "_evaluate_sobol_batch":
            sobol_task_batches.extend(batch_sizes)
            for task in tasks:
                yield worker_fn(task)
            return
        for task in tasks:
            yield worker_fn(task)

    monkeypatch.setattr(run_module, "_iter_parallel_map", fake_iter_parallel_map)

    run_module.run_lhs_uq(
        scenarios=scenarios,
        tire_parameters=tire_parameters,
        vehicle_parameters=run_module.VehicleParameters(),
        dt_s=0.2,
        uq=run_module.HighFidelityUQ(),
        lhs_samples=7,
        seed=77,
        diagnostics_stride=1,
        workers=3,
        progress_tracker=progress_tracker,
        pool_runner=object(),
    )
    assert lhs_task_batches == [3, 3, 1]

    progress_tracker = _RecordingProgressTracker()
    run_module.run_sobol_uq(
        scenario=next(s for s in scenarios if s.name == "combined_brake_corner"),
        tire_parameters=tire_parameters,
        vehicle_parameters=run_module.VehicleParameters(),
        dt_s=0.2,
        uq=run_module.HighFidelityUQ(),
        sobol_samples=3,
        seed=77,
        diagnostics_stride=1,
        workers=4,
        progress_tracker=progress_tracker,
        pool_runner=object(),
    )
    assert sobol_task_batches == [19, 19, 19, 18]


@pytest.mark.slow
def test_p8_end_to_end_harness_generates_results_and_report_fullish(tmp_path: Path) -> None:
    root, run_module, report_module = _load_modules()
    run_module.default_scenarios = lambda duration_scale=1.0: (
        run_module.ScenarioConfig(
            name="steady_corner",
            duration_s=0.6 * duration_scale,
            speed_mps=48.0,
            ax_mps2=0.0,
            ay_mps2=4.0,
            steering_angle_rad=0.04,
            yaw_rate_radps=0.08,
            brake_power_w=0.0,
            drive_power_w=14_000.0,
            ambient_temp_k=303.15,
            track_temp_k=316.15,
        ),
        run_module.ScenarioConfig(
            name="combined_brake_corner",
            duration_s=0.7 * duration_scale,
            speed_mps=45.0,
            ax_mps2=-3.2,
            ay_mps2=3.8,
            steering_angle_rad=0.05,
            yaw_rate_radps=0.09,
            brake_power_w=28_000.0,
            drive_power_w=0.0,
            ambient_temp_k=303.15,
            track_temp_k=316.15,
        ),
        run_module.ScenarioConfig(
            name="long_stint",
            duration_s=0.8 * duration_scale,
            speed_mps=42.0,
            ax_mps2=0.1,
            ay_mps2=2.0,
            steering_angle_rad=0.02,
            yaw_rate_radps=0.03,
            brake_power_w=2_500.0,
            drive_power_w=9_000.0,
            ambient_temp_k=304.15,
            track_temp_k=317.15,
            include_in_uq=False,
        ),
    )

    results_path = tmp_path / "hf_results.json"
    summary_path = tmp_path / "hf_summary.md"
    rerendered_summary_path = tmp_path / "hf_summary_rerendered.md"
    tire_params = HighFidelityTireModelParameters(
        use_2d_thermal_solver=True,
        no_op_thermal_step=False,
        radial_cells=4,
        theta_cells=8,
        internal_solver_dt_s=0.05,
    )

    artifact = run_module.run_high_fidelity_no_data(
        output_path=results_path,
        summary_path=summary_path,
        lhs_samples=2,
        sobol_samples=2,
        seed=77,
        dt_s=0.2,
        duration_scale=1.0,
        tire_parameters=tire_params,
    )

    assert results_path.exists()
    assert summary_path.exists()

    payload = json.loads(results_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["lhs_samples"] == 2
    assert payload["metadata"]["sobol_samples"] == 2
    assert payload["metadata"]["default_output_mode"] == "bands_plus_baseline"
    assert set(payload["baseline"]["scenario_summaries"]) == {
        "steady_corner",
        "combined_brake_corner",
        "long_stint",
    }
    assert set(payload["metadata"]["uq_scenario_names"]) == {
        "steady_corner",
        "combined_brake_corner",
    }
    assert "scenario_envelopes" in payload["uq"]["lhs"]
    assert "mean_surface_temp_c_trace" in payload["uq"]["lhs"]["scenario_envelopes"]["steady_corner"]
    assert len(payload["uq"]["sobol"]["indices"]) > 0
    assert "all_peak_core_finite" in payload["plausibility_checks"]
    assert "all_outputs_finite" in payload["plausibility_checks"]
    assert set(payload["timing"]["phases"]) >= {"baseline", "lhs", "sobol", "done"}
    assert "peak_mean_surface_temp_c" in payload["baseline"]["scenario_summaries"]["steady_corner"]
    assert (
        payload["baseline"]["scenario_summaries"]["long_stint"]["end_mean_core_temp_c"]
        > payload["baseline"]["scenario_summaries"]["long_stint"]["peak_mean_core_temp_c"] - 1e-9
    )
    assert (
        payload["baseline"]["scenario_summaries"]["long_stint"]["end_mean_core_temp_c"]
        > 30.0
    )
    assert artifact["metadata"]["seed"] == 77

    summary_text = report_module.write_high_fidelity_no_data_report(
        results_path=results_path,
        output_path=rerendered_summary_path,
    )
    assert rerendered_summary_path.exists()
    assert summary_text.startswith("# High-Fidelity No-Data Summary")
    assert "## UQ-First Scenario Metrics" in summary_text
    assert "## Sobol Ranking" in summary_text
    assert "## Runtime Breakdown" in summary_text
    assert "peak_mean_surface_temp_c" in summary_text
