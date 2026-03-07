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
    return normalized


def _normalized_summary(text: str) -> str:
    return "\n".join(
        line
        for line in text.splitlines()
        if not line.startswith("- created_at_utc:")
        and not line.startswith("- results_json:")
        and not line.startswith("- summary_md:")
    )


class _RecordingProgressTracker:
    def __init__(self) -> None:
        self.phases: list[str] = []
        self.advances: list[int] = []

    def set_phase(self, label: str) -> None:
        self.phases.append(label)

    def advance(self, amount: int = 1) -> None:
        self.advances.append(amount)


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
    assert progress_tracker.advances == [1, 1, 1]
    assert len(result["scenario_envelopes"]) == len(scenarios)


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
    assert "peak_mean_surface_temp_c" in summary_text
