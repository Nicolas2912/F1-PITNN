from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

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


def test_p8_end_to_end_harness_generates_results_and_report(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    run_module = _load_module(
        "run_high_fidelity_no_data",
        root / "scripts" / "run_high_fidelity_no_data.py",
    )
    report_module = _load_module(
        "report_high_fidelity_no_data",
        root / "scripts" / "report_high_fidelity_no_data.py",
    )

    results_path = tmp_path / "hf_results.json"
    summary_path = tmp_path / "hf_summary.md"
    rerendered_summary_path = tmp_path / "hf_summary_rerendered.md"
    tire_params = HighFidelityTireModelParameters(
        use_2d_thermal_solver=True,
        no_op_thermal_step=False,
        radial_cells=6,
        theta_cells=12,
        internal_solver_dt_s=0.02,
    )

    artifact = run_module.run_high_fidelity_no_data(
        output_path=results_path,
        summary_path=summary_path,
        lhs_samples=4,
        sobol_samples=16,
        seed=77,
        dt_s=0.2,
        duration_scale=0.2,
        tire_parameters=tire_params,
    )

    assert results_path.exists()
    assert summary_path.exists()

    payload = json.loads(results_path.read_text(encoding="utf-8"))
    assert payload["metadata"]["lhs_samples"] == 4
    assert payload["metadata"]["sobol_samples"] == 16
    assert set(payload["baseline"]["scenario_summaries"]) == {
        "steady_corner",
        "straight_braking",
        "straight_acceleration",
        "combined_brake_corner",
        "cooldown",
        "long_stint",
    }
    assert set(payload["metadata"]["uq_scenario_names"]) == {
        "steady_corner",
        "straight_braking",
        "straight_acceleration",
        "combined_brake_corner",
        "cooldown",
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
    assert "## Sobol Ranking" in summary_text
    assert "peak_mean_surface_temp_c" in summary_text
