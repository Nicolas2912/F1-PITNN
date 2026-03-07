from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from models.high_fidelity import HighFidelityTireModelParameters, ThermalFieldSolver2D
from models.high_fidelity.native_diffusion import (
    native_diffusion_available,
    run_native_build_source_and_diffuse_implicit,
    run_native_build_source_field,
    run_native_diffuse_vectorized_implicit,
)
from models.high_fidelity.native_simulator_kernels import native_simulator_kernels_available


FIXTURE = Path(__file__).resolve().parent / "fixtures" / "hf_diffusion_reference.npz"


def _load_run_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "run_high_fidelity_no_data.py"
    spec = importlib.util.spec_from_file_location("run_high_fidelity_no_data_native_test", module_path)
    if spec is None or spec.loader is None:
        msg = f"Unable to load module from {module_path}"
        raise RuntimeError(msg)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _normalized_artifact(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    for key in ("created_at_utc", "results_path", "summary_path"):
        metadata.pop(key, None)
    return payload


def _normalized_summary(path: Path) -> str:
    return "\n".join(
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if not line.startswith("- created_at_utc:")
        and not line.startswith("- results_json:")
        and not line.startswith("- summary_md:")
    )


def _layer_source_weights_array() -> np.ndarray:
    return np.asarray([12_000.0, 8_500.0, 7_200.0, 5_600.0, 4_300.0], dtype=float)


def test_diffusion_kernel_python_matches_stored_reference() -> None:
    payload = np.load(FIXTURE)
    params = HighFidelityTireModelParameters(
        radial_cells=int(payload["radial_cells"]),
        theta_cells=int(payload["theta_cells"]),
        width_zones=int(payload["width_zones"]),
        diffusion_max_iterations=int(payload["diffusion_max_iterations"]),
        diffusion_tolerance_k=float(payload["diffusion_tolerance_k"]),
    )
    solver = ThermalFieldSolver2D(params)

    actual_field, actual_iterations = solver._diffuse_vectorized_implicit_python(
        np.array(payload["field"], dtype=float, copy=True),
        source_w_per_m3=np.array(payload["source_w_per_m3"], dtype=float, copy=True),
        rho_cp=np.array(payload["rho_cp"], dtype=float, copy=True),
        k_r=np.array(payload["k_r"], dtype=float, copy=True),
        k_theta=np.array(payload["k_theta"], dtype=float, copy=True),
        k_w=np.array(payload["k_w"], dtype=float, copy=True),
        dt_s=float(payload["dt_s"]),
    )

    assert actual_iterations == int(payload["iterations"])
    assert np.array_equal(actual_field, payload["expected_field"])


def test_diffusion_kernel_native_matches_python_reference_when_available() -> None:
    if not native_diffusion_available():
        return

    payload = np.load(FIXTURE)
    params = HighFidelityTireModelParameters(
        radial_cells=int(payload["radial_cells"]),
        theta_cells=int(payload["theta_cells"]),
        width_zones=int(payload["width_zones"]),
        diffusion_max_iterations=int(payload["diffusion_max_iterations"]),
        diffusion_tolerance_k=float(payload["diffusion_tolerance_k"]),
    )
    solver = ThermalFieldSolver2D(params)

    python_field, python_iterations = solver._diffuse_vectorized_implicit_python(
        np.array(payload["field"], dtype=float, copy=True),
        source_w_per_m3=np.array(payload["source_w_per_m3"], dtype=float, copy=True),
        rho_cp=np.array(payload["rho_cp"], dtype=float, copy=True),
        k_r=np.array(payload["k_r"], dtype=float, copy=True),
        k_theta=np.array(payload["k_theta"], dtype=float, copy=True),
        k_w=np.array(payload["k_w"], dtype=float, copy=True),
        dt_s=float(payload["dt_s"]),
    )
    native_field, native_iterations = run_native_diffuse_vectorized_implicit(
        field=np.array(payload["field"], dtype=float, copy=True),
        source_w_per_m3=np.array(payload["source_w_per_m3"], dtype=float, copy=True),
        rho_cp=np.array(payload["rho_cp"], dtype=float, copy=True),
        k_r=np.array(payload["k_r"], dtype=float, copy=True),
        k_theta=np.array(payload["k_theta"], dtype=float, copy=True),
        k_w=np.array(payload["k_w"], dtype=float, copy=True),
        dt_s=float(payload["dt_s"]),
        radial_coeff_minus=np.array(solver._radial_coeff_minus, dtype=float, copy=True),
        radial_coeff_plus=np.array(solver._radial_coeff_plus, dtype=float, copy=True),
        theta_coeff=np.array(solver._theta_coeff, dtype=float, copy=True),
        width_coeff_minus=np.array(solver._width_coeff_minus, dtype=float, copy=True),
        width_coeff_plus=np.array(solver._width_coeff_plus, dtype=float, copy=True),
        diffusion_max_iterations=int(solver.parameters.diffusion_max_iterations),
        diffusion_tolerance_k=float(solver.parameters.diffusion_tolerance_k),
    )

    max_delta = float(np.max(np.abs(native_field - python_field)))
    assert native_iterations == python_iterations
    assert np.array_equal(native_field, python_field), f"max delta {max_delta:.3e}"


def test_native_source_field_matches_python_reference_when_available() -> None:
    if not native_diffusion_available():
        return

    params = HighFidelityTireModelParameters(
        radial_cells=8,
        theta_cells=16,
        width_zones=3,
        source_patch_theta_fraction=0.20,
        source_patch_radial_fraction=0.30,
    )
    solver = ThermalFieldSolver2D(params)
    _, _, _, _, layer_index = solver.layer_property_maps(
        wear=0.08,
        grain_index_w=np.array([0.05, 0.02, 0.04], dtype=float),
        blister_index_w=np.array([0.01, 0.03, 0.02], dtype=float),
        age_index=0.15,
    )
    zone_weights = np.array([0.22, 0.51, 0.27], dtype=float)
    layer_source_weights = {
        "tread": 12_000.0,
        "belt": 8_500.0,
        "carcass": 7_200.0,
        "sidewall": 5_600.0,
        "inner_liner": 4_300.0,
    }

    python_source = solver.source_field_w_per_m3(
        volumetric_source_w_per_m3=18_500.0,
        wheel_angular_speed_radps=187.0,
        time_s=0.37,
        zone_weights=zone_weights,
        layer_source_weights=layer_source_weights,
    )
    native_source = run_native_build_source_field(
        radial_cells=params.radial_cells,
        theta_cells=params.theta_cells,
        width_zones=params.width_zones,
        source_volumetric_fraction=params.source_volumetric_fraction,
        volumetric_source_w_per_m3=18_500.0,
        wheel_angular_speed_radps=187.0,
        time_s=0.37,
        theta_delta_rad=solver._theta_delta_rad,
        patch_radial_indices=solver._patch_radial_indices,
        theta_offsets=solver._theta_offsets,
        width_indices=solver._width_indices,
        layer_index=layer_index,
        zone_weights=zone_weights,
        layer_source_weights=_layer_source_weights_array(),
    )

    assert np.array_equal(native_source, python_source)


def test_native_source_and_diffusion_fused_path_matches_python_reference_when_available() -> None:
    if not native_diffusion_available():
        return

    params = HighFidelityTireModelParameters(
        radial_cells=8,
        theta_cells=16,
        width_zones=3,
        diffusion_max_iterations=10,
        diffusion_tolerance_k=1e-6,
    )
    solver = ThermalFieldSolver2D(params)
    rng = np.random.default_rng(20260308)
    field = rng.normal(loc=343.15, scale=7.5, size=(params.radial_cells, params.theta_cells, params.width_zones)).astype(float)
    extra_source = rng.normal(loc=420.0, scale=35.0, size=field.shape).astype(float)
    rho_cp, k_r, k_theta, k_w, layer_index = solver.layer_property_maps(
        wear=0.08,
        grain_index_w=np.array([0.05, 0.02, 0.04], dtype=float),
        blister_index_w=np.array([0.01, 0.03, 0.02], dtype=float),
        age_index=0.15,
    )
    zone_weights = np.array([0.22, 0.51, 0.27], dtype=float)
    layer_source_weights = {
        "tread": 12_000.0,
        "belt": 8_500.0,
        "carcass": 7_200.0,
        "sidewall": 5_600.0,
        "inner_liner": 4_300.0,
    }

    python_source = solver.source_field_w_per_m3(
        volumetric_source_w_per_m3=18_500.0,
        wheel_angular_speed_radps=187.0,
        time_s=0.37,
        zone_weights=zone_weights,
        layer_source_weights=layer_source_weights,
    ) + extra_source
    python_field, python_iterations = solver._diffuse_vectorized_implicit_python(
        np.array(field, dtype=float, copy=True),
        source_w_per_m3=np.array(python_source, dtype=float, copy=True),
        rho_cp=np.array(rho_cp, dtype=float, copy=True),
        k_r=np.array(k_r, dtype=float, copy=True),
        k_theta=np.array(k_theta, dtype=float, copy=True),
        k_w=np.array(k_w, dtype=float, copy=True),
        dt_s=0.01,
    )
    native_field, native_iterations, native_source = run_native_build_source_and_diffuse_implicit(
        field=np.array(field, dtype=float, copy=True),
        extra_source_w_per_m3=np.array(extra_source, dtype=float, copy=True),
        rho_cp=np.array(rho_cp, dtype=float, copy=True),
        k_r=np.array(k_r, dtype=float, copy=True),
        k_theta=np.array(k_theta, dtype=float, copy=True),
        k_w=np.array(k_w, dtype=float, copy=True),
        dt_s=0.01,
        radial_coeff_minus=np.array(solver._radial_coeff_minus, dtype=float, copy=True),
        radial_coeff_plus=np.array(solver._radial_coeff_plus, dtype=float, copy=True),
        theta_coeff=np.array(solver._theta_coeff, dtype=float, copy=True),
        width_coeff_minus=np.array(solver._width_coeff_minus, dtype=float, copy=True),
        width_coeff_plus=np.array(solver._width_coeff_plus, dtype=float, copy=True),
        diffusion_max_iterations=int(params.diffusion_max_iterations),
        diffusion_tolerance_k=float(params.diffusion_tolerance_k),
        source_volumetric_fraction=float(params.source_volumetric_fraction),
        volumetric_source_w_per_m3=18_500.0,
        wheel_angular_speed_radps=187.0,
        time_s=0.37,
        theta_delta_rad=solver._theta_delta_rad,
        patch_radial_indices=solver._patch_radial_indices,
        theta_offsets=solver._theta_offsets,
        width_indices=solver._width_indices,
        layer_index=layer_index,
        zone_weights=zone_weights,
        layer_source_weights=_layer_source_weights_array(),
    )

    assert native_iterations == python_iterations
    assert np.array_equal(native_source, python_source)
    assert np.array_equal(native_field, python_field)


def test_native_diffusion_dispatch_matches_python_for_single_scenario(monkeypatch) -> None:
    if not native_diffusion_available():
        return

    run_module = _load_run_module()
    scenario = run_module.ScenarioConfig(
        name="native_probe",
        duration_s=0.6,
        speed_mps=48.0,
        ax_mps2=-1.2,
        ay_mps2=4.0,
        steering_angle_rad=0.04,
        yaw_rate_radps=0.08,
        brake_power_w=12_000.0,
        drive_power_w=8_000.0,
        ambient_temp_k=303.15,
        track_temp_k=316.15,
    )
    tire_params = run_module.default_tire_parameters(radial_cells=4, theta_cells=8, internal_solver_dt_s=0.05)
    vehicle_params = run_module.VehicleParameters()

    monkeypatch.delenv("PITNN_USE_NATIVE_DIFFUSION", raising=False)
    python_result = run_module.run_single_scenario(
        scenario=scenario,
        tire_parameters=tire_params,
        vehicle_parameters=vehicle_params,
        dt_s=0.2,
        diagnostics_stride=1,
    )

    monkeypatch.setenv("PITNN_USE_NATIVE_DIFFUSION", "1")
    native_result = run_module.run_single_scenario(
        scenario=scenario,
        tire_parameters=tire_params,
        vehicle_parameters=vehicle_params,
        dt_s=0.2,
        diagnostics_stride=1,
    )

    assert python_result == native_result


def test_native_diffusion_dispatch_matches_python_for_reduced_harness(tmp_path: Path, monkeypatch) -> None:
    if not native_diffusion_available():
        return

    run_module = _load_run_module()
    python_json = tmp_path / "python.json"
    python_md = tmp_path / "python.md"
    native_json = tmp_path / "native.json"
    native_md = tmp_path / "native.md"

    monkeypatch.delenv("PITNN_USE_NATIVE_DIFFUSION", raising=False)
    run_module.run_high_fidelity_no_data(
        preset="smoke",
        output_path=python_json,
        summary_path=python_md,
        duration_scale=0.05,
        lhs_samples=2,
        sobol_samples=2,
        workers=1,
        progress=False,
    )

    monkeypatch.setenv("PITNN_USE_NATIVE_DIFFUSION", "1")
    run_module.run_high_fidelity_no_data(
        preset="smoke",
        output_path=native_json,
        summary_path=native_md,
        duration_scale=0.05,
        lhs_samples=2,
        sobol_samples=2,
        workers=1,
        progress=False,
    )

    assert _normalized_artifact(python_json) == _normalized_artifact(native_json)
    assert _normalized_summary(python_md) == _normalized_summary(native_md)


@pytest.mark.parametrize("workers", [1, 2, 4, 8])
def test_combined_native_flags_match_python_baseline_for_smoke_harness(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    workers: int,
) -> None:
    if not native_diffusion_available() or not native_simulator_kernels_available():
        return

    run_module = _load_run_module()
    python_json = tmp_path / f"python_{workers}.json"
    python_md = tmp_path / f"python_{workers}.md"
    native_json = tmp_path / f"native_{workers}.json"
    native_md = tmp_path / f"native_{workers}.md"

    monkeypatch.delenv("PITNN_USE_NATIVE_DIFFUSION", raising=False)
    monkeypatch.delenv("PITNN_USE_NATIVE_SIMULATOR_KERNELS", raising=False)
    run_module.run_high_fidelity_no_data(
        preset="smoke",
        output_path=python_json,
        summary_path=python_md,
        duration_scale=0.05,
        lhs_samples=2,
        sobol_samples=2,
        workers=workers,
        progress=False,
    )

    monkeypatch.setenv("PITNN_USE_NATIVE_DIFFUSION", "1")
    monkeypatch.setenv("PITNN_USE_NATIVE_SIMULATOR_KERNELS", "1")
    run_module.run_high_fidelity_no_data(
        preset="smoke",
        output_path=native_json,
        summary_path=native_md,
        duration_scale=0.05,
        lhs_samples=2,
        sobol_samples=2,
        workers=workers,
        progress=False,
    )

    assert _normalized_artifact(python_json) == _normalized_artifact(native_json)
    assert _normalized_summary(python_md) == _normalized_summary(native_md)
