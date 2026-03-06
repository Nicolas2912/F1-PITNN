from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.high_fidelity import (  # noqa: E402
    HighFidelityTireModelParameters,
    HighFidelityUQ,
    HighFidelityVehicleSimulator,
)
from models.high_fidelity.reporting import render_high_fidelity_no_data_summary  # noqa: E402
from models.vehicle_thermal import VehicleInputs, VehicleParameters  # noqa: E402

OUTPUT_DIR = ROOT / "reports" / "results"
RESULTS_FILE = OUTPUT_DIR / "high_fidelity_no_data_results.json"
SUMMARY_FILE = OUTPUT_DIR / "high_fidelity_no_data_summary.md"


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
    include_in_uq: bool = True


def default_tire_parameters(
    *,
    radial_cells: int = 24,
    theta_cells: int = 72,
    internal_solver_dt_s: float = 0.01,
) -> HighFidelityTireModelParameters:
    return HighFidelityTireModelParameters(
        use_2d_thermal_solver=True,
        no_op_thermal_step=False,
        radial_cells=radial_cells,
        theta_cells=theta_cells,
        internal_solver_dt_s=internal_solver_dt_s,
    )


def default_scenarios(*, duration_scale: float = 1.0) -> tuple[ScenarioConfig, ...]:
    return (
        ScenarioConfig(
            name="steady_corner",
            duration_s=6.0 * duration_scale,
            speed_mps=56.0,
            ax_mps2=0.0,
            ay_mps2=6.5,
            steering_angle_rad=0.075,
            yaw_rate_radps=0.115,
            brake_power_w=0.0,
            drive_power_w=18_000.0,
            ambient_temp_k=303.15,
            track_temp_k=317.15,
        ),
        ScenarioConfig(
            name="straight_braking",
            duration_s=5.0 * duration_scale,
            speed_mps=64.0,
            ax_mps2=-6.8,
            ay_mps2=0.0,
            steering_angle_rad=0.0,
            yaw_rate_radps=0.0,
            brake_power_w=82_000.0,
            drive_power_w=0.0,
            ambient_temp_k=303.15,
            track_temp_k=317.15,
        ),
        ScenarioConfig(
            name="straight_acceleration",
            duration_s=5.0 * duration_scale,
            speed_mps=48.0,
            ax_mps2=4.5,
            ay_mps2=0.0,
            steering_angle_rad=0.0,
            yaw_rate_radps=0.0,
            brake_power_w=0.0,
            drive_power_w=72_000.0,
            ambient_temp_k=303.15,
            track_temp_k=317.15,
        ),
        ScenarioConfig(
            name="combined_brake_corner",
            duration_s=6.0 * duration_scale,
            speed_mps=54.0,
            ax_mps2=-4.8,
            ay_mps2=5.3,
            steering_angle_rad=0.070,
            yaw_rate_radps=0.102,
            brake_power_w=54_000.0,
            drive_power_w=0.0,
            ambient_temp_k=303.15,
            track_temp_k=317.15,
        ),
        ScenarioConfig(
            name="cooldown",
            duration_s=5.0 * duration_scale,
            speed_mps=32.0,
            ax_mps2=-1.2,
            ay_mps2=0.0,
            steering_angle_rad=0.0,
            yaw_rate_radps=0.0,
            brake_power_w=6_000.0,
            drive_power_w=0.0,
            ambient_temp_k=301.15,
            track_temp_k=311.15,
        ),
        ScenarioConfig(
            name="long_stint",
            duration_s=30.0 * duration_scale,
            speed_mps=52.0,
            ax_mps2=0.2,
            ay_mps2=4.2,
            steering_angle_rad=0.050,
            yaw_rate_radps=0.082,
            brake_power_w=4_000.0,
            drive_power_w=12_000.0,
            ambient_temp_k=304.15,
            track_temp_k=319.15,
            include_in_uq=False,
        ),
    )


def run_high_fidelity_no_data(
    *,
    output_path: Path = RESULTS_FILE,
    summary_path: Path = SUMMARY_FILE,
    lhs_samples: int = 400,
    sobol_samples: int = 2048,
    seed: int = 2026,
    dt_s: float = 0.2,
    duration_scale: float = 1.0,
    tire_parameters: HighFidelityTireModelParameters | None = None,
    vehicle_parameters: VehicleParameters | None = None,
) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    tire_params = tire_parameters if tire_parameters is not None else default_tire_parameters()
    vehicle_params = vehicle_parameters if vehicle_parameters is not None else VehicleParameters()
    scenarios = default_scenarios(duration_scale=duration_scale)
    baseline = run_scenario_pack(
        scenarios=scenarios,
        tire_parameters=tire_params,
        vehicle_parameters=vehicle_params,
        dt_s=dt_s,
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
    )
    sobol_result = run_sobol_uq(
        scenario=next(s for s in scenarios if s.name == "combined_brake_corner"),
        tire_parameters=tire_params,
        vehicle_parameters=vehicle_params,
        dt_s=dt_s,
        uq=uq,
        sobol_samples=sobol_samples,
        seed=seed,
    )

    artifact = {
        "metadata": {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "dt_s": float(dt_s),
            "duration_scale": float(duration_scale),
            "lhs_samples": int(lhs_samples),
            "sobol_samples": int(sobol_samples),
            "seed": int(seed),
            "radial_cells": int(tire_params.radial_cells),
            "theta_cells": int(tire_params.theta_cells),
            "internal_solver_dt_s": float(tire_params.internal_solver_dt_s),
            "results_path": str(output_path),
            "summary_path": str(summary_path),
            "scenario_names": [scenario.name for scenario in scenarios],
            "uq_scenario_names": [scenario.name for scenario in scenarios if scenario.include_in_uq],
        },
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
    return artifact


def run_scenario_pack(
    *,
    scenarios: tuple[ScenarioConfig, ...],
    tire_parameters: HighFidelityTireModelParameters,
    vehicle_parameters: VehicleParameters,
    dt_s: float,
) -> dict:
    scenario_traces: dict[str, dict] = {}
    scenario_summaries: dict[str, dict] = {}
    for scenario in scenarios:
        result = run_single_scenario(
            scenario=scenario,
            tire_parameters=tire_parameters,
            vehicle_parameters=vehicle_parameters,
            dt_s=dt_s,
        )
        scenario_traces[scenario.name] = result["trace"]
        scenario_summaries[scenario.name] = result["summary"]
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
) -> dict:
    simulator = HighFidelityVehicleSimulator(
        parameters=vehicle_parameters,
        tire_parameters_by_wheel={wheel: tire_parameters for wheel in ("FL", "FR", "RL", "RR")},
    )
    state = simulator.initial_state(ambient_temp_k=scenario.ambient_temp_k)
    inputs = VehicleInputs(
        speed_mps=scenario.speed_mps,
        ax_mps2=scenario.ax_mps2,
        ay_mps2=scenario.ay_mps2,
        steering_angle_rad=scenario.steering_angle_rad,
        yaw_rate_radps=scenario.yaw_rate_radps,
        brake_power_w=scenario.brake_power_w,
        drive_power_w=scenario.drive_power_w,
        ambient_temp_k=scenario.ambient_temp_k,
        track_temp_k=scenario.track_temp_k,
    )

    steps = max(int(round(scenario.duration_s / dt_s)), 1)
    time_s = [0.0]
    mean_core_temp_c = []
    mean_surface_temp_c = []
    load_error_pct = []
    max_energy_residual_pct = []
    coupling_converged_fraction = []
    any_non_finite = False

    initial_diag = simulator.diagnostics(state, inputs)
    mean_core_temp_c.append(_mean_dict_value(initial_diag.wheel_core_temp_c))
    mean_surface_temp_c.append(_mean_dict_value(initial_diag.wheel_surface_temp_c))
    load_error_pct.append(initial_diag.load_conservation_error_pct)
    max_energy_residual_pct.append(_max_energy_residual(initial_diag))
    coupling_converged_fraction.append(_coupling_converged_fraction(initial_diag))
    any_non_finite = any_non_finite or _diag_has_non_finite(initial_diag)

    converged_wheel_steps = 0
    total_wheel_steps = 0

    for step in range(1, steps + 1):
        state = simulator.step(state, inputs, dt_s=dt_s)
        diag = simulator.diagnostics(state, inputs)
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


def run_lhs_uq(
    *,
    scenarios: tuple[ScenarioConfig, ...],
    tire_parameters: HighFidelityTireModelParameters,
    vehicle_parameters: VehicleParameters,
    dt_s: float,
    uq: HighFidelityUQ,
    lhs_samples: int,
    seed: int,
) -> dict:
    priors = uq.default_tire_priors(parameters=tire_parameters)
    unit_samples = uq.latin_hypercube(priors=priors, sample_count=lhs_samples, seed=seed)
    mapped = uq.map_priors(unit_samples=unit_samples, priors=priors)

    scenario_core_traces: dict[str, list[np.ndarray]] = {scenario.name: [] for scenario in scenarios}
    scenario_surface_traces: dict[str, list[np.ndarray]] = {scenario.name: [] for scenario in scenarios}
    scenario_end_metrics: dict[str, list[float]] = {scenario.name: [] for scenario in scenarios}
    scenario_peak_metrics: dict[str, list[float]] = {scenario.name: [] for scenario in scenarios}
    scenario_end_surface_metrics: dict[str, list[float]] = {scenario.name: [] for scenario in scenarios}
    scenario_peak_surface_metrics: dict[str, list[float]] = {scenario.name: [] for scenario in scenarios}

    for sample_idx in range(lhs_samples):
        sample = {prior.name: float(mapped[prior.name][sample_idx]) for prior in priors}
        sampled_tire_parameters = uq.apply_sample(base=tire_parameters, sample=sample)
        for scenario in scenarios:
            result = run_single_scenario(
                scenario=scenario,
                tire_parameters=sampled_tire_parameters,
                vehicle_parameters=vehicle_parameters,
                dt_s=dt_s,
            )
            core_trace = np.asarray(result["trace"]["mean_core_temp_c"], dtype=float)
            surface_trace = np.asarray(result["trace"]["mean_surface_temp_c"], dtype=float)
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
) -> dict:
    priors = uq.default_tire_priors(parameters=tire_parameters)

    def objective(sample: dict[str, float]) -> float:
        sampled_tire_parameters = uq.apply_sample(base=tire_parameters, sample=sample)
        result = run_single_scenario(
            scenario=scenario,
            tire_parameters=sampled_tire_parameters,
            vehicle_parameters=vehicle_parameters,
            dt_s=dt_s,
        )
        return float(result["summary"]["peak_mean_surface_temp_c"])

    sobol = uq.sobol_indices(
        priors=priors,
        model_fn=objective,
        sample_count=sobol_samples,
        seed=seed,
    )
    return {
        "objective_metric": f"{scenario.name}.peak_mean_surface_temp_c",
        "variance": sobol.variance,
        "indices": [asdict(index) for index in sobol.indices],
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
    parser.add_argument("--output-json", type=Path, default=RESULTS_FILE)
    parser.add_argument("--output-summary", type=Path, default=SUMMARY_FILE)
    parser.add_argument("--lhs-samples", type=int, default=400)
    parser.add_argument("--sobol-samples", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--dt-s", type=float, default=0.2)
    parser.add_argument("--duration-scale", type=float, default=1.0)
    parser.add_argument("--radial-cells", type=int, default=24)
    parser.add_argument("--theta-cells", type=int, default=72)
    parser.add_argument("--internal-dt-s", type=float, default=0.01)
    args = parser.parse_args()

    tire_parameters = default_tire_parameters(
        radial_cells=args.radial_cells,
        theta_cells=args.theta_cells,
        internal_solver_dt_s=args.internal_dt_s,
    )
    run_high_fidelity_no_data(
        output_path=args.output_json,
        summary_path=args.output_summary,
        lhs_samples=args.lhs_samples,
        sobol_samples=args.sobol_samples,
        seed=args.seed,
        dt_s=args.dt_s,
        duration_scale=args.duration_scale,
        tire_parameters=tire_parameters,
    )


if __name__ == "__main__":
    main()
