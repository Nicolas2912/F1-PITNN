from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.high_fidelity import HighFidelityUQ  # noqa: E402
from models.high_fidelity.native_diffusion import native_diffusion_available, native_diffusion_enabled  # noqa: E402
from models.high_fidelity.native_simulator_kernels import native_simulator_kernels_available, native_simulator_kernels_enabled  # noqa: E402
from models.vehicle_thermal import VehicleParameters  # noqa: E402
from run_high_fidelity_no_data import (  # noqa: E402
    ProcessPoolRunner,
    ScenarioConfig,
    UQSurrogateConfig,
    _vehicle_inputs_for_scenario,
    _vehicle_simulator,
    default_scenarios,
    default_tire_parameters,
    fidelity_preset,
    run_lhs_uq,
    run_scenario_pack,
    run_sobol_uq,
)


def _cap_scenario_steps(
    scenarios: tuple[ScenarioConfig, ...],
    *,
    dt_s: float,
    max_steps_per_scenario: int | None,
) -> tuple[ScenarioConfig, ...]:
    if max_steps_per_scenario is None:
        return scenarios
    capped_duration_s = max(int(max_steps_per_scenario), 1) * dt_s
    return tuple(
        replace(scenario, duration_s=min(float(scenario.duration_s), float(capped_duration_s)))
        for scenario in scenarios
    )


def _profile_single_scenario_step(
    *,
    scenario,
    tire_parameters,
    vehicle_parameters: VehicleParameters,
    dt_s: float,
    profile_steps: int,
) -> dict[str, object]:
    profiled_tire_parameters = replace(tire_parameters, enable_profiling=True)
    simulator = _vehicle_simulator(
        tire_parameters=profiled_tire_parameters,
        vehicle_parameters=vehicle_parameters,
    )
    inputs = _vehicle_inputs_for_scenario(scenario)
    prepared_inputs = simulator.prepare_inputs(inputs)
    state = simulator.initial_state(ambient_temp_k=scenario.ambient_temp_k)
    wheel_samples: dict[str, list[dict[str, float | int | None]]] = {}
    wall_times_s: list[float] = []

    for _ in range(max(int(profile_steps), 1)):
        start = time.perf_counter()
        state = simulator.step(state, inputs, dt_s=dt_s, prepared_inputs=prepared_inputs)
        wall_times_s.append(time.perf_counter() - start)
        for wheel, wheel_state in state.wheel_states.items():
            wheel_samples.setdefault(wheel, []).append(
                {
                    "solver_substeps": int(wheel_state.last_solver_substeps),
                    "solver_diffusion_iterations": int(wheel_state.last_solver_diffusion_iterations or 0),
                    "solver_advection_time_s": float(wheel_state.last_solver_advection_time_s or 0.0),
                    "solver_diffusion_time_s": float(wheel_state.last_solver_diffusion_time_s or 0.0),
                    "wheel_coupling_time_s": float(wheel_state.last_wheel_coupling_time_s or 0.0),
                }
            )

    wheel_summary: dict[str, dict[str, float]] = {}
    diffusion_means: list[float] = []
    coupling_means: list[float] = []
    advection_means: list[float] = []
    wall_mean = statistics.mean(wall_times_s)
    for wheel, samples in wheel_samples.items():
        diffusion = statistics.mean(float(sample["solver_diffusion_time_s"]) for sample in samples)
        coupling = statistics.mean(float(sample["wheel_coupling_time_s"]) for sample in samples)
        advection = statistics.mean(float(sample["solver_advection_time_s"]) for sample in samples)
        diffusion_means.append(diffusion)
        coupling_means.append(coupling)
        advection_means.append(advection)
        wheel_summary[wheel] = {
            "mean_solver_substeps": statistics.mean(int(sample["solver_substeps"]) for sample in samples),
            "mean_solver_diffusion_iterations": statistics.mean(
                int(sample["solver_diffusion_iterations"]) for sample in samples
            ),
            "mean_solver_advection_time_s": advection,
            "mean_solver_diffusion_time_s": diffusion,
            "mean_wheel_coupling_time_s": coupling,
        }

    aggregate_diffusion = statistics.mean(diffusion_means)
    aggregate_coupling = statistics.mean(coupling_means)
    aggregate_advection = statistics.mean(advection_means)
    return {
        "scenario": scenario.name,
        "profile_steps": max(int(profile_steps), 1),
        "mean_wall_time_per_vehicle_step_s": wall_mean,
        "mean_wall_time_per_tire_step_s": wall_mean / 4.0,
        "mean_per_wheel_solver_diffusion_time_s": aggregate_diffusion,
        "mean_per_wheel_solver_advection_time_s": aggregate_advection,
        "mean_per_wheel_wheel_coupling_time_s": aggregate_coupling,
        "estimated_vehicle_step_diffusion_share_pct": (
            100.0 * (4.0 * aggregate_diffusion) / wall_mean if wall_mean > 0.0 else 0.0
        ),
        "estimated_vehicle_step_coupling_share_pct": (
            100.0 * (4.0 * aggregate_coupling) / wall_mean if wall_mean > 0.0 else 0.0
        ),
        "per_wheel": wheel_summary,
    }


def _run_phase(
    *,
    label: str,
    fn,
    units: int,
) -> tuple[dict, dict[str, float | int | None]]:
    start = time.perf_counter()
    result = fn()
    elapsed_s = time.perf_counter() - start
    return result, {
        "label": label,
        "elapsed_s": elapsed_s,
        "units": int(units),
        "avg_seconds_per_unit": elapsed_s / units if units > 0 else None,
        "throughput_units_per_s": units / elapsed_s if elapsed_s > 0.0 else None,
    }


def _scenario_runtime_rows(
    *,
    scenarios,
    tire_parameters,
    vehicle_parameters: VehicleParameters,
    dt_s: float,
) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    for scenario in scenarios:
        simulator = _vehicle_simulator(
            tire_parameters=tire_parameters,
            vehicle_parameters=vehicle_parameters,
        )
        inputs = _vehicle_inputs_for_scenario(scenario)
        prepared_inputs = simulator.prepare_inputs(inputs)
        start = time.perf_counter()
        from run_high_fidelity_no_data import run_single_scenario  # noqa: E402

        run_single_scenario(
            scenario=scenario,
            tire_parameters=tire_parameters,
            vehicle_parameters=vehicle_parameters,
            dt_s=dt_s,
            diagnostics_stride=1,
            simulator=simulator,
            inputs=inputs,
            prepared_inputs=prepared_inputs,
        )
        elapsed_s = time.perf_counter() - start
        steps = max(int(round(scenario.duration_s / dt_s)), 1)
        rows.append(
            {
                "scenario": scenario.name,
                "duration_s": float(scenario.duration_s),
                "steps": steps,
                "elapsed_s": elapsed_s,
                "seconds_per_step": elapsed_s / steps,
            }
        )
    return rows


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Profile high-fidelity runtime and native solver breakdowns.")
    parser.add_argument("--preset", choices=["smoke", "dev", "full"], default="full")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--dt-s", type=float, default=0.2)
    parser.add_argument("--duration-scale", type=float, default=1.0)
    parser.add_argument("--lhs-samples", type=int, default=None)
    parser.add_argument("--sobol-samples", type=int, default=None)
    parser.add_argument("--pilot-lhs-samples", type=int, default=None)
    parser.add_argument("--pilot-sobol-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--profile-steps", type=int, default=3)
    parser.add_argument("--max-steps-per-scenario", type=int, default=None)
    parser.add_argument("--diagnostics-stride", type=int, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--run-full-lhs", action="store_true")
    parser.add_argument("--run-full-sobol", action="store_true")
    parser.add_argument("--uq-surrogate", choices=["none", "quadratic_ridge", "extra_trees"], default="none")
    parser.add_argument("--uq-surrogate-sobol-train-samples", type=int, default=None)
    parser.add_argument("--uq-surrogate-sobol-validation-samples", type=int, default=None)
    parser.add_argument("--uq-surrogate-ridge-alpha", type=float, default=1e-6)
    parser.add_argument("--uq-surrogate-max-rmse-c", type=float, default=0.75)
    parser.add_argument("--uq-surrogate-max-abs-error-c", type=float, default=2.0)
    parser.add_argument("--uq-surrogate-min-prediction-samples", type=int, default=32)
    parser.add_argument("--uq-surrogate-extra-trees-estimators", type=int, default=600)
    args = parser.parse_args()

    preset = fidelity_preset(args.preset)
    lhs_samples = preset.lhs_samples if args.lhs_samples is None else args.lhs_samples
    sobol_samples = preset.sobol_samples if args.sobol_samples is None else args.sobol_samples
    diagnostics_stride = preset.diagnostics_stride if args.diagnostics_stride is None else args.diagnostics_stride
    pilot_lhs_samples = lhs_samples if args.run_full_lhs else (
        min(lhs_samples, max(2 * max(args.workers, 1), 12))
        if args.pilot_lhs_samples is None
        else args.pilot_lhs_samples
    )
    pilot_sobol_samples = sobol_samples if args.run_full_sobol else (
        min(sobol_samples, max(4 * max(args.workers, 1), 24))
        if args.pilot_sobol_samples is None
        else args.pilot_sobol_samples
    )

    tire_parameters = default_tire_parameters(
        radial_cells=preset.radial_cells,
        theta_cells=preset.theta_cells,
        internal_solver_dt_s=preset.internal_solver_dt_s,
    )
    vehicle_parameters = VehicleParameters()
    scenarios = _cap_scenario_steps(
        default_scenarios(duration_scale=args.duration_scale),
        dt_s=args.dt_s,
        max_steps_per_scenario=args.max_steps_per_scenario,
    )
    uq_scenarios = tuple(s for s in scenarios if s.include_in_uq)
    combined_brake_corner = next(s for s in scenarios if s.name == "combined_brake_corner")
    uq = HighFidelityUQ()
    priors = uq.default_tire_priors(parameters=tire_parameters)
    sobol_eval_count = sobol_samples * (2 + len(priors))
    pilot_sobol_eval_count = pilot_sobol_samples * (2 + len(priors))
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

    overall_start = time.perf_counter()
    pool_runner = ProcessPoolRunner(workers=args.workers) if args.workers > 1 else None
    try:
        baseline_result, baseline_timing = _run_phase(
            label="baseline",
            units=len(scenarios),
            fn=lambda: run_scenario_pack(
                scenarios=scenarios,
                tire_parameters=tire_parameters,
                vehicle_parameters=vehicle_parameters,
                dt_s=args.dt_s,
                diagnostics_stride=diagnostics_stride,
                workers=args.workers,
                pool_runner=pool_runner,
            ),
        )
        lhs_result, lhs_timing = _run_phase(
            label="lhs_pilot" if pilot_lhs_samples != lhs_samples else "lhs",
            units=pilot_lhs_samples,
            fn=lambda: run_lhs_uq(
                scenarios=uq_scenarios,
                tire_parameters=tire_parameters,
                vehicle_parameters=vehicle_parameters,
                dt_s=args.dt_s,
                uq=uq,
                lhs_samples=pilot_lhs_samples,
                seed=args.seed,
                diagnostics_stride=diagnostics_stride,
                workers=args.workers,
                pool_runner=pool_runner,
            ),
        )
        sobol_result, sobol_timing = _run_phase(
            label="sobol_pilot" if pilot_sobol_samples != sobol_samples else "sobol",
            units=pilot_sobol_eval_count,
            fn=lambda: run_sobol_uq(
                scenario=combined_brake_corner,
                tire_parameters=tire_parameters,
                vehicle_parameters=vehicle_parameters,
                dt_s=args.dt_s,
                uq=uq,
                sobol_samples=pilot_sobol_samples,
                seed=args.seed,
                diagnostics_stride=diagnostics_stride,
                workers=args.workers,
                pool_runner=pool_runner,
                surrogate_config=surrogate_config,
            ),
        )
    finally:
        if pool_runner is not None:
            pool_runner.close()
    overall_elapsed_s = time.perf_counter() - overall_start

    scenario_rows = _scenario_runtime_rows(
        scenarios=scenarios,
        tire_parameters=tire_parameters,
        vehicle_parameters=vehicle_parameters,
        dt_s=args.dt_s,
    )
    native_step_profile = _profile_single_scenario_step(
        scenario=combined_brake_corner,
        tire_parameters=tire_parameters,
        vehicle_parameters=vehicle_parameters,
        dt_s=args.dt_s,
        profile_steps=args.profile_steps,
    )

    lhs_estimated_full_s = (
        float(lhs_timing["avg_seconds_per_unit"]) * lhs_samples
        if lhs_timing["avg_seconds_per_unit"] is not None
        else None
    )
    sobol_estimated_full_s = (
        float(sobol_timing["avg_seconds_per_unit"]) * sobol_eval_count
        if sobol_timing["avg_seconds_per_unit"] is not None
        else None
    )
    estimated_full_total_s = baseline_timing["elapsed_s"]
    if lhs_estimated_full_s is not None:
        estimated_full_total_s += lhs_estimated_full_s
    if sobol_estimated_full_s is not None:
        estimated_full_total_s += sobol_estimated_full_s

    payload = {
        "config": {
            "preset": preset.name,
            "workers": int(args.workers),
            "dt_s": float(args.dt_s),
            "duration_scale": float(args.duration_scale),
            "lhs_samples": int(lhs_samples),
            "sobol_samples": int(sobol_samples),
            "pilot_lhs_samples": int(pilot_lhs_samples),
            "pilot_sobol_samples": int(pilot_sobol_samples),
            "diagnostics_stride": int(diagnostics_stride),
            "max_steps_per_scenario": (
                None if args.max_steps_per_scenario is None else int(args.max_steps_per_scenario)
            ),
            "radial_cells": int(tire_parameters.radial_cells),
            "theta_cells": int(tire_parameters.theta_cells),
            "internal_solver_dt_s": float(tire_parameters.internal_solver_dt_s),
            "prior_count": len(priors),
            "sobol_eval_count_full": int(sobol_eval_count),
            "sobol_eval_count_measured": int(pilot_sobol_eval_count),
            "uq_surrogate": asdict(surrogate_config),
        },
        "native_runtime": {
            "native_diffusion_available": native_diffusion_available(),
            "native_diffusion_enabled": native_diffusion_enabled(),
            "native_simulator_kernels_available": native_simulator_kernels_available(),
            "native_simulator_kernels_enabled": native_simulator_kernels_enabled(),
        },
        "timing": {
            "overall_elapsed_s": overall_elapsed_s,
            "baseline": baseline_timing,
            "lhs_measured": lhs_timing,
            "sobol_measured": sobol_timing,
            "estimated_full_lhs_elapsed_s": lhs_estimated_full_s,
            "estimated_full_sobol_elapsed_s": sobol_estimated_full_s,
            "estimated_full_total_elapsed_s": estimated_full_total_s,
        },
        "baseline_result_keys": list(baseline_result["scenario_summaries"]),
        "lhs_result_keys": list(lhs_result["scenario_envelopes"]),
        "sobol_summary": {
            "objective_metric": sobol_result["objective_metric"],
            "variance": sobol_result["variance"],
            "top_indices": sobol_result["indices"][:5],
        },
        "scenario_runtime_rows": scenario_rows,
        "native_step_profile": native_step_profile,
        "priors": [asdict(prior) for prior in priors],
    }

    text = json.dumps(payload, indent=2)
    if args.output_json is not None:
        args.output_json.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
