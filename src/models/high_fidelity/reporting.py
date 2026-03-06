from __future__ import annotations


def render_high_fidelity_no_data_summary(artifact: dict) -> str:
    metadata = artifact["metadata"]
    scenario_summaries = artifact["baseline"]["scenario_summaries"]
    plausibility = artifact["plausibility_checks"]
    sobol = artifact["uq"]["sobol"]

    lines = [
        "# High-Fidelity No-Data Summary",
        "",
        "## Run Configuration",
        "",
        f"- created_at_utc: {metadata['created_at_utc']}",
        f"- dt_s: {metadata['dt_s']:.3f}",
        f"- duration_scale: {metadata['duration_scale']:.3f}",
        f"- lhs_samples: {metadata['lhs_samples']}",
        f"- sobol_samples: {metadata['sobol_samples']}",
        f"- seed: {metadata['seed']}",
        f"- radial_cells: {metadata['radial_cells']}",
        f"- theta_cells: {metadata['theta_cells']}",
        f"- internal_solver_dt_s: {metadata['internal_solver_dt_s']:.3f}",
        f"- default_output_mode: {metadata.get('default_output_mode', 'bands_plus_baseline')}",
        "",
        "## UQ-First Scenario Metrics",
        "",
        "| scenario | core_q05_c | core_q50_c | core_q95_c | surface_q50_c | surface_q95_c | baseline_core_end_c | baseline_surface_peak_c |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, envelope in artifact["uq"]["lhs"]["scenario_envelopes"].items():
        summary = scenario_summaries[name]
        lines.append(
            "| "
            + f"{name} | {envelope['end_mean_core_temp_c']['q05']:.3f} | "
            + f"{envelope['end_mean_core_temp_c']['q50']:.3f} | "
            + f"{envelope['end_mean_core_temp_c']['q95']:.3f} | "
            + f"{envelope['peak_mean_surface_temp_c']['q50']:.3f} | "
            + f"{envelope['peak_mean_surface_temp_c']['q95']:.3f} | "
            + f"{summary['end_mean_core_temp_c']:.3f} | "
            + f"{summary['peak_mean_surface_temp_c']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Deterministic Baseline",
            "",
            "| scenario | end_mean_core_temp_c | peak_mean_core_temp_c | end_mean_surface_temp_c | peak_mean_surface_temp_c | max_load_error_pct | max_energy_residual_pct | coupling_convergence_rate |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for name, summary in scenario_summaries.items():
        lines.append(
            "| "
            + f"{name} | {summary['end_mean_core_temp_c']:.3f} | "
            + f"{summary['peak_mean_core_temp_c']:.3f} | "
            + f"{summary['end_mean_surface_temp_c']:.3f} | "
            + f"{summary['peak_mean_surface_temp_c']:.3f} | "
            + f"{summary['max_load_error_pct']:.6f} | "
            + f"{summary['max_energy_residual_pct']:.6f} | "
            + f"{summary['coupling_convergence_rate']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Sobol Ranking",
            "",
            f"Objective metric: `{sobol['objective_metric']}`",
            "",
            "| parameter | first_order | total_order |",
            "|---|---:|---:|",
        ]
    )
    for entry in sobol["indices"]:
        lines.append(
            f"| {entry['name']} | {entry['first_order']:.6f} | {entry['total_order']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Plausibility Checks",
            "",
        ]
    )
    for name, value in plausibility.items():
        lines.append(f"- {name}: {value}")

    lines.extend(
        [
            "",
            "## Artifact Paths",
            "",
            f"- results_json: {metadata['results_path']}",
            f"- summary_md: {metadata['summary_path']}",
            "",
        ]
    )
    return "\n".join(lines)
