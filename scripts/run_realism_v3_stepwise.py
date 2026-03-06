from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.physics import (  # noqa: E402
    TireInputs,
    TireModelParameters,
    TireThermalSimulator,
    celsius_to_kelvin,
    kelvin_to_celsius,
)
from models.vehicle_thermal import (  # noqa: E402
    VehicleInputs,
    VehicleParameters,
    VehicleThermalSimulator,
)

INPUT_PROFILE = ROOT / "reports" / "results" / "core_temp_simulation_calibrated.csv"
BASELINE_METRICS_SOURCE = ROOT / "reports" / "results" / "realism_stepwise_metrics.csv"
OUTPUT_DIR = ROOT / "reports" / "results"

BASELINE_SNAPSHOT_FILE = OUTPUT_DIR / "realism_v3_baseline_snapshot.csv"
STEPWISE_METRICS_FILE = OUTPUT_DIR / "realism_v3_stepwise_metrics.csv"
STEPWISE_TEXT_FILE = OUTPUT_DIR / "realism_v3_stepwise_metrics.txt"
V3_SINGLE_TIRE_FILE = OUTPUT_DIR / "core_temp_simulation_realism_v3.csv"
VEHICLE_VALIDATION_FILE = OUTPUT_DIR / "vehicle_4wheel_validation_summary.md"

STAGE_LABELS: dict[int, str] = {
    0: "baseline",
    1: "sidewall_rr",
    2: "slip_transients",
    3: "quasi_2d_patch",
    4: "friction_partition",
    5: "vehicle_4wheel_coupling",
}
VALIDATION_GATES_FILE = OUTPUT_DIR / "realism_v3_validation_gates.csv"


@dataclass(frozen=True)
class SyntheticManeuver:
    name: str
    speed_mps: float
    ax_mps2: float
    ay_mps2: float


def stage_parameters(stage: int) -> TireModelParameters:
    if stage not in range(0, 5):
        msg = f"Single-tire stages must be in [0, 4], got {stage}"
        raise ValueError(msg)
    # Phases are enabled cumulatively to preserve stage comparability.
    return TireModelParameters(
        use_gauge_patch_model=stage >= 3,
        use_combined_slip_model=False,
        use_zone_lateral_conduction=False,
        use_temperature_dependent_properties=False,
        use_rotating_internal_gas_model=False,
        use_alignment_zone_effects=False,
        use_sidewall_rr_split_model=stage >= 1,
        use_slip_transient_model=stage >= 2,
        use_quasi_2d_patch_model=stage >= 3,
        use_friction_partition_model=stage >= 4,
    )


def run_single_tire_stage(df_profile: pd.DataFrame, stage: int) -> pd.DataFrame:
    params = stage_parameters(stage)
    simulator = TireThermalSimulator(params)

    dt_s = float(df_profile["time_s"].diff().dropna().median())
    first = df_profile.iloc[0]
    state = simulator.initial_state(
        ambient_temp_k=celsius_to_kelvin(float(first["t_carcass_core_c"])),
        brake_temp_k=celsius_to_kelvin(float(first["t_brake_c"])),
        wear=float(first["wear"]),
    )

    rows: list[dict[str, float]] = []
    for _, row in df_profile.iterrows():
        inputs = TireInputs(
            speed_mps=float(row["speed_mps"]),
            wheel_angular_speed_radps=float(row["speed_mps"]) * (1.0 + float(row["slip_ratio"])) / 0.33,
            normal_load_n=float(row["normal_load_n"]),
            slip_ratio=float(row["slip_ratio"]),
            slip_angle_rad=float(row["slip_angle_rad"]),
            brake_power_w=float(row["brake_power_w"]),
            ambient_temp_k=celsius_to_kelvin(28.0),
            track_temp_k=celsius_to_kelvin(44.0),
            camber_rad=0.0,
            toe_rad=0.0,
            lateral_accel_mps2=0.0,
            longitudinal_accel_mps2=0.0,
            is_left_tire=True,
            is_front_tire=True,
        )
        state = simulator.step(state, inputs, dt_s=dt_s)
        diag = simulator.diagnostics(state, inputs)
        rows.append(
            {
                "time_s": float(row["time_s"]),
                "lap_index": float(row["lap_index"]),
                "speed_mps": float(row["speed_mps"]),
                "normal_load_n": float(row["normal_load_n"]),
                "slip_ratio": float(row["slip_ratio"]),
                "slip_angle_rad": float(row["slip_angle_rad"]),
                "brake_power_w": float(row["brake_power_w"]),
                "t_surface_middle_c": kelvin_to_celsius(state.t_surface_middle_k),
                "t_carcass_core_c": kelvin_to_celsius(state.t_carcass_k),
                "t_gas_c": kelvin_to_celsius(state.t_gas_k),
                "pressure_bar_g": diag.dynamic_pressure_bar_gauge,
                "contact_patch_m2": diag.contact_patch_area_m2,
                "h_int_w_m2k": diag.internal_htc_w_m2k,
                "wear": state.wear,
            }
        )

    return pd.DataFrame(rows)


def single_tire_metrics(df: pd.DataFrame) -> dict[str, float]:
    dt = df["time_s"].diff().replace(0.0, np.nan).fillna(df["time_s"].diff().dropna().median())
    core_rate = df["t_carcass_core_c"].diff().fillna(0.0) / dt
    gas_rate = df["t_gas_c"].diff().fillna(0.0) / dt
    surface_gap = df["t_surface_middle_c"] - df["t_carcass_core_c"]
    h_int_clip_fraction = (
        (df["h_int_w_m2k"] >= (df["h_int_w_m2k"].max() - 1e-6)).sum() / max(len(df), 1)
    )
    return {
        "core_end_c": float(df["t_carcass_core_c"].iloc[-1]),
        "core_delta_c": float(df["t_carcass_core_c"].iloc[-1] - df["t_carcass_core_c"].iloc[0]),
        "core_rate_95_c_per_s": float(core_rate.quantile(0.95)),
        "gas_rate_99_c_per_s": float(gas_rate.quantile(0.99)),
        "surface_core_gap_mean_c": float(surface_gap.mean()),
        "surface_core_gap_max_c": float(surface_gap.max()),
        "pressure_end_bar_g": float(df["pressure_bar_g"].iloc[-1]),
        "pressure_delta_bar": float(df["pressure_bar_g"].iloc[-1] - df["pressure_bar_g"].iloc[0]),
        "patch_pressure_corr": float(df["contact_patch_m2"].corr(df["pressure_bar_g"])),
        "h_int_clip_fraction": float(h_int_clip_fraction),
        "wear_delta": float(df["wear"].iloc[-1] - df["wear"].iloc[0]),
    }


def freeze_baseline_snapshot() -> pd.DataFrame:
    baseline_df = pd.read_csv(BASELINE_METRICS_SOURCE)
    baseline_df.to_csv(BASELINE_SNAPSHOT_FILE, index=False)
    return baseline_df


def _expected_sign(value: float) -> int:
    if abs(value) < 1e-9:
        return 0
    return 1 if value > 0.0 else -1


def _sign_check(actual_value: float, expected_sign: int) -> bool:
    if expected_sign == 0:
        return abs(actual_value) <= 1e-6
    return _expected_sign(actual_value) == expected_sign


def run_vehicle_4wheel_validation() -> tuple[pd.DataFrame, dict[str, float]]:
    vehicle_params = VehicleParameters()
    tire_params = TireModelParameters(
        **{
            **stage_parameters(4).__dict__,
            "use_rotating_internal_gas_model": True,
        }
    )
    wheel_tire_params = {wheel: tire_params for wheel in ("FL", "FR", "RL", "RR")}
    simulator = VehicleThermalSimulator(
        parameters=vehicle_params,
        tire_parameters_by_wheel=wheel_tire_params,
    )

    maneuvers = [
        SyntheticManeuver(name="steady_corner", speed_mps=55.0, ax_mps2=0.0, ay_mps2=7.0),
        SyntheticManeuver(name="brake_in_line", speed_mps=62.0, ax_mps2=-7.5, ay_mps2=0.0),
        SyntheticManeuver(name="accel_in_line", speed_mps=48.0, ax_mps2=4.0, ay_mps2=0.0),
        SyntheticManeuver(name="combined_brake_corner", speed_mps=52.0, ax_mps2=-5.5, ay_mps2=5.2),
    ]
    maneuver_rows: list[dict[str, float]] = []
    all_h_int_samples: list[float] = []
    all_core_end_samples: list[float] = []
    all_pressure_end_samples: list[float] = []
    all_wear_end_samples: list[float] = []

    for maneuver in maneuvers:
        steering = 0.0
        if abs(maneuver.ay_mps2) > 1e-6:
            steering = math.atan2(
                maneuver.ay_mps2 * vehicle_params.wheelbase_m,
                max(maneuver.speed_mps * maneuver.speed_mps, 1.0),
            )
        yaw_rate = maneuver.ay_mps2 / max(maneuver.speed_mps, 1.0)
        brake_power = max(-maneuver.ax_mps2, 0.0) / 8.0 * 140_000.0
        drive_power = max(maneuver.ax_mps2, 0.0) / 5.0 * 100_000.0
        control = VehicleInputs(
            speed_mps=maneuver.speed_mps,
            ax_mps2=maneuver.ax_mps2,
            ay_mps2=maneuver.ay_mps2,
            steering_angle_rad=steering,
            yaw_rate_radps=yaw_rate,
            brake_power_w=brake_power,
            drive_power_w=drive_power,
            ambient_temp_k=celsius_to_kelvin(28.0),
            track_temp_k=celsius_to_kelvin(44.0),
        )
        state = simulator.initial_state(
            ambient_temp_k=celsius_to_kelvin(28.0),
            wear=0.05,
        )

        load_errors: list[float] = []
        front_transfer_sign: list[float] = []
        right_transfer_sign: list[float] = []
        wheel_positive: list[float] = []
        wheel_h_int_samples: list[float] = []
        diag = simulator.diagnostics(state, control)
        for _ in range(250):
            state = simulator.step(state, control, dt_s=0.1)
            diag = simulator.diagnostics(state, control)
            load_errors.append(diag.load_conservation_error_pct)
            static_front = vehicle_params.mass_kg * vehicle_params.gravity_mps2 * vehicle_params.front_static_weight_fraction
            static_front += (
                vehicle_params.aero_downforce_coeff_n_per_mps2
                * maneuver.speed_mps
                * maneuver.speed_mps
                * vehicle_params.aero_front_fraction
            )
            front_delta_vs_static = diag.front_axle_load_n - static_front
            front_transfer_sign.append(float(_sign_check(front_delta_vs_static, _expected_sign(-maneuver.ax_mps2))))
            right_transfer_sign.append(
                float(_sign_check(diag.right_minus_left_load_n, _expected_sign(maneuver.ay_mps2)))
            )
            wheel_positive.append(float(min(diag.wheel_load_n.values()) > 0.0))
            wheel_h_int_samples.extend(diag.wheel_internal_htc_w_m2k.values())

        all_h_int_samples.extend(wheel_h_int_samples)
        all_core_end_samples.extend(diag.wheel_core_temp_c.values())
        all_pressure_end_samples.extend(diag.wheel_pressure_bar_g.values())
        all_wear_end_samples.extend(wheel_state.wear for wheel_state in state.wheel_states.values())
        h_int_clip_fraction = (
            sum(v >= tire_params.max_internal_htc_w_per_m2k - 1e-6 for v in wheel_h_int_samples)
            / max(len(wheel_h_int_samples), 1)
        )
        maneuver_rows.append(
            {
                "maneuver": maneuver.name,
                "speed_mps": maneuver.speed_mps,
                "ax_mps2": maneuver.ax_mps2,
                "ay_mps2": maneuver.ay_mps2,
                "fl_n": diag.wheel_load_n["FL"],
                "fr_n": diag.wheel_load_n["FR"],
                "rl_n": diag.wheel_load_n["RL"],
                "rr_n": diag.wheel_load_n["RR"],
                "load_conservation_error_pct": float(max(load_errors)),
                "front_delta_vs_static_n": float(diag.front_axle_load_n - static_front),
                "right_minus_left_delta_n": float(diag.right_minus_left_load_n),
                "front_transfer_sign_ok": float(np.mean(front_transfer_sign)),
                "right_transfer_sign_ok": float(np.mean(right_transfer_sign)),
                "all_wheel_loads_positive": float(np.mean(wheel_positive)),
                "h_int_clip_fraction": float(h_int_clip_fraction),
                "core_temp_mean_c": float(np.mean(list(diag.wheel_core_temp_c.values()))),
                "pressure_mean_bar_g": float(np.mean(list(diag.wheel_pressure_bar_g.values()))),
            }
        )

    df = pd.DataFrame(maneuver_rows)
    overall_h_int_clip_fraction = (
        sum(v >= tire_params.max_internal_htc_w_per_m2k - 1e-6 for v in all_h_int_samples)
        / max(len(all_h_int_samples), 1)
    )
    summary = {
        "load_conservation_max_error_pct": float(df["load_conservation_error_pct"].max()),
        "front_transfer_sign_pass_rate": float(df["front_transfer_sign_ok"].mean()),
        "right_transfer_sign_pass_rate": float(df["right_transfer_sign_ok"].mean()),
        "wheel_positive_pass_rate": float(df["all_wheel_loads_positive"].mean()),
        "h_int_clip_fraction": float(overall_h_int_clip_fraction),
        "core_end_c": float(np.mean(all_core_end_samples)) if all_core_end_samples else float("nan"),
        "pressure_end_bar_g": float(np.mean(all_pressure_end_samples)) if all_pressure_end_samples else float("nan"),
        "wear_delta": float(np.mean(all_wear_end_samples) - 0.05) if all_wear_end_samples else float("nan"),
    }
    return df, summary


def evaluate_validation_gates(
    *,
    stage_metrics_df: pd.DataFrame,
    stage_outputs: dict[int, pd.DataFrame],
    vehicle_validation_df: pd.DataFrame,
) -> pd.DataFrame:
    baseline = stage_metrics_df.loc[stage_metrics_df["stage"] == 0].iloc[0]
    stage4 = stage_metrics_df.loc[stage_metrics_df["stage"] == 4].iloc[0]
    stage5 = stage_metrics_df.loc[stage_metrics_df["stage"] == 5].iloc[0]

    single_tire_finite = all(
        np.isfinite(df.select_dtypes(include=[np.number]).to_numpy()).all() for df in stage_outputs.values()
    )
    vehicle_finite = np.isfinite(vehicle_validation_df.select_dtypes(include=[np.number]).to_numpy()).all()
    no_nan_inf_anywhere = bool(single_tire_finite and vehicle_finite)

    core_drift = float(stage4["core_end_c"] - baseline["core_end_c"])
    pressure_drift = float(stage4["pressure_end_bar_g"] - baseline["pressure_end_bar_g"])
    patch_corr = float(stage4["patch_pressure_corr"])
    h_int_clip_stage5 = float(stage5["h_int_clip_fraction"])
    load_err_stage5 = float(stage5["load_conservation_max_error_pct"])

    checks = [
        ("no_nan_inf_anywhere", no_nan_inf_anywhere, float("nan"), "all outputs finite"),
        ("core_drift_within_4C", abs(core_drift) <= 4.0, core_drift, "abs(stage4-core0)<=4.0"),
        ("pressure_drift_within_0.12bar", abs(pressure_drift) <= 0.12, pressure_drift, "abs(stage4-stage0)<=0.12"),
        ("patch_pressure_corr_le_-0.40", patch_corr <= -0.40, patch_corr, "stage4<=-0.40"),
        ("h_int_clip_fraction_lt_0.05", h_int_clip_stage5 < 0.05, h_int_clip_stage5, "stage5<0.05"),
        ("load_conservation_lt_0.5pct", load_err_stage5 < 0.5, load_err_stage5, "stage5<0.5%"),
    ]
    return pd.DataFrame(
        [
            {
                "gate": gate,
                "passed": passed,
                "value": value,
                "criterion": criterion,
            }
            for gate, passed, value, criterion in checks
        ]
    )


def write_vehicle_validation_summary(validation_df: pd.DataFrame, summary: dict[str, float]) -> None:
    pretty = validation_df.copy()
    bool_columns = ["front_transfer_sign_ok", "right_transfer_sign_ok", "all_wheel_loads_positive"]
    for col in bool_columns:
        pretty[col] = pretty[col].map(lambda v: "yes" if v >= 0.999 else "no")

    lines = [
        "# Vehicle 4-Wheel Validation Summary",
        "",
        "Phase 5 deterministic maneuver checks using VehicleThermalSimulator:",
        "- steady corner",
        "- brake-in-line",
        "- accel-in-line",
        "- combined brake+corner",
        "",
        f"load_conservation_max_error_pct: {summary['load_conservation_max_error_pct']:.6f}",
        f"front_transfer_sign_pass_rate: {summary['front_transfer_sign_pass_rate']:.3f}",
        f"right_transfer_sign_pass_rate: {summary['right_transfer_sign_pass_rate']:.3f}",
        f"wheel_positive_pass_rate: {summary['wheel_positive_pass_rate']:.3f}",
        f"h_int_clip_fraction: {summary['h_int_clip_fraction']:.6f}",
        f"core_end_c_mean: {summary['core_end_c']:.3f}",
        f"pressure_end_bar_g_mean: {summary['pressure_end_bar_g']:.3f}",
        f"wear_delta_mean: {summary['wear_delta']:.6f}",
        "",
        pretty.to_string(index=False),
        "",
    ]
    VEHICLE_VALIDATION_FILE.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    freeze_baseline_snapshot()
    profile = pd.read_csv(INPUT_PROFILE)

    stage_rows: list[dict[str, float | int | str]] = []
    stage_outputs: dict[int, pd.DataFrame] = {}
    for stage in range(0, 5):
        stage_df = run_single_tire_stage(profile, stage=stage)
        stage_outputs[stage] = stage_df
        stage_metric = single_tire_metrics(stage_df)
        stage_rows.append(
            {
                "stage": stage,
                "label": STAGE_LABELS[stage],
                "mode": "single_tire_replay",
                **stage_metric,
                "load_conservation_max_error_pct": np.nan,
                "front_transfer_sign_pass_rate": np.nan,
                "right_transfer_sign_pass_rate": np.nan,
                "wheel_positive_pass_rate": np.nan,
            }
        )

    vehicle_validation_df, vehicle_summary = run_vehicle_4wheel_validation()
    stage_rows.append(
        {
            "stage": 5,
            "label": STAGE_LABELS[5],
            "mode": "vehicle_4wheel_coupling",
            "core_end_c": np.nan,
            "core_delta_c": np.nan,
            "core_rate_95_c_per_s": np.nan,
            "gas_rate_99_c_per_s": np.nan,
            "surface_core_gap_mean_c": np.nan,
            "surface_core_gap_max_c": np.nan,
            "pressure_end_bar_g": np.nan,
            "pressure_delta_bar": np.nan,
            "patch_pressure_corr": np.nan,
            "h_int_clip_fraction": vehicle_summary["h_int_clip_fraction"],
            "wear_delta": vehicle_summary["wear_delta"],
            "load_conservation_max_error_pct": vehicle_summary["load_conservation_max_error_pct"],
            "front_transfer_sign_pass_rate": vehicle_summary["front_transfer_sign_pass_rate"],
            "right_transfer_sign_pass_rate": vehicle_summary["right_transfer_sign_pass_rate"],
            "wheel_positive_pass_rate": vehicle_summary["wheel_positive_pass_rate"],
        }
    )

    stage_metrics_df = pd.DataFrame(stage_rows)
    stage_metrics_df.to_csv(STEPWISE_METRICS_FILE, index=False)
    stage_outputs[4].to_csv(V3_SINGLE_TIRE_FILE, index=False)
    gate_df = evaluate_validation_gates(
        stage_metrics_df=stage_metrics_df,
        stage_outputs=stage_outputs,
        vehicle_validation_df=vehicle_validation_df,
    )
    gate_df.to_csv(VALIDATION_GATES_FILE, index=False)

    summary_lines = [
        "Realism v3 stepwise metrics",
        stage_metrics_df.to_string(index=False),
        "",
        "Stage 5 maneuver details:",
        vehicle_validation_df.to_string(index=False),
        "",
        "Validation gates:",
        gate_df.to_string(index=False),
    ]
    STEPWISE_TEXT_FILE.write_text("\n".join(summary_lines), encoding="utf-8")
    write_vehicle_validation_summary(vehicle_validation_df, vehicle_summary)


if __name__ == "__main__":
    main()
