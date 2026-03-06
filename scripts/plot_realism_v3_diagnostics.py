from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from models.physics import TireModelParameters, celsius_to_kelvin  # noqa: E402
from models.vehicle_thermal import (  # noqa: E402
    VehicleInputs,
    VehicleParameters,
    VehicleThermalSimulator,
)

RESULTS_DIR = ROOT / "reports" / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
STEPWISE_METRICS_FILE = RESULTS_DIR / "realism_v3_stepwise_metrics.csv"
VALIDATION_GATES_FILE = RESULTS_DIR / "realism_v3_validation_gates.csv"
STAGE4_SINGLE_TIRE_FILE = RESULTS_DIR / "core_temp_simulation_realism_v3.csv"


@dataclass(frozen=True)
class SyntheticManeuver:
    name: str
    speed_mps: float
    ax_mps2: float
    ay_mps2: float


def _stage4_with_rotating_gas_parameters() -> TireModelParameters:
    return TireModelParameters(
        use_gauge_patch_model=True,
        use_combined_slip_model=False,
        use_zone_lateral_conduction=False,
        use_temperature_dependent_properties=False,
        use_rotating_internal_gas_model=True,
        use_alignment_zone_effects=False,
        use_sidewall_rr_split_model=True,
        use_slip_transient_model=True,
        use_quasi_2d_patch_model=True,
        use_friction_partition_model=True,
    )


def collect_stage5_h_int_samples() -> pd.DataFrame:
    vehicle_params = VehicleParameters()
    tire_params = _stage4_with_rotating_gas_parameters()
    simulator = VehicleThermalSimulator(
        parameters=vehicle_params,
        tire_parameters_by_wheel={wheel: tire_params for wheel in ("FL", "FR", "RL", "RR")},
    )

    maneuvers = [
        SyntheticManeuver(name="steady_corner", speed_mps=55.0, ax_mps2=0.0, ay_mps2=7.0),
        SyntheticManeuver(name="brake_in_line", speed_mps=62.0, ax_mps2=-7.5, ay_mps2=0.0),
        SyntheticManeuver(name="accel_in_line", speed_mps=48.0, ax_mps2=4.0, ay_mps2=0.0),
        SyntheticManeuver(name="combined_brake_corner", speed_mps=52.0, ax_mps2=-5.5, ay_mps2=5.2),
    ]

    rows: list[dict[str, float | str | int]] = []
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

        for step in range(250):
            state = simulator.step(state, control, dt_s=0.1)
            diag = simulator.diagnostics(state, control)
            for wheel, h_int in diag.wheel_internal_htc_w_m2k.items():
                rows.append(
                    {
                        "maneuver": maneuver.name,
                        "wheel": wheel,
                        "step": step,
                        "h_int_w_m2k": float(h_int),
                    }
                )

    return pd.DataFrame(rows)


def build_stage_overview_figure(stage_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Realism v3 Stage Metrics Overview", fontsize=14)

    stage_single = stage_df[stage_df["stage"] <= 4]
    stage_all = stage_df.copy()

    ax = axes[0, 0]
    ax.plot(stage_single["stage"], stage_single["patch_pressure_corr"], marker="o")
    ax.axhline(-0.40, color="crimson", linestyle="--", linewidth=1.2, label="target <= -0.40")
    ax.set_title("Stage Patch-Pressure Correlation")
    ax.set_xlabel("stage")
    ax.set_ylabel("corr(contact_patch, pressure)")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    ax = axes[0, 1]
    ax.plot(stage_all["stage"], stage_all["h_int_clip_fraction"], marker="o")
    ax.axhline(0.05, color="crimson", linestyle="--", linewidth=1.2, label="target < 0.05")
    ax.set_title("Internal HTC Clip Fraction")
    ax.set_xlabel("stage")
    ax.set_ylabel("clip fraction")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)

    ax = axes[1, 0]
    ax.plot(stage_single["stage"], stage_single["core_end_c"], marker="o")
    ax.set_title("Single-Tire Core End Temperature")
    ax.set_xlabel("stage")
    ax.set_ylabel("core_end_c")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(stage_single["stage"], stage_single["pressure_end_bar_g"], marker="o")
    ax.set_title("Single-Tire Pressure End")
    ax.set_xlabel("stage")
    ax.set_ylabel("pressure_end_bar_g")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "realism_v3_stage_metrics_overview.png", dpi=180)
    plt.close(fig)


def build_stage4_patch_pressure_scatter(stage4_df: pd.DataFrame) -> None:
    x = stage4_df["pressure_bar_g"].to_numpy(dtype=float)
    y = stage4_df["contact_patch_m2"].to_numpy(dtype=float)
    corr = float(np.corrcoef(x, y)[0, 1])

    m, b = np.polyfit(x, y, deg=1)
    xx = np.linspace(float(np.min(x)), float(np.max(x)), 120)
    yy = m * xx + b

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(x, y, s=10, alpha=0.35, label="samples")
    ax.plot(xx, yy, color="crimson", linewidth=1.6, label=f"linear fit (corr={corr:.4f})")
    ax.set_title("Stage 4: Contact Patch vs Pressure")
    ax.set_xlabel("pressure_bar_g")
    ax.set_ylabel("contact_patch_m2")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "realism_v3_stage4_patch_vs_pressure.png", dpi=180)
    plt.close(fig)


def build_gate_values_figure(gate_df: pd.DataFrame) -> None:
    numeric = gate_df[gate_df["value"].notna()].copy()
    labels = numeric["gate"].to_list()
    values = numeric["value"].to_numpy(dtype=float)
    thresholds = np.array([4.0, 0.12, -0.40, 0.05, 0.5], dtype=float)
    threshold_labels = [
        "abs <= 4.0",
        "abs <= 0.12",
        "<= -0.40",
        "< 0.05",
        "< 0.5",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    y_pos = np.arange(len(labels))
    colors = ["#2a9d8f" if bool(v) else "#e76f51" for v in gate_df[gate_df["value"].notna()]["passed"]]
    ax.barh(y_pos, values, color=colors, alpha=0.85)
    ax.set_yticks(y_pos, labels)
    ax.set_title("Validation Gate Values")
    ax.grid(axis="x", alpha=0.3)

    ax2 = axes[1]
    normalized_margin = np.array(
        [
            (4.0 - abs(values[0])) / 4.0,
            (0.12 - abs(values[1])) / 0.12,
            (-0.40 - values[2]) / 0.40,
            (0.05 - values[3]) / 0.05,
            (0.5 - values[4]) / 0.5,
        ],
        dtype=float,
    )
    ax2.barh(y_pos, normalized_margin, color=["#2a9d8f" if m >= 0 else "#e76f51" for m in normalized_margin])
    ax2.axvline(0.0, color="black", linewidth=1.0)
    ax2.set_yticks(y_pos, threshold_labels)
    ax2.set_title("Gate Margin (normalized, >0 means pass)")
    ax2.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "realism_v3_validation_gate_values.png", dpi=180)
    plt.close(fig)


def build_stage5_h_int_figure(h_int_df: pd.DataFrame) -> None:
    maneuvers = list(dict.fromkeys(h_int_df["maneuver"].to_list()))
    data = [h_int_df.loc[h_int_df["maneuver"] == m, "h_int_w_m2k"].to_numpy(dtype=float) for m in maneuvers]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    bp = ax.boxplot(data, tick_labels=maneuvers, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#90caf9")
        patch.set_alpha(0.8)
    ax.set_title("Phase 5 Internal HTC by Maneuver")
    ax.set_ylabel("h_int_w_m2k")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.hist(h_int_df["h_int_w_m2k"].to_numpy(dtype=float), bins=30, color="#8ecae6", edgecolor="white")
    ax.set_title("Phase 5 Internal HTC Distribution (all wheels)")
    ax.set_xlabel("h_int_w_m2k")
    ax.set_ylabel("count")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "realism_v3_stage5_h_int_distribution.png", dpi=180)
    plt.close(fig)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    stage_df = pd.read_csv(STEPWISE_METRICS_FILE)
    gate_df = pd.read_csv(VALIDATION_GATES_FILE)
    stage4_df = pd.read_csv(STAGE4_SINGLE_TIRE_FILE)
    h_int_df = collect_stage5_h_int_samples()
    h_int_df.to_csv(FIGURES_DIR / "realism_v3_stage5_h_int_samples.csv", index=False)

    build_stage_overview_figure(stage_df)
    build_stage4_patch_pressure_scatter(stage4_df)
    build_gate_values_figure(gate_df)
    build_stage5_h_int_figure(h_int_df)


if __name__ == "__main__":
    main()
