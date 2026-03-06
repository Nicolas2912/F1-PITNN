from __future__ import annotations

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
INPUT_PROFILE = ROOT / "reports" / "results" / "core_temp_simulation_calibrated.csv"
OUTPUT_DIR = ROOT / "reports" / "results"


def stage_parameters(stage: int) -> TireModelParameters:
    return TireModelParameters(
        use_gauge_patch_model=stage >= 1,
        use_combined_slip_model=stage >= 2,
        use_zone_lateral_conduction=stage >= 3,
        use_temperature_dependent_properties=stage >= 4,
        use_rotating_internal_gas_model=stage >= 5,
        use_alignment_zone_effects=stage >= 6,
    )


def _alignment_features(df: pd.DataFrame, dt_s: float) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    longitudinal_accel = (
        df["speed_mps"].diff().fillna(0.0) / df["time_s"].diff().replace(0.0, np.nan).fillna(dt_s)
    )
    centered_load = df["normal_load_n"] - df["normal_load_n"].rolling(25, min_periods=1).mean()
    lateral_accel_proxy = np.sign(df["slip_angle_rad"]) * centered_load / df["normal_load_n"].mean() * 12.0
    camber = pd.Series(np.full(len(df), -2.8 * np.pi / 180.0), index=df.index)
    toe = pd.Series(np.full(len(df), 0.12 * np.pi / 180.0), index=df.index)
    return camber, toe, lateral_accel_proxy, longitudinal_accel


def run_stage(df_profile: pd.DataFrame, stage: int) -> pd.DataFrame:
    params = stage_parameters(stage)
    simulator = TireThermalSimulator(params)

    dt_s = float(df_profile["time_s"].diff().dropna().median())
    first = df_profile.iloc[0]
    state = simulator.initial_state(
        ambient_temp_k=celsius_to_kelvin(float(first["t_carcass_core_c"])),
        brake_temp_k=celsius_to_kelvin(float(first["t_brake_c"])),
        wear=float(first["wear"]),
    )

    camber, toe, lateral_accel, longitudinal_accel = _alignment_features(df_profile, dt_s)

    rows: list[dict[str, float]] = []
    for idx, row in df_profile.iterrows():
        if stage >= 6:
            camber_i = float(camber.loc[idx])
            toe_i = float(toe.loc[idx])
            lat_i = float(lateral_accel.loc[idx])
            long_i = float(longitudinal_accel.loc[idx])
        else:
            camber_i = 0.0
            toe_i = 0.0
            lat_i = 0.0
            long_i = 0.0

        inputs = TireInputs(
            speed_mps=float(row["speed_mps"]),
            wheel_angular_speed_radps=float(row["speed_mps"]) * (1.0 + float(row["slip_ratio"])) / 0.33,
            normal_load_n=float(row["normal_load_n"]),
            slip_ratio=float(row["slip_ratio"]),
            slip_angle_rad=float(row["slip_angle_rad"]),
            brake_power_w=float(row["brake_power_w"]),
            ambient_temp_k=celsius_to_kelvin(28.0),
            track_temp_k=celsius_to_kelvin(44.0),
            camber_rad=camber_i,
            toe_rad=toe_i,
            lateral_accel_mps2=lat_i,
            longitudinal_accel_mps2=long_i,
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


def metrics(df: pd.DataFrame) -> dict[str, float]:
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


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    profile = pd.read_csv(INPUT_PROFILE)
    stage_rows: list[dict[str, float | str]] = []
    stage_outputs: dict[int, pd.DataFrame] = {}

    for stage in range(0, 7):
        stage_df = run_stage(profile, stage=stage)
        stage_outputs[stage] = stage_df
        stage_metric = metrics(stage_df)
        stage_rows.append(
            {
                "stage": stage,
                "label": f"stage_{stage}",
                **stage_metric,
            }
        )

    stage_metrics_df = pd.DataFrame(stage_rows)
    stage_metrics_df.to_csv(OUTPUT_DIR / "realism_stepwise_metrics.csv", index=False)
    stage_outputs[6].to_csv(OUTPUT_DIR / "core_temp_simulation_realism_v2.csv", index=False)

    summary_lines = ["Stepwise realism metrics", stage_metrics_df.to_string(index=False)]
    (OUTPUT_DIR / "realism_stepwise_metrics.txt").write_text("\n".join(summary_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
