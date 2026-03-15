from __future__ import annotations

from dataclasses import replace
import math
import numpy as np
import pytest

from models.physics import TireInputs, TireModelParameters, TireState, TireThermalSimulator, celsius_to_kelvin


def _baseline_input(**overrides: float) -> TireInputs:
    payload: dict[str, float] = {
        "speed_mps": 72.0,
        "wheel_angular_speed_radps": 220.0,
        "normal_load_n": 3_800.0,
        "slip_ratio": 0.08,
        "slip_angle_rad": 0.05,
        "brake_power_w": 4_000.0,
        "ambient_temp_k": celsius_to_kelvin(30.0),
        "track_temp_k": celsius_to_kelvin(44.0),
    }
    payload.update(overrides)
    return TireInputs(**payload)


def test_core_temperature_rises_under_sustained_energy_input() -> None:
    simulator = TireThermalSimulator()
    initial_state = simulator.initial_state(ambient_temp_k=celsius_to_kelvin(30.0))
    state = initial_state
    control = _baseline_input()

    for _ in range(300):
        state = simulator.step(state, control, dt_s=0.1)

    assert state.core_temperature_k > initial_state.core_temperature_k + 0.7
    assert simulator.dynamic_pressure_pa(state, control) > simulator.dynamic_pressure_pa(
        initial_state,
        control,
    )


def test_hotter_gas_reduces_contact_patch_via_pressure_feedback() -> None:
    simulator = TireThermalSimulator()
    base = simulator.initial_state(ambient_temp_k=celsius_to_kelvin(25.0))
    control = _baseline_input(brake_power_w=0.0)

    hot = replace(base, t_gas_k=base.t_gas_k + 45.0)

    diag_base = simulator.diagnostics(base, control)
    diag_hot = simulator.diagnostics(hot, control)

    assert diag_hot.dynamic_pressure_pa > diag_base.dynamic_pressure_pa
    assert diag_hot.contact_patch_area_m2 < diag_base.contact_patch_area_m2


def test_compression_work_term_increases_gas_heating_rate() -> None:
    simulator = TireThermalSimulator()
    state = simulator.initial_state(ambient_temp_k=celsius_to_kelvin(28.0))
    no_compression = _baseline_input(
        volume_change_rate_m3ps=0.0,
        normal_load_rate_nps=0.0,
        wheel_angular_accel_radps2=0.0,
    )
    with_compression = _baseline_input(
        volume_change_rate_m3ps=-8.0e-4,
        normal_load_rate_nps=12_000.0,
        wheel_angular_accel_radps2=0.0,
    )

    rates_nominal = simulator.temperature_rates_k_per_s(state, no_compression)
    rates_compression = simulator.temperature_rates_k_per_s(state, with_compression)

    assert rates_compression["gas"] > rates_nominal["gas"]


def test_long_run_remains_finite_and_physical() -> None:
    simulator = TireThermalSimulator()
    state = simulator.initial_state(ambient_temp_k=celsius_to_kelvin(27.0))
    control = _baseline_input(brake_power_w=2_000.0)

    for _ in range(1_500):
        state = simulator.step(state, control, dt_s=0.05)

    assert all(math.isfinite(value) for value in state.as_vector())
    assert 0.0 <= state.wear <= 1.0
    assert state.t_carcass_k > celsius_to_kelvin(27.0)


def test_phase1_rr_split_conserves_rolling_resistance_power() -> None:
    params = TireModelParameters(
        use_sidewall_rr_split_model=True,
        rolling_resistance_coeff=0.012,
        rr_belt_tread_fraction=0.85,
        rr_sidewall_fraction=0.15,
    )
    simulator = TireThermalSimulator(params)
    state = simulator.initial_state(ambient_temp_k=celsius_to_kelvin(30.0))
    control = _baseline_input(speed_mps=68.0, slip_ratio=0.0, slip_angle_rad=0.0, brake_power_w=0.0)

    diag = simulator.diagnostics(state, control)

    assert diag.rolling_resistance_total_w > 0.0
    assert math.isclose(
        diag.rolling_resistance_total_w,
        diag.rolling_resistance_belt_tread_w + diag.rolling_resistance_sidewall_w,
        rel_tol=1e-9,
        abs_tol=1e-9,
    )
    assert math.isclose(diag.rolling_resistance_belt_tread_w / diag.rolling_resistance_total_w, 0.85, rel_tol=1e-9)
    assert math.isclose(diag.rolling_resistance_sidewall_w / diag.rolling_resistance_total_w, 0.15, rel_tol=1e-9)


def test_p0_rr_heating_active_without_sidewall_split_model() -> None:
    params = TireModelParameters(
        use_sidewall_rr_split_model=False,
        rolling_resistance_coeff=0.012,
    )
    simulator = TireThermalSimulator(params)
    state = simulator.initial_state(ambient_temp_k=celsius_to_kelvin(30.0))
    control = _baseline_input(speed_mps=65.0, slip_ratio=0.0, slip_angle_rad=0.0, brake_power_w=0.0)

    diag = simulator.diagnostics(state, control)

    assert diag.rolling_resistance_total_w > 0.0
    assert math.isclose(
        diag.rolling_resistance_total_w,
        diag.rolling_resistance_belt_tread_w,
        rel_tol=1e-9,
        abs_tol=1e-9,
    )
    assert math.isclose(diag.rolling_resistance_sidewall_w, 0.0, rel_tol=0.0, abs_tol=1e-12)


def test_phase1_sidewall_node_behaves_as_carcass_heat_sink() -> None:
    state = TireThermalSimulator().initial_state(ambient_temp_k=celsius_to_kelvin(30.0))
    heated_carcass_state = replace(state, t_carcass_k=state.t_carcass_k + 35.0, t_rim_k=state.t_rim_k + 20.0)
    control = _baseline_input(speed_mps=60.0, slip_ratio=0.0, slip_angle_rad=0.0, brake_power_w=0.0)

    legacy_sim = TireThermalSimulator(TireModelParameters(use_sidewall_rr_split_model=False))
    phase1_sim = TireThermalSimulator(TireModelParameters(use_sidewall_rr_split_model=True))

    rates_legacy = legacy_sim.temperature_rates_k_per_s(heated_carcass_state, control)
    rates_phase1 = phase1_sim.temperature_rates_k_per_s(heated_carcass_state, control)

    assert rates_phase1["sidewall"] > 0.0
    assert rates_phase1["carcass"] < rates_legacy["carcass"]


def test_phase6_state_vector_roundtrip_requires_full_state_dimension() -> None:
    full_state = np.array(
        [
            celsius_to_kelvin(40.0),
            celsius_to_kelvin(41.0),
            celsius_to_kelvin(42.0),
            celsius_to_kelvin(43.0),
            celsius_to_kelvin(44.0),
            celsius_to_kelvin(45.0),
            celsius_to_kelvin(46.0),
            celsius_to_kelvin(47.0),
            celsius_to_kelvin(48.0),
            0.03,
            0.02,
            0.13,
        ],
        dtype=float,
    )

    state = TireState.from_vector(full_state, time_s=2.5)

    assert math.isclose(state.t_sidewall_k, full_state[8], rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(state.kappa_dyn, full_state[9], rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(state.alpha_dyn_rad, full_state[10], rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(state.wear, full_state[11], rel_tol=0.0, abs_tol=1e-12)

    with pytest.raises(ValueError):
        TireState.from_vector(full_state[:10], time_s=1.5)


def test_phase2_slip_transient_lags_then_tracks_command() -> None:
    params = TireModelParameters(
        use_slip_transient_model=True,
        relaxation_length_long_m=10.0,
        relaxation_length_lat_m=12.0,
        slip_transition_speed_mps=4.0,
        max_slip_relaxation_rate_per_s=10.0,
    )
    simulator = TireThermalSimulator(params)
    state = simulator.initial_state(ambient_temp_k=celsius_to_kelvin(30.0))
    control = _baseline_input(
        speed_mps=65.0,
        slip_ratio=0.14,
        slip_angle_rad=0.09,
        slip_ratio_cmd=0.14,
        slip_angle_cmd_rad=0.09,
        brake_power_w=0.0,
    )

    diag_0 = simulator.diagnostics(state, control)
    assert diag_0.effective_slip_ratio < 0.14
    assert abs(diag_0.effective_slip_angle_rad) < 0.09

    state_1 = simulator.step(state, control, dt_s=0.1)
    assert 0.0 < state_1.kappa_dyn < 0.14
    assert 0.0 < state_1.alpha_dyn_rad < 0.09

    state_n = state_1
    for _ in range(80):
        state_n = simulator.step(state_n, control, dt_s=0.1)
    assert abs(state_n.kappa_dyn - 0.14) < 0.01
    assert abs(state_n.alpha_dyn_rad - 0.09) < 0.01


def test_phase2_low_speed_blends_to_algebraic_slip_without_stiffness() -> None:
    params = TireModelParameters(
        use_slip_transient_model=True,
        relaxation_length_long_m=10.0,
        relaxation_length_lat_m=12.0,
        slip_transition_speed_mps=5.0,
        max_slip_relaxation_rate_per_s=10.0,
    )
    simulator = TireThermalSimulator(params)
    state = simulator.initial_state(ambient_temp_k=celsius_to_kelvin(30.0))
    control = _baseline_input(
        speed_mps=0.3,
        slip_ratio=0.12,
        slip_angle_rad=0.08,
        slip_ratio_cmd=0.12,
        slip_angle_cmd_rad=0.08,
        brake_power_w=0.0,
    )

    rates = simulator.temperature_rates_k_per_s(state, control)
    diag = simulator.diagnostics(state, control)

    assert abs(rates["slip_ratio_dyn"]) < 0.01
    assert abs(rates["slip_angle_dyn_rad"]) < 0.01
    assert abs(diag.effective_slip_ratio - 0.12) < 0.01
    assert abs(diag.effective_slip_angle_rad - 0.08) < 0.01


def test_phase3_quasi_2d_patch_force_and_power_consistency() -> None:
    params = TireModelParameters(
        use_slip_transient_model=True,
        use_quasi_2d_patch_model=True,
        friction_heat_fraction=0.22,
        slip_transition_speed_mps=4.0,
    )
    simulator = TireThermalSimulator(params)
    state = simulator.initial_state(ambient_temp_k=celsius_to_kelvin(30.0))
    control = _baseline_input(
        speed_mps=58.0,
        normal_load_n=3900.0,
        slip_ratio=0.10,
        slip_angle_rad=0.07,
        slip_ratio_cmd=0.10,
        slip_angle_cmd_rad=0.07,
        camber_rad=-0.03,
        longitudinal_accel_mps2=-4.5,
        brake_power_w=0.0,
    )

    diag = simulator.diagnostics(state, control)

    assert diag.patch_cell_area_m2 > 0.0
    assert all(value >= 0.0 for row in diag.patch_pressure_grid_pa for value in row)
    assert all(value >= 0.0 for row in diag.patch_shear_grid_pa for value in row)
    assert all(value >= 0.0 for value in diag.zone_friction_power_w)

    total_patch_area = diag.patch_cell_area_m2 * 9.0
    assert math.isclose(total_patch_area, diag.contact_patch_area_m2, rel_tol=1e-9, abs_tol=1e-9)

    integrated_normal_force = (
        sum(sum(row) for row in diag.patch_pressure_grid_pa) * diag.patch_cell_area_m2
    )
    assert math.isclose(integrated_normal_force, control.normal_load_n, rel_tol=1e-9, abs_tol=1e-6)

    integrated_friction_power = (
        sum(sum(row) for row in diag.patch_shear_grid_pa) * diag.patch_cell_area_m2 * abs(diag.slip_speed_mps)
    )
    assert math.isclose(
        integrated_friction_power * params.friction_heat_fraction,
        sum(diag.zone_friction_power_w),
        rel_tol=1e-9,
        abs_tol=1e-9,
    )


def test_p0_friction_heat_fraction_scales_surface_heating_across_models() -> None:
    def surface_middle_rate(use_quasi_2d_patch_model: bool, friction_heat_fraction: float) -> float:
        params = TireModelParameters(
            use_quasi_2d_patch_model=use_quasi_2d_patch_model,
            friction_heat_fraction=friction_heat_fraction,
        )
        simulator = TireThermalSimulator(params)
        state = simulator.initial_state(ambient_temp_k=celsius_to_kelvin(30.0))
        control = _baseline_input(
            speed_mps=58.0,
            normal_load_n=3900.0,
            slip_ratio=0.10,
            slip_angle_rad=0.07,
            slip_ratio_cmd=0.10,
            slip_angle_cmd_rad=0.07,
            brake_power_w=0.0,
            ambient_temp_k=celsius_to_kelvin(30.0),
            track_temp_k=celsius_to_kelvin(30.0),
        )
        return simulator.temperature_rates_k_per_s(state, control)["surface_middle"]

    high_fraction = 1.0
    low_fraction = 0.25
    for use_quasi_2d_patch_model in (False, True):
        high_rate = surface_middle_rate(use_quasi_2d_patch_model, high_fraction)
        low_rate = surface_middle_rate(use_quasi_2d_patch_model, low_fraction)

        assert high_rate > 0.0
        assert low_rate > 0.0
        assert math.isclose(
            low_rate / high_rate,
            low_fraction / high_fraction,
            rel_tol=2e-3,
            abs_tol=2e-3,
        )


def test_phase4_friction_partition_normalization_and_conservation() -> None:
    params = TireModelParameters(
        use_quasi_2d_patch_model=True,
        use_friction_partition_model=True,
        friction_partition_adhesion=0.60,
        friction_partition_hysteresis=0.50,
        friction_partition_flash=0.50,
    )
    simulator = TireThermalSimulator(params)
    state = simulator.initial_state(ambient_temp_k=celsius_to_kelvin(30.0))
    control = _baseline_input(
        speed_mps=62.0,
        slip_ratio=0.09,
        slip_angle_rad=0.06,
        brake_power_w=0.0,
    )

    diag = simulator.diagnostics(state, control)
    assert diag.friction_power_total_w > 0.0
    assert math.isclose(
        diag.friction_power_total_w,
        diag.friction_power_adhesion_w + diag.friction_power_hysteresis_w + diag.friction_power_flash_w,
        rel_tol=1e-9,
        abs_tol=1e-9,
    )
    # 0.60/0.50/0.50 normalized to 0.375/0.3125/0.3125
    assert math.isclose(diag.friction_power_adhesion_w / diag.friction_power_total_w, 0.375, rel_tol=1e-9)
    assert math.isclose(diag.friction_power_hysteresis_w / diag.friction_power_total_w, 0.3125, rel_tol=1e-9)
    assert math.isclose(diag.friction_power_flash_w / diag.friction_power_total_w, 0.3125, rel_tol=1e-9)


def test_phase4_flash_partition_limits_surface_heating_response() -> None:
    baseline_params = TireModelParameters(
        use_quasi_2d_patch_model=True,
        use_friction_partition_model=False,
    )
    flash_heavy_params = TireModelParameters(
        use_quasi_2d_patch_model=True,
        use_friction_partition_model=True,
        friction_partition_adhesion=0.05,
        friction_partition_hysteresis=0.05,
        friction_partition_flash=0.90,
        flash_surface_time_constant_s=0.10,
    )

    baseline_sim = TireThermalSimulator(baseline_params)
    flash_sim = TireThermalSimulator(flash_heavy_params)
    state = baseline_sim.initial_state(ambient_temp_k=celsius_to_kelvin(30.0))
    control = _baseline_input(
        speed_mps=66.0,
        slip_ratio=0.11,
        slip_angle_rad=0.08,
        brake_power_w=0.0,
    )

    baseline_state = state
    flash_state = state
    for _ in range(180):
        baseline_state = baseline_sim.step(baseline_state, control, dt_s=0.1)
        flash_state = flash_sim.step(flash_state, control, dt_s=0.1)

    assert flash_state.t_surface_middle_k < baseline_state.t_surface_middle_k
