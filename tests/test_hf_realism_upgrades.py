from __future__ import annotations

from dataclasses import replace

import numpy as np

from models.high_fidelity import (
    HighFidelityTireInputs,
    HighFidelityTireModelParameters,
    HighFidelityTireSimulator,
    ThermalFieldSolver2D,
    WheelForceCouplingModel,
)
from models.high_fidelity.types import (
    ConstructionParameters,
    InternalCouplingParameters,
    LayerMaterialParameters,
    LayerStackParameters,
    LocalContactParameters,
)
from models.physics import celsius_to_kelvin


def _base_inputs(*, speed_mps: float = 58.0, wheel_angular_speed_radps: float = 185.0, brake_power_w: float = 0.0) -> HighFidelityTireInputs:
    return HighFidelityTireInputs(
        speed_mps=speed_mps,
        wheel_angular_speed_radps=wheel_angular_speed_radps,
        normal_load_n=3_900.0,
        slip_ratio_cmd=0.10,
        slip_angle_cmd_rad=0.05,
        ambient_temp_k=celsius_to_kelvin(28.0),
        track_temp_k=celsius_to_kelvin(42.0),
        road_surface_temp_k=celsius_to_kelvin(42.0),
        road_bulk_temp_k=celsius_to_kelvin(40.0),
        brake_power_w=brake_power_w,
    )


def _hf_params(**overrides: object) -> HighFidelityTireModelParameters:
    defaults = dict(
        use_2d_thermal_solver=True,
        no_op_thermal_step=False,
        use_reduced_patch_mechanics=True,
        use_local_temp_friction_partition=True,
        use_structural_hysteresis_model=True,
        radial_cells=8,
        theta_cells=16,
        internal_solver_dt_s=0.01,
    )
    defaults.update(overrides)
    return HighFidelityTireModelParameters(**defaults)


def test_internal_coupling_gas_response_increases_with_speed_sensitive_mixing() -> None:
    params = _hf_params(
        internal_coupling=InternalCouplingParameters(
            enabled=True,
            gas_mixing_speed_gain_w_per_k_per_radps=0.05,
        ),
    )
    sim = HighFidelityTireSimulator(params)
    hot_inner = sim._thermal_solver.initial_temperature_field(celsius_to_kelvin(28.0))
    hot_inner[0, :, :] = celsius_to_kelvin(110.0)

    base_state = replace(
        sim.initial_state(ambient_temp_k=celsius_to_kelvin(28.0)),
        thermal_field_rtw_k=hot_inner,
        thermal_field_rt_k=np.mean(hot_inner, axis=2),
    )
    low_speed = sim.step(base_state, _base_inputs(speed_mps=20.0, wheel_angular_speed_radps=70.0), dt_s=0.08)
    high_speed = sim.step(base_state, _base_inputs(speed_mps=72.0, wheel_angular_speed_radps=240.0), dt_s=0.08)

    assert high_speed.temperature_nodes_k[5] > low_speed.temperature_nodes_k[5]
    assert high_speed.last_effective_gas_inner_liner_htc_w_per_k > low_speed.last_effective_gas_inner_liner_htc_w_per_k


def test_internal_coupling_rim_temperature_rises_with_stronger_brake_to_rim_coupling() -> None:
    weak = _hf_params(
        internal_coupling=InternalCouplingParameters(enabled=True, brake_to_rim_conductance_w_per_k=45.0),
    )
    strong = _hf_params(
        internal_coupling=InternalCouplingParameters(enabled=True, brake_to_rim_conductance_w_per_k=180.0),
    )
    weak_sim = HighFidelityTireSimulator(weak)
    strong_sim = HighFidelityTireSimulator(strong)

    weak_state = weak_sim.step(
        weak_sim.initial_state(ambient_temp_k=celsius_to_kelvin(28.0)),
        _base_inputs(brake_power_w=48_000.0),
        dt_s=0.08,
    )
    strong_state = strong_sim.step(
        strong_sim.initial_state(ambient_temp_k=celsius_to_kelvin(28.0)),
        _base_inputs(brake_power_w=48_000.0),
        dt_s=0.08,
    )

    assert strong_state.temperature_nodes_k[6] > weak_state.temperature_nodes_k[6]
    assert strong_state.last_brake_disc_to_rim_heat_w > weak_state.last_brake_disc_to_rim_heat_w


def test_internal_coupling_brake_disc_heats_substantially_under_heavy_braking() -> None:
    params = _hf_params(
        internal_coupling=InternalCouplingParameters(enabled=True),
    )
    sim = HighFidelityTireSimulator(params)
    state = sim.initial_state(ambient_temp_k=celsius_to_kelvin(28.0))

    for _ in range(35):
        state = sim.step(state, _base_inputs(speed_mps=72.0, wheel_angular_speed_radps=220.0, brake_power_w=40_000.0), dt_s=0.2)

    assert state.last_brake_disc_temp_k is not None
    assert state.last_brake_disc_temp_k - celsius_to_kelvin(28.0) > 200.0


def test_heat_balance_audit_is_positive_in_heating_case() -> None:
    params = _hf_params(
        internal_coupling=InternalCouplingParameters(enabled=True),
        local_contact=LocalContactParameters(enabled=True),
        construction=ConstructionParameters(enabled=True),
    )
    sim = HighFidelityTireSimulator(params)
    state = sim.initial_state(ambient_temp_k=celsius_to_kelvin(28.0))

    for _ in range(20):
        state = sim.step(state, _base_inputs(speed_mps=60.0, wheel_angular_speed_radps=200.0), dt_s=0.1)

    assert state.last_heat_source_total_w > 0.0
    assert state.last_net_heat_to_tire_w > 0.0


def test_stronger_default_path_heats_more_than_disabled_upgrades() -> None:
    strong_params = _hf_params(
        internal_coupling=InternalCouplingParameters(enabled=True),
        local_contact=LocalContactParameters(enabled=True),
        construction=ConstructionParameters(enabled=True),
    )
    weak_params = _hf_params(
        internal_coupling=InternalCouplingParameters(enabled=False),
        local_contact=LocalContactParameters(enabled=False),
        construction=ConstructionParameters(enabled=False),
    )
    strong_sim = HighFidelityTireSimulator(strong_params)
    weak_sim = HighFidelityTireSimulator(weak_params)
    inputs = replace(
        _base_inputs(speed_mps=42.0, wheel_angular_speed_radps=140.0),
        speed_mps=58.0,
        wheel_angular_speed_radps=185.0,
        slip_ratio_cmd=0.16,
        brake_power_w=18_000.0,
    )
    strong_state = strong_sim.initial_state(ambient_temp_k=celsius_to_kelvin(28.0))
    weak_state = weak_sim.initial_state(ambient_temp_k=celsius_to_kelvin(28.0))

    for _ in range(30):
        strong_state = strong_sim.step(strong_state, inputs, dt_s=0.1)
        weak_state = weak_sim.step(weak_state, inputs, dt_s=0.1)

    strong_diag = strong_sim.diagnostics(strong_state, inputs)
    weak_diag = weak_sim.diagnostics(weak_state, inputs)
    assert strong_diag.surface_temperature_k > weak_diag.surface_temperature_k
    assert strong_diag.flash_surface_temperature_k is not None
    assert weak_diag.flash_surface_temperature_k is not None
    assert strong_diag.flash_surface_temperature_k > weak_diag.flash_surface_temperature_k


def test_local_contact_law_reduces_effective_mu_when_flash_temperature_is_too_hot() -> None:
    params = _hf_params(
        local_contact=LocalContactParameters(enabled=True),
    )
    model = WheelForceCouplingModel(params)
    inputs = _base_inputs()
    cool = model.solve(
        inputs=inputs,
        surface_temp_k=celsius_to_kelvin(88.0),
        bulk_surface_temp_k=celsius_to_kelvin(86.0),
        flash_surface_temp_k=celsius_to_kelvin(95.0),
        zone_bulk_surface_temp_k=(celsius_to_kelvin(86.0),) * 3,
        zone_flash_surface_temp_k=(celsius_to_kelvin(95.0),) * 3,
        dynamic_pressure_pa=240_000.0,
    )
    hot = model.solve(
        inputs=inputs,
        surface_temp_k=celsius_to_kelvin(112.0),
        bulk_surface_temp_k=celsius_to_kelvin(108.0),
        flash_surface_temp_k=celsius_to_kelvin(145.0),
        zone_bulk_surface_temp_k=(celsius_to_kelvin(108.0),) * 3,
        zone_flash_surface_temp_k=(celsius_to_kelvin(145.0),) * 3,
        dynamic_pressure_pa=240_000.0,
    )

    assert hot.effective_contact_temperature_k > cool.effective_contact_temperature_k
    assert hot.effective_mu < cool.effective_mu


def test_explicit_disabled_upgrade_flags_match_default_behavior() -> None:
    default_params = _hf_params()
    explicit_params = _hf_params(
        internal_coupling=InternalCouplingParameters(enabled=False),
        local_contact=LocalContactParameters(enabled=False),
        construction=ConstructionParameters(enabled=False),
    )
    default_sim = HighFidelityTireSimulator(default_params)
    explicit_sim = HighFidelityTireSimulator(explicit_params)
    inputs = _base_inputs()

    default_state = default_sim.step(default_sim.initial_state(ambient_temp_k=celsius_to_kelvin(28.0)), inputs, dt_s=0.05)
    explicit_state = explicit_sim.step(explicit_sim.initial_state(ambient_temp_k=celsius_to_kelvin(28.0)), inputs, dt_s=0.05)

    assert np.allclose(default_state.temperature_nodes_k, explicit_state.temperature_nodes_k, atol=1e-9, rtol=0.0)
    assert np.allclose(default_state.thermal_field_rtw_k, explicit_state.thermal_field_rtw_k, atol=1e-9, rtol=0.0)
    assert default_state.last_friction_to_tire_w == explicit_state.last_friction_to_tire_w


def test_construction_anisotropy_changes_layer_transport_by_width_zone() -> None:
    layer_stack = LayerStackParameters(
        tread=LayerMaterialParameters(
            thickness_m=0.012,
            volumetric_heat_capacity_j_per_m3k=1.95e6,
            k_r_w_per_mk=0.26,
            k_theta_w_per_mk=0.29,
            k_w_w_per_mk=0.22,
            shoulder_conductivity_bias=1.35,
            center_conductivity_bias=0.95,
            bead_conductivity_bias=1.15,
            cord_angle_deg=20.0,
            reinforcement_density_factor=1.2,
        ),
    )
    params = _hf_params(
        construction=ConstructionParameters(enabled=True),
        layer_stack=layer_stack,
    )
    solver = ThermalFieldSolver2D(params)
    _, _, _, k_w, layer_index = solver.layer_property_maps(wear=0.0)
    tread_mask = layer_index == 0
    shoulder_left = float(np.mean(k_w[:, :, 0][tread_mask[:, :, 0]]))
    center = float(np.mean(k_w[:, :, 1][tread_mask[:, :, 1]]))
    shoulder_right = float(np.mean(k_w[:, :, 2][tread_mask[:, :, 2]]))

    assert shoulder_left > center
    assert shoulder_right > center


def test_neutral_construction_modifiers_reproduce_default_layer_properties() -> None:
    default_solver = ThermalFieldSolver2D(_hf_params())
    neutral_solver = ThermalFieldSolver2D(
        _hf_params(construction=ConstructionParameters(enabled=True))
    )

    default_maps = default_solver.layer_property_maps(wear=0.15)
    neutral_maps = neutral_solver.layer_property_maps(wear=0.15)
    for default_map, neutral_map in zip(default_maps[:4], neutral_maps[:4]):
        assert np.allclose(default_map, neutral_map, atol=1e-12, rtol=0.0)
