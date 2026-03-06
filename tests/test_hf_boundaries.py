from __future__ import annotations

from dataclasses import replace
import math

import numpy as np

from models.high_fidelity import (
    BoundaryConditionModel,
    HighFidelityBoundaryParameters,
    HighFidelityTireInputs,
    HighFidelityTireModelParameters,
    HighFidelityTireSimulator,
)
from models.physics import celsius_to_kelvin


def _inputs(
    *,
    road_surface_temp_k: float,
    road_bulk_temp_k: float | None = None,
) -> HighFidelityTireInputs:
    return HighFidelityTireInputs(
        speed_mps=60.0,
        wheel_angular_speed_radps=190.0,
        normal_load_n=3_850.0,
        slip_ratio_cmd=0.10,
        slip_angle_cmd_rad=0.06,
        ambient_temp_k=celsius_to_kelvin(30.0),
        track_temp_k=road_surface_temp_k,
        road_surface_temp_k=road_surface_temp_k,
        road_bulk_temp_k=road_surface_temp_k if road_bulk_temp_k is None else road_bulk_temp_k,
    )


def _simulator(
    *,
    boundary: HighFidelityBoundaryParameters,
) -> HighFidelityTireSimulator:
    params = HighFidelityTireModelParameters(
        use_2d_thermal_solver=True,
        no_op_thermal_step=False,
        radial_cells=10,
        theta_cells=24,
        internal_solver_dt_s=0.01,
        boundary=boundary,
    )
    return HighFidelityTireSimulator(params)


def test_p4_partition_conserves_contact_friction_power() -> None:
    model = BoundaryConditionModel(HighFidelityBoundaryParameters(eta_tire=0.64))
    total_power = 18_500.0
    tire_power, road_power = model.partition_friction_power(
        total_friction_power_w=total_power
    )

    assert math.isclose(tire_power + road_power, total_power, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(tire_power, total_power * 0.64, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(road_power, total_power * 0.36, rel_tol=0.0, abs_tol=1e-12)


def test_p4_contact_and_rim_flux_signs_are_consistent() -> None:
    model = BoundaryConditionModel(HighFidelityBoundaryParameters())

    road_flux_hot_tire = model.road_conduction_power_w(
        tire_surface_temp_k=celsius_to_kelvin(85.0),
        road_surface_temp_k=celsius_to_kelvin(40.0),
    )
    road_flux_hot_road = model.road_conduction_power_w(
        tire_surface_temp_k=celsius_to_kelvin(40.0),
        road_surface_temp_k=celsius_to_kelvin(85.0),
    )
    assert road_flux_hot_tire > 0.0
    assert road_flux_hot_road < 0.0

    rim_flux_hot_tire, h_c_low = model.rim_conduction_power_w(
        tire_inner_temp_k=celsius_to_kelvin(80.0),
        rim_temp_k=celsius_to_kelvin(40.0),
        normal_load_n=2_500.0,
    )
    rim_flux_hot_rim, h_c_high = model.rim_conduction_power_w(
        tire_inner_temp_k=celsius_to_kelvin(40.0),
        rim_temp_k=celsius_to_kelvin(80.0),
        normal_load_n=5_500.0,
    )
    assert rim_flux_hot_tire > 0.0
    assert rim_flux_hot_rim < 0.0
    assert h_c_high > h_c_low


def test_p4_sensitivity_trends_for_eta_hcp_hbead_and_road_temperature() -> None:
    base_boundary = HighFidelityBoundaryParameters(
        eta_tire=0.60,
        h_cp_w_per_m2k=1_600.0,
        h_c_bead_w_per_m2k=900.0,
    )
    sim_low_eta = _simulator(boundary=base_boundary)
    sim_high_eta = _simulator(
        boundary=replace(base_boundary, eta_tire=0.90)
    )
    control = _inputs(road_surface_temp_k=celsius_to_kelvin(20.0))

    low_eta_state = sim_low_eta.step(
        sim_low_eta.initial_state(ambient_temp_k=celsius_to_kelvin(30.0)),
        control,
        dt_s=0.05,
    )
    high_eta_state = sim_high_eta.step(
        sim_high_eta.initial_state(ambient_temp_k=celsius_to_kelvin(30.0)),
        control,
        dt_s=0.05,
    )
    low_eta_diag = sim_low_eta.diagnostics(low_eta_state, control)
    high_eta_diag = sim_high_eta.diagnostics(high_eta_state, control)
    assert high_eta_diag.friction_power_tire_w > low_eta_diag.friction_power_tire_w
    assert high_eta_diag.friction_power_road_w < low_eta_diag.friction_power_road_w

    sim_low_hcp = _simulator(boundary=replace(base_boundary, h_cp_w_per_m2k=800.0))
    sim_high_hcp = _simulator(boundary=replace(base_boundary, h_cp_w_per_m2k=3_200.0))
    low_hcp_state = sim_low_hcp.step(
        sim_low_hcp.initial_state(ambient_temp_k=celsius_to_kelvin(30.0)),
        control,
        dt_s=0.05,
    )
    high_hcp_state = sim_high_hcp.step(
        sim_high_hcp.initial_state(ambient_temp_k=celsius_to_kelvin(30.0)),
        control,
        dt_s=0.05,
    )
    low_hcp_diag = sim_low_hcp.diagnostics(low_hcp_state, control)
    high_hcp_diag = sim_high_hcp.diagnostics(high_hcp_state, control)
    assert high_hcp_diag.road_conduction_w > low_hcp_diag.road_conduction_w

    sim_low_hbead = _simulator(boundary=replace(base_boundary, h_c_bead_w_per_m2k=400.0))
    sim_high_hbead = _simulator(boundary=replace(base_boundary, h_c_bead_w_per_m2k=1_800.0))
    low_hbead_initial = sim_low_hbead.initial_state(ambient_temp_k=celsius_to_kelvin(30.0))
    high_hbead_initial = sim_high_hbead.initial_state(ambient_temp_k=celsius_to_kelvin(30.0))
    low_nodes = low_hbead_initial.temperature_nodes_k.copy()
    high_nodes = high_hbead_initial.temperature_nodes_k.copy()
    low_nodes[6] = celsius_to_kelvin(20.0)
    high_nodes[6] = celsius_to_kelvin(20.0)
    low_hbead_initial = replace(low_hbead_initial, temperature_nodes_k=low_nodes)
    high_hbead_initial = replace(high_hbead_initial, temperature_nodes_k=high_nodes)

    low_hbead_state = sim_low_hbead.step(low_hbead_initial, control, dt_s=0.05)
    high_hbead_state = sim_high_hbead.step(high_hbead_initial, control, dt_s=0.05)
    low_hbead_diag = sim_low_hbead.diagnostics(low_hbead_state, control)
    high_hbead_diag = sim_high_hbead.diagnostics(high_hbead_state, control)
    assert high_hbead_diag.rim_conduction_w > low_hbead_diag.rim_conduction_w

    sim_road_temp = _simulator(boundary=base_boundary)
    cold_road = _inputs(road_surface_temp_k=celsius_to_kelvin(20.0))
    hot_road = _inputs(road_surface_temp_k=celsius_to_kelvin(55.0))

    cold_state = sim_road_temp.step(
        sim_road_temp.initial_state(ambient_temp_k=celsius_to_kelvin(30.0)),
        cold_road,
        dt_s=0.05,
    )
    hot_state = sim_road_temp.step(
        sim_road_temp.initial_state(ambient_temp_k=celsius_to_kelvin(30.0)),
        hot_road,
        dt_s=0.05,
    )
    cold_diag = sim_road_temp.diagnostics(cold_state, cold_road)
    hot_diag = sim_road_temp.diagnostics(hot_state, hot_road)
    assert cold_diag.road_conduction_w > hot_diag.road_conduction_w
    assert cold_state.road_surface_temp_k is not None
    assert hot_state.road_surface_temp_k is not None
    assert np.isfinite(cold_state.road_surface_temp_k)
    assert np.isfinite(hot_state.road_surface_temp_k)
