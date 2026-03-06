from __future__ import annotations

import numpy as np

from ..physics import TireInputs, TireState
from .types import HighFidelityTireInputs, HighFidelityTireState


def inputs_from_legacy(inputs: TireInputs) -> HighFidelityTireInputs:
    """Map legacy tire inputs onto the high-fidelity input schema."""
    return HighFidelityTireInputs(
        speed_mps=inputs.speed_mps,
        wheel_angular_speed_radps=inputs.wheel_angular_speed_radps,
        normal_load_n=inputs.normal_load_n,
        slip_ratio_cmd=inputs.slip_ratio if inputs.slip_ratio_cmd is None else inputs.slip_ratio_cmd,
        slip_angle_cmd_rad=(
            inputs.slip_angle_rad if inputs.slip_angle_cmd_rad is None else inputs.slip_angle_cmd_rad
        ),
        brake_power_w=inputs.brake_power_w,
        ambient_temp_k=inputs.ambient_temp_k,
        track_temp_k=inputs.track_temp_k,
        road_surface_temp_k=inputs.track_temp_k,
    )


def state_from_legacy(state: TireState) -> HighFidelityTireState:
    """Map legacy tire state onto the high-fidelity state schema."""
    nodes = np.array(
        [
            state.t_surface_inner_k,
            state.t_surface_middle_k,
            state.t_surface_outer_k,
            state.t_belt_k,
            state.t_carcass_k,
            state.t_gas_k,
            state.t_rim_k,
            state.t_brake_k,
            state.t_sidewall_k,
        ],
        dtype=float,
    )
    return HighFidelityTireState(
        temperature_nodes_k=nodes,
        wear=state.wear,
        time_s=state.time_s,
    )


def state_to_legacy(state: HighFidelityTireState) -> TireState:
    """Map high-fidelity state back to the legacy tire state schema."""
    if state.temperature_nodes_k.shape[0] < 9:
        msg = f"Expected at least 9 thermal nodes, got {state.temperature_nodes_k.shape[0]}"
        raise ValueError(msg)

    return TireState(
        t_surface_inner_k=float(state.temperature_nodes_k[0]),
        t_surface_middle_k=float(state.temperature_nodes_k[1]),
        t_surface_outer_k=float(state.temperature_nodes_k[2]),
        t_belt_k=float(state.temperature_nodes_k[3]),
        t_carcass_k=float(state.temperature_nodes_k[4]),
        t_gas_k=float(state.temperature_nodes_k[5]),
        t_rim_k=float(state.temperature_nodes_k[6]),
        t_brake_k=float(state.temperature_nodes_k[7]),
        t_sidewall_k=float(state.temperature_nodes_k[8]),
        kappa_dyn=0.0,
        alpha_dyn_rad=0.0,
        wear=state.wear,
        time_s=state.time_s,
    )

