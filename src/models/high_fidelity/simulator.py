from __future__ import annotations

from dataclasses import replace
from typing import Iterable

import numpy as np

from ..physics import celsius_to_kelvin
from .boundary_conditions import BoundaryConditionModel, BoundaryState
from .materials import ViscoelasticMaterialModel
from .thermal_solver import ThermalFieldSolver2D
from .types import (
    HighFidelityTireDiagnostics,
    HighFidelityTireInputs,
    HighFidelityTireModelParameters,
    HighFidelityTireState,
)
from .wheel_coupling import WheelCouplingResult, WheelForceCouplingModel


class HighFidelityTireSimulator:
    """
    High-fidelity tire simulator skeleton.

    P1 keeps an additive API skeleton.
    P3 optionally enables a 2D r-theta thermal field step when
    `use_2d_thermal_solver=True` and `no_op_thermal_step=False`.
    """

    def __init__(self, parameters: HighFidelityTireModelParameters | None = None) -> None:
        self.parameters = (
            parameters if parameters is not None else HighFidelityTireModelParameters()
        )
        self._materials = ViscoelasticMaterialModel(self.parameters)
        self._thermal_solver = ThermalFieldSolver2D(self.parameters)
        self._boundary_model = BoundaryConditionModel(self.parameters.boundary)
        self._wheel_coupling = WheelForceCouplingModel(self.parameters)

    def initial_state(
        self,
        *,
        ambient_temp_k: float = celsius_to_kelvin(25.0),
        wear: float = 0.0,
    ) -> HighFidelityTireState:
        nodes = np.full(self.parameters.thermal_node_count, float(ambient_temp_k), dtype=float)
        thermal_field = (
            self._thermal_solver.initial_temperature_field(ambient_temp_k)
            if self.parameters.use_2d_thermal_solver
            else None
        )
        return HighFidelityTireState(
            temperature_nodes_k=nodes,
            thermal_field_rt_k=thermal_field,
            road_surface_temp_k=None,
            road_subsurface_temp_k=None,
            wear=float(np.clip(wear, 0.0, 1.0)),
            last_energy_residual_pct=0.0,
            last_solver_substeps=0,
            last_friction_total_w=0.0,
            last_friction_to_tire_w=0.0,
            last_friction_to_road_w=0.0,
            last_road_conduction_w=0.0,
            last_rim_conduction_w=0.0,
            last_brake_heat_to_tire_w=0.0,
            last_brake_heat_to_rim_w=0.0,
            last_effective_bead_htc_w_per_m2k=0.0,
            last_effective_slip_ratio=0.0,
            last_effective_slip_angle_rad=0.0,
            last_longitudinal_force_n=0.0,
            last_lateral_force_n=0.0,
            last_torque_residual_nm=0.0,
            last_lateral_force_residual_n=0.0,
            last_coupling_iterations=0,
            last_coupling_converged=False,
            time_s=0.0,
        )

    def diagnostics(
        self,
        state: HighFidelityTireState,
        inputs: HighFidelityTireInputs,
    ) -> HighFidelityTireDiagnostics:
        effective_inputs = replace(
            inputs,
            slip_ratio_cmd=state.last_effective_slip_ratio,
            slip_angle_cmd_rad=state.last_effective_slip_angle_rad,
        )
        frequency_hz, loss_modulus_pa, q_hyst_w_per_m3 = self._materials.hysteresis_source_summary(
            temperature_k=state.core_temperature_k,
            inputs=effective_inputs,
        )
        energy_source_total_w = q_hyst_w_per_m3 * max(self.parameters.hysteresis_active_volume_m3, 0.0)
        if state.thermal_field_rt_k is not None:
            surface_temperature_k = float(np.mean(state.thermal_field_rt_k[-1, :]))
            mean_temperature_k = float(np.mean(state.thermal_field_rt_k))
            thermal_grid_shape: tuple[int, int] | None = (
                int(state.thermal_field_rt_k.shape[0]),
                int(state.thermal_field_rt_k.shape[1]),
            )
        else:
            surface_temperature_k = float(np.mean(state.temperature_nodes_k[:3]))
            mean_temperature_k = float(np.mean(state.temperature_nodes_k))
            thermal_grid_shape = None
        return HighFidelityTireDiagnostics(
            core_temperature_k=state.core_temperature_k,
            core_temperature_c=state.core_temperature_c,
            surface_temperature_k=surface_temperature_k,
            mean_temperature_k=mean_temperature_k,
            no_op_mode=self.parameters.no_op_thermal_step,
            excitation_frequency_hz=frequency_hz,
            loss_modulus_pa=loss_modulus_pa,
            hysteresis_power_density_w_per_m3=q_hyst_w_per_m3,
            energy_source_total_w=energy_source_total_w,
            energy_residual_pct=state.last_energy_residual_pct,
            solver_substeps=state.last_solver_substeps,
            thermal_grid_shape=thermal_grid_shape,
            road_surface_temp_k=state.road_surface_temp_k,
            road_subsurface_temp_k=state.road_subsurface_temp_k,
            friction_power_total_w=state.last_friction_total_w,
            friction_power_tire_w=state.last_friction_to_tire_w,
            friction_power_road_w=state.last_friction_to_road_w,
            road_conduction_w=state.last_road_conduction_w,
            rim_conduction_w=state.last_rim_conduction_w,
            brake_heat_to_tire_w=state.last_brake_heat_to_tire_w,
            brake_heat_to_rim_w=state.last_brake_heat_to_rim_w,
            effective_bead_htc_w_per_m2k=state.last_effective_bead_htc_w_per_m2k,
            effective_slip_ratio=state.last_effective_slip_ratio,
            effective_slip_angle_rad=state.last_effective_slip_angle_rad,
            longitudinal_force_n=state.last_longitudinal_force_n,
            lateral_force_n=state.last_lateral_force_n,
            torque_residual_nm=state.last_torque_residual_nm,
            lateral_force_residual_n=state.last_lateral_force_residual_n,
            coupling_iterations=state.last_coupling_iterations,
            coupling_converged=state.last_coupling_converged,
        )

    def step(
        self,
        state: HighFidelityTireState,
        _inputs: HighFidelityTireInputs,
        dt_s: float,
    ) -> HighFidelityTireState:
        if dt_s <= 0.0:
            msg = f"dt_s must be positive, got {dt_s}"
            raise ValueError(msg)

        y_next = state.temperature_nodes_k.copy()
        thermal_field_next = state.thermal_field_rt_k
        road_state = self._resolve_road_state(state=state, inputs=_inputs)
        coupling_result = self._resolved_wheel_coupling(
            temperature_nodes_k=y_next,
            thermal_field_rt_k=thermal_field_next,
            inputs=_inputs,
        )
        effective_inputs = replace(
            _inputs,
            slip_ratio_cmd=coupling_result.effective_slip_ratio,
            slip_angle_cmd_rad=coupling_result.effective_slip_angle_rad,
        )
        last_energy_residual_pct = 0.0
        last_solver_substeps = 0
        last_friction_total_w = 0.0
        last_friction_to_tire_w = 0.0
        last_friction_to_road_w = 0.0
        last_road_conduction_w = 0.0
        last_rim_conduction_w = 0.0
        last_brake_heat_to_tire_w = 0.0
        last_brake_heat_to_rim_w = 0.0
        last_effective_bead_htc = 0.0
        if self.parameters.use_2d_thermal_solver and not self.parameters.no_op_thermal_step:
            thermal_field = thermal_field_next
            if thermal_field is None:
                thermal_field = self._thermal_solver.initial_temperature_field(float(np.mean(y_next)))

            tire_surface_temp_k = float(np.mean(thermal_field[-1, :]))
            tire_inner_temp_k = float(np.mean(thermal_field[0, :]))
            rim_temp_k = float(y_next[6])

            last_friction_total_w = coupling_result.friction_power_w
            (
                last_friction_to_tire_w,
                last_friction_to_road_w,
            ) = self._boundary_model.partition_friction_power(
                total_friction_power_w=last_friction_total_w
            )
            last_road_conduction_w = self._boundary_model.road_conduction_power_w(
                tire_surface_temp_k=tire_surface_temp_k,
                road_surface_temp_k=road_state.road_surface_temp_k,
            )
            (
                last_rim_conduction_w,
                last_effective_bead_htc,
            ) = self._boundary_model.rim_conduction_power_w(
                tire_inner_temp_k=tire_inner_temp_k,
                rim_temp_k=rim_temp_k,
                normal_load_n=_inputs.normal_load_n,
            )
            last_brake_heat_to_tire_w, last_brake_heat_to_rim_w = self._brake_heat_flows(
                brake_power_w=effective_inputs.brake_power_w
            )

            _, _, q_hyst_w_per_m3 = self._materials.hysteresis_source_summary(
                temperature_k=state.core_temperature_k,
                inputs=effective_inputs,
            )
            boundary_source = self._boundary_source_field_w_per_m3(
                thermal_field_rt_k=thermal_field,
                friction_to_tire_w=last_friction_to_tire_w,
                road_conduction_w=last_road_conduction_w,
                rim_conduction_w=last_rim_conduction_w,
                brake_heat_to_tire_w=last_brake_heat_to_tire_w,
                wheel_angular_speed_radps=effective_inputs.wheel_angular_speed_radps,
                time_s=state.time_s,
            )
            solver_result = self._thermal_solver.step(
                temperature_field_rt_k=thermal_field,
                inputs=effective_inputs,
                time_s=state.time_s,
                dt_s=dt_s,
                volumetric_source_w_per_m3=q_hyst_w_per_m3,
                extra_source_w_per_m3=boundary_source,
            )
            thermal_field_next = solver_result.temperature_field_rt_k
            y_next = self._temperature_nodes_from_field(thermal_field_next, y_next, dt_s=dt_s)
            y_next[6] = self._boundary_model.step_rim_temperature(
                rim_temp_k=float(y_next[6]),
                ambient_temp_k=_inputs.ambient_temp_k,
                heat_input_w=last_rim_conduction_w + last_brake_heat_to_rim_w,
                dt_s=dt_s,
            )
            road_state = self._boundary_model.step_road_slab(
                state=road_state,
                dt_s=dt_s,
                heat_input_surface_w=last_friction_to_road_w + last_road_conduction_w,
                road_bulk_temp_k=effective_inputs.road_bulk_temp_k,
            )
            last_energy_residual_pct = solver_result.energy_residual_pct
            last_solver_substeps = solver_result.substeps
        elif self.parameters.use_2d_thermal_solver and thermal_field_next is None:
            thermal_field_next = self._thermal_solver.initial_temperature_field(float(np.mean(y_next)))

        y_next = np.clip(
            y_next,
            self.parameters.minimum_temperature_k,
            self.parameters.maximum_temperature_k,
        )
        return replace(
            state,
            temperature_nodes_k=y_next,
            thermal_field_rt_k=thermal_field_next,
            road_surface_temp_k=road_state.road_surface_temp_k,
            road_subsurface_temp_k=road_state.road_subsurface_temp_k,
            wear=float(np.clip(state.wear, 0.0, 1.0)),
            last_energy_residual_pct=last_energy_residual_pct,
            last_solver_substeps=last_solver_substeps,
            last_friction_total_w=last_friction_total_w,
            last_friction_to_tire_w=last_friction_to_tire_w,
            last_friction_to_road_w=last_friction_to_road_w,
            last_road_conduction_w=last_road_conduction_w,
            last_rim_conduction_w=last_rim_conduction_w,
            last_brake_heat_to_tire_w=last_brake_heat_to_tire_w,
            last_brake_heat_to_rim_w=last_brake_heat_to_rim_w,
            last_effective_bead_htc_w_per_m2k=last_effective_bead_htc,
            last_effective_slip_ratio=coupling_result.effective_slip_ratio,
            last_effective_slip_angle_rad=coupling_result.effective_slip_angle_rad,
            last_longitudinal_force_n=coupling_result.longitudinal_force_n,
            last_lateral_force_n=coupling_result.lateral_force_n,
            last_torque_residual_nm=coupling_result.torque_residual_nm,
            last_lateral_force_residual_n=coupling_result.lateral_force_residual_n,
            last_coupling_iterations=coupling_result.iterations,
            last_coupling_converged=coupling_result.converged,
            time_s=state.time_s + dt_s,
        )

    def _resolved_wheel_coupling(
        self,
        *,
        temperature_nodes_k: np.ndarray,
        thermal_field_rt_k: np.ndarray | None,
        inputs: HighFidelityTireInputs,
    ) -> WheelCouplingResult:
        surface_temp_k = self._surface_temperature_k(
            temperature_nodes_k=temperature_nodes_k,
            thermal_field_rt_k=thermal_field_rt_k,
        )
        if self.parameters.use_wheel_coupling:
            return self._wheel_coupling.solve(
                inputs=inputs,
                surface_temp_k=surface_temp_k,
            )
        return WheelCouplingResult(
            effective_slip_ratio=inputs.slip_ratio_cmd,
            effective_slip_angle_rad=inputs.slip_angle_cmd_rad,
            longitudinal_force_n=0.0,
            lateral_force_n=0.0,
            net_wheel_torque_nm=0.0,
            torque_residual_nm=0.0,
            lateral_force_residual_n=0.0,
            friction_power_w=self._boundary_model.frictional_contact_power_w(
                speed_mps=inputs.speed_mps,
                slip_ratio=inputs.slip_ratio_cmd,
                slip_angle_rad=inputs.slip_angle_cmd_rad,
                normal_load_n=inputs.normal_load_n,
            ),
            converged=False,
            iterations=0,
        )

    def _surface_temperature_k(
        self,
        *,
        temperature_nodes_k: np.ndarray,
        thermal_field_rt_k: np.ndarray | None,
    ) -> float:
        if thermal_field_rt_k is not None:
            return float(np.mean(thermal_field_rt_k[-1, :]))
        return float(np.mean(temperature_nodes_k[:3]))

    def _resolve_road_state(
        self,
        *,
        state: HighFidelityTireState,
        inputs: HighFidelityTireInputs,
    ) -> BoundaryState:
        if state.road_surface_temp_k is None:
            road_surface_temp_k = (
                inputs.track_temp_k
                if inputs.road_surface_temp_k is None
                else inputs.road_surface_temp_k
            )
        else:
            road_surface_temp_k = state.road_surface_temp_k

        if state.road_subsurface_temp_k is None:
            road_subsurface_temp_k = (
                road_surface_temp_k
                if inputs.road_bulk_temp_k is None
                else inputs.road_bulk_temp_k
            )
        else:
            road_subsurface_temp_k = state.road_subsurface_temp_k

        return self._boundary_model.initial_state(
            road_surface_temp_k=road_surface_temp_k,
            road_subsurface_temp_k=road_subsurface_temp_k,
        )

    def _boundary_source_field_w_per_m3(
        self,
        *,
        thermal_field_rt_k: np.ndarray,
        friction_to_tire_w: float,
        road_conduction_w: float,
        rim_conduction_w: float,
        brake_heat_to_tire_w: float,
        wheel_angular_speed_radps: float,
        time_s: float,
    ) -> np.ndarray:
        source = np.zeros_like(thermal_field_rt_k)
        cell_volumes = self._thermal_solver.cell_volumes_m3

        radial_indices, theta_indices = self._thermal_solver.contact_patch_indices(
            wheel_angular_speed_radps=wheel_angular_speed_radps,
            time_s=time_s,
        )
        patch_index = np.ix_(radial_indices, theta_indices)
        patch_volume = float(np.sum(cell_volumes[patch_index]))
        patch_power_w = friction_to_tire_w - road_conduction_w
        source[patch_index] += patch_power_w / max(patch_volume, 1e-12)

        inner_ring_volume = float(np.sum(cell_volumes[0, :]))
        source[0, :] += (brake_heat_to_tire_w - rim_conduction_w) / max(inner_ring_volume, 1e-12)
        return source

    def _brake_heat_flows(self, *, brake_power_w: float) -> tuple[float, float]:
        brake_power_w = max(brake_power_w, 0.0)
        return (
            brake_power_w * max(self.parameters.brake_heat_to_tire_fraction, 0.0),
            brake_power_w * max(self.parameters.brake_heat_to_rim_fraction, 0.0),
        )

    def _temperature_nodes_from_field(
        self,
        thermal_field_rt_k: np.ndarray,
        current_nodes_k: np.ndarray,
        *,
        dt_s: float,
    ) -> np.ndarray:
        radial_cells = thermal_field_rt_k.shape[0]
        surface_idx = radial_cells - 1
        belt_idx = int(round(0.72 * (radial_cells - 1)))
        carcass_idx = int(round(0.45 * (radial_cells - 1)))
        inner_idx = int(round(0.20 * (radial_cells - 1)))

        surface_temp = float(np.mean(thermal_field_rt_k[surface_idx, :]))
        belt_temp = float(np.mean(thermal_field_rt_k[belt_idx, :]))
        carcass_temp = float(np.mean(thermal_field_rt_k[carcass_idx, :]))
        inner_temp = float(np.mean(thermal_field_rt_k[inner_idx, :]))
        core_bulk_temp = self._core_bulk_temperature_k(thermal_field_rt_k)

        nodes = current_nodes_k.copy()
        nodes[0] = surface_temp
        nodes[1] = surface_temp
        nodes[2] = surface_temp
        nodes[3] = belt_temp
        nodes[4] = self._update_core_temperature_k(
            current_core_temp_k=float(nodes[4]),
            bulk_temp_k=core_bulk_temp,
            belt_temp_k=belt_temp,
            carcass_temp_k=carcass_temp,
            gas_temp_k=float(nodes[5]),
            dt_s=dt_s,
        )
        nodes[5] = 0.70 * nodes[5] + 0.30 * inner_temp
        nodes[8] = 0.5 * (carcass_temp + inner_temp)
        return nodes

    def _core_bulk_temperature_k(self, thermal_field_rt_k: np.ndarray) -> float:
        radial_cells = thermal_field_rt_k.shape[0]
        params = self.parameters
        inner_idx = int(round(params.core_probe_inner_fraction * (radial_cells - 1)))
        outer_idx = int(round(params.core_probe_outer_fraction * (radial_cells - 1)))
        inner_idx = max(0, min(inner_idx, radial_cells - 1))
        outer_idx = max(inner_idx, min(outer_idx, radial_cells - 1))
        window = thermal_field_rt_k[inner_idx : outer_idx + 1, :]
        volumes = self._thermal_solver.cell_volumes_m3[inner_idx : outer_idx + 1, :]
        return float(np.sum(window * volumes) / max(np.sum(volumes), 1e-12))

    def _update_core_temperature_k(
        self,
        *,
        current_core_temp_k: float,
        bulk_temp_k: float,
        belt_temp_k: float,
        carcass_temp_k: float,
        gas_temp_k: float,
        dt_s: float,
    ) -> float:
        params = self.parameters
        belt_weight = max(params.core_probe_belt_weight, 0.0)
        carcass_weight = max(params.core_probe_carcass_weight, 0.0)
        gas_weight = max(params.core_probe_gas_weight, 0.0)
        bulk_weight = max(1.0 - belt_weight - carcass_weight - gas_weight, 0.0)
        total_weight = max(bulk_weight + belt_weight + carcass_weight + gas_weight, 1e-12)
        target_core_temp_k = (
            bulk_weight * bulk_temp_k
            + belt_weight * belt_temp_k
            + carcass_weight * carcass_temp_k
            + gas_weight * gas_temp_k
        ) / total_weight
        response_time_s = max(params.core_probe_response_time_s, 1e-6)
        alpha = 1.0 - np.exp(-dt_s / response_time_s)
        return float(current_core_temp_k + alpha * (target_core_temp_k - current_core_temp_k))

    def simulate(
        self,
        initial_state: HighFidelityTireState,
        inputs_stream: Iterable[HighFidelityTireInputs],
        dt_s: float,
    ) -> list[HighFidelityTireState]:
        states = [initial_state]
        current = initial_state
        for inputs in inputs_stream:
            current = self.step(current, inputs, dt_s=dt_s)
            states.append(current)
        return states
