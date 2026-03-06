from __future__ import annotations

from dataclasses import replace
import math
import time
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
    """Layered high-fidelity tire simulator with width-aware thermal transport."""

    def __init__(self, parameters: HighFidelityTireModelParameters | None = None) -> None:
        self.parameters = parameters if parameters is not None else HighFidelityTireModelParameters()
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
        width_zones = self.parameters.width_zones
        thermal_field_rtw = (
            self._thermal_solver.initial_temperature_field(ambient_temp_k)
            if self.parameters.use_2d_thermal_solver
            else None
        )
        thermal_field_rt = (
            np.mean(thermal_field_rtw, axis=2)
            if thermal_field_rtw is not None
            else None
        )
        flash_field_tw = np.full((self.parameters.theta_cells, width_zones), float(ambient_temp_k), dtype=float)
        sidewall_field_tw = np.full((self.parameters.theta_cells, width_zones), float(ambient_temp_k), dtype=float)
        zero_w = np.zeros(width_zones, dtype=float)
        pressure_patch = self.parameters.pressure_patch
        dynamic_volume_m3 = max(
            pressure_patch.base_volume_m3 - pressure_patch.wear_volume_loss_m3 * np.clip(wear, 0.0, 1.0),
            pressure_patch.minimum_volume_m3,
        )
        dynamic_pressure_pa = (
            pressure_patch.resolved_gas_mass_kg()
            * pressure_patch.gas_specific_constant_j_per_kgk
            * ambient_temp_k
            / max(dynamic_volume_m3, 1e-12)
        )
        return HighFidelityTireState(
            temperature_nodes_k=nodes,
            thermal_field_rt_k=thermal_field_rt,
            thermal_field_rtw_k=thermal_field_rtw,
            flash_temperature_field_tw_k=flash_field_tw,
            sidewall_field_tw_k=sidewall_field_tw,
            road_surface_temp_k=None,
            road_subsurface_temp_k=None,
            road_surface_temp_w_k=None,
            road_subsurface_temp_w_k=None,
            road_moisture_w=np.zeros(width_zones, dtype=float),
            wear=float(np.clip(wear, 0.0, 1.0)),
            age_index=0.0,
            grain_index_w=np.zeros(width_zones, dtype=float),
            blister_index_w=np.zeros(width_zones, dtype=float),
            dynamic_pressure_pa=float(dynamic_pressure_pa),
            dynamic_volume_m3=float(dynamic_volume_m3),
            contact_patch_area_m2=pressure_patch.reference_contact_patch_area_m2,
            zone_contact_patch_area_m2=np.full(
                width_zones,
                pressure_patch.reference_contact_patch_area_m2 / max(width_zones, 1),
                dtype=float,
            ),
            effective_rolling_radius_m=self.parameters.wheel_effective_radius_m,
            last_energy_residual_pct=0.0,
            last_solver_substeps=0,
            last_friction_total_w=0.0,
            last_friction_to_tire_w=0.0,
            last_friction_to_road_w=0.0,
            last_road_conduction_w=0.0,
            last_rim_conduction_w=0.0,
            last_sidewall_heat_w=0.0,
            last_brake_heat_to_tire_w=0.0,
            last_brake_heat_to_rim_w=0.0,
            last_brake_heat_to_sidewall_w=0.0,
            last_effective_bead_htc_w_per_m2k=0.0,
            last_effective_slip_ratio=0.0,
            last_effective_slip_angle_rad=0.0,
            last_longitudinal_force_n=0.0,
            last_lateral_force_n=0.0,
            last_torque_residual_nm=0.0,
            last_lateral_force_residual_n=0.0,
            last_coupling_iterations=0,
            last_coupling_converged=False,
            last_contact_patch_length_m=0.0,
            last_contact_patch_width_m=0.0,
            last_sliding_fraction=0.0,
            last_effective_mu=0.0,
            last_hysteresis_strain_amplitude=0.0,
            last_zone_effective_mu=zero_w.copy(),
            last_zone_friction_power_w=zero_w.copy(),
            last_zone_friction_power_tire_w=zero_w.copy(),
            last_zone_friction_power_road_w=zero_w.copy(),
            last_zone_tire_heat_partition=zero_w.copy(),
            last_zone_sliding_fraction=zero_w.copy(),
            last_zone_flash_to_bulk_delta_k=zero_w.copy(),
            last_solver_advection_time_s=None,
            last_solver_diffusion_time_s=None,
            last_solver_diffusion_iterations=None,
            last_wheel_coupling_time_s=None,
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
        hysteresis = self._materials.layer_hysteresis_summary(
            temperature_k=state.core_temperature_k,
            inputs=effective_inputs,
            age_index=state.age_index,
            wear=state.wear,
            dynamic_pressure_pa=state.dynamic_pressure_pa,
        )
        frequency_hz = hysteresis.excitation_frequency_hz
        loss_modulus_pa = hysteresis.equivalent_loss_modulus_pa
        q_hyst_w_per_m3 = hysteresis.total_power_density_w_per_m3
        deformation = hysteresis.deformation
        energy_source_total_w = q_hyst_w_per_m3 * max(self.parameters.hysteresis_active_volume_m3, 0.0)
        if state.thermal_field_rtw_k is not None:
            surface_temperature_k = float(np.mean(state.thermal_field_rtw_k[-1, :, :]))
            mean_temperature_k = float(np.mean(state.thermal_field_rtw_k))
            thermal_grid_shape: tuple[int, int, int] | None = tuple(int(v) for v in state.thermal_field_rtw_k.shape)
            per_width_surface = tuple(float(np.mean(state.thermal_field_rtw_k[-1, :, idx])) for idx in range(self.parameters.width_zones))
            bulk_core_by_width = tuple(
                float(np.mean(self._core_window_temperature(state.thermal_field_rtw_k)[:, idx]))
                for idx in range(self.parameters.width_zones)
            )
            layer_mean = self._thermal_solver.layer_mean_temperatures_k(
                temperature_field_rtw_k=state.thermal_field_rtw_k,
                wear=state.wear,
            )
        else:
            surface_temperature_k = float(np.mean(state.temperature_nodes_k[:3]))
            mean_temperature_k = float(np.mean(state.temperature_nodes_k))
            thermal_grid_shape = None
            per_width_surface = (surface_temperature_k, surface_temperature_k, surface_temperature_k)
            bulk_core_by_width = (state.core_temperature_k, state.core_temperature_k, state.core_temperature_k)
            layer_mean = {
                "tread": surface_temperature_k,
                "belt": float(state.temperature_nodes_k[3]),
                "carcass": float(state.temperature_nodes_k[4]),
                "inner_liner": float(state.temperature_nodes_k[5]),
            }
        if state.flash_temperature_field_tw_k is not None:
            per_width_flash_surface = self._per_width_flash_surface_temp_k(
                flash_field_tw_k=state.flash_temperature_field_tw_k,
                wheel_angular_speed_radps=inputs.wheel_angular_speed_radps,
                time_s=state.time_s,
                fallback=per_width_surface,
            )
            flash_surface_temperature_k = float(np.mean(per_width_flash_surface))
        else:
            flash_surface_temperature_k = surface_temperature_k
            per_width_flash_surface = per_width_surface
        zone_contact = self._tuple3(state.zone_contact_patch_area_m2)
        road_surface_w = self._tuple3(state.road_surface_temp_w_k)
        road_subsurface_w = self._tuple3(state.road_subsurface_temp_w_k)
        road_moisture_w = self._tuple3(state.road_moisture_w)
        grain_w = self._tuple3(state.grain_index_w)
        blister_w = self._tuple3(state.blister_index_w)
        zone_effective_mu = self._tuple3(state.last_zone_effective_mu)
        zone_friction = self._tuple3(state.last_zone_friction_power_w)
        zone_friction_tire = self._tuple3(state.last_zone_friction_power_tire_w)
        zone_friction_road = self._tuple3(state.last_zone_friction_power_road_w)
        zone_eta = self._tuple3(state.last_zone_tire_heat_partition)
        zone_sliding = self._tuple3(state.last_zone_sliding_fraction)
        zone_flash_to_bulk = self._tuple3(state.last_zone_flash_to_bulk_delta_k)
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
            contact_patch_length_m=state.last_contact_patch_length_m,
            contact_patch_width_m=state.last_contact_patch_width_m,
            sliding_fraction=state.last_sliding_fraction,
            effective_mu=state.last_effective_mu,
            hysteresis_strain_amplitude=state.last_hysteresis_strain_amplitude,
            per_zone_effective_mu=zone_effective_mu,
            per_zone_friction_power_w=zone_friction,
            per_zone_friction_power_tire_w=zone_friction_tire,
            per_zone_friction_power_road_w=zone_friction_road,
            per_zone_tire_heat_partition=zone_eta,
            per_zone_sliding_fraction=zone_sliding,
            per_zone_flash_to_bulk_delta_k=zone_flash_to_bulk,
            hysteresis_strain_by_layer=dict(state.last_hysteresis_strain_by_layer),
            hysteresis_loss_modulus_by_layer_pa=dict(state.last_hysteresis_loss_modulus_by_layer_pa),
            hysteresis_power_by_layer_w=dict(state.last_hysteresis_power_by_layer_w),
            bulk_core_temperature_k=float(np.mean(self._core_window_temperature(state.thermal_field_rtw_k))) if state.thermal_field_rtw_k is not None else state.core_temperature_k,
            cavity_gas_temperature_k=float(state.temperature_nodes_k[5]),
            core_temperature_compare_k=float(
                (1.0 - self.parameters.core_sensor.cavity_gas_comparison_fraction) * state.core_temperature_k
                + self.parameters.core_sensor.cavity_gas_comparison_fraction * state.temperature_nodes_k[5]
            ),
            flash_surface_temperature_k=flash_surface_temperature_k,
            dynamic_pressure_pa=state.dynamic_pressure_pa,
            dynamic_pressure_bar_gauge=(
                (state.dynamic_pressure_pa - self.parameters.pressure_patch.atmospheric_pressure_pa) / 100_000.0
            ),
            dynamic_volume_m3=state.dynamic_volume_m3,
            contact_patch_area_m2=state.contact_patch_area_m2,
            zone_contact_patch_area_m2=zone_contact,
            effective_rolling_radius_m=state.effective_rolling_radius_m,
            per_width_surface_temp_k=per_width_surface,
            per_width_flash_surface_temp_k=per_width_flash_surface,
            per_width_bulk_temp_k=bulk_core_by_width,
            layer_mean_temp_k=layer_mean,
            road_surface_temp_w_k=road_surface_w,
            road_subsurface_temp_w_k=road_subsurface_w,
            road_moisture_w=road_moisture_w,
            age_index=state.age_index,
            grain_index_w=grain_w,
            blister_index_w=blister_w,
            brake_heat_to_sidewall_w=state.last_brake_heat_to_sidewall_w,
            brake_duct_cooling_factor=inputs.brake_duct_cooling_factor,
            wheel_wake_factor=inputs.wheel_wake_factor,
            solver_advection_time_s=state.last_solver_advection_time_s,
            solver_diffusion_time_s=state.last_solver_diffusion_time_s,
            solver_diffusion_iterations=state.last_solver_diffusion_iterations,
            wheel_coupling_time_s=state.last_wheel_coupling_time_s,
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

        road_state = self._resolve_road_state(state=state, inputs=_inputs)
        thermal_field_rtw = state.thermal_field_rtw_k
        if self.parameters.use_2d_thermal_solver and thermal_field_rtw is None:
            thermal_field_rtw = self._thermal_solver.initial_temperature_field(float(np.mean(state.temperature_nodes_k)))
        thermal_field_rt = state.thermal_field_rt_k
        if thermal_field_rt is None and thermal_field_rtw is not None:
            thermal_field_rt = np.mean(thermal_field_rtw, axis=2)
        dynamic_volume_m3_pre = self._dynamic_volume_m3(state=state, inputs=_inputs)
        dynamic_pressure_pa_pre = self._dynamic_pressure_pa(
            gas_temp_k=float(state.temperature_nodes_k[5]),
            dynamic_volume_m3=dynamic_volume_m3_pre,
        )
        coupling_start = time.perf_counter() if self.parameters.enable_profiling else 0.0
        coupling_result = self._resolved_wheel_coupling(
            temperature_nodes_k=state.temperature_nodes_k,
            thermal_field_rtw_k=thermal_field_rtw,
            flash_field_tw_k=state.flash_temperature_field_tw_k,
            dynamic_pressure_pa=dynamic_pressure_pa_pre,
            time_s=state.time_s,
            inputs=_inputs,
        )
        coupling_time_s = time.perf_counter() - coupling_start if self.parameters.enable_profiling else None
        effective_inputs = replace(
            _inputs,
            slip_ratio_cmd=coupling_result.effective_slip_ratio,
            slip_angle_cmd_rad=coupling_result.effective_slip_angle_rad,
        )

        if self.parameters.no_op_thermal_step or not self.parameters.use_2d_thermal_solver:
            return replace(
                state,
                time_s=state.time_s + dt_s,
                last_effective_slip_ratio=coupling_result.effective_slip_ratio,
                last_effective_slip_angle_rad=coupling_result.effective_slip_angle_rad,
                last_longitudinal_force_n=coupling_result.longitudinal_force_n,
                last_lateral_force_n=coupling_result.lateral_force_n,
                last_torque_residual_nm=coupling_result.torque_residual_nm,
                last_lateral_force_residual_n=coupling_result.lateral_force_residual_n,
                last_coupling_iterations=coupling_result.iterations,
                last_coupling_converged=coupling_result.converged,
                last_contact_patch_length_m=coupling_result.contact_patch_length_m,
                last_contact_patch_width_m=coupling_result.contact_patch_width_m,
                last_sliding_fraction=coupling_result.sliding_fraction,
                last_effective_mu=coupling_result.effective_mu,
                last_zone_effective_mu=self._zone_array(coupling_result.zone_effective_mu, width_zones=self.parameters.width_zones),
                last_zone_friction_power_w=self._zone_array(coupling_result.zone_friction_power_w, width_zones=self.parameters.width_zones),
                last_zone_friction_power_tire_w=np.zeros(self.parameters.width_zones, dtype=float),
                last_zone_friction_power_road_w=np.zeros(self.parameters.width_zones, dtype=float),
                last_zone_tire_heat_partition=np.zeros(self.parameters.width_zones, dtype=float),
                last_zone_sliding_fraction=self._zone_array(coupling_result.zone_sliding_fraction, width_zones=self.parameters.width_zones),
                last_zone_flash_to_bulk_delta_k=np.zeros(self.parameters.width_zones, dtype=float),
                last_wheel_coupling_time_s=coupling_time_s,
            )

        width_zones = self.parameters.width_zones
        sidewall_field_tw = (
            np.full((self.parameters.theta_cells, width_zones), float(np.mean(state.temperature_nodes_k)), dtype=float)
            if state.sidewall_field_tw_k is None
            else np.array(state.sidewall_field_tw_k, dtype=float, copy=True)
        )
        flash_field_tw = (
            np.full((self.parameters.theta_cells, width_zones), float(np.mean(state.temperature_nodes_k[:3])), dtype=float)
            if state.flash_temperature_field_tw_k is None
            else np.array(state.flash_temperature_field_tw_k, dtype=float, copy=True)
        )
        zone_weights = self._zone_weights(inputs=effective_inputs)
        dynamic_volume_m3 = self._dynamic_volume_m3(state=state, inputs=effective_inputs)
        dynamic_pressure_pa = self._dynamic_pressure_pa(gas_temp_k=float(state.temperature_nodes_k[5]), dynamic_volume_m3=dynamic_volume_m3)
        effective_radius_m = self._effective_rolling_radius_m(dynamic_pressure_pa=dynamic_pressure_pa)
        contact_patch_area_m2 = self._contact_patch_area_m2(
            normal_load_n=effective_inputs.normal_load_n,
            pressure_pa=dynamic_pressure_pa,
            surface_temp_k=self._surface_temperature_k(
                temperature_nodes_k=state.temperature_nodes_k,
                thermal_field_rtw_k=thermal_field_rtw,
            ),
        )
        zone_contact_patch_area_m2 = contact_patch_area_m2 * zone_weights
        current_grain = np.zeros(width_zones, dtype=float) if state.grain_index_w is None else np.array(state.grain_index_w, dtype=float, copy=True)
        current_blister = np.zeros(width_zones, dtype=float) if state.blister_index_w is None else np.array(state.blister_index_w, dtype=float, copy=True)
        zone_contact_patch_area_m2 *= np.clip(
            1.0 - self.parameters.surface_state.graining_contact_penalty * current_grain,
            0.35,
            1.0,
        )
        zone_contact_patch_area_m2 *= np.clip(1.0 - 0.08 * current_blister, 0.45, 1.0)

        tire_surface_temp_w_k = np.mean(thermal_field_rtw[-1, :, :], axis=0)
        tire_inner_temp_k = float(np.mean(thermal_field_rtw[0, :, :]))
        rim_temp_k = float(state.temperature_nodes_k[6])

        zone_bulk_surface = self._zone_array(
            self._per_width_patch_surface_temp_k(
                thermal_field_rtw_k=thermal_field_rtw,
                wheel_angular_speed_radps=effective_inputs.wheel_angular_speed_radps,
                time_s=state.time_s,
                fallback=self._per_width_bulk_surface_temp_k(
                    temperature_nodes_k=state.temperature_nodes_k,
                    thermal_field_rtw_k=thermal_field_rtw,
                ),
            ),
            width_zones,
        )
        zone_flash_surface = self._zone_array(
            self._per_width_flash_surface_temp_k(
                flash_field_tw_k=flash_field_tw,
                wheel_angular_speed_radps=effective_inputs.wheel_angular_speed_radps,
                time_s=state.time_s,
                fallback=self._tuple3(zone_bulk_surface),
            ),
            width_zones,
        )
        zone_flash_to_bulk_delta_k = np.maximum(zone_flash_surface - zone_bulk_surface, 0.0)
        zone_friction_power_w = self._zone_array(coupling_result.zone_friction_power_w, width_zones)
        last_friction_total_w = float(np.sum(zone_friction_power_w))
        if self.parameters.use_local_temp_friction_partition:
            zone_friction_to_tire_w, zone_friction_to_road_w, zone_tire_eta = self._boundary_model.partition_friction_power_by_zone(
                zone_friction_power_w=zone_friction_power_w,
                flash_temp_w_k=zone_flash_surface,
                bulk_temp_w_k=zone_bulk_surface,
                road_surface_temp_w_k=road_state.road_surface_temp_w_k,
                road_moisture_w=road_state.road_moisture_w,
                asphalt_effusivity=effective_inputs.asphalt_effusivity,
                rubbering_level=effective_inputs.rubbering_level,
            )
            last_friction_to_tire_w = float(np.sum(zone_friction_to_tire_w))
            last_friction_to_road_w = float(np.sum(zone_friction_to_road_w))
        else:
            last_friction_to_tire_w, last_friction_to_road_w = self._boundary_model.partition_friction_power(
                total_friction_power_w=last_friction_total_w,
                road_moisture=float(np.mean(road_state.road_moisture_w)),
                asphalt_effusivity=effective_inputs.asphalt_effusivity,
                rubbering_level=effective_inputs.rubbering_level,
            )
            eta_scalar = (
                last_friction_to_tire_w / max(last_friction_total_w, 1e-9)
                if last_friction_total_w > 0.0
                else 0.0
            )
            zone_tire_eta = np.full(width_zones, eta_scalar, dtype=float)
            zone_friction_to_tire_w = zone_friction_power_w * zone_tire_eta
            zone_friction_to_road_w = np.maximum(zone_friction_power_w - zone_friction_to_tire_w, 0.0)
        road_conduction_w_by_zone = self._boundary_model.road_conduction_power_w_by_zone(
            tire_surface_temp_w_k=tire_surface_temp_w_k,
            road_surface_temp_w_k=road_state.road_surface_temp_w_k,
            zone_contact_patch_area_m2=zone_contact_patch_area_m2,
            road_moisture_w=road_state.road_moisture_w,
            asphalt_effusivity=effective_inputs.asphalt_effusivity,
            wind_mps=effective_inputs.wind_mps,
        )
        last_road_conduction_w = float(np.sum(road_conduction_w_by_zone))
        last_rim_conduction_w, last_effective_bead_htc = self._boundary_model.rim_conduction_power_w(
            tire_inner_temp_k=tire_inner_temp_k,
            rim_temp_k=rim_temp_k,
            normal_load_n=effective_inputs.normal_load_n,
            dynamic_pressure_pa=dynamic_pressure_pa,
            contact_area_m2=float(np.sum(zone_contact_patch_area_m2)) * 0.55,
        )
        last_brake_heat_to_tire_w, last_brake_heat_to_rim_w, last_brake_heat_to_sidewall_w = self._brake_heat_flows(
            brake_power_w=effective_inputs.brake_power_w
        )

        hysteresis = self._materials.layer_hysteresis_summary(
            temperature_k=state.core_temperature_k,
            inputs=effective_inputs,
            age_index=state.age_index,
            wear=state.wear,
            dynamic_pressure_pa=dynamic_pressure_pa,
        )
        q_hyst_w_per_m3 = hysteresis.total_power_density_w_per_m3
        deformation = hysteresis.deformation
        boundary_source = self._boundary_source_field_w_per_m3(
            thermal_field_rtw_k=thermal_field_rtw,
            zone_friction_to_tire_w=zone_friction_to_tire_w,
            road_conduction_w_by_zone=road_conduction_w_by_zone,
            rim_conduction_w=last_rim_conduction_w,
            brake_heat_to_tire_w=last_brake_heat_to_tire_w,
            zone_weights=zone_weights,
            wheel_angular_speed_radps=effective_inputs.wheel_angular_speed_radps,
            time_s=state.time_s,
        )
        solver_result = self._thermal_solver.step(
            temperature_field_rtw_k=thermal_field_rtw,
            inputs=effective_inputs,
            time_s=state.time_s,
            dt_s=dt_s,
            volumetric_source_w_per_m3=q_hyst_w_per_m3,
            extra_source_w_per_m3=boundary_source,
            layer_source_weights=hysteresis.power_density_by_layer_w_per_m3,
            wear=state.wear,
            grain_index_w=current_grain,
            blister_index_w=current_blister,
            age_index=state.age_index,
            zone_source_weights=zone_weights,
        )
        thermal_field_rtw_next = solver_result.temperature_field_rtw_k
        thermal_field_rt_next = np.mean(thermal_field_rtw_next, axis=2)
        flash_field_tw_next = self._step_flash_field(
            flash_field_tw_k=flash_field_tw,
            thermal_field_rtw_k=thermal_field_rtw_next,
            road_state=road_state,
            ambient_temp_k=effective_inputs.ambient_temp_k,
            friction_to_tire_w=last_friction_to_tire_w,
            zone_weights=zone_weights,
            wheel_angular_speed_radps=effective_inputs.wheel_angular_speed_radps,
            time_s=state.time_s,
            dt_s=dt_s,
        )
        sidewall_field_tw_next, last_sidewall_heat_w = self._step_sidewall_field(
            sidewall_field_tw_k=sidewall_field_tw,
            thermal_field_rtw_k=thermal_field_rtw_next,
            inputs=effective_inputs,
            rim_temp_k=rim_temp_k,
            brake_heat_to_sidewall_w=last_brake_heat_to_sidewall_w,
            dt_s=dt_s,
        )
        y_next = self._temperature_nodes_from_field(
            thermal_field_rtw_k=thermal_field_rtw_next,
            sidewall_field_tw_k=sidewall_field_tw_next,
            current_nodes_k=state.temperature_nodes_k,
            dt_s=dt_s,
        )
        y_next[6] = self._boundary_model.step_rim_temperature(
            rim_temp_k=float(y_next[6]),
            ambient_temp_k=effective_inputs.ambient_temp_k,
            heat_input_w=last_rim_conduction_w + last_brake_heat_to_rim_w,
            dt_s=dt_s,
            brake_duct_cooling_factor=effective_inputs.brake_duct_cooling_factor,
            wheel_wake_factor=effective_inputs.wheel_wake_factor,
        )
        y_next[5] = self._update_gas_temperature_k(
            current_gas_temp_k=float(state.temperature_nodes_k[5]),
            carcass_inner_temp_k=float(np.mean(thermal_field_rtw_next[0, :, :])),
            rim_temp_k=float(y_next[6]),
            dynamic_pressure_pa=dynamic_pressure_pa,
            inputs=effective_inputs,
            dt_s=dt_s,
        )
        road_state = self._boundary_model.step_road_slab(
            state=road_state,
            dt_s=dt_s,
            heat_input_surface_w=road_conduction_w_by_zone + zone_friction_to_road_w,
            road_bulk_temp_k=effective_inputs.road_bulk_temp_k if effective_inputs.road_bulk_temp_k is not None else effective_inputs.track_temp_k,
            asphalt_effusivity=effective_inputs.asphalt_effusivity,
            solar_w_m2=effective_inputs.solar_w_m2,
            wind_mps=effective_inputs.wind_mps,
        )
        wear_next, age_next, grain_next, blister_next = self._update_surface_state(
            state=state,
            inputs=effective_inputs,
            thermal_field_rtw_k=thermal_field_rtw_next,
            friction_to_tire_w=last_friction_to_tire_w,
            zone_contact_patch_area_m2=zone_contact_patch_area_m2,
            dt_s=dt_s,
        )

        y_next = np.clip(y_next, self.parameters.minimum_temperature_k, self.parameters.maximum_temperature_k)
        return replace(
            state,
            temperature_nodes_k=y_next,
            thermal_field_rt_k=thermal_field_rt_next,
            thermal_field_rtw_k=thermal_field_rtw_next,
            flash_temperature_field_tw_k=flash_field_tw_next,
            sidewall_field_tw_k=sidewall_field_tw_next,
            road_surface_temp_k=float(np.mean(road_state.road_surface_temp_w_k)),
            road_subsurface_temp_k=float(np.mean(road_state.road_subsurface_temp_w_k)),
            road_surface_temp_w_k=road_state.road_surface_temp_w_k,
            road_subsurface_temp_w_k=road_state.road_subsurface_temp_w_k,
            road_moisture_w=road_state.road_moisture_w,
            wear=wear_next,
            age_index=age_next,
            grain_index_w=grain_next,
            blister_index_w=blister_next,
            dynamic_pressure_pa=float(dynamic_pressure_pa),
            dynamic_volume_m3=float(dynamic_volume_m3),
            contact_patch_area_m2=float(contact_patch_area_m2),
            zone_contact_patch_area_m2=zone_contact_patch_area_m2,
            effective_rolling_radius_m=float(effective_radius_m),
            last_energy_residual_pct=solver_result.energy_residual_pct,
            last_solver_substeps=solver_result.substeps,
            last_friction_total_w=last_friction_total_w,
            last_friction_to_tire_w=last_friction_to_tire_w,
            last_friction_to_road_w=last_friction_to_road_w,
            last_road_conduction_w=last_road_conduction_w,
            last_rim_conduction_w=last_rim_conduction_w,
            last_sidewall_heat_w=last_sidewall_heat_w,
            last_brake_heat_to_tire_w=last_brake_heat_to_tire_w,
            last_brake_heat_to_rim_w=last_brake_heat_to_rim_w,
            last_brake_heat_to_sidewall_w=last_brake_heat_to_sidewall_w,
            last_effective_bead_htc_w_per_m2k=last_effective_bead_htc,
            last_effective_slip_ratio=coupling_result.effective_slip_ratio,
            last_effective_slip_angle_rad=coupling_result.effective_slip_angle_rad,
            last_longitudinal_force_n=coupling_result.longitudinal_force_n,
            last_lateral_force_n=coupling_result.lateral_force_n,
            last_torque_residual_nm=coupling_result.torque_residual_nm,
            last_lateral_force_residual_n=coupling_result.lateral_force_residual_n,
            last_coupling_iterations=coupling_result.iterations,
            last_coupling_converged=coupling_result.converged,
            last_contact_patch_length_m=coupling_result.contact_patch_length_m,
            last_contact_patch_width_m=coupling_result.contact_patch_width_m,
            last_sliding_fraction=coupling_result.sliding_fraction,
            last_effective_mu=coupling_result.effective_mu,
            last_hysteresis_strain_amplitude=deformation.equivalent_strain_amplitude,
            last_zone_effective_mu=self._zone_array(coupling_result.zone_effective_mu, width_zones),
            last_zone_friction_power_w=zone_friction_power_w,
            last_zone_friction_power_tire_w=zone_friction_to_tire_w,
            last_zone_friction_power_road_w=zone_friction_to_road_w,
            last_zone_tire_heat_partition=zone_tire_eta,
            last_zone_sliding_fraction=self._zone_array(coupling_result.zone_sliding_fraction, width_zones),
            last_zone_flash_to_bulk_delta_k=zone_flash_to_bulk_delta_k,
            last_hysteresis_strain_by_layer=dict(hysteresis.strain_by_layer),
            last_hysteresis_loss_modulus_by_layer_pa=dict(hysteresis.loss_modulus_by_layer_pa),
            last_hysteresis_power_by_layer_w=self._hysteresis_power_w_by_layer(hysteresis.power_density_by_layer_w_per_m3),
            last_solver_advection_time_s=solver_result.advection_time_s if self.parameters.enable_profiling else None,
            last_solver_diffusion_time_s=solver_result.diffusion_time_s if self.parameters.enable_profiling else None,
            last_solver_diffusion_iterations=solver_result.diffusion_iterations if self.parameters.enable_profiling else None,
            last_wheel_coupling_time_s=coupling_time_s,
            time_s=state.time_s + dt_s,
        )

    def _resolved_wheel_coupling(
        self,
        *,
        temperature_nodes_k: np.ndarray,
        thermal_field_rtw_k: np.ndarray | None,
        flash_field_tw_k: np.ndarray | None,
        dynamic_pressure_pa: float | None,
        time_s: float,
        inputs: HighFidelityTireInputs,
    ) -> WheelCouplingResult:
        zone_bulk_surface = self._per_width_patch_surface_temp_k(
            thermal_field_rtw_k=thermal_field_rtw_k,
            wheel_angular_speed_radps=inputs.wheel_angular_speed_radps,
            time_s=time_s,
            fallback=self._per_width_bulk_surface_temp_k(
                temperature_nodes_k=temperature_nodes_k,
                thermal_field_rtw_k=thermal_field_rtw_k,
            ),
        )
        zone_flash_surface = self._per_width_flash_surface_temp_k(
            flash_field_tw_k=flash_field_tw_k,
            wheel_angular_speed_radps=inputs.wheel_angular_speed_radps,
            time_s=time_s,
            fallback=zone_bulk_surface,
        )
        bulk_surface_temp_k = float(np.mean(zone_bulk_surface))
        surface_temp_k = float(np.mean(zone_flash_surface))
        if self.parameters.use_wheel_coupling:
            return self._wheel_coupling.solve(
                inputs=inputs,
                surface_temp_k=surface_temp_k,
                bulk_surface_temp_k=bulk_surface_temp_k,
                flash_surface_temp_k=surface_temp_k,
                zone_bulk_surface_temp_k=zone_bulk_surface,
                zone_flash_surface_temp_k=zone_flash_surface,
                dynamic_pressure_pa=dynamic_pressure_pa,
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
                road_moisture=inputs.road_moisture,
                rubbering_level=inputs.rubbering_level,
                asphalt_roughness=inputs.asphalt_roughness,
            ),
            converged=False,
            iterations=0,
            contact_patch_length_m=0.0,
            contact_patch_width_m=0.0,
            sliding_fraction=0.0,
            effective_mu=0.0,
            zone_effective_mu=np.zeros(self.parameters.width_zones, dtype=float),
            zone_friction_power_w=np.full(
                self.parameters.width_zones,
                self._boundary_model.frictional_contact_power_w(
                    speed_mps=inputs.speed_mps,
                    slip_ratio=inputs.slip_ratio_cmd,
                    slip_angle_rad=inputs.slip_angle_cmd_rad,
                    normal_load_n=inputs.normal_load_n,
                    road_moisture=inputs.road_moisture,
                    rubbering_level=inputs.rubbering_level,
                    asphalt_roughness=inputs.asphalt_roughness,
                ) / max(self.parameters.width_zones, 1),
                dtype=float,
            ),
            zone_sliding_fraction=np.zeros(self.parameters.width_zones, dtype=float),
        )

    def _surface_temperature_k(
        self,
        *,
        temperature_nodes_k: np.ndarray,
        thermal_field_rtw_k: np.ndarray | None,
    ) -> float:
        if thermal_field_rtw_k is not None:
            return float(np.mean(thermal_field_rtw_k[-1, :, :]))
        return float(np.mean(temperature_nodes_k[:3]))

    def _flash_surface_temperature_k(
        self,
        *,
        flash_field_tw_k: np.ndarray | None,
        bulk_surface_temp_k: float,
    ) -> float:
        if flash_field_tw_k is None:
            return bulk_surface_temp_k
        return float(np.mean(flash_field_tw_k))

    def _per_width_bulk_surface_temp_k(
        self,
        *,
        temperature_nodes_k: np.ndarray,
        thermal_field_rtw_k: np.ndarray | None,
    ) -> tuple[float, float, float]:
        if thermal_field_rtw_k is None:
            return self._tuple3(temperature_nodes_k[:3])
        return self._tuple3(np.mean(thermal_field_rtw_k[-1, :, :], axis=0))

    def _per_width_flash_surface_temp_k(
        self,
        *,
        flash_field_tw_k: np.ndarray | None,
        wheel_angular_speed_radps: float,
        time_s: float,
        fallback: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        if flash_field_tw_k is None:
            return fallback
        _, theta_indices, width_indices = self._thermal_solver.contact_patch_indices(
            wheel_angular_speed_radps=wheel_angular_speed_radps,
            time_s=time_s,
        )
        patch_means = np.mean(flash_field_tw_k[np.ix_(theta_indices, width_indices)], axis=0)
        return self._tuple3(patch_means)

    def _per_width_patch_surface_temp_k(
        self,
        *,
        thermal_field_rtw_k: np.ndarray | None,
        wheel_angular_speed_radps: float,
        time_s: float,
        fallback: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        if thermal_field_rtw_k is None:
            return fallback
        _, theta_indices, width_indices = self._thermal_solver.contact_patch_indices(
            wheel_angular_speed_radps=wheel_angular_speed_radps,
            time_s=time_s,
        )
        patch_means = np.mean(thermal_field_rtw_k[-1, :, :][np.ix_(theta_indices, width_indices)], axis=0)
        return self._tuple3(patch_means)

    def _resolve_road_state(
        self,
        *,
        state: HighFidelityTireState,
        inputs: HighFidelityTireInputs,
    ) -> BoundaryState:
        if state.road_surface_temp_w_k is not None and state.road_subsurface_temp_w_k is not None and state.road_moisture_w is not None:
            return BoundaryState(
                road_surface_temp_w_k=np.array(state.road_surface_temp_w_k, dtype=float, copy=True),
                road_subsurface_temp_w_k=np.array(state.road_subsurface_temp_w_k, dtype=float, copy=True),
                road_moisture_w=np.array(state.road_moisture_w, dtype=float, copy=True),
            )
        road_surface_temp_k = inputs.track_temp_k if inputs.road_surface_temp_k is None else inputs.road_surface_temp_k
        road_subsurface_temp_k = road_surface_temp_k if inputs.road_bulk_temp_k is None else inputs.road_bulk_temp_k
        return self._boundary_model.initial_state(
            road_surface_temp_k=road_surface_temp_k,
            road_subsurface_temp_k=road_subsurface_temp_k,
            road_moisture=inputs.road_moisture,
            width_zones=self.parameters.width_zones,
        )

    def _dynamic_volume_m3(self, *, state: HighFidelityTireState, inputs: HighFidelityTireInputs) -> float:
        pp = self.parameters.pressure_patch
        volume = pp.base_volume_m3
        volume += pp.centrifugal_volume_gain_coeff_m3_per_radps2 * inputs.wheel_angular_speed_radps**2
        volume -= pp.deflection_volume_loss_coeff_m3_per_n * max(inputs.normal_load_n, 0.0)
        volume -= pp.wear_volume_loss_m3 * np.clip(state.wear, 0.0, 1.0)
        volume += inputs.volume_change_rate_m3ps * 0.10
        return float(max(volume, pp.minimum_volume_m3))

    def _dynamic_pressure_pa(self, *, gas_temp_k: float, dynamic_volume_m3: float) -> float:
        pp = self.parameters.pressure_patch
        gas_mass = pp.resolved_gas_mass_kg()
        pressure_pa = gas_mass * pp.gas_specific_constant_j_per_kgk * gas_temp_k / max(dynamic_volume_m3, 1e-12)
        return float(max(pressure_pa, pp.atmospheric_pressure_pa))

    def _effective_rolling_radius_m(self, *, dynamic_pressure_pa: float) -> float:
        pp = self.parameters.pressure_patch
        delta_bar = (dynamic_pressure_pa - pp.reference_pressure_pa) / 100_000.0
        return float(
            max(
                self.parameters.wheel_effective_radius_m + pp.effective_radius_pressure_gain_m_per_bar * delta_bar,
                0.25,
            )
        )

    def _contact_patch_area_m2(
        self,
        normal_load_n: float,
        pressure_pa: float,
        *,
        surface_temp_k: float,
    ) -> float:
        pp = self.parameters.pressure_patch
        gauge_pressure_pa = max(pressure_pa - pp.atmospheric_pressure_pa, 0.0)
        load_ratio = max(0.4, min(2.0, normal_load_n / max(pp.reference_normal_load_n, 1e-6)))
        carcass_load_term = 1.0 + pp.carcass_support_load_gain * (load_ratio - 1.0)
        temp_gain = 1.0 + pp.carcass_support_temp_gain_per_k * (surface_temp_k - celsius_to_kelvin(80.0))
        effective_pressure_pa = max(
            gauge_pressure_pa + pp.carcass_support_pressure_pa * max(carcass_load_term, 0.2) * max(temp_gain, 0.2),
            pp.min_effective_contact_pressure_pa,
        )
        area = max(normal_load_n, 0.0) / max(effective_pressure_pa * pp.contact_pressure_factor, 1.0)
        return float(np.clip(area, pp.min_contact_patch_area_m2, pp.max_contact_patch_area_m2))

    def _zone_weights(self, *, inputs: HighFidelityTireInputs) -> np.ndarray:
        inner, middle, outer = [max(v, 1e-6) for v in inputs.zone_load_split]
        shift = 0.0
        shift += 0.26 * math.tanh(inputs.camber_rad / 0.12)
        shift += 0.12 * math.tanh(inputs.toe_rad / 0.12)
        lateral_sign = -1.0 if inputs.is_left_tire else 1.0
        shift += lateral_sign * 0.20 * math.tanh(inputs.lateral_accel_mps2 / 9.81)
        shift += 0.10 * math.tanh(inputs.slip_angle_cmd_rad / 0.12)
        shift = float(np.clip(shift, -0.35, 0.35))
        inner *= 1.0 - shift
        outer *= 1.0 + shift
        middle *= 1.0 + 0.10 * abs(math.tanh(inputs.longitudinal_accel_mps2 / 9.81))
        values = np.array([inner, middle, outer], dtype=float)
        values /= max(float(np.sum(values)), 1e-12)
        return values

    def _boundary_source_field_w_per_m3(
        self,
        *,
        thermal_field_rtw_k: np.ndarray,
        zone_friction_to_tire_w: np.ndarray,
        road_conduction_w_by_zone: np.ndarray,
        rim_conduction_w: float,
        brake_heat_to_tire_w: float,
        zone_weights: np.ndarray,
        wheel_angular_speed_radps: float,
        time_s: float,
    ) -> np.ndarray:
        source = np.zeros_like(thermal_field_rtw_k)
        flash_fraction = 0.0
        if self.parameters.flash_layer.enabled:
            flash_fraction = float(np.clip(self.parameters.flash_layer.friction_fraction, 0.0, 0.95))
        bulk_fraction = 1.0 - flash_fraction
        cell_volumes = self._thermal_solver.cell_volumes_m3
        radial_indices, theta_indices, width_indices = self._thermal_solver.contact_patch_indices(
            wheel_angular_speed_radps=wheel_angular_speed_radps,
            time_s=time_s,
        )
        for zone_idx, width_idx in enumerate(width_indices):
            patch_index = np.ix_(radial_indices, theta_indices, np.array([width_idx], dtype=int))
            patch_volume = float(np.sum(cell_volumes[patch_index]))
            patch_power_w = bulk_fraction * zone_friction_to_tire_w[zone_idx] - road_conduction_w_by_zone[zone_idx]
            source[patch_index] += patch_power_w / max(patch_volume, 1e-12)
        inner_ring_volume = float(np.sum(cell_volumes[0, :, :]))
        source[0, :, :] += (brake_heat_to_tire_w - rim_conduction_w) / max(inner_ring_volume, 1e-12)
        return source

    def _zone_array(self, values: np.ndarray | tuple[float, float, float] | None, width_zones: int) -> np.ndarray:
        if values is None:
            return np.zeros(width_zones, dtype=float)
        arr = np.asarray(values, dtype=float)
        if arr.shape == (width_zones,):
            return np.array(arr, dtype=float, copy=True)
        if arr.size == 1:
            return np.full(width_zones, float(arr[0]), dtype=float)
        return np.resize(arr, width_zones).astype(float, copy=True)

    def _hysteresis_power_w_by_layer(self, power_density_by_layer_w_per_m3: dict[str, float]) -> dict[str, float]:
        active_volume_m3 = max(self.parameters.hysteresis_active_volume_m3, 0.0)
        return {
            name: active_volume_m3 * max(power_density, 0.0)
            for name, power_density in power_density_by_layer_w_per_m3.items()
        }

    def _step_flash_field(
        self,
        *,
        flash_field_tw_k: np.ndarray,
        thermal_field_rtw_k: np.ndarray,
        road_state: BoundaryState,
        ambient_temp_k: float,
        friction_to_tire_w: float,
        zone_weights: np.ndarray,
        wheel_angular_speed_radps: float,
        time_s: float,
        dt_s: float,
    ) -> np.ndarray:
        params = self.parameters.flash_layer
        bulk_surface_tw = np.mean(thermal_field_rtw_k[-1, :, :], axis=0)
        if not params.enabled:
            return np.repeat(bulk_surface_tw[None, :], self.parameters.theta_cells, axis=0)

        next_field = np.array(flash_field_tw_k, dtype=float, copy=True)
        surface_cell_areas = self._thermal_solver.surface_cell_areas_m2
        _, theta_indices, width_indices = self._thermal_solver.contact_patch_indices(
            wheel_angular_speed_radps=wheel_angular_speed_radps,
            time_s=time_s,
        )
        patch_area_by_zone = np.sum(surface_cell_areas[np.ix_(theta_indices, width_indices)], axis=0)
        flash_fraction = float(np.clip(params.friction_fraction, 0.0, 0.95))
        patch_theta_set = set(int(idx) for idx in np.asarray(theta_indices, dtype=int))

        for zone_idx, width_idx in enumerate(width_indices):
            q_flash_zone = flash_fraction * friction_to_tire_w * zone_weights[zone_idx]
            zone_patch_area_m2 = max(float(patch_area_by_zone[zone_idx]), 1e-9)
            q_patch_flux_w_per_m2 = q_flash_zone / zone_patch_area_m2
            min_flash_excess_k = 0.0
            if q_flash_zone > 0.0:
                min_flash_excess_k = 0.4 + 6.0 * min(q_patch_flux_w_per_m2 / 180_000.0, 1.0)
            for theta_idx in range(self.parameters.theta_cells):
                area_m2 = max(float(surface_cell_areas[theta_idx, width_idx]), 1e-9)
                temp_k = float(next_field[theta_idx, width_idx])
                bulk_temp_k = float(bulk_surface_tw[width_idx])
                q_bulk = area_m2 * (bulk_temp_k - temp_k) / max(params.bulk_coupling_time_s, 1e-6)
                q_ambient = area_m2 * (ambient_temp_k - temp_k) / max(params.ambient_cooling_time_s, 1e-6)
                q_patch = 0.0
                q_road = 0.0
                if theta_idx in patch_theta_set:
                    q_patch = q_flash_zone * area_m2 / zone_patch_area_m2
                    q_road = area_m2 * (
                        float(road_state.road_surface_temp_w_k[width_idx]) - temp_k
                    ) / max(params.road_cooling_time_s, 1e-6)
                delta_k = dt_s * (q_patch + q_bulk + q_ambient + q_road) / max(
                    params.areal_heat_capacity_j_per_m2k * area_m2,
                    1e-9,
                )
                temp_next_k = temp_k + delta_k
                if theta_idx in patch_theta_set and q_flash_zone > 0.0:
                    temp_next_k = max(temp_next_k, bulk_temp_k + min_flash_excess_k)
                temp_next_k = min(temp_next_k, bulk_temp_k + params.max_delta_above_bulk_k)
                temp_next_k = max(temp_next_k, min(bulk_temp_k, ambient_temp_k) - 15.0)
                next_field[theta_idx, width_idx] = temp_next_k

        return next_field

    def _step_sidewall_field(
        self,
        *,
        sidewall_field_tw_k: np.ndarray,
        thermal_field_rtw_k: np.ndarray,
        inputs: HighFidelityTireInputs,
        rim_temp_k: float,
        brake_heat_to_sidewall_w: float,
        dt_s: float,
    ) -> tuple[np.ndarray, float]:
        next_field = np.array(sidewall_field_tw_k, dtype=float, copy=True)
        shoulder_temp_by_zone = np.mean(thermal_field_rtw_k[int(round(0.45 * (thermal_field_rtw_k.shape[0] - 1))), :, :], axis=0)
        gas_temp_k = float(np.mean(thermal_field_rtw_k[0, :, :]))
        total_heat_w = 0.0
        per_zone_brake = brake_heat_to_sidewall_w / max(self.parameters.width_zones, 1)
        capacity = 1.65e5
        for zone_idx in range(self.parameters.width_zones):
            shoulder_bias = 1.10 if zone_idx != 1 else 0.85
            for theta_idx in range(self.parameters.theta_cells):
                temp = next_field[theta_idx, zone_idx]
                q_carcass = 72.0 * shoulder_bias * (shoulder_temp_by_zone[zone_idx] - temp)
                q_rim = 24.0 * (rim_temp_k - temp)
                q_gas = 16.0 * (gas_temp_k - temp)
                h_amb = 18.0 + 4.5 * inputs.wind_mps * (1.0 + 0.12 * abs(math.sin(inputs.wind_yaw_rad)))
                h_amb *= 1.0 + 0.15 * max(inputs.wheel_wake_factor - 1.0, -0.8)
                q_amb = h_amb * (temp - inputs.ambient_temp_k)
                q_solar = 0.06 * inputs.solar_w_m2 if zone_idx != 1 else 0.04 * inputs.solar_w_m2
                q_brake = per_zone_brake / max(self.parameters.theta_cells, 1)
                delta = dt_s * (q_carcass + q_rim + q_gas + q_solar + q_brake - q_amb) / max(capacity, 1.0)
                next_field[theta_idx, zone_idx] = temp + delta
                total_heat_w += q_carcass + q_rim + q_gas + q_brake
        return next_field, float(total_heat_w / max(self.parameters.theta_cells, 1))

    def _update_gas_temperature_k(
        self,
        *,
        current_gas_temp_k: float,
        carcass_inner_temp_k: float,
        rim_temp_k: float,
        dynamic_pressure_pa: float,
        inputs: HighFidelityTireInputs,
        dt_s: float,
    ) -> float:
        pp = self.parameters.pressure_patch
        gas_mass = pp.resolved_gas_mass_kg()
        q_carcass = 24.0 * (carcass_inner_temp_k - current_gas_temp_k)
        q_rim = 18.0 * (rim_temp_k - current_gas_temp_k)
        q_pdv = dynamic_pressure_pa * inputs.volume_change_rate_m3ps
        delta = dt_s * (q_carcass + q_rim - q_pdv) / max(gas_mass * pp.gas_cv_j_per_kgk, 1.0)
        return float(current_gas_temp_k + delta)

    def _update_surface_state(
        self,
        *,
        state: HighFidelityTireState,
        inputs: HighFidelityTireInputs,
        thermal_field_rtw_k: np.ndarray,
        friction_to_tire_w: float,
        zone_contact_patch_area_m2: np.ndarray,
        dt_s: float,
    ) -> tuple[float, float, np.ndarray, np.ndarray]:
        surface = self.parameters.surface_state
        per_width_surface = np.mean(thermal_field_rtw_k[-1, :, :], axis=0)
        heat_factor = max(friction_to_tire_w / max(surface.wear_reference_heat_w, 1.0), 0.0)
        age_rate = (
            surface.age_energy_gain * surface.wear_reference_heat_w * min(heat_factor, 3.0)
            + surface.age_temperature_gain * max(np.mean(per_width_surface) - surface.age_reference_temperature_k, 0.0)
        )
        age_next = max(state.age_index + dt_s * age_rate, 0.0)

        grain = np.zeros(self.parameters.width_zones, dtype=float) if state.grain_index_w is None else np.array(state.grain_index_w, dtype=float, copy=True)
        blister = np.zeros(self.parameters.width_zones, dtype=float) if state.blister_index_w is None else np.array(state.blister_index_w, dtype=float, copy=True)
        for idx in range(self.parameters.width_zones):
            cool_margin = max(surface.graining_cooling_temp_k - per_width_surface[idx], 0.0) / 20.0
            grain_drive = surface.graining_gain * min(heat_factor, 2.0) * cool_margin
            grain_relax = grain[idx] / max(surface.graining_relaxation_s, 1e-6)
            grain[idx] = float(np.clip(grain[idx] + dt_s * (grain_drive - grain_relax), 0.0, 1.0))

            blister_drive = (
                surface.blister_gain * max(per_width_surface[idx] - surface.blister_threshold_temp_k, 0.0) / 14.0
                + surface.blister_energy_gain * min(heat_factor, 2.5)
            )
            blister_relax = blister[idx] / max(surface.blister_relaxation_s, 1e-6)
            blister[idx] = float(np.clip(blister[idx] + dt_s * (blister_drive - blister_relax), 0.0, 1.0))

        wear_driver = inputs.speed_mps * (0.02 + abs(inputs.slip_ratio_cmd) + abs(math.tan(inputs.slip_angle_cmd_rad)))
        wear_rate = surface.wear_rate_coefficient * wear_driver * max(friction_to_tire_w / max(surface.wear_reference_heat_w, 1.0), 0.2)
        wear_rate *= 1.0 + 0.15 * age_next + 0.10 * float(np.mean(blister))
        wear_next = float(np.clip(state.wear + dt_s * wear_rate, 0.0, 1.0))
        return wear_next, float(age_next), grain, blister

    def _temperature_nodes_from_field(
        self,
        *,
        thermal_field_rtw_k: np.ndarray,
        sidewall_field_tw_k: np.ndarray,
        current_nodes_k: np.ndarray,
        dt_s: float,
    ) -> np.ndarray:
        radial_cells = thermal_field_rtw_k.shape[0]
        surface_idx = radial_cells - 1
        belt_idx = int(round(0.72 * (radial_cells - 1)))
        carcass_idx = int(round(0.45 * (radial_cells - 1)))
        inner_idx = int(round(0.10 * (radial_cells - 1)))

        per_width_surface = np.mean(thermal_field_rtw_k[surface_idx, :, :], axis=0)
        belt_temp = float(np.mean(thermal_field_rtw_k[belt_idx, :, :]))
        carcass_temp = float(np.mean(thermal_field_rtw_k[carcass_idx, :, :]))
        inner_temp = float(np.mean(thermal_field_rtw_k[inner_idx, :, :]))
        core_bulk_temp = self._core_window_temperature(thermal_field_rtw_k)

        nodes = current_nodes_k.copy()
        nodes[0] = float(per_width_surface[0])
        nodes[1] = float(per_width_surface[1])
        nodes[2] = float(per_width_surface[2])
        nodes[3] = belt_temp
        nodes[4] = self._update_core_temperature_k(
            current_core_temp_k=float(nodes[4]),
            thermal_field_rtw_k=thermal_field_rtw_k,
            sidewall_field_tw_k=sidewall_field_tw_k,
            gas_temp_k=float(nodes[5]),
            dt_s=dt_s,
        )
        nodes[5] = 0.65 * nodes[5] + 0.35 * inner_temp
        nodes[8] = float(np.mean(sidewall_field_tw_k))
        return nodes

    def _core_window_temperature(self, thermal_field_rtw_k: np.ndarray | None) -> np.ndarray:
        if thermal_field_rtw_k is None:
            return np.zeros((self.parameters.theta_cells, self.parameters.width_zones), dtype=float)
        radial_cells = thermal_field_rtw_k.shape[0]
        sensor = self.parameters.core_sensor
        depth_fraction = np.clip(sensor.probe_depth_fraction_from_outer, 0.05, 0.95)
        center_idx = int(round((1.0 - depth_fraction) * (radial_cells - 1)))
        half_window = max(int(round(0.10 * radial_cells)), 1)
        start = max(center_idx - half_window, 0)
        end = min(center_idx + half_window + 1, radial_cells)
        return np.mean(thermal_field_rtw_k[start:end, :, :], axis=0)

    def _update_core_temperature_k(
        self,
        *,
        current_core_temp_k: float,
        thermal_field_rtw_k: np.ndarray,
        sidewall_field_tw_k: np.ndarray,
        gas_temp_k: float,
        dt_s: float,
    ) -> float:
        sensor = self.parameters.core_sensor
        bulk_window_tw = self._core_window_temperature(thermal_field_rtw_k)
        bulk_window_w = np.mean(bulk_window_tw, axis=0)
        width_weights = np.asarray(sensor.width_weights, dtype=float)
        width_weights = width_weights / max(float(np.sum(width_weights)), 1e-12)
        bulk_temp_k = float(np.dot(bulk_window_w, width_weights))
        belt_temp_k = float(np.mean(thermal_field_rtw_k[int(round(0.72 * (thermal_field_rtw_k.shape[0] - 1))), :, :]))
        carcass_temp_k = float(np.mean(thermal_field_rtw_k[int(round(0.45 * (thermal_field_rtw_k.shape[0] - 1))), :, :]))
        sidewall_bead_temp_k = float(np.mean(sidewall_field_tw_k[:, (0, 2)]))
        target_core_temp_k = (
            sensor.tread_side_weight * bulk_temp_k
            + sensor.bead_side_weight * sidewall_bead_temp_k
            + sensor.belt_weight * belt_temp_k
            + sensor.carcass_weight * carcass_temp_k
            + sensor.gas_weight * gas_temp_k
        ) / max(
            sensor.tread_side_weight
            + sensor.bead_side_weight
            + sensor.belt_weight
            + sensor.carcass_weight
            + sensor.gas_weight,
            1e-12,
        )
        alpha = 1.0 - np.exp(-dt_s / max(sensor.response_time_s, 1e-6))
        return float(current_core_temp_k + alpha * (target_core_temp_k - current_core_temp_k))

    def _brake_heat_flows(self, *, brake_power_w: float) -> tuple[float, float, float]:
        brake_power_w = max(brake_power_w, 0.0)
        cooling = self.parameters.brake_cooling
        return (
            brake_power_w * max(cooling.brake_heat_to_tire_fraction, 0.0),
            brake_power_w * max(cooling.brake_heat_to_rim_fraction, 0.0),
            brake_power_w * max(cooling.brake_heat_to_sidewall_fraction, 0.0),
        )

    def _tuple3(self, values: np.ndarray | None) -> tuple[float, float, float]:
        if values is None:
            return (0.0, 0.0, 0.0)
        arr = np.asarray(values, dtype=float)
        if arr.shape[0] >= 3:
            return (float(arr[0]), float(arr[1]), float(arr[2]))
        if arr.shape[0] == 1:
            value = float(arr[0])
            return (value, value, value)
        if arr.shape[0] == 2:
            return (float(arr[0]), float(np.mean(arr)), float(arr[1]))
        return (0.0, 0.0, 0.0)

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
