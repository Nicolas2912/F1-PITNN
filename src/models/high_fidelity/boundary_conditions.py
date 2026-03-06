from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class HighFidelityBoundaryParameters:
    """Boundary, road, and environment parameters for the layered HF model."""

    eta_tire: float = 0.72
    h_cp_w_per_m2k: float = 2_600.0
    h_c_bead_w_per_m2k: float = 1_100.0
    use_pressure_sensitive_bead: bool = True
    bead_pressure_exponent: float = 0.35
    reference_normal_load_n: float = 3_500.0
    reference_pressure_pa: float = 240_000.0
    contact_patch_area_m2: float = 0.018
    bead_contact_area_m2: float = 0.010
    road_surface_layer_depth_m: float = 0.003
    road_subsurface_layer_depth_m: float = 0.040
    road_thermal_conductivity_w_per_mk: float = 1.15
    road_volumetric_heat_capacity_j_per_m3k: float = 2.0e6
    road_bulk_coupling_w_per_k: float = 40.0
    rim_heat_capacity_j_per_k: float = 8_200.0
    rim_to_ambient_conductance_w_per_k: float = 14.0
    friction_coefficient: float = 1.35
    max_slip_speed_mps: float = 45.0
    wind_cooling_gain: float = 0.12
    solar_absorptivity: float = 0.78
    humidity_cooling_gain: float = 0.10
    moisture_evaporation_gain: float = 0.08
    roughness_mu_gain: float = 0.10
    rubbering_mu_gain: float = 0.06
    effusivity_eta_gain: float = 0.08
    moisture_hcp_gain: float = 0.45


@dataclass(frozen=True)
class BoundaryState:
    road_surface_temp_w_k: np.ndarray
    road_subsurface_temp_w_k: np.ndarray
    road_moisture_w: np.ndarray

    @property
    def road_surface_temp_k(self) -> float:
        return float(np.mean(self.road_surface_temp_w_k))

    @property
    def road_subsurface_temp_k(self) -> float:
        return float(np.mean(self.road_subsurface_temp_w_k))


@dataclass(frozen=True)
class BoundaryHeatFlows:
    friction_total_w: float
    friction_to_tire_w: float
    friction_to_road_w: float
    road_conduction_w: float
    road_conduction_w_by_zone: np.ndarray
    rim_conduction_w: float
    effective_bead_htc_w_per_m2k: float


class BoundaryConditionModel:
    """Layered road/rim/environment model with per-width road state."""

    def __init__(self, parameters: HighFidelityBoundaryParameters) -> None:
        self.parameters = parameters

    def initial_state(
        self,
        *,
        road_surface_temp_k: float,
        road_subsurface_temp_k: float | None = None,
        road_moisture: float = 0.0,
        width_zones: int = 3,
    ) -> BoundaryState:
        subsurface = road_surface_temp_k if road_subsurface_temp_k is None else float(road_subsurface_temp_k)
        road_surface_temp_w_k = np.full(width_zones, float(road_surface_temp_k), dtype=float)
        road_subsurface_temp_w_k = np.full(width_zones, float(subsurface), dtype=float)
        road_moisture_w = np.full(width_zones, float(np.clip(road_moisture, 0.0, 1.0)), dtype=float)
        return BoundaryState(
            road_surface_temp_w_k=road_surface_temp_w_k,
            road_subsurface_temp_w_k=road_subsurface_temp_w_k,
            road_moisture_w=road_moisture_w,
        )

    def frictional_contact_power_w(
        self,
        *,
        speed_mps: float,
        slip_ratio: float,
        slip_angle_rad: float,
        normal_load_n: float,
        road_moisture: float = 0.0,
        rubbering_level: float = 0.0,
        asphalt_roughness: float = 1.0,
    ) -> float:
        slip_speed = abs(speed_mps) * math.sqrt((slip_ratio**2) + math.tan(slip_angle_rad) ** 2)
        slip_speed = min(slip_speed, self.parameters.max_slip_speed_mps)
        mu = self._effective_mu(
            road_moisture=road_moisture,
            rubbering_level=rubbering_level,
            asphalt_roughness=asphalt_roughness,
        )
        return max(mu * max(normal_load_n, 0.0) * slip_speed, 0.0)

    def partition_friction_power(
        self,
        *,
        total_friction_power_w: float,
        road_moisture: float = 0.0,
        asphalt_effusivity: float = 1.0,
        rubbering_level: float = 0.0,
    ) -> tuple[float, float]:
        eta = np.clip(
            self.parameters.eta_tire
            - 0.15 * np.clip(road_moisture, 0.0, 1.0)
            + self.parameters.effusivity_eta_gain * (asphalt_effusivity - 1.0)
            + 0.04 * np.clip(rubbering_level, 0.0, 1.0),
            0.20,
            0.98,
        )
        tire = max(total_friction_power_w, 0.0) * eta
        road = max(total_friction_power_w, 0.0) - tire
        return tire, road

    def partition_friction_power_by_zone(
        self,
        *,
        zone_friction_power_w: np.ndarray,
        flash_temp_w_k: np.ndarray,
        bulk_temp_w_k: np.ndarray,
        road_surface_temp_w_k: np.ndarray,
        road_moisture_w: np.ndarray,
        asphalt_effusivity: float = 1.0,
        rubbering_level: float = 0.0,
        zone_sliding_fraction: np.ndarray | None = None,
        zone_contact_temp_w_k: np.ndarray | None = None,
        zone_contact_pressure_factor: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        zone_friction = np.asarray(zone_friction_power_w, dtype=float)
        flash_temp = np.asarray(flash_temp_w_k, dtype=float)
        bulk_temp = np.asarray(bulk_temp_w_k, dtype=float)
        road_temp = np.asarray(road_surface_temp_w_k, dtype=float)
        moisture = np.asarray(road_moisture_w, dtype=float)
        sliding_fraction = np.zeros_like(zone_friction, dtype=float) if zone_sliding_fraction is None else np.asarray(zone_sliding_fraction, dtype=float)
        contact_temp = np.asarray(bulk_temp, dtype=float) if zone_contact_temp_w_k is None else np.asarray(zone_contact_temp_w_k, dtype=float)
        pressure_factor = np.ones_like(zone_friction, dtype=float) if zone_contact_pressure_factor is None else np.asarray(zone_contact_pressure_factor, dtype=float)
        if not (
            zone_friction.shape
            == flash_temp.shape
            == bulk_temp.shape
            == road_temp.shape
            == moisture.shape
            == sliding_fraction.shape
            == contact_temp.shape
            == pressure_factor.shape
        ):
            msg = "Zone-local partition inputs must have matching shapes"
            raise ValueError(msg)

        eta_w = np.zeros_like(zone_friction, dtype=float)
        tire_w = np.zeros_like(zone_friction, dtype=float)
        road_w = np.zeros_like(zone_friction, dtype=float)
        for idx in range(zone_friction.shape[0]):
            delta_flash = max(float(flash_temp[idx] - bulk_temp[idx]), 0.0)
            delta_road = float(bulk_temp[idx] - road_temp[idx])
            eta = self.parameters.eta_tire
            eta += 0.0018 * delta_flash
            eta += 0.0008 * delta_road
            eta -= 0.15 * np.clip(float(moisture[idx]), 0.0, 1.0)
            eta += self.parameters.effusivity_eta_gain * (asphalt_effusivity - 1.0)
            eta += 0.04 * np.clip(rubbering_level, 0.0, 1.0)
            eta += 0.08 * np.clip(float(sliding_fraction[idx]), 0.0, 1.0)
            eta += 0.0006 * max(float(contact_temp[idx]) - float(bulk_temp[idx]), 0.0)
            eta += 0.05 * (float(pressure_factor[idx]) - 1.0)
            eta = float(np.clip(eta, 0.20, 0.98))
            power = max(float(zone_friction[idx]), 0.0)
            eta_w[idx] = eta
            tire_w[idx] = power * eta
            road_w[idx] = power - tire_w[idx]
        return tire_w, road_w, eta_w

    def road_conduction_power_w(
        self,
        *,
        tire_surface_temp_k: float,
        road_surface_temp_k: float,
        contact_patch_area_m2: float | None = None,
        road_moisture: float = 0.0,
        asphalt_effusivity: float = 1.0,
        wind_mps: float = 0.0,
    ) -> float:
        area = self.parameters.contact_patch_area_m2 if contact_patch_area_m2 is None else contact_patch_area_m2
        h_cp = self._effective_contact_htc(
            road_moisture=road_moisture,
            asphalt_effusivity=asphalt_effusivity,
            wind_mps=wind_mps,
        )
        return h_cp * max(area, 1e-9) * (tire_surface_temp_k - road_surface_temp_k)

    def road_conduction_power_w_by_zone(
        self,
        *,
        tire_surface_temp_w_k: np.ndarray,
        road_surface_temp_w_k: np.ndarray,
        zone_contact_patch_area_m2: np.ndarray,
        road_moisture_w: np.ndarray,
        asphalt_effusivity: float,
        wind_mps: float,
    ) -> np.ndarray:
        outputs = np.zeros_like(tire_surface_temp_w_k, dtype=float)
        for idx in range(tire_surface_temp_w_k.shape[0]):
            outputs[idx] = self.road_conduction_power_w(
                tire_surface_temp_k=float(tire_surface_temp_w_k[idx]),
                road_surface_temp_k=float(road_surface_temp_w_k[idx]),
                contact_patch_area_m2=float(zone_contact_patch_area_m2[idx]),
                road_moisture=float(road_moisture_w[idx]),
                asphalt_effusivity=asphalt_effusivity,
                wind_mps=wind_mps,
            )
        return outputs

    def effective_bead_htc_w_per_m2k(
        self,
        *,
        normal_load_n: float,
        dynamic_pressure_pa: float | None = None,
    ) -> float:
        base = max(self.parameters.h_c_bead_w_per_m2k, 0.0)
        if not self.parameters.use_pressure_sensitive_bead:
            return base
        load_ratio = max(normal_load_n, 0.0) / max(self.parameters.reference_normal_load_n, 1e-9)
        pressure_ratio = 1.0
        if dynamic_pressure_pa is not None:
            pressure_ratio = max(dynamic_pressure_pa, 1.0) / max(self.parameters.reference_pressure_pa, 1.0)
        return base * (load_ratio ** self.parameters.bead_pressure_exponent) * (pressure_ratio ** 0.12)

    def rim_conduction_power_w(
        self,
        *,
        tire_inner_temp_k: float,
        rim_temp_k: float,
        normal_load_n: float,
        dynamic_pressure_pa: float | None = None,
        contact_area_m2: float | None = None,
    ) -> tuple[float, float]:
        h_c = self.effective_bead_htc_w_per_m2k(
            normal_load_n=normal_load_n,
            dynamic_pressure_pa=dynamic_pressure_pa,
        )
        area = self.parameters.bead_contact_area_m2 if contact_area_m2 is None else contact_area_m2
        return h_c * max(area, 1e-9) * (tire_inner_temp_k - rim_temp_k), h_c

    def step_road_slab(
        self,
        *,
        state: BoundaryState,
        dt_s: float,
        heat_input_surface_w: float | np.ndarray,
        road_bulk_temp_k: float | np.ndarray | None,
        asphalt_effusivity: float = 1.0,
        solar_w_m2: float = 0.0,
        wind_mps: float = 0.0,
    ) -> BoundaryState:
        if dt_s <= 0.0:
            msg = f"dt_s must be positive, got {dt_s}"
            raise ValueError(msg)

        params = self.parameters
        width_zones = state.road_surface_temp_w_k.shape[0]
        area_per_zone = max(params.contact_patch_area_m2, 1e-9) / max(width_zones, 1)
        depth_surface = max(params.road_surface_layer_depth_m, 1e-6)
        depth_subsurface = max(params.road_subsurface_layer_depth_m, 1e-6)
        heat_input_surface_w = self._broadcast_width_values(heat_input_surface_w, width_zones)
        t_bulk = state.road_subsurface_temp_w_k if road_bulk_temp_k is None else self._broadcast_width_values(
            road_bulk_temp_k,
            width_zones,
        )

        surface = state.road_surface_temp_w_k.copy()
        subsurface = state.road_subsurface_temp_w_k.copy()
        moisture = state.road_moisture_w.copy()
        for idx in range(width_zones):
            conductivity = params.road_thermal_conductivity_w_per_mk * np.clip(asphalt_effusivity, 0.5, 1.8)
            c_surface = params.road_volumetric_heat_capacity_j_per_m3k * area_per_zone * depth_surface
            c_subsurface = params.road_volumetric_heat_capacity_j_per_m3k * area_per_zone * depth_subsurface
            effective_spacing = 0.5 * (depth_surface + depth_subsurface)
            g_surface_sub = conductivity * area_per_zone / max(effective_spacing, 1e-9)
            g_bulk = max(params.road_bulk_coupling_w_per_k, 0.0)
            q_surface_to_sub = g_surface_sub * (surface[idx] - subsurface[idx])
            q_sub_to_bulk = g_bulk * (subsurface[idx] - t_bulk[idx])
            q_solar = params.solar_absorptivity * solar_w_m2 * area_per_zone
            q_evap = (
                params.moisture_evaporation_gain
                * max(wind_mps, 0.0)
                * moisture[idx]
                * max(surface[idx] - 273.15, 0.0)
            )
            surface[idx] = surface[idx] + dt_s * (
                heat_input_surface_w[idx] + q_solar - q_surface_to_sub - q_evap
            ) / max(c_surface, 1e-9)
            subsurface[idx] = subsurface[idx] + dt_s * (q_surface_to_sub - q_sub_to_bulk) / max(c_subsurface, 1e-9)
            moisture[idx] = float(
                np.clip(
                    moisture[idx] - dt_s * 0.0025 * max(surface[idx] - 290.0, 0.0) * (1.0 + 0.15 * wind_mps),
                    0.0,
                    1.0,
                )
            )
        return BoundaryState(
            road_surface_temp_w_k=surface,
            road_subsurface_temp_w_k=subsurface,
            road_moisture_w=moisture,
        )

    def step_rim_temperature(
        self,
        *,
        rim_temp_k: float,
        ambient_temp_k: float,
        heat_input_w: float,
        dt_s: float,
        brake_duct_cooling_factor: float = 1.0,
        wheel_wake_factor: float = 1.0,
        wheel_angular_speed_radps: float = 0.0,
        external_cooling_gain: float = 1.0,
    ) -> float:
        c_rim = max(self.parameters.rim_heat_capacity_j_per_k, 1e-6)
        g_rim_amb = max(self.parameters.rim_to_ambient_conductance_w_per_k, 0.0)
        cooling_gain = (
            1.0
            + 0.25 * max(brake_duct_cooling_factor - 1.0, -0.8)
            + 0.20 * max(wheel_wake_factor - 1.0, -0.8)
        )
        speed_gain = 1.0 + 0.02 * min(max(abs(wheel_angular_speed_radps), 0.0) / 200.0, 2.5)
        q_amb = g_rim_amb * cooling_gain * speed_gain * max(external_cooling_gain, 0.2) * (rim_temp_k - ambient_temp_k)
        return float(rim_temp_k + dt_s * (heat_input_w - q_amb) / c_rim)

    def _effective_mu(
        self,
        *,
        road_moisture: float,
        rubbering_level: float,
        asphalt_roughness: float,
    ) -> float:
        mu = self.parameters.friction_coefficient
        mu *= 1.0 - 0.35 * np.clip(road_moisture, 0.0, 1.0)
        mu *= 1.0 + self.parameters.roughness_mu_gain * (asphalt_roughness - 1.0)
        mu *= 1.0 + self.parameters.rubbering_mu_gain * np.clip(rubbering_level, 0.0, 1.0)
        return max(mu, 0.20)

    def _effective_contact_htc(
        self,
        *,
        road_moisture: float,
        asphalt_effusivity: float,
        wind_mps: float,
    ) -> float:
        h_cp = max(self.parameters.h_cp_w_per_m2k, 0.0)
        h_cp *= 1.0 + self.parameters.moisture_hcp_gain * np.clip(road_moisture, 0.0, 1.0)
        h_cp *= np.clip(asphalt_effusivity, 0.5, 1.8)
        h_cp *= 1.0 + self.parameters.wind_cooling_gain * min(max(wind_mps, 0.0) / 20.0, 1.0)
        return h_cp

    def _broadcast_width_values(
        self,
        values: float | np.ndarray,
        width_zones: int,
    ) -> np.ndarray:
        if isinstance(values, np.ndarray):
            if values.shape == (width_zones,):
                return values.astype(float, copy=True)
            msg = f"Expected width array shape ({width_zones},), got {values.shape}"
            raise ValueError(msg)
        return np.full(width_zones, float(values), dtype=float)
