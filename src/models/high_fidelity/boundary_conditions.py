from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class HighFidelityBoundaryParameters:
    """Boundary/contact parameters for the P4 thermal model."""

    eta_tire: float = 0.72
    h_cp_w_per_m2k: float = 2_600.0
    h_c_bead_w_per_m2k: float = 1_100.0
    use_pressure_sensitive_bead: bool = True
    bead_pressure_exponent: float = 0.35
    reference_normal_load_n: float = 3_500.0
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


@dataclass(frozen=True)
class BoundaryState:
    road_surface_temp_k: float
    road_subsurface_temp_k: float


@dataclass(frozen=True)
class BoundaryHeatFlows:
    friction_total_w: float
    friction_to_tire_w: float
    friction_to_road_w: float
    road_conduction_w: float
    rim_conduction_w: float
    effective_bead_htc_w_per_m2k: float


class BoundaryConditionModel:
    """P4 boundary/contact model with tire-road and tire-rim heat transfer."""

    def __init__(self, parameters: HighFidelityBoundaryParameters) -> None:
        self.parameters = parameters

    def initial_state(
        self,
        *,
        road_surface_temp_k: float,
        road_subsurface_temp_k: float | None = None,
    ) -> BoundaryState:
        subsurface = (
            road_surface_temp_k
            if road_subsurface_temp_k is None
            else float(road_subsurface_temp_k)
        )
        return BoundaryState(
            road_surface_temp_k=float(road_surface_temp_k),
            road_subsurface_temp_k=subsurface,
        )

    def frictional_contact_power_w(
        self,
        *,
        speed_mps: float,
        slip_ratio: float,
        slip_angle_rad: float,
        normal_load_n: float,
    ) -> float:
        slip_speed = abs(speed_mps) * math.sqrt((slip_ratio**2) + math.tan(slip_angle_rad) ** 2)
        slip_speed = min(slip_speed, self.parameters.max_slip_speed_mps)
        return max(
            self.parameters.friction_coefficient * max(normal_load_n, 0.0) * slip_speed,
            0.0,
        )

    def partition_friction_power(
        self,
        *,
        total_friction_power_w: float,
    ) -> tuple[float, float]:
        eta_tire = min(max(self.parameters.eta_tire, 0.0), 1.0)
        tire = max(total_friction_power_w, 0.0) * eta_tire
        road = max(total_friction_power_w, 0.0) - tire
        return tire, road

    def road_conduction_power_w(
        self,
        *,
        tire_surface_temp_k: float,
        road_surface_temp_k: float,
    ) -> float:
        h_cp = max(self.parameters.h_cp_w_per_m2k, 0.0)
        area = max(self.parameters.contact_patch_area_m2, 1e-9)
        return h_cp * area * (tire_surface_temp_k - road_surface_temp_k)

    def effective_bead_htc_w_per_m2k(self, *, normal_load_n: float) -> float:
        base = max(self.parameters.h_c_bead_w_per_m2k, 0.0)
        if not self.parameters.use_pressure_sensitive_bead:
            return base

        load_ratio = max(normal_load_n, 0.0) / max(self.parameters.reference_normal_load_n, 1e-9)
        return base * (load_ratio ** self.parameters.bead_pressure_exponent)

    def rim_conduction_power_w(
        self,
        *,
        tire_inner_temp_k: float,
        rim_temp_k: float,
        normal_load_n: float,
    ) -> tuple[float, float]:
        h_c = self.effective_bead_htc_w_per_m2k(normal_load_n=normal_load_n)
        area = max(self.parameters.bead_contact_area_m2, 1e-9)
        return h_c * area * (tire_inner_temp_k - rim_temp_k), h_c

    def step_road_slab(
        self,
        *,
        state: BoundaryState,
        dt_s: float,
        heat_input_surface_w: float,
        road_bulk_temp_k: float | None,
    ) -> BoundaryState:
        if dt_s <= 0.0:
            msg = f"dt_s must be positive, got {dt_s}"
            raise ValueError(msg)

        params = self.parameters
        area = max(params.contact_patch_area_m2, 1e-9)
        depth_surface = max(params.road_surface_layer_depth_m, 1e-6)
        depth_subsurface = max(params.road_subsurface_layer_depth_m, 1e-6)

        c_surface = params.road_volumetric_heat_capacity_j_per_m3k * area * depth_surface
        c_subsurface = params.road_volumetric_heat_capacity_j_per_m3k * area * depth_subsurface

        effective_spacing = 0.5 * (depth_surface + depth_subsurface)
        g_surface_sub = params.road_thermal_conductivity_w_per_mk * area / max(effective_spacing, 1e-9)
        g_bulk = max(params.road_bulk_coupling_w_per_k, 0.0)
        t_bulk = state.road_subsurface_temp_k if road_bulk_temp_k is None else road_bulk_temp_k

        q_surface_to_sub = g_surface_sub * (state.road_surface_temp_k - state.road_subsurface_temp_k)
        q_sub_to_bulk = g_bulk * (state.road_subsurface_temp_k - t_bulk)

        t_surface_next = state.road_surface_temp_k + dt_s * (
            heat_input_surface_w - q_surface_to_sub
        ) / max(c_surface, 1e-9)
        t_subsurface_next = state.road_subsurface_temp_k + dt_s * (
            q_surface_to_sub - q_sub_to_bulk
        ) / max(c_subsurface, 1e-9)
        return BoundaryState(
            road_surface_temp_k=float(t_surface_next),
            road_subsurface_temp_k=float(t_subsurface_next),
        )

    def step_rim_temperature(
        self,
        *,
        rim_temp_k: float,
        ambient_temp_k: float,
        heat_input_w: float,
        dt_s: float,
    ) -> float:
        c_rim = max(self.parameters.rim_heat_capacity_j_per_k, 1e-6)
        g_rim_amb = max(self.parameters.rim_to_ambient_conductance_w_per_k, 0.0)
        q_amb = g_rim_amb * (rim_temp_k - ambient_temp_k)
        return float(rim_temp_k + dt_s * (heat_input_w - q_amb) / c_rim)

