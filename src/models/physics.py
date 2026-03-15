from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

KELVIN_OFFSET = 273.15
STANDARD_ATMOSPHERIC_PRESSURE_PA = 101_325.0
STEFAN_BOLTZMANN_CONSTANT = 5.670_374_419e-8

SURFACE_INNER_INDEX = 0
SURFACE_MIDDLE_INDEX = 1
SURFACE_OUTER_INDEX = 2
BELT_INDEX = 3
CARCASS_INDEX = 4
GAS_INDEX = 5
RIM_INDEX = 6
BRAKE_INDEX = 7
SIDEWALL_INDEX = 8
KAPPA_DYN_INDEX = 9
ALPHA_DYN_INDEX = 10
WEAR_INDEX = 11
STATE_DIMENSION = 12
PatchGrid3x3 = tuple[
    tuple[float, float, float],
    tuple[float, float, float],
    tuple[float, float, float],
]


def celsius_to_kelvin(value_c: float) -> float:
    return value_c + KELVIN_OFFSET


def kelvin_to_celsius(value_k: float) -> float:
    return value_k - KELVIN_OFFSET


@dataclass(frozen=True)
class TireState:
    """Thermal state (9 thermal nodes + 2 slip states + scalar wear state)."""

    t_surface_inner_k: float
    t_surface_middle_k: float
    t_surface_outer_k: float
    t_belt_k: float
    t_carcass_k: float
    t_gas_k: float
    t_rim_k: float
    t_brake_k: float
    t_sidewall_k: float
    kappa_dyn: float = 0.0
    alpha_dyn_rad: float = 0.0
    wear: float = 0.0
    time_s: float = 0.0

    @property
    def core_temperature_k(self) -> float:
        # "Core" is best approximated by the carcass/inner-liner node.
        return self.t_carcass_k

    @property
    def core_temperature_c(self) -> float:
        return kelvin_to_celsius(self.core_temperature_k)

    def as_vector(self) -> np.ndarray:
        return np.array(
            [
                self.t_surface_inner_k,
                self.t_surface_middle_k,
                self.t_surface_outer_k,
                self.t_belt_k,
                self.t_carcass_k,
                self.t_gas_k,
                self.t_rim_k,
                self.t_brake_k,
                self.t_sidewall_k,
                self.kappa_dyn,
                self.alpha_dyn_rad,
                self.wear,
            ],
            dtype=float,
        )

    @classmethod
    def from_vector(cls, values: np.ndarray, time_s: float) -> TireState:
        if values.shape != (STATE_DIMENSION,):
            msg = f"Expected shape {(STATE_DIMENSION,)}, got {values.shape}"
            raise ValueError(msg)
        return cls(
            t_surface_inner_k=float(values[SURFACE_INNER_INDEX]),
            t_surface_middle_k=float(values[SURFACE_MIDDLE_INDEX]),
            t_surface_outer_k=float(values[SURFACE_OUTER_INDEX]),
            t_belt_k=float(values[BELT_INDEX]),
            t_carcass_k=float(values[CARCASS_INDEX]),
            t_gas_k=float(values[GAS_INDEX]),
            t_rim_k=float(values[RIM_INDEX]),
            t_brake_k=float(values[BRAKE_INDEX]),
            t_sidewall_k=float(values[SIDEWALL_INDEX]),
            kappa_dyn=float(values[KAPPA_DYN_INDEX]),
            alpha_dyn_rad=float(values[ALPHA_DYN_INDEX]),
            wear=float(values[WEAR_INDEX]),
            time_s=time_s,
        )


@dataclass(frozen=True)
class TireInputs:
    """External excitations for a single integration step."""

    speed_mps: float
    wheel_angular_speed_radps: float
    normal_load_n: float
    slip_ratio: float
    slip_angle_rad: float
    slip_ratio_cmd: float | None = None
    slip_angle_cmd_rad: float | None = None
    brake_power_w: float = 0.0
    ambient_temp_k: float = celsius_to_kelvin(25.0)
    track_temp_k: float = celsius_to_kelvin(35.0)
    volume_change_rate_m3ps: float = 0.0
    normal_load_rate_nps: float = 0.0
    wheel_angular_accel_radps2: float = 0.0
    zone_load_split: tuple[float, float, float] = (0.33, 0.34, 0.33)
    camber_rad: float = 0.0
    toe_rad: float = 0.0
    lateral_accel_mps2: float = 0.0
    longitudinal_accel_mps2: float = 0.0
    is_left_tire: bool = True
    is_front_tire: bool = True


@dataclass(frozen=True)
class TireDiagnostics:
    dynamic_pressure_pa: float
    dynamic_pressure_bar_gauge: float
    volume_m3: float
    volume_rate_m3ps: float
    contact_patch_area_m2: float
    zone_contact_areas_m2: tuple[float, float, float]
    effective_slip_ratio: float
    effective_slip_angle_rad: float
    slip_speed_mps: float
    external_htc_w_m2k: tuple[float, float, float]
    internal_htc_w_m2k: float
    tread_mass_kg: float
    tread_thickness_m: float
    rolling_resistance_total_w: float
    rolling_resistance_belt_tread_w: float
    rolling_resistance_sidewall_w: float
    friction_power_total_w: float
    friction_power_adhesion_w: float
    friction_power_hysteresis_w: float
    friction_power_flash_w: float
    patch_pressure_grid_pa: PatchGrid3x3
    patch_shear_grid_pa: PatchGrid3x3
    patch_cell_area_m2: float
    zone_friction_power_w: tuple[float, float, float]


@dataclass(frozen=True)
class TireModelParameters:
    # Feature toggles for staged realism upgrades.
    use_gauge_patch_model: bool = True
    use_combined_slip_model: bool = True
    use_zone_lateral_conduction: bool = True
    use_temperature_dependent_properties: bool = True
    use_rotating_internal_gas_model: bool = True
    use_alignment_zone_effects: bool = True
    use_sidewall_rr_split_model: bool = False
    use_slip_transient_model: bool = False
    use_quasi_2d_patch_model: bool = False
    use_friction_partition_model: bool = False

    # Gas and pressure model.
    pressure_static_bar_gauge: float = 1.4
    atmospheric_pressure_pa: float = STANDARD_ATMOSPHERIC_PRESSURE_PA
    gas_specific_constant_j_per_kgk: float = 287.05
    gas_cv_j_per_kgk: float = 718.0
    reference_fill_temp_k: float = celsius_to_kelvin(20.0)
    gas_mass_kg: float | None = None

    # Mechanical volume dynamics.
    base_volume_m3: float = 0.030
    minimum_volume_m3: float = 0.016
    centrifugal_volume_gain_coeff_m3_per_radps2: float = 8.0e-9
    deflection_volume_loss_coeff_m3_per_n: float = 1.2e-7
    wear_volume_loss_m3: float = 0.0015

    # Contact patch and friction model.
    contact_pressure_factor: float = 1.0
    min_contact_patch_area_m2: float = 5.0e-4
    max_contact_patch_area_m2: float = 0.045
    carcass_support_pressure_pa: float = 72_000.0
    carcass_support_load_gain: float = 0.22
    carcass_support_temp_gain_per_k: float = 1.5e-3
    min_effective_contact_pressure_pa: float = 20_000.0
    shape_sensitivity: float = 0.35
    base_shear_stress_pa: float = 8.4e4
    reference_load_n: float = 3_500.0
    shear_load_gain: float = 0.35
    heating_drop_temp_k: float = celsius_to_kelvin(82.0)
    heating_drop_width_k: float = 6.0
    heating_min_factor: float = 0.08
    mu_peak_reference: float = 1.78
    mu_load_sensitivity: float = 0.12
    mu_temperature_peak_k: float = celsius_to_kelvin(92.0)
    mu_temperature_width_k: float = 22.0
    mu_min_fraction: float = 0.45
    mu_patch_sensitivity: float = 0.08
    reference_contact_patch_area_m2: float = 0.018
    combined_slip_shape: float = 1.55
    slip_ratio_reference: float = 0.11
    slip_angle_reference_rad: float = 0.09
    toe_slip_gain: float = 0.45
    friction_heat_fraction: float = 0.72

    # Rolling resistance and sidewall split model.
    rolling_resistance_coeff: float = 0.0018
    rr_belt_tread_fraction: float = 0.85
    rr_sidewall_fraction: float = 0.15
    sidewall_hysteresis_fraction: float = 0.12

    # Slip transients (relaxation-length model).
    relaxation_length_long_m: float = 12.0
    relaxation_length_lat_m: float = 15.0
    slip_transition_speed_mps: float = 5.0
    max_slip_relaxation_rate_per_s: float = 12.0

    # Quasi-2D patch pressure/shear model.
    patch_camber_pressure_skew: float = 0.24
    patch_longitudinal_pressure_skew: float = 0.18
    patch_pressure_min_fraction: float = 0.25
    patch_pressure_norm_tol_n: float = 1e-3

    # Friction heat partition (adhesion / hysteresis / flash).
    friction_partition_adhesion: float = 0.45
    friction_partition_hysteresis: float = 0.35
    friction_partition_flash: float = 0.20
    flash_surface_time_constant_s: float = 0.18

    # Material properties and geometric areas.
    cp_rubber_j_per_kgk: float = 1_850.0
    k_rubber_w_per_mk: float = 0.28
    k_belt_w_per_mk: float = 45.0
    reference_material_temp_k: float = celsius_to_kelvin(80.0)
    cp_temp_coeff_per_k: float = 1.2e-3
    k_rubber_temp_coeff_per_k: float = 8.5e-4
    k_belt_temp_coeff_per_k: float = 2.0e-4
    min_cp_scale: float = 0.85
    max_cp_scale: float = 1.22
    min_k_scale: float = 0.75
    max_k_scale: float = 1.20
    m_tread_fresh_kg: float = 3.5
    m_tread_worn_kg: float = 1.5
    tread_thickness_fresh_m: float = 0.012
    tread_thickness_worn_m: float = 0.004
    m_belt_kg: float = 2.9
    c_belt_j_per_kgk: float = 500.0
    m_carcass_kg: float = 3.8
    m_sidewall_kg: float = 1.6
    belt_to_carcass_area_m2: float = 0.115
    belt_to_carcass_path_m: float = 0.0045
    belt_to_rim_conductance_w_per_k: float = 100.0
    carcass_external_area_m2: float = 0.26
    carcass_external_h_base_w_per_m2k: float = 30.0
    carcass_external_h_speed_gain: float = 0.35
    internal_area_m2: float = 0.47
    rim_to_gas_area_m2: float = 0.22
    sidewall_heat_capacity_j_per_kgk: float = 1_850.0
    sidewall_area_m2: float = 0.18
    sidewall_to_carcass_conductance_w_per_k: float = 85.0
    sidewall_to_rim_conductance_w_per_k: float = 55.0
    sidewall_external_h_base_w_per_m2k: float = 24.0
    sidewall_external_h_speed_gain: float = 0.30
    sidewall_internal_h_fraction: float = 0.45

    # External convection (power-law Nu correlation).
    char_length_m: float = 0.27
    air_density_kg_per_m3: float = 1.18
    air_dynamic_viscosity_pa_s: float = 1.85e-5
    air_prandtl: float = 0.71
    air_thermal_conductivity_w_per_mk: float = 0.026
    min_external_htc_w_per_m2k: float = 8.0
    zone_external_h_multipliers: tuple[float, float, float] = (1.30, 1.0, 0.95)
    exposed_surface_area_m2: float = 0.24
    track_contact_htc_w_per_m2k: float = 2_600.0
    surface_emissivity: float = 0.94
    tread_to_belt_area_fraction: float = 0.32
    zone_lateral_conductance_w_per_k: float = 18.0
    zone_lateral_conductance_wear_gain: float = 0.70

    # Internal convection.
    gas_thermal_conductivity_w_per_mk: float = 0.028
    gas_dynamic_viscosity_pa_s: float = 1.9e-5
    gas_prandtl: float = 0.71
    internal_flow_diameter_m: float = 0.56
    internal_gas_gap_m: float = 0.055
    internal_nusselt_floor: float = 3.0
    gas_mixing_time_constant_s: float = 0.35
    gas_mixing_floor: float = 0.24
    min_internal_htc_w_per_m2k: float = 8.0
    max_internal_htc_w_per_m2k: float = 180.0

    # Hysteresis and wear.
    loss_tangent: float = 0.30
    loss_tangent_peak_temp_k: float = celsius_to_kelvin(88.0)
    loss_tangent_temp_width_k: float = 18.0
    loss_tangent_min_fraction: float = 0.55
    loss_tangent_freq_gain: float = 0.08
    reference_angular_speed_radps: float = 180.0
    hysteresis_gain: float = 4.0e-4
    hysteresis_drop_temp_k: float = celsius_to_kelvin(88.0)
    hysteresis_drop_width_k: float = 8.0
    wear_rate_coefficient: float = 3.4e-4
    wear_reference_heat_w: float = 30_000.0

    # Rim and brake nodes.
    m_rim_kg: float = 9.0
    c_rim_j_per_kgk: float = 880.0
    rim_external_area_m2: float = 0.16
    h_rim_external_w_per_m2k: float = 55.0
    h_rim_to_gas_w_per_m2k: float = 34.0
    m_brake_kg: float = 7.0
    c_brake_j_per_kgk: float = 530.0
    brake_area_m2: float = 0.20
    brake_emissivity: float = 0.9
    h_brake_conv_w_per_m2k: float = 65.0
    brake_to_rim_fraction: float = 0.40

    # Numerical safety.
    minimum_temperature_k: float = 160.0
    maximum_temperature_k: float = 2_200.0

    # Zone split sensitivities for camber/toe/lateral acceleration.
    zone_camber_sensitivity: float = 0.26
    zone_toe_sensitivity: float = 0.12
    zone_lateral_accel_sensitivity: float = 0.20
    zone_longitudinal_accel_mid_gain: float = 0.12
    zone_angle_scale_rad: float = 0.12

    def resolved_gas_mass_kg(self) -> float:
        if self.gas_mass_kg is not None:
            return self.gas_mass_kg
        static_abs_pressure = self.pressure_static_bar_gauge * 100_000.0 + self.atmospheric_pressure_pa
        return (
            static_abs_pressure
            * self.base_volume_m3
            / (self.gas_specific_constant_j_per_kgk * self.reference_fill_temp_k)
        )

    def tread_mass_kg(self, wear: float) -> float:
        wear_clamped = min(max(wear, 0.0), 1.0)
        return self.m_tread_fresh_kg + wear_clamped * (self.m_tread_worn_kg - self.m_tread_fresh_kg)

    def tread_thickness_m(self, wear: float) -> float:
        wear_clamped = min(max(wear, 0.0), 1.0)
        return self.tread_thickness_fresh_m + wear_clamped * (
            self.tread_thickness_worn_m - self.tread_thickness_fresh_m
        )

    def rr_split_fractions(self) -> tuple[float, float]:
        belt = max(self.rr_belt_tread_fraction, 0.0)
        sidewall = max(self.rr_sidewall_fraction, 0.0)
        total = belt + sidewall
        if total <= 1e-12:
            return (0.85, 0.15)
        return (belt / total, sidewall / total)


class TireThermalSimulator:
    """9-node tire thermal simulator with pressure-volume feedback and RK4 integration."""

    def __init__(self, parameters: TireModelParameters | None = None) -> None:
        self.parameters = parameters if parameters is not None else TireModelParameters()
        self._gas_mass_kg = self.parameters.resolved_gas_mass_kg()

    def initial_state(
        self,
        *,
        ambient_temp_k: float = celsius_to_kelvin(25.0),
        brake_temp_k: float | None = None,
        wear: float = 0.0,
    ) -> TireState:
        brake_k = ambient_temp_k if brake_temp_k is None else brake_temp_k
        return TireState(
            t_surface_inner_k=ambient_temp_k,
            t_surface_middle_k=ambient_temp_k,
            t_surface_outer_k=ambient_temp_k,
            t_belt_k=ambient_temp_k,
            t_carcass_k=ambient_temp_k,
            t_gas_k=ambient_temp_k,
            t_rim_k=ambient_temp_k,
            t_brake_k=brake_k,
            t_sidewall_k=ambient_temp_k,
            kappa_dyn=0.0,
            alpha_dyn_rad=0.0,
            wear=min(max(wear, 0.0), 1.0),
            time_s=0.0,
        )

    def dynamic_pressure_pa(self, state: TireState, inputs: TireInputs) -> float:
        volume_m3 = self._dynamic_volume_m3(
            wheel_speed_radps=inputs.wheel_angular_speed_radps,
            normal_load_n=inputs.normal_load_n,
            wear=state.wear,
        )
        pressure_pa = self._gas_mass_kg * self.parameters.gas_specific_constant_j_per_kgk * state.t_gas_k
        return max(pressure_pa / max(volume_m3, 1e-12), self.parameters.atmospheric_pressure_pa)

    def diagnostics(self, state: TireState, inputs: TireInputs) -> TireDiagnostics:
        slip_ratio_eff, slip_angle_base_eff, _, _ = self._resolved_slip_state(state, inputs)
        effective_slip_angle = (
            slip_angle_base_eff + self.parameters.toe_slip_gain * inputs.toe_rad
            if self.parameters.use_alignment_zone_effects
            else slip_angle_base_eff
        )
        zone_weights = self._zone_weights(
            inputs.zone_load_split,
            effective_slip_angle,
            camber_rad=inputs.camber_rad,
            toe_rad=inputs.toe_rad,
            lateral_accel_mps2=inputs.lateral_accel_mps2,
            longitudinal_accel_mps2=inputs.longitudinal_accel_mps2,
            is_left_tire=inputs.is_left_tire,
        )
        pressure_pa = self.dynamic_pressure_pa(state, inputs)
        volume_m3 = self._dynamic_volume_m3(
            wheel_speed_radps=inputs.wheel_angular_speed_radps,
            normal_load_n=inputs.normal_load_n,
            wear=state.wear,
        )
        volume_rate_m3ps = self._volume_rate_m3ps(inputs)
        surface_temp_avg = (
            state.t_surface_inner_k + state.t_surface_middle_k + state.t_surface_outer_k
        ) / 3.0
        contact_patch_area_m2 = self._contact_patch_area_m2(
            normal_load_n=inputs.normal_load_n,
            pressure_pa=pressure_pa,
            surface_temp_k=surface_temp_avg,
        )
        zone_contact_areas_m2 = (
            contact_patch_area_m2 * zone_weights[0],
            contact_patch_area_m2 * zone_weights[1],
            contact_patch_area_m2 * zone_weights[2],
        )
        slip_speed_mps = self._slip_speed_mps(inputs.speed_mps, slip_ratio_eff, effective_slip_angle)
        external_htc = self._external_htc_w_per_m2k(inputs.speed_mps)
        internal_htc = self._internal_htc_w_per_m2k(
            pressure_pa=pressure_pa,
            gas_temp_k=state.t_gas_k,
            wheel_speed_radps=inputs.wheel_angular_speed_radps,
        )
        tread_mass_kg = self.parameters.tread_mass_kg(state.wear)
        tread_thickness_m = self.parameters.tread_thickness_m(state.wear)
        (
            q_rr_total_w,
            q_rr_belt_tread_w,
            q_rr_sidewall_w,
        ) = self._rolling_resistance_heating_w(inputs)
        (
            patch_pressure_grid_pa,
            patch_shear_grid_pa,
            patch_cell_area_m2,
            zone_friction_power_mechanical_w,
        ) = self._quasi_2d_patch_fields(
            inputs=inputs,
            contact_patch_area_m2=contact_patch_area_m2,
            effective_slip_ratio=slip_ratio_eff,
            effective_slip_angle_rad=effective_slip_angle,
            slip_speed_mps=slip_speed_mps,
            surface_temp_k=surface_temp_avg,
        )
        zone_friction_power_w = self._friction_power_to_tire(zone_friction_power_mechanical_w)
        friction_total_w = sum(zone_friction_power_w)
        (
            c_adh,
            c_hys,
            c_flash,
        ) = self._friction_partition_coefficients()
        friction_adhesion_w = friction_total_w * c_adh
        friction_hysteresis_w = friction_total_w * c_hys
        friction_flash_w = friction_total_w * c_flash
        return TireDiagnostics(
            dynamic_pressure_pa=pressure_pa,
            dynamic_pressure_bar_gauge=(pressure_pa - self.parameters.atmospheric_pressure_pa) / 100_000.0,
            volume_m3=volume_m3,
            volume_rate_m3ps=volume_rate_m3ps,
            contact_patch_area_m2=contact_patch_area_m2,
            zone_contact_areas_m2=zone_contact_areas_m2,
            effective_slip_ratio=slip_ratio_eff,
            effective_slip_angle_rad=effective_slip_angle,
            slip_speed_mps=slip_speed_mps,
            external_htc_w_m2k=external_htc,
            internal_htc_w_m2k=internal_htc,
            tread_mass_kg=tread_mass_kg,
            tread_thickness_m=tread_thickness_m,
            rolling_resistance_total_w=q_rr_total_w,
            rolling_resistance_belt_tread_w=q_rr_belt_tread_w,
            rolling_resistance_sidewall_w=q_rr_sidewall_w,
            friction_power_total_w=friction_total_w,
            friction_power_adhesion_w=friction_adhesion_w,
            friction_power_hysteresis_w=friction_hysteresis_w,
            friction_power_flash_w=friction_flash_w,
            patch_pressure_grid_pa=patch_pressure_grid_pa,
            patch_shear_grid_pa=patch_shear_grid_pa,
            patch_cell_area_m2=patch_cell_area_m2,
            zone_friction_power_w=zone_friction_power_w,
        )

    def temperature_rates_k_per_s(self, state: TireState, inputs: TireInputs) -> dict[str, float]:
        rates = self._derivative_vector(state.as_vector(), inputs)
        return {
            "surface_inner": float(rates[SURFACE_INNER_INDEX]),
            "surface_middle": float(rates[SURFACE_MIDDLE_INDEX]),
            "surface_outer": float(rates[SURFACE_OUTER_INDEX]),
            "belt": float(rates[BELT_INDEX]),
            "carcass": float(rates[CARCASS_INDEX]),
            "gas": float(rates[GAS_INDEX]),
            "rim": float(rates[RIM_INDEX]),
            "brake": float(rates[BRAKE_INDEX]),
            "sidewall": float(rates[SIDEWALL_INDEX]),
            "slip_ratio_dyn": float(rates[KAPPA_DYN_INDEX]),
            "slip_angle_dyn_rad": float(rates[ALPHA_DYN_INDEX]),
        }

    def step(self, state: TireState, inputs: TireInputs, dt_s: float) -> TireState:
        if dt_s <= 0.0:
            msg = f"dt_s must be positive, got {dt_s}"
            raise ValueError(msg)

        y0 = state.as_vector()
        k1 = self._derivative_vector(y0, inputs)
        k2 = self._derivative_vector(y0 + 0.5 * dt_s * k1, inputs)
        k3 = self._derivative_vector(y0 + 0.5 * dt_s * k2, inputs)
        k4 = self._derivative_vector(y0 + dt_s * k3, inputs)
        y_next = y0 + (dt_s / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

        y_next[:SIDEWALL_INDEX + 1] = np.clip(
            y_next[:SIDEWALL_INDEX + 1],
            self.parameters.minimum_temperature_k,
            self.parameters.maximum_temperature_k,
        )
        y_next[KAPPA_DYN_INDEX] = float(np.clip(y_next[KAPPA_DYN_INDEX], -3.0, 3.0))
        y_next[ALPHA_DYN_INDEX] = float(np.clip(y_next[ALPHA_DYN_INDEX], -1.2, 1.2))
        y_next[WEAR_INDEX] = float(np.clip(y_next[WEAR_INDEX], 0.0, 1.0))

        return TireState.from_vector(y_next, time_s=state.time_s + dt_s)

    def simulate(
        self,
        initial_state: TireState,
        inputs_stream: Iterable[TireInputs],
        dt_s: float,
    ) -> list[TireState]:
        states = [initial_state]
        current = initial_state
        for inputs in inputs_stream:
            current = self.step(current, inputs, dt_s=dt_s)
            states.append(current)
        return states

    def _derivative_vector(self, state_vector: np.ndarray, inputs: TireInputs) -> np.ndarray:
        state = TireState.from_vector(state_vector, time_s=0.0)
        diag = self.diagnostics(state, inputs)
        params = self.parameters

        slip_ratio_eff, slip_angle_base_eff, dkappa_dyn_dt, dalpha_dyn_dt = self._resolved_slip_state(
            state,
            inputs,
        )
        effective_slip_angle = (
            slip_angle_base_eff + params.toe_slip_gain * inputs.toe_rad
            if params.use_alignment_zone_effects
            else slip_angle_base_eff
        )

        zone_weights = self._zone_weights(
            inputs.zone_load_split,
            effective_slip_angle,
            camber_rad=inputs.camber_rad,
            toe_rad=inputs.toe_rad,
            lateral_accel_mps2=inputs.lateral_accel_mps2,
            longitudinal_accel_mps2=inputs.longitudinal_accel_mps2,
            is_left_tire=inputs.is_left_tire,
        )
        zone_surface_area = tuple(params.exposed_surface_area_m2 * w for w in zone_weights)
        zone_mass = tuple(diag.tread_mass_kg * (1.0 / 3.0) for _ in zone_weights)
        surface_temp_avg = (
            state.t_surface_inner_k + state.t_surface_middle_k + state.t_surface_outer_k
        ) / 3.0
        tread_cp = (
            self._rubber_cp_j_per_kgk(surface_temp_avg)
            if params.use_temperature_dependent_properties
            else params.cp_rubber_j_per_kgk
        )

        load_ratio = max(0.5, min(1.6, inputs.normal_load_n / params.reference_load_n))
        slip_ratio_mag = abs(slip_ratio_eff)
        slip_angle_eff = abs(math.tan(effective_slip_angle))
        slip_factor = slip_ratio_mag + slip_angle_eff
        heating_factor = params.heating_min_factor + (1.0 - params.heating_min_factor) / (
            1.0 + math.exp((surface_temp_avg - params.heating_drop_temp_k) / params.heating_drop_width_k)
        )
        if params.use_quasi_2d_patch_model:
            q_friction_base = diag.zone_friction_power_w
        elif params.use_combined_slip_model:
            mu_eff = self._friction_coefficient(
                load_ratio=load_ratio,
                surface_temp_k=surface_temp_avg,
                contact_patch_area_m2=diag.contact_patch_area_m2,
            )
            slip_mag = math.sqrt(
                (slip_ratio_mag / max(params.slip_ratio_reference, 1e-4)) ** 2
                + (
                    slip_angle_eff
                    / max(math.tan(params.slip_angle_reference_rad), 1e-4)
                )
                ** 2
            )
            slip_utilization = 1.0 - math.exp(-params.combined_slip_shape * slip_mag)
            friction_force_total = mu_eff * max(inputs.normal_load_n, 0.0) * slip_utilization
            q_friction_total = friction_force_total * diag.slip_speed_mps
            q_friction_base = self._friction_power_to_tire(
                (
                    q_friction_total * zone_weights[0],
                    q_friction_total * zone_weights[1],
                    q_friction_total * zone_weights[2],
                )
            )
        else:
            tau_shear_pa = params.base_shear_stress_pa * (0.2 + slip_factor)
            tau_shear_pa *= 1.0 + params.shear_load_gain * (load_ratio - 1.0)
            tau_shear_pa *= heating_factor
            q_friction_base = self._friction_power_to_tire(
                (
                    tau_shear_pa * diag.zone_contact_areas_m2[0] * diag.slip_speed_mps,
                    tau_shear_pa * diag.zone_contact_areas_m2[1] * diag.slip_speed_mps,
                    tau_shear_pa * diag.zone_contact_areas_m2[2] * diag.slip_speed_mps,
                )
            )

        q_convection = tuple(
            h * a * (t - inputs.ambient_temp_k)
            for h, a, t in zip(
                diag.external_htc_w_m2k,
                zone_surface_area,
                (
                    state.t_surface_inner_k,
                    state.t_surface_middle_k,
                    state.t_surface_outer_k,
                ),
                strict=True,
            )
        )
        q_radiation = tuple(
            STEFAN_BOLTZMANN_CONSTANT
            * params.surface_emissivity
            * a
            * (t**4 - inputs.ambient_temp_k**4)
            for a, t in zip(
                zone_surface_area,
                (
                    state.t_surface_inner_k,
                    state.t_surface_middle_k,
                    state.t_surface_outer_k,
                ),
                strict=True,
            )
        )
        q_track = tuple(
            params.track_contact_htc_w_per_m2k * a * (t - inputs.track_temp_k)
            for a, t in zip(
                diag.zone_contact_areas_m2,
                (
                    state.t_surface_inner_k,
                    state.t_surface_middle_k,
                    state.t_surface_outer_k,
                ),
                strict=True,
            )
        )

        diffusion_area = tuple(
            max(a_cp, params.exposed_surface_area_m2 * (1.0 / 3.0) * params.tread_to_belt_area_fraction)
            for a_cp in diag.zone_contact_areas_m2
        )
        k_rubber_eff = (
            self._rubber_k_w_per_mk(surface_temp_avg)
            if params.use_temperature_dependent_properties
            else params.k_rubber_w_per_mk
        )
        tread_to_belt_conductance = tuple(
            k_rubber_eff * max(a, 1e-5) / max(diag.tread_thickness_m, 1e-4)
            for a in diffusion_area
        )
        q_surface_to_belt = tuple(
            g * (t - state.t_belt_k)
            for g, t in zip(
                tread_to_belt_conductance,
                (
                    state.t_surface_inner_k,
                    state.t_surface_middle_k,
                    state.t_surface_outer_k,
                ),
                strict=True,
            )
        )
        lateral_conductance = 0.0
        if params.use_zone_lateral_conduction:
            wear_gain = 1.0 + params.zone_lateral_conductance_wear_gain * state.wear
            thickness_gain = params.tread_thickness_fresh_m / max(diag.tread_thickness_m, 1e-4)
            lateral_conductance = params.zone_lateral_conductance_w_per_k * wear_gain * thickness_gain

        q_inner_to_middle = lateral_conductance * (state.t_surface_inner_k - state.t_surface_middle_k)
        q_outer_to_middle = lateral_conductance * (state.t_surface_outer_k - state.t_surface_middle_k)
        surface_storage = tuple(max(m * tread_cp, 1.0) for m in zone_mass)

        if params.use_friction_partition_model:
            c_adh, c_hys_surf, c_flash = self._friction_partition_coefficients()
            q_friction_adh = tuple(c_adh * q for q in q_friction_base)
            q_friction_hys_belt = tuple(c_hys_surf * q for q in q_friction_base)
            q_friction_flash = tuple(c_flash * q for q in q_friction_base)
            flash_track_capacity = tuple(
                params.track_contact_htc_w_per_m2k * area * max(t_surface - inputs.track_temp_k, 0.0)
                for area, t_surface in zip(
                    diag.zone_contact_areas_m2,
                    (
                        state.t_surface_inner_k,
                        state.t_surface_middle_k,
                        state.t_surface_outer_k,
                    ),
                    strict=True,
                )
            )
            q_flash_to_track = tuple(
                min(q_flash, q_capacity)
                for q_flash, q_capacity in zip(
                    q_friction_flash,
                    flash_track_capacity,
                    strict=True,
                )
            )
            q_friction = tuple(
                q_adh + (q_flash - q_track)
                for q_adh, q_flash, q_track in zip(
                    q_friction_adh,
                    q_friction_flash,
                    q_flash_to_track,
                    strict=True,
                )
            )
        else:
            q_friction = q_friction_base
            q_friction_hys_belt = (0.0, 0.0, 0.0)
            q_flash_to_track = (0.0, 0.0, 0.0)

        dt_surface = (
            (
                q_friction[0]
                - q_convection[0]
                - q_radiation[0]
                - q_track[0]
                - q_flash_to_track[0]
                - q_surface_to_belt[0]
                - q_inner_to_middle
            )
            / surface_storage[0],
            (
                q_friction[1]
                - q_convection[1]
                - q_radiation[1]
                - q_track[1]
                - q_flash_to_track[1]
                - q_surface_to_belt[1]
                + q_inner_to_middle
                + q_outer_to_middle
            )
            / surface_storage[1],
            (
                q_friction[2]
                - q_convection[2]
                - q_radiation[2]
                - q_track[2]
                - q_flash_to_track[2]
                - q_surface_to_belt[2]
                - q_outer_to_middle
            )
            / surface_storage[2],
        )

        hysteresis_factor = 0.20 + 0.80 / (
            1.0 + math.exp((state.t_belt_k - params.hysteresis_drop_temp_k) / params.hysteresis_drop_width_k)
        )
        loss_tangent_eff = (
            self._loss_tangent(
                belt_temp_k=state.t_belt_k,
                wheel_angular_speed_radps=inputs.wheel_angular_speed_radps,
            )
            if params.use_temperature_dependent_properties
            else params.loss_tangent
        )
        q_hysteresis = (
            params.hysteresis_gain
            * loss_tangent_eff
            * abs(inputs.wheel_angular_speed_radps)
            * inputs.normal_load_n
            * (1.0 + 2.0 * slip_ratio_mag)
            * hysteresis_factor
        )
        (
            _q_rr_total,
            q_rr_belt_tread,
            q_rr_sidewall,
        ) = self._rolling_resistance_heating_w(inputs)
        q_hysteresis_sidewall = (
            params.sidewall_hysteresis_fraction * q_hysteresis
            if params.use_sidewall_rr_split_model
            else 0.0
        )
        q_hysteresis_belt = q_hysteresis - q_hysteresis_sidewall

        q_surfaces_to_belt = sum(q_surface_to_belt)
        k_belt_eff = (
            self._belt_k_w_per_mk(state.t_belt_k)
            if params.use_temperature_dependent_properties
            else params.k_belt_w_per_mk
        )
        belt_to_carcass_conductance = (
            k_belt_eff
            * params.belt_to_carcass_area_m2
            / max(params.belt_to_carcass_path_m, 1e-4)
        )
        q_belt_to_carcass = belt_to_carcass_conductance * (state.t_belt_k - state.t_carcass_k)
        q_belt_to_rim = params.belt_to_rim_conductance_w_per_k * (state.t_belt_k - state.t_rim_k)

        dt_belt = (
            q_surfaces_to_belt
            + q_hysteresis_belt
            + q_rr_belt_tread
            + sum(q_friction_hys_belt)
            - q_belt_to_carcass
            - q_belt_to_rim
        ) / max(params.m_belt_kg * params.c_belt_j_per_kgk, 1.0)

        q_carcass_to_sidewall = 0.0
        q_rim_to_sidewall = 0.0
        dt_sidewall = 0.0
        if params.use_sidewall_rr_split_model:
            q_carcass_to_sidewall = (
                params.sidewall_to_carcass_conductance_w_per_k * (state.t_carcass_k - state.t_sidewall_k)
            )
            q_rim_to_sidewall = params.sidewall_to_rim_conductance_w_per_k * (
                state.t_rim_k - state.t_sidewall_k
            )
            h_sidewall_external = (
                params.sidewall_external_h_base_w_per_m2k
                + params.sidewall_external_h_speed_gain * diag.external_htc_w_m2k[SURFACE_MIDDLE_INDEX]
            )
            h_sidewall_internal = params.sidewall_internal_h_fraction * diag.internal_htc_w_m2k
            q_sidewall_to_ambient = h_sidewall_external * params.sidewall_area_m2 * (
                state.t_sidewall_k - inputs.ambient_temp_k
            )
            q_sidewall_to_gas = h_sidewall_internal * params.sidewall_area_m2 * (
                state.t_sidewall_k - state.t_gas_k
            )
            dt_sidewall = (
                q_rr_sidewall
                + q_hysteresis_sidewall
                + q_carcass_to_sidewall
                + q_rim_to_sidewall
                - q_sidewall_to_ambient
                - q_sidewall_to_gas
            ) / max(params.m_sidewall_kg * params.sidewall_heat_capacity_j_per_kgk, 1.0)

        q_int_conv = diag.internal_htc_w_m2k * params.internal_area_m2 * (
            state.t_carcass_k - state.t_gas_k
        )
        h_carcass_ext = (
            params.carcass_external_h_base_w_per_m2k
            + params.carcass_external_h_speed_gain * diag.external_htc_w_m2k[SURFACE_MIDDLE_INDEX]
        )
        q_carcass_to_ambient = h_carcass_ext * params.carcass_external_area_m2 * (
            state.t_carcass_k - inputs.ambient_temp_k
        )
        carcass_cp = (
            self._rubber_cp_j_per_kgk(state.t_carcass_k)
            if params.use_temperature_dependent_properties
            else params.cp_rubber_j_per_kgk
        )
        dt_carcass = (q_belt_to_carcass - q_int_conv - q_carcass_to_ambient) / max(
            params.m_carcass_kg * carcass_cp,
            1.0,
        )
        if params.use_sidewall_rr_split_model:
            dt_carcass -= q_carcass_to_sidewall / max(params.m_carcass_kg * carcass_cp, 1.0)

        q_rim_to_gas = params.h_rim_to_gas_w_per_m2k * params.rim_to_gas_area_m2 * (
            state.t_rim_k - state.t_gas_k
        )
        q_pdv = diag.dynamic_pressure_pa * diag.volume_rate_m3ps
        dt_gas = (q_int_conv + q_rim_to_gas - q_pdv) / max(
            self._gas_mass_kg * params.gas_cv_j_per_kgk,
            1.0,
        )

        q_brake_to_rim = params.brake_to_rim_fraction * max(inputs.brake_power_w, 0.0)
        q_rim_to_ambient = params.h_rim_external_w_per_m2k * params.rim_external_area_m2 * (
            state.t_rim_k - inputs.ambient_temp_k
        )
        dt_rim = (q_brake_to_rim + q_belt_to_rim - q_rim_to_gas - q_rim_to_ambient) / max(
            params.m_rim_kg * params.c_rim_j_per_kgk,
            1.0,
        )
        if params.use_sidewall_rr_split_model:
            dt_rim -= q_rim_to_sidewall / max(params.m_rim_kg * params.c_rim_j_per_kgk, 1.0)

        q_brake_radiation = (
            STEFAN_BOLTZMANN_CONSTANT
            * params.brake_emissivity
            * params.brake_area_m2
            * (state.t_brake_k**4 - inputs.ambient_temp_k**4)
        )
        q_brake_conv = params.h_brake_conv_w_per_m2k * params.brake_area_m2 * (
            state.t_brake_k - inputs.ambient_temp_k
        )
        dt_brake = (
            max(inputs.brake_power_w, 0.0) - q_brake_to_rim - q_brake_radiation - q_brake_conv
        ) / max(params.m_brake_kg * params.c_brake_j_per_kgk, 1.0)

        avg_surface_temp = surface_temp_avg
        wear_driver = diag.slip_speed_mps * (0.02 + slip_ratio_mag + slip_angle_eff)
        temperature_factor = max(0.35, avg_surface_temp / celsius_to_kelvin(85.0))
        friction_energy_factor = max(0.2, sum(q_friction_base) / params.wear_reference_heat_w)
        load_factor = max(0.75, inputs.normal_load_n / params.reference_load_n)
        dwear_dt = (
            params.wear_rate_coefficient
            * wear_driver
            * temperature_factor
            * friction_energy_factor
            * load_factor
        )
        if state.wear >= 1.0 and dwear_dt > 0.0:
            dwear_dt = 0.0

        return np.array(
            [
                dt_surface[0],
                dt_surface[1],
                dt_surface[2],
                dt_belt,
                dt_carcass,
                dt_gas,
                dt_rim,
                dt_brake,
                dt_sidewall,
                dkappa_dyn_dt,
                dalpha_dyn_dt,
                dwear_dt,
            ],
            dtype=float,
        )

    def _slip_commands(self, inputs: TireInputs) -> tuple[float, float]:
        slip_ratio_cmd = inputs.slip_ratio if inputs.slip_ratio_cmd is None else inputs.slip_ratio_cmd
        slip_angle_cmd = (
            inputs.slip_angle_rad if inputs.slip_angle_cmd_rad is None else inputs.slip_angle_cmd_rad
        )
        return (slip_ratio_cmd, slip_angle_cmd)

    def _resolved_slip_state(
        self,
        state: TireState,
        inputs: TireInputs,
    ) -> tuple[float, float, float, float]:
        params = self.parameters
        slip_ratio_cmd, slip_angle_cmd = self._slip_commands(inputs)
        if not params.use_slip_transient_model:
            return (slip_ratio_cmd, slip_angle_cmd, 0.0, 0.0)

        speed_abs = abs(inputs.speed_mps)
        long_length = max(params.relaxation_length_long_m, 1e-3)
        lat_length = max(params.relaxation_length_lat_m, 1e-3)
        gain_long = min(speed_abs / long_length, params.max_slip_relaxation_rate_per_s)
        gain_lat = min(speed_abs / lat_length, params.max_slip_relaxation_rate_per_s)

        dkappa_dyn_dt = gain_long * (slip_ratio_cmd - state.kappa_dyn)
        dalpha_dyn_dt = gain_lat * (slip_angle_cmd - state.alpha_dyn_rad)

        transition_speed = max(params.slip_transition_speed_mps, 1e-3)
        blend = speed_abs / (speed_abs + transition_speed)
        slip_ratio_eff = blend * state.kappa_dyn + (1.0 - blend) * slip_ratio_cmd
        slip_angle_eff = blend * state.alpha_dyn_rad + (1.0 - blend) * slip_angle_cmd
        return (slip_ratio_eff, slip_angle_eff, dkappa_dyn_dt, dalpha_dyn_dt)

    def _quasi_2d_patch_fields(
        self,
        *,
        inputs: TireInputs,
        contact_patch_area_m2: float,
        effective_slip_ratio: float,
        effective_slip_angle_rad: float,
        slip_speed_mps: float,
        surface_temp_k: float,
    ) -> tuple[PatchGrid3x3, PatchGrid3x3, float, tuple[float, float, float]]:
        params = self.parameters
        cell_area = max(contact_patch_area_m2, 0.0) / 9.0
        if cell_area <= 0.0 or inputs.normal_load_n <= 0.0:
            zero_grid: PatchGrid3x3 = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
            return (zero_grid, zero_grid, max(cell_area, 0.0), (0.0, 0.0, 0.0))

        camber_scale = max(self.parameters.zone_angle_scale_rad, 1e-5)
        width_skew = params.patch_camber_pressure_skew * math.tanh(inputs.camber_rad / camber_scale)
        length_skew = params.patch_longitudinal_pressure_skew * math.tanh(inputs.longitudinal_accel_mps2 / 9.81)
        width_factors = np.array([1.0 - width_skew, 1.0, 1.0 + width_skew], dtype=float)
        length_factors = np.array([1.0 + length_skew, 1.0, 1.0 - length_skew], dtype=float)
        width_factors = np.clip(width_factors, params.patch_pressure_min_fraction, None)
        length_factors = np.clip(length_factors, params.patch_pressure_min_fraction, None)

        pressure_shape = np.outer(width_factors, length_factors)
        scale = max(inputs.normal_load_n, 0.0) / max(float(np.sum(pressure_shape) * cell_area), 1e-12)
        pressure_grid = np.maximum(pressure_shape * scale, 0.0)

        target_force_n = max(inputs.normal_load_n, 0.0)
        force_from_grid_n = float(np.sum(pressure_grid) * cell_area)
        force_error_n = force_from_grid_n - target_force_n
        if abs(force_error_n) > params.patch_pressure_norm_tol_n:
            pressure_grid *= target_force_n / max(force_from_grid_n, 1e-12)

        load_ratio = max(0.5, min(1.6, inputs.normal_load_n / params.reference_load_n))
        slip_ratio_mag = abs(effective_slip_ratio)
        slip_angle_tan = abs(math.tan(effective_slip_angle_rad))
        slip_factor = slip_ratio_mag + slip_angle_tan
        heating_factor = params.heating_min_factor + (1.0 - params.heating_min_factor) / (
            1.0 + math.exp((surface_temp_k - params.heating_drop_temp_k) / params.heating_drop_width_k)
        )

        if params.use_combined_slip_model:
            mu_eff = self._friction_coefficient(
                load_ratio=load_ratio,
                surface_temp_k=surface_temp_k,
                contact_patch_area_m2=contact_patch_area_m2,
            )
            slip_mag = math.sqrt(
                (slip_ratio_mag / max(params.slip_ratio_reference, 1e-4)) ** 2
                + (slip_angle_tan / max(math.tan(params.slip_angle_reference_rad), 1e-4)) ** 2
            )
            slip_utilization = 1.0 - math.exp(-params.combined_slip_shape * slip_mag)
            shear_mu = mu_eff * slip_utilization
            shear_grid = pressure_grid * shear_mu
        else:
            tau_ref_pa = params.base_shear_stress_pa * (0.2 + slip_factor)
            tau_ref_pa *= 1.0 + params.shear_load_gain * (load_ratio - 1.0)
            tau_ref_pa *= heating_factor
            pressure_ref_pa = max(target_force_n / max(contact_patch_area_m2, 1e-12), 1.0)
            shear_grid = tau_ref_pa * (pressure_grid / pressure_ref_pa)

        friction_power_grid_w = shear_grid * cell_area * abs(slip_speed_mps)
        zone_friction_power = tuple(float(np.sum(friction_power_grid_w[row_idx, :])) for row_idx in range(3))
        return (
            self._to_patch_grid(pressure_grid),
            self._to_patch_grid(shear_grid),
            cell_area,
            (zone_friction_power[0], zone_friction_power[1], zone_friction_power[2]),
        )

    def _to_patch_grid(self, values: np.ndarray) -> PatchGrid3x3:
        return (
            (float(values[0, 0]), float(values[0, 1]), float(values[0, 2])),
            (float(values[1, 0]), float(values[1, 1]), float(values[1, 2])),
            (float(values[2, 0]), float(values[2, 1]), float(values[2, 2])),
        )

    def _friction_partition_coefficients(self) -> tuple[float, float, float]:
        params = self.parameters
        if not params.use_friction_partition_model:
            return (1.0, 0.0, 0.0)

        c_adh = max(params.friction_partition_adhesion, 0.0)
        c_hys = max(params.friction_partition_hysteresis, 0.0)
        c_flash = max(params.friction_partition_flash, 0.0)
        total = c_adh + c_hys + c_flash
        if total <= 1e-12:
            return (0.45, 0.35, 0.20)
        return (c_adh / total, c_hys / total, c_flash / total)

    def _friction_power_to_tire(
        self,
        zone_friction_power_mechanical_w: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        heat_fraction = max(self.parameters.friction_heat_fraction, 0.0)
        return (
            heat_fraction * zone_friction_power_mechanical_w[0],
            heat_fraction * zone_friction_power_mechanical_w[1],
            heat_fraction * zone_friction_power_mechanical_w[2],
        )

    def _rolling_resistance_heating_w(self, inputs: TireInputs) -> tuple[float, float, float]:
        params = self.parameters

        q_rr_total = (
            max(params.rolling_resistance_coeff, 0.0)
            * max(inputs.normal_load_n, 0.0)
            * abs(inputs.speed_mps)
        )
        if params.use_sidewall_rr_split_model:
            belt_fraction, sidewall_fraction = params.rr_split_fractions()
        else:
            belt_fraction, sidewall_fraction = (1.0, 0.0)
        q_rr_belt_tread = q_rr_total * belt_fraction
        q_rr_sidewall = q_rr_total * sidewall_fraction
        return (q_rr_total, q_rr_belt_tread, q_rr_sidewall)

    def _dynamic_volume_m3(self, wheel_speed_radps: float, normal_load_n: float, wear: float) -> float:
        params = self.parameters
        volume = params.base_volume_m3
        volume += params.centrifugal_volume_gain_coeff_m3_per_radps2 * (wheel_speed_radps**2)
        volume -= params.deflection_volume_loss_coeff_m3_per_n * max(normal_load_n, 0.0)
        volume -= params.wear_volume_loss_m3 * min(max(wear, 0.0), 1.0)
        return max(volume, params.minimum_volume_m3)

    def _volume_rate_m3ps(self, inputs: TireInputs) -> float:
        params = self.parameters
        dvolume_dwheelspeed = (
            2.0
            * params.centrifugal_volume_gain_coeff_m3_per_radps2
            * inputs.wheel_angular_speed_radps
            * inputs.wheel_angular_accel_radps2
        )
        dvolume_dload = -params.deflection_volume_loss_coeff_m3_per_n * inputs.normal_load_rate_nps
        return dvolume_dwheelspeed + dvolume_dload + inputs.volume_change_rate_m3ps

    def _contact_patch_area_m2(
        self,
        normal_load_n: float,
        pressure_pa: float,
        *,
        surface_temp_k: float,
    ) -> float:
        params = self.parameters
        if params.use_gauge_patch_model:
            gauge_pressure_pa = max(pressure_pa - params.atmospheric_pressure_pa, 0.0)
            load_ratio = max(0.4, min(2.0, normal_load_n / max(params.reference_load_n, 1e-6)))
            carcass_load_term = 1.0 + params.carcass_support_load_gain * (load_ratio - 1.0)
            carcass_temp_term = 1.0 + params.carcass_support_temp_gain_per_k * (
                surface_temp_k - params.reference_material_temp_k
            )
            carcass_support_pa = params.carcass_support_pressure_pa * max(carcass_load_term, 0.2)
            carcass_support_pa *= max(carcass_temp_term, 0.2)
            effective_pressure_pa = max(
                gauge_pressure_pa + carcass_support_pa,
                params.min_effective_contact_pressure_pa,
            )
            area = max(normal_load_n, 0.0) / max(effective_pressure_pa * params.contact_pressure_factor, 1.0)
        else:
            area = max(normal_load_n, 0.0) / max(pressure_pa * params.contact_pressure_factor, 1.0)
        return min(max(area, params.min_contact_patch_area_m2), params.max_contact_patch_area_m2)

    def _slip_speed_mps(self, speed_mps: float, slip_ratio: float, slip_angle_rad: float) -> float:
        longitudinal = speed_mps * slip_ratio
        lateral = speed_mps * math.tan(slip_angle_rad)
        return math.sqrt(longitudinal**2 + lateral**2)

    def _external_htc_w_per_m2k(self, speed_mps: float) -> tuple[float, float, float]:
        params = self.parameters
        reynolds = (
            params.air_density_kg_per_m3
            * max(speed_mps, 0.1)
            * params.char_length_m
            / max(params.air_dynamic_viscosity_pa_s, 1e-10)
        )
        nusselt = 0.037 * (reynolds**0.8) * (params.air_prandtl**0.33)
        base_h = max(
            params.min_external_htc_w_per_m2k,
            nusselt * params.air_thermal_conductivity_w_per_mk / max(params.char_length_m, 1e-6),
        )
        return tuple(base_h * scale for scale in params.zone_external_h_multipliers)

    def _internal_htc_w_per_m2k(
        self,
        *,
        pressure_pa: float,
        gas_temp_k: float,
        wheel_speed_radps: float,
    ) -> float:
        params = self.parameters
        gas_density = pressure_pa / max(params.gas_specific_constant_j_per_kgk * gas_temp_k, 1e-6)
        # Cavity mixing is driven by the annular gas gap shear, not the full tire diameter.
        flow_speed_mps = abs(wheel_speed_radps) * params.internal_gas_gap_m
        reynolds_internal = (
            gas_density
            * flow_speed_mps
            * params.internal_gas_gap_m
            / max(params.gas_dynamic_viscosity_pa_s, 1e-10)
        )
        if params.use_rotating_internal_gas_model:
            radius_m = params.internal_flow_diameter_m * 0.5
            kinematic_viscosity = params.gas_dynamic_viscosity_pa_s / max(gas_density, 1e-10)
            reynolds_rotation = (
                abs(wheel_speed_radps)
                * radius_m
                * params.internal_flow_diameter_m
                / max(kinematic_viscosity, 1e-10)
            )
            taylor_number = (
                abs(wheel_speed_radps)
                * math.sqrt(max(radius_m * params.internal_gas_gap_m, 1e-10))
                * params.internal_gas_gap_m
                / max(kinematic_viscosity, 1e-10)
            )
            nu_axial = 0.023 * (max(reynolds_internal, 1.0) ** 0.8) * (params.gas_prandtl ** 0.4)
            nu_rotation = 0.11 * (max(reynolds_rotation, 1.0) ** 0.67) * (params.gas_prandtl ** 0.33)
            nu_taylor = 0.032 * (max(taylor_number, 1.0) ** 0.35)
            nusselt = max(
                params.internal_nusselt_floor,
                math.sqrt(nu_axial**2 + nu_rotation**2 + nu_taylor**2),
            )
            h_uncoupled = (
                nusselt
                * params.gas_thermal_conductivity_w_per_mk
                / max(params.internal_flow_diameter_m, 1e-6)
            )
            turnover_time_s = (2.0 * math.pi) / max(abs(wheel_speed_radps), 0.1)
            mixing_efficiency = 1.0 / (
                1.0 + params.gas_mixing_time_constant_s / max(turnover_time_s, 1e-6)
            )
            mixing_efficiency = params.gas_mixing_floor + (1.0 - params.gas_mixing_floor) * mixing_efficiency
            h_int = params.min_internal_htc_w_per_m2k + mixing_efficiency * (
                h_uncoupled - params.min_internal_htc_w_per_m2k
            )
        else:
            h_int = (
                0.015
                * params.gas_thermal_conductivity_w_per_mk
                / max(params.internal_flow_diameter_m, 1e-6)
                * (max(reynolds_internal, 1.0) ** 0.8)
            )
        return float(
            np.clip(h_int, params.min_internal_htc_w_per_m2k, params.max_internal_htc_w_per_m2k)
        )

    def _zone_weights(
        self,
        zone_load_split: tuple[float, float, float],
        slip_angle_rad: float,
        *,
        camber_rad: float = 0.0,
        toe_rad: float = 0.0,
        lateral_accel_mps2: float = 0.0,
        longitudinal_accel_mps2: float = 0.0,
        is_left_tire: bool = True,
    ) -> tuple[float, float, float]:
        inner, middle, outer = (max(v, 1e-6) for v in zone_load_split)
        shape_shift = self.parameters.shape_sensitivity * math.tanh(slip_angle_rad)
        if self.parameters.use_alignment_zone_effects:
            angle_scale = max(self.parameters.zone_angle_scale_rad, 1e-5)
            shape_shift += self.parameters.zone_camber_sensitivity * math.tanh(camber_rad / angle_scale)
            shape_shift += self.parameters.zone_toe_sensitivity * math.tanh(toe_rad / angle_scale)
            lateral_sign = -1.0 if is_left_tire else 1.0
            shape_shift += (
                lateral_sign
                * self.parameters.zone_lateral_accel_sensitivity
                * math.tanh(lateral_accel_mps2 / 9.81)
            )
            middle *= 1.0 + self.parameters.zone_longitudinal_accel_mid_gain * abs(
                math.tanh(longitudinal_accel_mps2 / 9.81)
            )

        shape_shift = params_clip(shape_shift, -0.35, 0.35)
        inner *= 1.0 - shape_shift
        outer *= 1.0 + shape_shift
        total = inner + middle + outer
        return (inner / total, middle / total, outer / total)

    def _rubber_cp_j_per_kgk(self, temperature_k: float) -> float:
        params = self.parameters
        scale = 1.0 + params.cp_temp_coeff_per_k * (temperature_k - params.reference_material_temp_k)
        scale = params_clip(scale, params.min_cp_scale, params.max_cp_scale)
        return params.cp_rubber_j_per_kgk * scale

    def _rubber_k_w_per_mk(self, temperature_k: float) -> float:
        params = self.parameters
        scale = 1.0 + params.k_rubber_temp_coeff_per_k * (
            temperature_k - params.reference_material_temp_k
        )
        scale = params_clip(scale, params.min_k_scale, params.max_k_scale)
        return params.k_rubber_w_per_mk * scale

    def _belt_k_w_per_mk(self, temperature_k: float) -> float:
        params = self.parameters
        scale = 1.0 + params.k_belt_temp_coeff_per_k * (temperature_k - params.reference_material_temp_k)
        scale = params_clip(scale, params.min_k_scale, params.max_k_scale)
        return params.k_belt_w_per_mk * scale

    def _loss_tangent(self, *, belt_temp_k: float, wheel_angular_speed_radps: float) -> float:
        params = self.parameters
        temp_error = (belt_temp_k - params.loss_tangent_peak_temp_k) / max(
            params.loss_tangent_temp_width_k,
            1e-6,
        )
        temp_factor = params.loss_tangent_min_fraction + (
            1.0 - params.loss_tangent_min_fraction
        ) * math.exp(-(temp_error**2))
        freq_factor = 1.0 + params.loss_tangent_freq_gain * math.log1p(
            abs(wheel_angular_speed_radps) / max(params.reference_angular_speed_radps, 1e-6)
        )
        return params.loss_tangent * temp_factor * max(freq_factor, 0.1)

    def _friction_coefficient(
        self,
        *,
        load_ratio: float,
        surface_temp_k: float,
        contact_patch_area_m2: float,
    ) -> float:
        params = self.parameters
        mu_load = params.mu_peak_reference * (load_ratio ** (-params.mu_load_sensitivity))
        temp_error = (surface_temp_k - params.mu_temperature_peak_k) / max(params.mu_temperature_width_k, 1e-6)
        mu_temp_factor = params.mu_min_fraction + (1.0 - params.mu_min_fraction) * math.exp(-(temp_error**2))
        patch_ratio = contact_patch_area_m2 / max(params.reference_contact_patch_area_m2, 1e-6)
        mu_patch_factor = patch_ratio ** params.mu_patch_sensitivity
        return max(0.1, mu_load * mu_temp_factor * params_clip(mu_patch_factor, 0.75, 1.25))


def params_clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)
