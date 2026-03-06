from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..physics import (
    STANDARD_ATMOSPHERIC_PRESSURE_PA,
    celsius_to_kelvin,
    kelvin_to_celsius,
)
from .boundary_conditions import HighFidelityBoundaryParameters


@dataclass(frozen=True)
class LayerMaterialParameters:
    thickness_m: float
    volumetric_heat_capacity_j_per_m3k: float
    k_r_w_per_mk: float
    k_theta_w_per_mk: float
    k_w_w_per_mk: float
    hysteresis_scale: float = 1.0
    cord_angle_deg: float = 0.0
    reinforcement_density_factor: float = 1.0
    shoulder_conductivity_bias: float = 1.0
    center_conductivity_bias: float = 1.0
    bead_conductivity_bias: float = 1.0
    temp_conductivity_sensitivity_per_k: float = 0.0
    wear_conductivity_sensitivity: float = 0.0


@dataclass(frozen=True)
class InternalCouplingParameters:
    enabled: bool = False
    gas_inner_liner_htc_w_per_k: float = 24.0
    gas_rim_htc_w_per_k: float = 18.0
    cavity_rim_htc_w_per_k: float = 9.0
    brake_disc_heat_capacity_j_per_k: float = 900.0
    brake_disc_to_ambient_conductance_w_per_k: float = 12.0
    brake_to_rim_conductance_w_per_k: float = 12.0
    brake_to_tire_conductance_w_per_k: float = 7.0
    brake_to_sidewall_conductance_w_per_k: float = 4.0
    gas_mixing_htc_w_per_k: float = 8.0
    gas_mixing_speed_gain_w_per_k_per_radps: float = 0.02
    gas_mixing_pressure_gain_per_bar: float = 0.10
    rim_cooling_speed_gain_per_radps: float = 0.0025
    rim_cooling_wake_gain: float = 0.20


@dataclass(frozen=True)
class LocalContactParameters:
    enabled: bool = False
    fallback_to_temperature_mu: bool = True
    adhesion_flash_weight: float = 0.35
    sliding_flash_weight: float = 0.82
    adhesion_temperature_peak_k: float = celsius_to_kelvin(92.0)
    adhesion_temperature_width_k: float = 20.0
    sliding_temperature_peak_k: float = celsius_to_kelvin(104.0)
    sliding_temperature_width_k: float = 28.0
    adhesion_min_fraction: float = 0.50
    sliding_min_fraction: float = 0.38
    pressure_mu_sensitivity: float = 0.10
    sliding_mu_drop_fraction: float = 0.22
    partition_sliding_gain: float = 0.10
    partition_pressure_gain: float = 0.05


@dataclass(frozen=True)
class ConstructionParameters:
    enabled: bool = False
    shoulder_width_fraction: float = 0.28
    bead_width_fraction: float = 0.18
    reinforcement_hysteresis_gain: float = 0.12
    cord_angle_hysteresis_gain: float = 0.20
    width_hysteresis_gain: float = 0.08
    temp_reference_k: float = celsius_to_kelvin(80.0)


@dataclass(frozen=True)
class LayerStackParameters:
    tread: LayerMaterialParameters = field(
        default_factory=lambda: LayerMaterialParameters(
            thickness_m=0.012,
            volumetric_heat_capacity_j_per_m3k=1.95e6,
            k_r_w_per_mk=0.26,
            k_theta_w_per_mk=0.29,
            k_w_w_per_mk=0.22,
            hysteresis_scale=1.0,
        )
    )
    belt: LayerMaterialParameters = field(
        default_factory=lambda: LayerMaterialParameters(
            thickness_m=0.008,
            volumetric_heat_capacity_j_per_m3k=2.05e6,
            k_r_w_per_mk=1.8,
            k_theta_w_per_mk=18.0,
            k_w_w_per_mk=4.5,
            hysteresis_scale=0.35,
        )
    )
    carcass: LayerMaterialParameters = field(
        default_factory=lambda: LayerMaterialParameters(
            thickness_m=0.068,
            volumetric_heat_capacity_j_per_m3k=1.85e6,
            k_r_w_per_mk=0.19,
            k_theta_w_per_mk=0.28,
            k_w_w_per_mk=0.18,
            hysteresis_scale=0.70,
        )
    )
    inner_liner: LayerMaterialParameters = field(
        default_factory=lambda: LayerMaterialParameters(
            thickness_m=0.004,
            volumetric_heat_capacity_j_per_m3k=1.65e6,
            k_r_w_per_mk=0.12,
            k_theta_w_per_mk=0.16,
            k_w_w_per_mk=0.12,
            hysteresis_scale=0.15,
        )
    )

    def total_thickness_m(self, *, tread_thickness_m: float | None = None) -> float:
        tread = self.tread.thickness_m if tread_thickness_m is None else tread_thickness_m
        return tread + self.belt.thickness_m + self.carcass.thickness_m + self.inner_liner.thickness_m


@dataclass(frozen=True)
class PressurePatchParameters:
    pressure_static_bar_gauge: float = 1.4
    atmospheric_pressure_pa: float = STANDARD_ATMOSPHERIC_PRESSURE_PA
    reference_pressure_pa: float = 240_000.0
    gas_specific_constant_j_per_kgk: float = 287.05
    gas_cv_j_per_kgk: float = 718.0
    reference_fill_temp_k: float = celsius_to_kelvin(20.0)
    gas_mass_kg: float | None = None
    base_volume_m3: float = 0.030
    minimum_volume_m3: float = 0.016
    centrifugal_volume_gain_coeff_m3_per_radps2: float = 8.0e-9
    deflection_volume_loss_coeff_m3_per_n: float = 1.2e-7
    wear_volume_loss_m3: float = 0.0015
    contact_pressure_factor: float = 1.0
    reference_contact_patch_area_m2: float = 0.018
    min_contact_patch_area_m2: float = 5.0e-4
    max_contact_patch_area_m2: float = 0.045
    reference_normal_load_n: float = 3_500.0
    carcass_support_pressure_pa: float = 72_000.0
    carcass_support_load_gain: float = 0.22
    carcass_support_temp_gain_per_k: float = 1.5e-3
    min_effective_contact_pressure_pa: float = 20_000.0
    patch_camber_pressure_skew: float = 0.24
    patch_toe_pressure_skew: float = 0.12
    patch_longitudinal_pressure_skew: float = 0.18
    patch_pressure_min_fraction: float = 0.25
    patch_pressure_norm_tol_n: float = 1e-3
    effective_radius_pressure_gain_m_per_bar: float = -0.0025

    def resolved_gas_mass_kg(self) -> float:
        if self.gas_mass_kg is not None:
            return self.gas_mass_kg
        static_abs_pressure = self.pressure_static_bar_gauge * 100_000.0 + self.atmospheric_pressure_pa
        return (
            static_abs_pressure
            * self.base_volume_m3
            / (self.gas_specific_constant_j_per_kgk * self.reference_fill_temp_k)
        )


@dataclass(frozen=True)
class CoreSensorParameters:
    probe_depth_fraction_from_outer: float = 0.22
    width_weights: tuple[float, float, float] = (0.25, 0.50, 0.25)
    bead_side_weight: float = 0.10
    tread_side_weight: float = 0.20
    belt_weight: float = 0.48
    carcass_weight: float = 0.17
    gas_weight: float = 0.05
    response_time_s: float = 3.0
    cavity_gas_comparison_fraction: float = 0.35


@dataclass(frozen=True)
class SurfaceStateParameters:
    width_zones: int = 3
    tread_thickness_fresh_m: float = 0.012
    tread_thickness_worn_m: float = 0.004
    tread_mass_fresh_kg: float = 3.5
    tread_mass_worn_kg: float = 1.5
    age_reference_temperature_k: float = celsius_to_kelvin(90.0)
    age_temperature_gain: float = 0.035
    age_energy_gain: float = 1.4e-5
    graining_gain: float = 2.0e-5
    graining_relaxation_s: float = 35.0
    graining_cooling_temp_k: float = celsius_to_kelvin(72.0)
    blister_gain: float = 1.6e-5
    blister_relaxation_s: float = 60.0
    blister_threshold_temp_k: float = celsius_to_kelvin(108.0)
    blister_energy_gain: float = 1.1e-5
    graining_contact_penalty: float = 0.22
    blister_conductivity_penalty: float = 0.32
    aging_hysteresis_shift_per_index: float = 0.18
    wear_rate_coefficient: float = 3.4e-4
    wear_reference_heat_w: float = 30_000.0

    def tread_thickness_m(self, wear: float) -> float:
        wear_clamped = min(max(wear, 0.0), 1.0)
        return self.tread_thickness_fresh_m + wear_clamped * (
            self.tread_thickness_worn_m - self.tread_thickness_fresh_m
        )

    def tread_mass_kg(self, wear: float) -> float:
        wear_clamped = min(max(wear, 0.0), 1.0)
        return self.tread_mass_fresh_kg + wear_clamped * (
            self.tread_mass_worn_kg - self.tread_mass_fresh_kg
        )


@dataclass(frozen=True)
class FlashLayerParameters:
    enabled: bool = True
    friction_fraction: float = 0.38
    patch_relaxation_time_s: float = 0.080
    bulk_coupling_time_s: float = 0.42
    ambient_cooling_time_s: float = 1.20
    road_cooling_time_s: float = 0.070
    areal_heat_capacity_j_per_m2k: float = 520.0
    max_delta_above_bulk_k: float = 140.0


@dataclass(frozen=True)
class BrakeCoolingParameters:
    brake_heat_to_tire_fraction: float = 0.03
    brake_heat_to_rim_fraction: float = 0.08
    brake_heat_to_sidewall_fraction: float = 0.02
    brake_disc_to_ambient_h_base_w_per_m2k: float = 65.0
    brake_duct_cooling_gain: float = 30.0
    wheel_wake_cooling_gain: float = 22.0
    rim_to_ambient_h_base_w_per_m2k: float = 55.0
    rim_to_ambient_wheel_wake_gain: float = 10.0


@dataclass(frozen=True)
class HighFidelityTireInputs:
    """Inputs for the layered high-fidelity tire model."""

    speed_mps: float
    wheel_angular_speed_radps: float
    normal_load_n: float
    slip_ratio_cmd: float
    slip_angle_cmd_rad: float
    drive_torque_nm: float | None = None
    brake_torque_nm: float | None = None
    lateral_force_target_n: float | None = None
    brake_power_w: float = 0.0
    ambient_temp_k: float = celsius_to_kelvin(25.0)
    track_temp_k: float = celsius_to_kelvin(35.0)
    wind_mps: float = 0.0
    wind_yaw_rad: float = 0.0
    humidity_rel: float = 0.50
    solar_w_m2: float = 0.0
    road_surface_temp_k: float | None = None
    road_bulk_temp_k: float | None = None
    road_moisture: float = 0.0
    rubbering_level: float = 0.0
    asphalt_roughness: float = 1.0
    asphalt_effusivity: float = 1.0
    brake_duct_cooling_factor: float = 1.0
    wheel_wake_factor: float = 1.0
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
class HighFidelityTireModelParameters:
    """Configuration for the layered high-fidelity tire model."""

    @dataclass(frozen=True)
    class PronyBranch:
        modulus_pa: float
        relaxation_time_s: float

    thermal_node_count: int = 9
    no_op_thermal_step: bool = True
    minimum_temperature_k: float = 160.0
    maximum_temperature_k: float = 2_200.0
    wlf_reference_temp_k: float = celsius_to_kelvin(80.0)
    wlf_c1: float = 8.86
    wlf_c2_k: float = 101.6
    reference_load_n: float = 3_500.0
    strain_amplitude_reference: float = 0.055
    strain_load_sensitivity: float = 0.32
    min_strain_amplitude: float = 0.01
    max_strain_amplitude: float = 0.16
    slip_frequency_length_m: float = 0.08
    min_excitation_frequency_hz: float = 0.20
    hysteresis_active_volume_m3: float = 2.1e-3
    use_2d_thermal_solver: bool = False
    enable_profiling: bool = False
    radial_cells: int = 36
    theta_cells: int = 72
    width_zones: int = 3
    tire_section_width_m: float = 0.33
    inner_radius_m: float = 0.230
    outer_radius_m: float = 0.340
    radial_spacing_bias: float = 2.2
    internal_solver_dt_s: float = 0.01
    advection_cfl_limit: float = 0.85
    max_solver_substeps: int = 400
    diffusion_max_iterations: int = 24
    diffusion_tolerance_k: float = 1e-6
    source_patch_theta_fraction: float = 0.12
    source_patch_radial_fraction: float = 0.18
    source_volumetric_fraction: float = 0.70
    use_wheel_coupling: bool = True
    use_local_temp_friction_partition: bool = False
    use_reduced_patch_mechanics: bool = False
    use_structural_hysteresis_model: bool = False
    wheel_effective_radius_m: float = 0.330
    max_coupling_iterations: int = 8
    coupling_relaxation: float = 0.55
    coupling_torque_tolerance_nm: float = 15.0
    coupling_force_tolerance_n: float = 60.0
    coupling_slip_perturbation: float = 1e-3
    coupling_angle_perturbation_rad: float = 5e-4
    max_effective_slip_ratio: float = 0.25
    max_effective_slip_angle_rad: float = 0.22
    contact_patch_rows: int = 7
    contact_patch_cols: int = 5
    contact_patch_length_scale: float = 0.26
    contact_patch_min_length_m: float = 0.055
    contact_patch_max_length_m: float = 0.18
    contact_patch_min_width_m: float = 0.18
    contact_patch_max_width_m: float = 0.36
    pressure_shape_longitudinal: float = 0.90
    pressure_shape_lateral: float = 0.65
    shear_stiffness_longitudinal_pa_per_m: float = 1.8e7
    shear_stiffness_lateral_pa_per_m: float = 1.4e7
    partial_slip_relaxation: float = 0.82
    trailing_edge_slip_gain: float = 0.35
    flash_temperature_weight: float = 0.72
    mu_sliding_drop_fraction: float = 0.16
    force_mu_peak: float = 1.78
    force_mu_load_sensitivity: float = 0.12
    force_mu_temperature_peak_k: float = celsius_to_kelvin(92.0)
    force_mu_temperature_width_k: float = 22.0
    force_mu_min_fraction: float = 0.45
    force_slip_ratio_reference: float = 0.11
    force_slip_angle_reference_rad: float = 0.09
    force_combined_shape: float = 1.55
    longitudinal_force_shape: float = 1.0
    lateral_force_shape: float = 1.0
    lateral_weight_gain: float = 1.0
    force_reference_speed_mps: float = 55.0
    thermal_diffusivity_m2_per_s: float = 1.6e-7
    volumetric_heat_capacity_j_per_m3k: float = 1.9e6
    radial_deflection_reference_m: float = 0.018
    belt_strain_gain: float = 0.55
    sidewall_strain_gain: float = 0.95
    slip_strain_gain: float = 0.30
    pressure_strain_gain: float = 0.22
    internal_coupling: InternalCouplingParameters = field(default_factory=InternalCouplingParameters)
    local_contact: LocalContactParameters = field(default_factory=LocalContactParameters)
    construction: ConstructionParameters = field(default_factory=ConstructionParameters)
    boundary: HighFidelityBoundaryParameters = field(
        default_factory=HighFidelityBoundaryParameters
    )
    layer_stack: LayerStackParameters = field(default_factory=LayerStackParameters)
    pressure_patch: PressurePatchParameters = field(default_factory=PressurePatchParameters)
    core_sensor: CoreSensorParameters = field(default_factory=CoreSensorParameters)
    surface_state: SurfaceStateParameters = field(default_factory=SurfaceStateParameters)
    flash_layer: FlashLayerParameters = field(default_factory=FlashLayerParameters)
    brake_cooling: BrakeCoolingParameters = field(default_factory=BrakeCoolingParameters)
    prony_branches: tuple[PronyBranch, ...] = (
        PronyBranch(modulus_pa=7.8e6, relaxation_time_s=2.5e-3),
        PronyBranch(modulus_pa=1.15e7, relaxation_time_s=1.8e-2),
        PronyBranch(modulus_pa=6.6e6, relaxation_time_s=9.5e-2),
        PronyBranch(modulus_pa=2.8e6, relaxation_time_s=5.2e-1),
    )


@dataclass(frozen=True)
class HighFidelityTireState:
    """State container for the layered high-fidelity tire model."""

    temperature_nodes_k: np.ndarray
    thermal_field_rt_k: np.ndarray | None = None
    thermal_field_rtw_k: np.ndarray | None = None
    flash_temperature_field_tw_k: np.ndarray | None = None
    sidewall_field_tw_k: np.ndarray | None = None
    road_surface_temp_k: float | None = None
    road_subsurface_temp_k: float | None = None
    road_surface_temp_w_k: np.ndarray | None = None
    road_subsurface_temp_w_k: np.ndarray | None = None
    road_moisture_w: np.ndarray | None = None
    wear: float = 0.0
    age_index: float = 0.0
    grain_index_w: np.ndarray | None = None
    blister_index_w: np.ndarray | None = None
    dynamic_pressure_pa: float = STANDARD_ATMOSPHERIC_PRESSURE_PA
    dynamic_volume_m3: float = 0.0
    contact_patch_area_m2: float = 0.0
    zone_contact_patch_area_m2: np.ndarray | None = None
    effective_rolling_radius_m: float = 0.0
    last_energy_residual_pct: float = 0.0
    last_solver_substeps: int = 0
    last_friction_total_w: float = 0.0
    last_friction_to_tire_w: float = 0.0
    last_friction_to_road_w: float = 0.0
    last_road_conduction_w: float = 0.0
    last_rim_conduction_w: float = 0.0
    last_sidewall_heat_w: float = 0.0
    last_brake_heat_to_tire_w: float = 0.0
    last_brake_heat_to_rim_w: float = 0.0
    last_brake_heat_to_sidewall_w: float = 0.0
    last_effective_bead_htc_w_per_m2k: float = 0.0
    last_effective_slip_ratio: float = 0.0
    last_effective_slip_angle_rad: float = 0.0
    last_longitudinal_force_n: float = 0.0
    last_lateral_force_n: float = 0.0
    last_torque_residual_nm: float = 0.0
    last_lateral_force_residual_n: float = 0.0
    last_coupling_iterations: int = 0
    last_coupling_converged: bool = False
    last_contact_patch_length_m: float = 0.0
    last_contact_patch_width_m: float = 0.0
    last_sliding_fraction: float = 0.0
    last_effective_mu: float = 0.0
    last_hysteresis_strain_amplitude: float = 0.0
    last_zone_effective_mu: np.ndarray | None = None
    last_zone_friction_power_w: np.ndarray | None = None
    last_zone_friction_power_tire_w: np.ndarray | None = None
    last_zone_friction_power_road_w: np.ndarray | None = None
    last_zone_tire_heat_partition: np.ndarray | None = None
    last_zone_sliding_fraction: np.ndarray | None = None
    last_zone_flash_to_bulk_delta_k: np.ndarray | None = None
    last_effective_gas_inner_liner_htc_w_per_k: float = 0.0
    last_effective_gas_rim_htc_w_per_k: float = 0.0
    last_effective_cavity_rim_htc_w_per_k: float = 0.0
    last_cavity_to_rim_heat_w: float = 0.0
    last_gas_to_inner_liner_heat_w: float = 0.0
    last_gas_to_rim_heat_w: float = 0.0
    last_brake_disc_temp_k: float | None = None
    last_brake_disc_to_rim_heat_w: float = 0.0
    last_brake_disc_to_tire_heat_w: float = 0.0
    last_brake_disc_to_sidewall_heat_w: float = 0.0
    last_effective_contact_temp_k: float = 0.0
    last_adhesion_power_w: float = 0.0
    last_sliding_power_w: float = 0.0
    last_contact_pressure_factor: float = 1.0
    last_layer_conductivity_scale_by_layer: dict[str, float] = field(default_factory=dict)
    last_layer_hysteresis_scale_by_layer: dict[str, float] = field(default_factory=dict)
    last_heat_source_total_w: float = 0.0
    last_heat_sink_total_w: float = 0.0
    last_net_heat_to_tire_w: float = 0.0
    last_hysteresis_strain_by_layer: dict[str, float] = field(default_factory=dict)
    last_hysteresis_loss_modulus_by_layer_pa: dict[str, float] = field(default_factory=dict)
    last_hysteresis_power_by_layer_w: dict[str, float] = field(default_factory=dict)
    last_solver_advection_time_s: float | None = None
    last_solver_diffusion_time_s: float | None = None
    last_solver_diffusion_iterations: int | None = None
    last_wheel_coupling_time_s: float | None = None
    time_s: float = 0.0

    @property
    def core_temperature_k(self) -> float:
        return float(self.temperature_nodes_k[4])

    @property
    def core_temperature_c(self) -> float:
        return kelvin_to_celsius(self.core_temperature_k)


@dataclass(frozen=True)
class HighFidelityTireDiagnostics:
    """Diagnostics emitted by the layered high-fidelity tire model."""

    core_temperature_k: float
    core_temperature_c: float
    surface_temperature_k: float
    mean_temperature_k: float
    no_op_mode: bool
    excitation_frequency_hz: float
    loss_modulus_pa: float
    hysteresis_power_density_w_per_m3: float
    energy_source_total_w: float
    energy_residual_pct: float
    solver_substeps: int
    thermal_grid_shape: tuple[int, int] | tuple[int, int, int] | None
    road_surface_temp_k: float | None
    road_subsurface_temp_k: float | None
    friction_power_total_w: float
    friction_power_tire_w: float
    friction_power_road_w: float
    road_conduction_w: float
    rim_conduction_w: float
    brake_heat_to_tire_w: float
    brake_heat_to_rim_w: float
    effective_bead_htc_w_per_m2k: float
    effective_slip_ratio: float
    effective_slip_angle_rad: float
    longitudinal_force_n: float
    lateral_force_n: float
    torque_residual_nm: float
    lateral_force_residual_n: float
    coupling_iterations: int
    coupling_converged: bool
    contact_patch_length_m: float = 0.0
    contact_patch_width_m: float = 0.0
    sliding_fraction: float = 0.0
    effective_mu: float = 0.0
    hysteresis_strain_amplitude: float = 0.0
    per_zone_effective_mu: tuple[float, float, float] = (0.0, 0.0, 0.0)
    per_zone_friction_power_w: tuple[float, float, float] = (0.0, 0.0, 0.0)
    per_zone_friction_power_tire_w: tuple[float, float, float] = (0.0, 0.0, 0.0)
    per_zone_friction_power_road_w: tuple[float, float, float] = (0.0, 0.0, 0.0)
    per_zone_tire_heat_partition: tuple[float, float, float] = (0.0, 0.0, 0.0)
    per_zone_sliding_fraction: tuple[float, float, float] = (0.0, 0.0, 0.0)
    per_zone_flash_to_bulk_delta_k: tuple[float, float, float] = (0.0, 0.0, 0.0)
    effective_gas_inner_liner_htc_w_per_k: float = 0.0
    effective_gas_rim_htc_w_per_k: float = 0.0
    effective_cavity_rim_htc_w_per_k: float = 0.0
    cavity_to_rim_heat_w: float = 0.0
    gas_to_inner_liner_heat_w: float = 0.0
    gas_to_rim_heat_w: float = 0.0
    brake_disc_temperature_k: float | None = None
    brake_disc_to_rim_heat_w: float = 0.0
    brake_disc_to_tire_heat_w: float = 0.0
    brake_disc_to_sidewall_heat_w: float = 0.0
    effective_contact_temperature_k: float = 0.0
    adhesion_power_w: float = 0.0
    sliding_power_w: float = 0.0
    contact_pressure_factor: float = 1.0
    hysteresis_strain_by_layer: dict[str, float] = field(default_factory=dict)
    hysteresis_loss_modulus_by_layer_pa: dict[str, float] = field(default_factory=dict)
    hysteresis_power_by_layer_w: dict[str, float] = field(default_factory=dict)
    layer_conductivity_scale_by_layer: dict[str, float] = field(default_factory=dict)
    layer_hysteresis_scale_by_layer: dict[str, float] = field(default_factory=dict)
    heat_source_total_w: float = 0.0
    heat_sink_total_w: float = 0.0
    net_heat_to_tire_w: float = 0.0
    bulk_core_temperature_k: float | None = None
    cavity_gas_temperature_k: float | None = None
    core_temperature_compare_k: float | None = None
    flash_surface_temperature_k: float | None = None
    dynamic_pressure_pa: float = STANDARD_ATMOSPHERIC_PRESSURE_PA
    dynamic_pressure_bar_gauge: float = 0.0
    dynamic_volume_m3: float = 0.0
    contact_patch_area_m2: float = 0.0
    zone_contact_patch_area_m2: tuple[float, float, float] = (0.0, 0.0, 0.0)
    effective_rolling_radius_m: float = 0.0
    per_width_surface_temp_k: tuple[float, float, float] = (0.0, 0.0, 0.0)
    per_width_flash_surface_temp_k: tuple[float, float, float] = (0.0, 0.0, 0.0)
    per_width_bulk_temp_k: tuple[float, float, float] = (0.0, 0.0, 0.0)
    layer_mean_temp_k: dict[str, float] = field(default_factory=dict)
    road_surface_temp_w_k: tuple[float, float, float] = (0.0, 0.0, 0.0)
    road_subsurface_temp_w_k: tuple[float, float, float] = (0.0, 0.0, 0.0)
    road_moisture_w: tuple[float, float, float] = (0.0, 0.0, 0.0)
    age_index: float = 0.0
    grain_index_w: tuple[float, float, float] = (0.0, 0.0, 0.0)
    blister_index_w: tuple[float, float, float] = (0.0, 0.0, 0.0)
    brake_heat_to_sidewall_w: float = 0.0
    brake_duct_cooling_factor: float = 1.0
    wheel_wake_factor: float = 1.0
    solver_advection_time_s: float | None = None
    solver_diffusion_time_s: float | None = None
    solver_diffusion_iterations: int | None = None
    wheel_coupling_time_s: float | None = None
