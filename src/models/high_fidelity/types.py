from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..physics import celsius_to_kelvin, kelvin_to_celsius
from .boundary_conditions import HighFidelityBoundaryParameters


@dataclass(frozen=True)
class HighFidelityTireInputs:
    """Inputs for the high-fidelity tire model skeleton."""

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
    humidity_rel: float = 0.50
    solar_w_m2: float = 0.0
    road_surface_temp_k: float | None = None
    road_bulk_temp_k: float | None = None


@dataclass(frozen=True)
class HighFidelityTireModelParameters:
    """Configuration for the high-fidelity tire model skeleton."""

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
    radial_cells: int = 24
    theta_cells: int = 72
    inner_radius_m: float = 0.230
    outer_radius_m: float = 0.340
    radial_spacing_bias: float = 2.2
    thermal_diffusivity_m2_per_s: float = 1.6e-7
    volumetric_heat_capacity_j_per_m3k: float = 1.9e6
    internal_solver_dt_s: float = 0.01
    advection_cfl_limit: float = 0.85
    max_solver_substeps: int = 400
    diffusion_max_iterations: int = 120
    diffusion_tolerance_k: float = 1e-6
    source_patch_theta_fraction: float = 0.12
    source_patch_radial_fraction: float = 0.18
    source_volumetric_fraction: float = 0.70
    brake_heat_to_tire_fraction: float = 0.03
    brake_heat_to_rim_fraction: float = 0.08
    # Assumption: the "core" telemetry available in race-tire workflows is usually an
    # embedded under-tread / belt-package temperature, not a full-thickness carcass
    # average and not the cavity gas temperature. Bias the proxy toward the heated
    # outer carcass with a modest first-order lag to mimic sensor inertia.
    core_probe_inner_fraction: float = 0.58
    core_probe_outer_fraction: float = 0.95
    core_probe_belt_weight: float = 0.72
    core_probe_carcass_weight: float = 0.12
    core_probe_gas_weight: float = 0.02
    core_probe_response_time_s: float = 3.0
    use_wheel_coupling: bool = True
    wheel_effective_radius_m: float = 0.330
    max_coupling_iterations: int = 8
    coupling_relaxation: float = 0.55
    coupling_torque_tolerance_nm: float = 15.0
    coupling_force_tolerance_n: float = 60.0
    coupling_slip_perturbation: float = 1e-3
    coupling_angle_perturbation_rad: float = 5e-4
    max_effective_slip_ratio: float = 0.25
    max_effective_slip_angle_rad: float = 0.22
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
    boundary: HighFidelityBoundaryParameters = field(
        default_factory=HighFidelityBoundaryParameters
    )
    prony_branches: tuple[PronyBranch, ...] = (
        PronyBranch(modulus_pa=7.8e6, relaxation_time_s=2.5e-3),
        PronyBranch(modulus_pa=1.15e7, relaxation_time_s=1.8e-2),
        PronyBranch(modulus_pa=6.6e6, relaxation_time_s=9.5e-2),
        PronyBranch(modulus_pa=2.8e6, relaxation_time_s=5.2e-1),
    )


@dataclass(frozen=True)
class HighFidelityTireState:
    """State container for the high-fidelity tire model skeleton."""

    temperature_nodes_k: np.ndarray
    thermal_field_rt_k: np.ndarray | None = None
    road_surface_temp_k: float | None = None
    road_subsurface_temp_k: float | None = None
    wear: float = 0.0
    last_energy_residual_pct: float = 0.0
    last_solver_substeps: int = 0
    last_friction_total_w: float = 0.0
    last_friction_to_tire_w: float = 0.0
    last_friction_to_road_w: float = 0.0
    last_road_conduction_w: float = 0.0
    last_rim_conduction_w: float = 0.0
    last_brake_heat_to_tire_w: float = 0.0
    last_brake_heat_to_rim_w: float = 0.0
    last_effective_bead_htc_w_per_m2k: float = 0.0
    last_effective_slip_ratio: float = 0.0
    last_effective_slip_angle_rad: float = 0.0
    last_longitudinal_force_n: float = 0.0
    last_lateral_force_n: float = 0.0
    last_torque_residual_nm: float = 0.0
    last_lateral_force_residual_n: float = 0.0
    last_coupling_iterations: int = 0
    last_coupling_converged: bool = False
    time_s: float = 0.0

    @property
    def core_temperature_k(self) -> float:
        return float(self.temperature_nodes_k[4])

    @property
    def core_temperature_c(self) -> float:
        return kelvin_to_celsius(self.core_temperature_k)


@dataclass(frozen=True)
class HighFidelityTireDiagnostics:
    """Diagnostics emitted by the high-fidelity tire model skeleton."""

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
    thermal_grid_shape: tuple[int, int] | None
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
