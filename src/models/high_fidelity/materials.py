from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .types import HighFidelityTireInputs, HighFidelityTireModelParameters


@dataclass(frozen=True)
class StructuralDeformationState:
    contact_patch_length_m: float
    contact_patch_width_m: float
    radial_deflection_m: float
    tread_strain: float
    belt_strain: float
    carcass_strain: float
    sidewall_strain: float
    equivalent_strain_amplitude: float


@dataclass(frozen=True)
class LayerHysteresisSummary:
    excitation_frequency_hz: float
    equivalent_loss_modulus_pa: float
    total_power_density_w_per_m3: float
    deformation: StructuralDeformationState
    strain_by_layer: dict[str, float]
    loss_modulus_by_layer_pa: dict[str, float]
    power_density_by_layer_w_per_m3: dict[str, float]


class ViscoelasticMaterialModel:
    """Prony + WLF material model for hysteresis source estimation."""

    def __init__(self, parameters: HighFidelityTireModelParameters) -> None:
        self.parameters = parameters

    def wlf_shift_factor(self, *, temperature_k: float) -> float:
        params = self.parameters
        delta_t = temperature_k - params.wlf_reference_temp_k
        denominator = params.wlf_c2_k + delta_t
        if abs(denominator) < 1e-9:
            denominator = 1e-9 if denominator >= 0.0 else -1e-9
        log10_a_t = -params.wlf_c1 * delta_t / denominator
        return 10.0 ** log10_a_t

    def excitation_frequency_hz(
        self,
        *,
        inputs: HighFidelityTireInputs,
        contact_patch_length_m: float | None = None,
    ) -> float:
        params = self.parameters
        f_rot_hz = abs(inputs.wheel_angular_speed_radps) / (2.0 * math.pi)
        slip_speed_mps = math.sqrt(
            (inputs.speed_mps * inputs.slip_ratio_cmd) ** 2
            + (inputs.speed_mps * math.tan(inputs.slip_angle_cmd_rad)) ** 2
        )
        slip_length_m = params.slip_frequency_length_m if contact_patch_length_m is None else max(contact_patch_length_m, 1e-6)
        f_slip_hz = slip_speed_mps / max(slip_length_m, 1e-6)
        return max(params.min_excitation_frequency_hz, f_rot_hz + f_slip_hz)

    def effective_strain_amplitude(
        self,
        *,
        inputs: HighFidelityTireInputs,
        dynamic_pressure_pa: float | None = None,
        wear: float = 0.0,
    ) -> StructuralDeformationState:
        params = self.parameters
        if not params.use_structural_hysteresis_model:
            return self._legacy_deformation_state(inputs=inputs)

        return self._structural_deformation_state(
            inputs=inputs,
            dynamic_pressure_pa=dynamic_pressure_pa,
            wear=wear,
        )

    def _legacy_deformation_state(
        self,
        *,
        inputs: HighFidelityTireInputs,
    ) -> StructuralDeformationState:
        params = self.parameters
        load_ratio = max(inputs.normal_load_n, 0.0) / max(params.reference_load_n, 1e-6)
        equivalent = params.strain_amplitude_reference * (load_ratio ** params.strain_load_sensitivity)
        equivalent = float(min(max(equivalent, params.min_strain_amplitude), params.max_strain_amplitude))
        slip_mag = abs(inputs.slip_ratio_cmd) + abs(math.tan(inputs.slip_angle_cmd_rad))
        tread_strain = 0.70 * equivalent + 0.18 * params.slip_strain_gain * slip_mag
        belt_strain = 0.55 * equivalent + 0.10 * params.slip_strain_gain * abs(inputs.slip_ratio_cmd)
        carcass_strain = 0.60 * equivalent
        sidewall_strain = 0.72 * equivalent + 0.15 * params.slip_strain_gain * abs(math.tan(inputs.slip_angle_cmd_rad))
        return StructuralDeformationState(
            contact_patch_length_m=params.slip_frequency_length_m,
            contact_patch_width_m=params.tire_section_width_m * 0.80,
            radial_deflection_m=params.radial_deflection_reference_m,
            tread_strain=float(max(tread_strain, 0.0)),
            belt_strain=float(max(belt_strain, 0.0)),
            carcass_strain=float(max(carcass_strain, 0.0)),
            sidewall_strain=float(max(sidewall_strain, 0.0)),
            equivalent_strain_amplitude=equivalent,
        )

    def _structural_deformation_state(
        self,
        *,
        inputs: HighFidelityTireInputs,
        dynamic_pressure_pa: float | None = None,
        wear: float = 0.0,
    ) -> StructuralDeformationState:
        params = self.parameters
        pp = params.pressure_patch
        gauge_pressure_pa = max((dynamic_pressure_pa or pp.reference_pressure_pa) - pp.atmospheric_pressure_pa, 20_000.0)
        effective_support_pa = gauge_pressure_pa + pp.carcass_support_pressure_pa
        patch_area_m2 = inputs.normal_load_n / max(effective_support_pa, 1.0)
        patch_area_m2 = float(max(min(patch_area_m2, pp.max_contact_patch_area_m2), pp.min_contact_patch_area_m2))
        width_m = float(
            max(
                min(
                    params.tire_section_width_m * (0.78 + 0.10 * min(inputs.normal_load_n / max(params.reference_load_n, 1e-6), 1.5)),
                    params.contact_patch_max_width_m,
                ),
                params.contact_patch_min_width_m,
            )
        )
        length_m = float(
            max(
                min(patch_area_m2 / max(width_m, 1e-6), params.contact_patch_max_length_m),
                params.contact_patch_min_length_m,
            )
        )
        load_ratio = max(inputs.normal_load_n, 0.0) / max(params.reference_load_n, 1e-6)
        pressure_ratio = max(pp.reference_pressure_pa - pp.atmospheric_pressure_pa, 1.0) / max(gauge_pressure_pa, 1.0)
        radial_deflection_m = params.radial_deflection_reference_m * (load_ratio ** 0.92) * (pressure_ratio ** 0.32)
        radial_deflection_m *= 0.85 + 0.15 * max(min(1.0 - wear, 1.0), 0.3)
        radial_deflection_m = float(max(min(radial_deflection_m, 0.028), 0.006))
        section_height_m = max(params.outer_radius_m - params.inner_radius_m, 1e-6)
        tread_strain = 0.42 * radial_deflection_m / section_height_m
        tread_strain += 0.70 * params.slip_strain_gain * abs(inputs.slip_ratio_cmd)
        tread_strain += 0.20 * params.slip_strain_gain * abs(math.tan(inputs.slip_angle_cmd_rad))
        belt_strain = params.belt_strain_gain * (length_m / max(2.0 * math.pi * params.wheel_effective_radius_m, 1e-6))
        belt_strain += 0.25 * params.slip_strain_gain * abs(inputs.slip_ratio_cmd)
        carcass_strain = 0.55 * radial_deflection_m / section_height_m
        carcass_strain += 0.18 * params.slip_strain_gain * (
            abs(inputs.slip_ratio_cmd) + abs(math.tan(inputs.slip_angle_cmd_rad))
        )
        sidewall_strain = params.sidewall_strain_gain * radial_deflection_m / section_height_m
        sidewall_strain += params.slip_strain_gain * abs(math.tan(inputs.slip_angle_cmd_rad))
        pressure_term = params.pressure_strain_gain * max(load_ratio - 1.0, -0.3)
        eq_strain = params.strain_amplitude_reference
        eq_strain += (
            0.10 * tread_strain
            + 0.08 * belt_strain
            + 0.10 * carcass_strain
            + 0.12 * sidewall_strain
            + 0.04 * pressure_term
        )
        eq_strain = float(max(min(eq_strain, params.max_strain_amplitude), params.min_strain_amplitude))
        return StructuralDeformationState(
            contact_patch_length_m=length_m,
            contact_patch_width_m=width_m,
            radial_deflection_m=radial_deflection_m,
            tread_strain=float(max(tread_strain, 0.0)),
            belt_strain=float(max(belt_strain, 0.0)),
            carcass_strain=float(max(carcass_strain, 0.0)),
            sidewall_strain=float(max(sidewall_strain, 0.0)),
            equivalent_strain_amplitude=eq_strain,
        )

    def loss_modulus_pa(
        self,
        *,
        temperature_k: float,
        frequency_hz: float,
    ) -> float:
        params = self.parameters
        omega = 2.0 * math.pi * max(frequency_hz, 0.0)
        a_t = self.wlf_shift_factor(temperature_k=temperature_k)

        e_double_prime = 0.0
        for branch in params.prony_branches:
            tau_t = max(branch.relaxation_time_s, 1e-9) * a_t
            omega_tau = omega * tau_t
            e_double_prime += max(branch.modulus_pa, 0.0) * omega_tau / (1.0 + omega_tau * omega_tau)
        return max(e_double_prime, 0.0)

    def hysteresis_power_density_w_per_m3(
        self,
        *,
        temperature_k: float,
        inputs: HighFidelityTireInputs,
        age_index: float = 0.0,
        wear: float = 0.0,
        dynamic_pressure_pa: float | None = None,
    ) -> float:
        summary = self.layer_hysteresis_summary(
            temperature_k=temperature_k,
            inputs=inputs,
            age_index=age_index,
            wear=wear,
            dynamic_pressure_pa=dynamic_pressure_pa,
        )
        return max(summary.total_power_density_w_per_m3, 0.0)

    def layer_hysteresis_summary(
        self,
        *,
        temperature_k: float,
        inputs: HighFidelityTireInputs,
        age_index: float = 0.0,
        wear: float = 0.0,
        dynamic_pressure_pa: float | None = None,
    ) -> LayerHysteresisSummary:
        deformation = self.effective_strain_amplitude(
            inputs=inputs,
            dynamic_pressure_pa=dynamic_pressure_pa,
            wear=wear,
        )
        frequency_hz = self.excitation_frequency_hz(
            inputs=inputs,
            contact_patch_length_m=deformation.contact_patch_length_m,
        )
        e_double_prime = self.loss_modulus_pa(
            temperature_k=temperature_k,
            frequency_hz=frequency_hz,
        )
        surface = self.parameters.surface_state
        stack = self.parameters.layer_stack
        tread_scale = max(surface.tread_thickness_m(wear), 1e-6) / max(surface.tread_thickness_fresh_m, 1e-6)
        age_scale = 1.0 + surface.aging_hysteresis_shift_per_index * max(age_index, 0.0)
        total_branch_modulus_pa = float(sum(max(branch.modulus_pa, 0.0) for branch in self.parameters.prony_branches))
        load_ratio = max(inputs.normal_load_n, 0.0) / max(self.parameters.reference_load_n, 1e-6)
        slip_mag = abs(inputs.slip_ratio_cmd) + abs(math.tan(inputs.slip_angle_cmd_rad))
        deflection_ratio = deformation.radial_deflection_m / max(self.parameters.radial_deflection_reference_m, 1e-6)
        loss_floor_fraction = 0.0012
        loss_floor_fraction += 0.0045 * min(slip_mag, 0.40)
        loss_floor_fraction += 0.0018 * max(load_ratio - 0.8, 0.0)
        loss_floor_fraction += 0.0014 * min(max(deflection_ratio - 0.6, 0.0), 1.5)
        loss_floor_fraction = min(max(loss_floor_fraction, 0.0012), 0.010)

        strain_by_layer = {
            "tread": deformation.tread_strain,
            "belt": deformation.belt_strain,
            "carcass": deformation.carcass_strain,
            "sidewall": deformation.sidewall_strain,
        }
        scale_by_layer = {
            "tread": stack.tread.hysteresis_scale * tread_scale,
            "belt": stack.belt.hysteresis_scale,
            "carcass": stack.carcass.hysteresis_scale,
            "sidewall": 0.85 * stack.carcass.hysteresis_scale,
        }
        temperature_offset_by_layer = {
            "tread": 4.0,
            "belt": 0.0,
            "carcass": -3.0,
            "sidewall": -6.0,
        }
        loss_modulus_by_layer_pa: dict[str, float] = {}
        power_density_by_layer_w_per_m3: dict[str, float] = {}
        for name, strain in strain_by_layer.items():
            layer_temp_k = temperature_k + temperature_offset_by_layer[name]
            layer_loss_pa = self.loss_modulus_pa(temperature_k=layer_temp_k, frequency_hz=frequency_hz)
            layer_loss_floor_pa = total_branch_modulus_pa * loss_floor_fraction * max(scale_by_layer[name], 0.05)
            layer_loss_pa = max(layer_loss_pa, layer_loss_floor_pa)
            loss_modulus_by_layer_pa[name] = layer_loss_pa
            power_density = math.pi * frequency_hz * layer_loss_pa * (max(strain, 0.0) ** 2)
            power_density *= age_scale * max(scale_by_layer[name], 0.05)
            power_density_by_layer_w_per_m3[name] = max(power_density, 0.0)

        total_power_density = float(np.sum(np.fromiter(power_density_by_layer_w_per_m3.values(), dtype=float)))
        return LayerHysteresisSummary(
            excitation_frequency_hz=frequency_hz,
            equivalent_loss_modulus_pa=e_double_prime,
            total_power_density_w_per_m3=total_power_density,
            deformation=deformation,
            strain_by_layer=strain_by_layer,
            loss_modulus_by_layer_pa=loss_modulus_by_layer_pa,
            power_density_by_layer_w_per_m3=power_density_by_layer_w_per_m3,
        )

    def hysteresis_source_summary(
        self,
        *,
        temperature_k: float,
        inputs: HighFidelityTireInputs,
        age_index: float = 0.0,
        wear: float = 0.0,
        dynamic_pressure_pa: float | None = None,
    ) -> tuple[float, float, float, StructuralDeformationState]:
        summary = self.layer_hysteresis_summary(
            temperature_k=temperature_k,
            inputs=inputs,
            age_index=age_index,
            wear=wear,
            dynamic_pressure_pa=dynamic_pressure_pa,
        )
        return (
            summary.excitation_frequency_hz,
            summary.equivalent_loss_modulus_pa,
            summary.total_power_density_w_per_m3,
            summary.deformation,
        )
