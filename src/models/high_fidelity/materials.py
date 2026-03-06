from __future__ import annotations

import math

from .types import HighFidelityTireInputs, HighFidelityTireModelParameters


class ViscoelasticMaterialModel:
    """Prony + WLF material model for hysteresis source estimation."""

    def __init__(self, parameters: HighFidelityTireModelParameters) -> None:
        self.parameters = parameters

    def wlf_shift_factor(self, *, temperature_k: float) -> float:
        """
        Return a_T from WLF.

        log10(a_T) = -C1 * (T - T_ref) / (C2 + (T - T_ref))
        """
        params = self.parameters
        delta_t = temperature_k - params.wlf_reference_temp_k
        denominator = params.wlf_c2_k + delta_t
        # Avoid singularity if T approaches T_ref - C2.
        if abs(denominator) < 1e-9:
            denominator = 1e-9 if denominator >= 0.0 else -1e-9
        log10_a_t = -params.wlf_c1 * delta_t / denominator
        return 10.0 ** log10_a_t

    def excitation_frequency_hz(
        self,
        *,
        inputs: HighFidelityTireInputs,
    ) -> float:
        params = self.parameters
        f_rot_hz = abs(inputs.wheel_angular_speed_radps) / (2.0 * math.pi)
        slip_speed_mps = math.sqrt(
            (inputs.speed_mps * inputs.slip_ratio_cmd) ** 2
            + (inputs.speed_mps * math.tan(inputs.slip_angle_cmd_rad)) ** 2
        )
        f_slip_hz = slip_speed_mps / max(params.slip_frequency_length_m, 1e-6)
        return max(params.min_excitation_frequency_hz, f_rot_hz + f_slip_hz)

    def effective_strain_amplitude(
        self,
        *,
        normal_load_n: float,
    ) -> float:
        params = self.parameters
        load_ratio = max(normal_load_n, 0.0) / max(params.reference_load_n, 1e-6)
        strain = params.strain_amplitude_reference * (load_ratio ** params.strain_load_sensitivity)
        return min(max(strain, params.min_strain_amplitude), params.max_strain_amplitude)

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
    ) -> float:
        frequency_hz = self.excitation_frequency_hz(inputs=inputs)
        e_double_prime = self.loss_modulus_pa(
            temperature_k=temperature_k,
            frequency_hz=frequency_hz,
        )
        strain_amp = self.effective_strain_amplitude(normal_load_n=inputs.normal_load_n)
        q_hyst = math.pi * frequency_hz * e_double_prime * (strain_amp**2)
        return max(q_hyst, 0.0)

    def hysteresis_source_summary(
        self,
        *,
        temperature_k: float,
        inputs: HighFidelityTireInputs,
    ) -> tuple[float, float, float]:
        """Returns (frequency_hz, loss_modulus_pa, volumetric_power_density)."""
        frequency_hz = self.excitation_frequency_hz(inputs=inputs)
        e_double_prime = self.loss_modulus_pa(
            temperature_k=temperature_k,
            frequency_hz=frequency_hz,
        )
        q_hyst = self.hysteresis_power_density_w_per_m3(
            temperature_k=temperature_k,
            inputs=inputs,
        )
        return (frequency_hz, e_double_prime, q_hyst)

