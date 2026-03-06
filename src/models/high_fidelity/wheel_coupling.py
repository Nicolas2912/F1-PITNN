from __future__ import annotations

from dataclasses import dataclass
import math

from .types import HighFidelityTireInputs, HighFidelityTireModelParameters


@dataclass(frozen=True)
class WheelCouplingResult:
    effective_slip_ratio: float
    effective_slip_angle_rad: float
    longitudinal_force_n: float
    lateral_force_n: float
    net_wheel_torque_nm: float
    torque_residual_nm: float
    lateral_force_residual_n: float
    friction_power_w: float
    converged: bool
    iterations: int


class WheelForceCouplingModel:
    """Closed-loop wheel force/slip coupling using torque and lateral-force equilibrium."""

    def __init__(self, parameters: HighFidelityTireModelParameters) -> None:
        self.parameters = parameters

    def solve(
        self,
        *,
        inputs: HighFidelityTireInputs,
        surface_temp_k: float,
    ) -> WheelCouplingResult:
        params = self.parameters
        kappa = float(
            min(
                max(inputs.slip_ratio_cmd, -params.max_effective_slip_ratio),
                params.max_effective_slip_ratio,
            )
        )
        alpha = float(
            min(
                max(inputs.slip_angle_cmd_rad, -params.max_effective_slip_angle_rad),
                params.max_effective_slip_angle_rad,
            )
        )

        command_forces = self._evaluate_forces(
            slip_ratio=kappa,
            slip_angle_rad=alpha,
            normal_load_n=inputs.normal_load_n,
            surface_temp_k=surface_temp_k,
            speed_mps=inputs.speed_mps,
        )
        target_torque_nm = self._target_torque_nm(inputs=inputs, command_fx_n=command_forces[0])
        target_lateral_force_n = (
            command_forces[1]
            if inputs.lateral_force_target_n is None
            else float(inputs.lateral_force_target_n)
        )

        best_result: WheelCouplingResult | None = None
        best_error = math.inf

        max_iterations = max(params.max_coupling_iterations, 1)
        for iteration in range(1, max_iterations + 1):
            fx_n, fy_n, friction_power_w = self._evaluate_forces(
                slip_ratio=kappa,
                slip_angle_rad=alpha,
                normal_load_n=inputs.normal_load_n,
                surface_temp_k=surface_temp_k,
                speed_mps=inputs.speed_mps,
            )
            torque_residual_nm = target_torque_nm - fx_n * params.wheel_effective_radius_m
            lateral_force_residual_n = target_lateral_force_n - fy_n

            error_norm = self._normalized_error(
                torque_residual_nm=torque_residual_nm,
                lateral_force_residual_n=lateral_force_residual_n,
            )
            current = WheelCouplingResult(
                effective_slip_ratio=kappa,
                effective_slip_angle_rad=alpha,
                longitudinal_force_n=fx_n,
                lateral_force_n=fy_n,
                net_wheel_torque_nm=target_torque_nm,
                torque_residual_nm=torque_residual_nm,
                lateral_force_residual_n=lateral_force_residual_n,
                friction_power_w=friction_power_w,
                converged=(
                    abs(torque_residual_nm) <= params.coupling_torque_tolerance_nm
                    and abs(lateral_force_residual_n) <= params.coupling_force_tolerance_n
                ),
                iterations=iteration,
            )
            if error_norm < best_error:
                best_error = error_norm
                best_result = current
            if current.converged:
                return current

            kappa = self._next_slip_ratio(
                current_slip_ratio=kappa,
                current_slip_angle_rad=alpha,
                torque_residual_nm=torque_residual_nm,
                normal_load_n=inputs.normal_load_n,
                surface_temp_k=surface_temp_k,
                speed_mps=inputs.speed_mps,
            )
            alpha = self._next_slip_angle(
                current_slip_ratio=kappa,
                current_slip_angle_rad=alpha,
                lateral_force_residual_n=lateral_force_residual_n,
                normal_load_n=inputs.normal_load_n,
                surface_temp_k=surface_temp_k,
                speed_mps=inputs.speed_mps,
            )

        if best_result is None:
            msg = "Wheel coupling failed to produce any iterate"
            raise RuntimeError(msg)
        return best_result

    def _target_torque_nm(
        self,
        *,
        inputs: HighFidelityTireInputs,
        command_fx_n: float,
    ) -> float:
        omega = abs(inputs.wheel_angular_speed_radps)
        drive_torque_nm = 0.0 if inputs.drive_torque_nm is None else float(inputs.drive_torque_nm)
        if inputs.brake_torque_nm is not None:
            brake_torque_nm = float(inputs.brake_torque_nm)
        elif abs(inputs.brake_power_w) > 0.0 and omega > 1e-6:
            brake_torque_nm = abs(inputs.brake_power_w) / omega
        else:
            brake_torque_nm = 0.0

        explicit_torque_nm = drive_torque_nm - brake_torque_nm
        if inputs.drive_torque_nm is None and inputs.brake_torque_nm is None and abs(inputs.brake_power_w) <= 0.0:
            return command_fx_n * self.parameters.wheel_effective_radius_m
        return explicit_torque_nm

    def _evaluate_forces(
        self,
        *,
        slip_ratio: float,
        slip_angle_rad: float,
        normal_load_n: float,
        surface_temp_k: float,
        speed_mps: float,
    ) -> tuple[float, float, float]:
        params = self.parameters
        normal_load_n = max(normal_load_n, 0.0)
        if normal_load_n <= 0.0:
            return (0.0, 0.0, 0.0)

        mu_peak = self._friction_coefficient(
            normal_load_n=normal_load_n,
            surface_temp_k=surface_temp_k,
        )
        tan_alpha = math.tan(slip_angle_rad)
        slip_mag = math.sqrt(
            (slip_ratio / max(params.force_slip_ratio_reference, 1e-6)) ** 2
            + (tan_alpha / max(math.tan(params.force_slip_angle_reference_rad), 1e-6)) ** 2
        )
        slip_utilization = 1.0 - math.exp(-params.force_combined_shape * abs(slip_mag))
        force_total_n = mu_peak * normal_load_n * slip_utilization

        stiffness_long = max(params.longitudinal_force_shape, 1e-6)
        stiffness_lat = max(params.lateral_force_shape, 1e-6)
        weight_long = abs(slip_ratio) / max(
            abs(slip_ratio) + params.lateral_weight_gain * abs(tan_alpha) + 1e-12,
            1e-12,
        )
        weight_lat = 1.0 - weight_long

        fx_n = math.copysign(force_total_n * (weight_long**(1.0 / stiffness_long)), slip_ratio)
        fy_n = math.copysign(force_total_n * (weight_lat**(1.0 / stiffness_lat)), slip_angle_rad)

        slip_speed_mps = math.sqrt(
            (max(speed_mps, 0.0) * abs(slip_ratio)) ** 2
            + (max(speed_mps, 0.0) * tan_alpha) ** 2
        )
        friction_power_w = abs(fx_n * max(speed_mps, 0.0) * slip_ratio) + abs(
            fy_n * max(speed_mps, 0.0) * tan_alpha
        )
        friction_power_w = max(friction_power_w, force_total_n * slip_speed_mps)
        return (fx_n, fy_n, friction_power_w)

    def _next_slip_ratio(
        self,
        *,
        current_slip_ratio: float,
        current_slip_angle_rad: float,
        torque_residual_nm: float,
        normal_load_n: float,
        surface_temp_k: float,
        speed_mps: float,
    ) -> float:
        params = self.parameters
        dkappa = params.coupling_slip_perturbation
        fx_plus, _, _ = self._evaluate_forces(
            slip_ratio=current_slip_ratio + dkappa,
            slip_angle_rad=current_slip_angle_rad,
            normal_load_n=normal_load_n,
            surface_temp_k=surface_temp_k,
            speed_mps=speed_mps,
        )
        fx_minus, _, _ = self._evaluate_forces(
            slip_ratio=current_slip_ratio - dkappa,
            slip_angle_rad=current_slip_angle_rad,
            normal_load_n=normal_load_n,
            surface_temp_k=surface_temp_k,
            speed_mps=speed_mps,
        )
        derivative_nm_per_slip = (
            (fx_plus - fx_minus) * params.wheel_effective_radius_m / max(2.0 * dkappa, 1e-12)
        )
        if abs(derivative_nm_per_slip) < 1e-6:
            derivative_nm_per_slip = math.copysign(1e-6, torque_residual_nm if torque_residual_nm != 0.0 else 1.0)
        next_slip_ratio = current_slip_ratio + (
            params.coupling_relaxation * torque_residual_nm / derivative_nm_per_slip
        )
        return float(
            min(
                max(next_slip_ratio, -params.max_effective_slip_ratio),
                params.max_effective_slip_ratio,
            )
        )

    def _next_slip_angle(
        self,
        *,
        current_slip_ratio: float,
        current_slip_angle_rad: float,
        lateral_force_residual_n: float,
        normal_load_n: float,
        surface_temp_k: float,
        speed_mps: float,
    ) -> float:
        params = self.parameters
        dalpha = params.coupling_angle_perturbation_rad
        _, fy_plus, _ = self._evaluate_forces(
            slip_ratio=current_slip_ratio,
            slip_angle_rad=current_slip_angle_rad + dalpha,
            normal_load_n=normal_load_n,
            surface_temp_k=surface_temp_k,
            speed_mps=speed_mps,
        )
        _, fy_minus, _ = self._evaluate_forces(
            slip_ratio=current_slip_ratio,
            slip_angle_rad=current_slip_angle_rad - dalpha,
            normal_load_n=normal_load_n,
            surface_temp_k=surface_temp_k,
            speed_mps=speed_mps,
        )
        derivative_n_per_rad = (fy_plus - fy_minus) / max(2.0 * dalpha, 1e-12)
        if abs(derivative_n_per_rad) < 1e-6:
            derivative_n_per_rad = math.copysign(
                1e-6,
                lateral_force_residual_n if lateral_force_residual_n != 0.0 else 1.0,
            )
        next_slip_angle_rad = current_slip_angle_rad + (
            params.coupling_relaxation * lateral_force_residual_n / derivative_n_per_rad
        )
        return float(
            min(
                max(next_slip_angle_rad, -params.max_effective_slip_angle_rad),
                params.max_effective_slip_angle_rad,
            )
        )

    def _normalized_error(
        self,
        *,
        torque_residual_nm: float,
        lateral_force_residual_n: float,
    ) -> float:
        params = self.parameters
        return math.sqrt(
            (torque_residual_nm / max(params.coupling_torque_tolerance_nm, 1e-6)) ** 2
            + (lateral_force_residual_n / max(params.coupling_force_tolerance_n, 1e-6)) ** 2
        )

    def _friction_coefficient(
        self,
        *,
        normal_load_n: float,
        surface_temp_k: float,
    ) -> float:
        params = self.parameters
        load_ratio = normal_load_n / max(params.reference_load_n, 1e-6)
        load_factor = 1.0 / (1.0 + params.force_mu_load_sensitivity * max(load_ratio - 1.0, -0.5))
        temp_term = (surface_temp_k - params.force_mu_temperature_peak_k) / max(
            params.force_mu_temperature_width_k,
            1e-6,
        )
        temp_factor = params.force_mu_min_fraction + (1.0 - params.force_mu_min_fraction) * math.exp(
            -(temp_term**2)
        )
        return max(params.force_mu_peak * load_factor * temp_factor, params.force_mu_peak * 0.1)
