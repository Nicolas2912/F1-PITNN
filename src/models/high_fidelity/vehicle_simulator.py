from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

from ..physics import celsius_to_kelvin
from ..vehicle_thermal import (
    VehicleInputs,
    VehicleParameters,
    WHEEL_IDS,
    WheelId,
    params_clip,
)
from .simulator import HighFidelityTireSimulator
from .types import (
    HighFidelityTireDiagnostics,
    HighFidelityTireInputs,
    HighFidelityTireModelParameters,
    HighFidelityTireState,
)


@dataclass(frozen=True)
class HighFidelityVehicleState:
    wheel_states: dict[WheelId, HighFidelityTireState]
    time_s: float = 0.0


@dataclass(frozen=True)
class HighFidelityVehicleDiagnostics:
    wheel_load_n: dict[WheelId, float]
    wheel_speed_mps: dict[WheelId, float]
    wheel_slip_ratio_cmd: dict[WheelId, float]
    wheel_slip_angle_cmd_rad: dict[WheelId, float]
    wheel_drive_torque_nm: dict[WheelId, float]
    wheel_brake_torque_nm: dict[WheelId, float]
    wheel_core_temp_c: dict[WheelId, float]
    wheel_surface_temp_c: dict[WheelId, float]
    wheel_effective_slip_ratio: dict[WheelId, float]
    wheel_effective_slip_angle_rad: dict[WheelId, float]
    wheel_coupling_converged: dict[WheelId, bool]
    tire_diagnostics: dict[WheelId, HighFidelityTireDiagnostics]
    total_vertical_force_n: float
    load_conservation_error_pct: float
    front_axle_load_n: float
    rear_axle_load_n: float
    right_minus_left_load_n: float


class HighFidelityVehicleSimulator:
    def __init__(
        self,
        parameters: VehicleParameters | None = None,
        tire_parameters_by_wheel: Mapping[WheelId, HighFidelityTireModelParameters] | None = None,
    ) -> None:
        self.parameters = parameters if parameters is not None else VehicleParameters()
        tire_params = tire_parameters_by_wheel if tire_parameters_by_wheel is not None else {}
        self._tire_parameters_by_wheel: dict[WheelId, HighFidelityTireModelParameters] = {
            wheel: tire_params.get(wheel, HighFidelityTireModelParameters()) for wheel in WHEEL_IDS
        }
        self._tire_simulators: dict[WheelId, HighFidelityTireSimulator] = {
            wheel: HighFidelityTireSimulator(self._tire_parameters_by_wheel[wheel]) for wheel in WHEEL_IDS
        }

    def initial_state(
        self,
        *,
        ambient_temp_k: float = celsius_to_kelvin(25.0),
        wear: float = 0.0,
        wear_by_wheel: Mapping[WheelId, float] | None = None,
    ) -> HighFidelityVehicleState:
        states: dict[WheelId, HighFidelityTireState] = {}
        for wheel in WHEEL_IDS:
            wheel_wear = wear if wear_by_wheel is None else wear_by_wheel.get(wheel, wear)
            states[wheel] = self._tire_simulators[wheel].initial_state(
                ambient_temp_k=ambient_temp_k,
                wear=wheel_wear,
            )
        return HighFidelityVehicleState(wheel_states=states, time_s=0.0)

    def step(
        self,
        state: HighFidelityVehicleState,
        inputs: VehicleInputs,
        dt_s: float,
    ) -> HighFidelityVehicleState:
        if dt_s <= 0.0:
            msg = f"dt_s must be positive, got {dt_s}"
            raise ValueError(msg)

        wheel_loads = self._wheel_vertical_loads(inputs)
        wheel_slip_ratio = self._wheel_slip_ratio_commands(inputs, wheel_loads)
        wheel_slip_angle = self._wheel_slip_angle_commands(inputs)
        wheel_speed = self._wheel_longitudinal_speeds(inputs)
        wheel_drive_torque = self._wheel_drive_torque(inputs, wheel_speed, wheel_slip_ratio)
        wheel_brake_torque = self._wheel_brake_torque(inputs, wheel_speed, wheel_slip_ratio)

        next_wheel_states: dict[WheelId, HighFidelityTireState] = {}
        for wheel in WHEEL_IDS:
            tire_inputs = self._tire_inputs_for_wheel(
                wheel=wheel,
                inputs=inputs,
                wheel_speed_mps=wheel_speed[wheel],
                normal_load_n=wheel_loads[wheel],
                slip_ratio_cmd=wheel_slip_ratio[wheel],
                slip_angle_cmd_rad=wheel_slip_angle[wheel],
                drive_torque_nm=wheel_drive_torque[wheel],
                brake_torque_nm=wheel_brake_torque[wheel],
            )
            next_wheel_states[wheel] = self._tire_simulators[wheel].step(
                state.wheel_states[wheel],
                tire_inputs,
                dt_s=dt_s,
            )
        return HighFidelityVehicleState(wheel_states=next_wheel_states, time_s=state.time_s + dt_s)

    def simulate(
        self,
        initial_state: HighFidelityVehicleState,
        inputs_stream: list[VehicleInputs],
        dt_s: float,
    ) -> list[HighFidelityVehicleState]:
        states = [initial_state]
        current = initial_state
        for inputs in inputs_stream:
            current = self.step(current, inputs, dt_s=dt_s)
            states.append(current)
        return states

    def diagnostics(
        self,
        state: HighFidelityVehicleState,
        inputs: VehicleInputs,
    ) -> HighFidelityVehicleDiagnostics:
        wheel_loads = self._wheel_vertical_loads(inputs)
        wheel_slip_ratio = self._wheel_slip_ratio_commands(inputs, wheel_loads)
        wheel_slip_angle = self._wheel_slip_angle_commands(inputs)
        wheel_speed = self._wheel_longitudinal_speeds(inputs)
        wheel_drive_torque = self._wheel_drive_torque(inputs, wheel_speed, wheel_slip_ratio)
        wheel_brake_torque = self._wheel_brake_torque(inputs, wheel_speed, wheel_slip_ratio)

        tire_diags: dict[WheelId, HighFidelityTireDiagnostics] = {}
        wheel_core_temp: dict[WheelId, float] = {}
        wheel_surface_temp: dict[WheelId, float] = {}
        wheel_effective_slip_ratio: dict[WheelId, float] = {}
        wheel_effective_slip_angle: dict[WheelId, float] = {}
        wheel_coupling_converged: dict[WheelId, bool] = {}
        for wheel in WHEEL_IDS:
            tire_inputs = self._tire_inputs_for_wheel(
                wheel=wheel,
                inputs=inputs,
                wheel_speed_mps=wheel_speed[wheel],
                normal_load_n=wheel_loads[wheel],
                slip_ratio_cmd=wheel_slip_ratio[wheel],
                slip_angle_cmd_rad=wheel_slip_angle[wheel],
                drive_torque_nm=wheel_drive_torque[wheel],
                brake_torque_nm=wheel_brake_torque[wheel],
            )
            diag = self._tire_simulators[wheel].diagnostics(state.wheel_states[wheel], tire_inputs)
            tire_diags[wheel] = diag
            wheel_core_temp[wheel] = state.wheel_states[wheel].core_temperature_c
            wheel_surface_temp[wheel] = diag.surface_temperature_k - 273.15
            wheel_effective_slip_ratio[wheel] = diag.effective_slip_ratio
            wheel_effective_slip_angle[wheel] = diag.effective_slip_angle_rad
            wheel_coupling_converged[wheel] = diag.coupling_converged

        total_vertical_force_n = self._total_vertical_force(inputs)
        load_sum = sum(wheel_loads.values())
        load_error_pct = abs(load_sum - total_vertical_force_n) / max(total_vertical_force_n, 1.0) * 100.0
        front_axle = wheel_loads["FL"] + wheel_loads["FR"]
        rear_axle = wheel_loads["RL"] + wheel_loads["RR"]
        right_minus_left = (wheel_loads["FR"] + wheel_loads["RR"]) - (wheel_loads["FL"] + wheel_loads["RL"])
        return HighFidelityVehicleDiagnostics(
            wheel_load_n=wheel_loads,
            wheel_speed_mps=wheel_speed,
            wheel_slip_ratio_cmd=wheel_slip_ratio,
            wheel_slip_angle_cmd_rad=wheel_slip_angle,
            wheel_drive_torque_nm=wheel_drive_torque,
            wheel_brake_torque_nm=wheel_brake_torque,
            wheel_core_temp_c=wheel_core_temp,
            wheel_surface_temp_c=wheel_surface_temp,
            wheel_effective_slip_ratio=wheel_effective_slip_ratio,
            wheel_effective_slip_angle_rad=wheel_effective_slip_angle,
            wheel_coupling_converged=wheel_coupling_converged,
            tire_diagnostics=tire_diags,
            total_vertical_force_n=total_vertical_force_n,
            load_conservation_error_pct=load_error_pct,
            front_axle_load_n=front_axle,
            rear_axle_load_n=rear_axle,
            right_minus_left_load_n=right_minus_left,
        )

    def _tire_inputs_for_wheel(
        self,
        *,
        wheel: WheelId,
        inputs: VehicleInputs,
        wheel_speed_mps: float,
        normal_load_n: float,
        slip_ratio_cmd: float,
        slip_angle_cmd_rad: float,
        drive_torque_nm: float,
        brake_torque_nm: float,
    ) -> HighFidelityTireInputs:
        wheel_omega = self._wheel_angular_speed(
            wheel=wheel,
            wheel_speed_mps=wheel_speed_mps,
            slip_ratio_cmd=slip_ratio_cmd,
            external_wheel_speed=inputs.wheel_angular_speed_radps,
        )
        return HighFidelityTireInputs(
            speed_mps=abs(wheel_speed_mps),
            wheel_angular_speed_radps=wheel_omega,
            normal_load_n=max(normal_load_n, self.parameters.min_wheel_load_n),
            slip_ratio_cmd=slip_ratio_cmd,
            slip_angle_cmd_rad=slip_angle_cmd_rad,
            drive_torque_nm=drive_torque_nm,
            brake_torque_nm=brake_torque_nm,
            lateral_force_target_n=None,
            brake_power_w=max(brake_torque_nm, 0.0) * abs(wheel_omega),
            ambient_temp_k=inputs.ambient_temp_k,
            track_temp_k=inputs.track_temp_k,
            road_surface_temp_k=inputs.track_temp_k,
            road_bulk_temp_k=inputs.track_temp_k,
            wind_mps=0.0,
            humidity_rel=0.50,
            solar_w_m2=0.0,
        )

    def _wheel_vertical_loads(self, inputs: VehicleInputs) -> dict[WheelId, float]:
        p = self.parameters
        total_vertical_force = self._total_vertical_force(inputs)
        downforce = p.aero_downforce_coeff_n_per_mps2 * max(inputs.speed_mps, 0.0) ** 2
        static_front = p.mass_kg * p.gravity_mps2 * p.front_static_weight_fraction + downforce * p.aero_front_fraction
        static_rear = total_vertical_force - static_front

        longitudinal_transfer = p.mass_kg * inputs.ax_mps2 * p.cg_height_m / max(p.wheelbase_m, 1e-6)
        front_axle = static_front - longitudinal_transfer
        rear_axle = static_rear + longitudinal_transfer

        front_lat_transfer = (
            p.mass_kg
            * inputs.ay_mps2
            * p.cg_height_m
            * p.front_roll_stiffness_fraction
            / max(p.front_track_m, 1e-6)
        )
        rear_lat_transfer = (
            p.mass_kg
            * inputs.ay_mps2
            * p.cg_height_m
            * (1.0 - p.front_roll_stiffness_fraction)
            / max(p.rear_track_m, 1e-6)
        )
        return {
            "FL": 0.5 * front_axle - 0.5 * front_lat_transfer,
            "FR": 0.5 * front_axle + 0.5 * front_lat_transfer,
            "RL": 0.5 * rear_axle - 0.5 * rear_lat_transfer,
            "RR": 0.5 * rear_axle + 0.5 * rear_lat_transfer,
        }

    def _wheel_slip_angle_commands(self, inputs: VehicleInputs) -> dict[WheelId, float]:
        p = self.parameters
        speed = max(abs(inputs.speed_mps), 1e-3)
        l_f = p.wheelbase_m * (1.0 - p.front_static_weight_fraction)
        l_r = p.wheelbase_m - l_f
        front_center_alpha = inputs.steering_angle_rad - math.atan2(l_f * inputs.yaw_rate_radps, speed)
        rear_center_alpha = -math.atan2(-l_r * inputs.yaw_rate_radps, speed)
        front_lr_offset = math.atan2(inputs.yaw_rate_radps * (0.5 * p.front_track_m), speed)
        rear_lr_offset = math.atan2(inputs.yaw_rate_radps * (0.5 * p.rear_track_m), speed)
        return {
            "FL": front_center_alpha + front_lr_offset,
            "FR": front_center_alpha - front_lr_offset,
            "RL": rear_center_alpha + rear_lr_offset,
            "RR": rear_center_alpha - rear_lr_offset,
        }

    def _wheel_slip_ratio_commands(
        self,
        inputs: VehicleInputs,
        wheel_load_n: Mapping[WheelId, float],
    ) -> dict[WheelId, float]:
        p = self.parameters
        wheel_speed_mps = self._wheel_longitudinal_speeds(inputs)
        wheel_drive_torque = self._wheel_drive_torque(inputs, wheel_speed_mps, None)
        wheel_brake_torque = self._wheel_brake_torque(inputs, wheel_speed_mps, None)
        slip_commands: dict[WheelId, float] = {}
        for wheel in WHEEL_IDS:
            speed_abs = abs(wheel_speed_mps[wheel])
            net_torque_nm = wheel_drive_torque[wheel] - wheel_brake_torque[wheel]
            power_based_force = net_torque_nm / max(p.wheel_radius_m, 1e-6)
            mu_demand = power_based_force / max(abs(wheel_load_n[wheel]), p.min_wheel_load_n)
            kappa_from_power = params_clip(
                p.power_to_slip_gain * mu_demand,
                -p.max_power_based_slip_ratio,
                p.max_power_based_slip_ratio,
            )
            if inputs.wheel_angular_speed_radps is None:
                slip_commands[wheel] = kappa_from_power
                continue

            omega = inputs.wheel_angular_speed_radps.get(wheel, 0.0)
            kappa_from_speed = (omega * p.wheel_radius_m - wheel_speed_mps[wheel]) / max(speed_abs, 1.0)
            blend = speed_abs / (speed_abs + max(p.slip_blend_speed_mps, 1e-3))
            slip_commands[wheel] = blend * kappa_from_speed + (1.0 - blend) * kappa_from_power
        return slip_commands

    def _wheel_drive_torque(
        self,
        inputs: VehicleInputs,
        wheel_speed_mps: Mapping[WheelId, float],
        wheel_slip_ratio_cmd: Mapping[WheelId, float] | None,
    ) -> dict[WheelId, float]:
        p = self.parameters
        drive_total = max(inputs.drive_power_w, 0.0)
        rear_total = drive_total * params_clip(p.drive_bias_rear, 0.0, 1.0)
        front_total = drive_total - rear_total
        drive_power = {
            "FL": 0.5 * front_total,
            "FR": 0.5 * front_total,
            "RL": 0.5 * rear_total,
            "RR": 0.5 * rear_total,
        }
        torques: dict[WheelId, float] = {}
        for wheel in WHEEL_IDS:
            slip_ratio_cmd = 0.0 if wheel_slip_ratio_cmd is None else wheel_slip_ratio_cmd.get(wheel, 0.0)
            omega = self._wheel_angular_speed(
                wheel=wheel,
                wheel_speed_mps=wheel_speed_mps[wheel],
                slip_ratio_cmd=slip_ratio_cmd,
                external_wheel_speed=inputs.wheel_angular_speed_radps,
            )
            omega_ref = max(abs(omega), p.min_speed_for_slip_mps / max(p.wheel_radius_m, 1e-6))
            torques[wheel] = drive_power[wheel] / omega_ref
        return torques

    def _wheel_brake_torque(
        self,
        inputs: VehicleInputs,
        wheel_speed_mps: Mapping[WheelId, float],
        wheel_slip_ratio_cmd: Mapping[WheelId, float] | None,
    ) -> dict[WheelId, float]:
        p = self.parameters
        brake_total = max(inputs.brake_power_w, 0.0)
        front_total = brake_total * params_clip(p.brake_bias_front, 0.0, 1.0)
        rear_total = brake_total - front_total
        brake_power = {
            "FL": 0.5 * front_total,
            "FR": 0.5 * front_total,
            "RL": 0.5 * rear_total,
            "RR": 0.5 * rear_total,
        }
        torques: dict[WheelId, float] = {}
        for wheel in WHEEL_IDS:
            slip_ratio_cmd = 0.0 if wheel_slip_ratio_cmd is None else wheel_slip_ratio_cmd.get(wheel, 0.0)
            omega = self._wheel_angular_speed(
                wheel=wheel,
                wheel_speed_mps=wheel_speed_mps[wheel],
                slip_ratio_cmd=slip_ratio_cmd,
                external_wheel_speed=inputs.wheel_angular_speed_radps,
            )
            omega_ref = max(abs(omega), p.min_speed_for_slip_mps / max(p.wheel_radius_m, 1e-6))
            torques[wheel] = brake_power[wheel] / omega_ref
        return torques

    def _wheel_longitudinal_speeds(self, inputs: VehicleInputs) -> dict[WheelId, float]:
        p = self.parameters
        v = inputs.speed_mps
        front_delta = inputs.yaw_rate_radps * (0.5 * p.front_track_m)
        rear_delta = inputs.yaw_rate_radps * (0.5 * p.rear_track_m)
        return {
            "FL": v - front_delta,
            "FR": v + front_delta,
            "RL": v - rear_delta,
            "RR": v + rear_delta,
        }

    def _wheel_angular_speed(
        self,
        *,
        wheel: WheelId,
        wheel_speed_mps: float,
        slip_ratio_cmd: float,
        external_wheel_speed: Mapping[WheelId, float] | None,
    ) -> float:
        if external_wheel_speed is not None and wheel in external_wheel_speed:
            return external_wheel_speed[wheel]
        return wheel_speed_mps * (1.0 + slip_ratio_cmd) / max(self.parameters.wheel_radius_m, 1e-6)

    def _total_vertical_force(self, inputs: VehicleInputs) -> float:
        p = self.parameters
        downforce = p.aero_downforce_coeff_n_per_mps2 * max(inputs.speed_mps, 0.0) ** 2
        return p.mass_kg * p.gravity_mps2 + downforce
