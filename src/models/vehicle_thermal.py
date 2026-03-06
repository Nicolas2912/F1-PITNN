from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Mapping

from .physics import (
    TireDiagnostics,
    TireInputs,
    TireModelParameters,
    TireState,
    TireThermalSimulator,
    celsius_to_kelvin,
)

WheelId = Literal["FL", "FR", "RL", "RR"]
WHEEL_IDS: tuple[WheelId, WheelId, WheelId, WheelId] = ("FL", "FR", "RL", "RR")


@dataclass(frozen=True)
class VehicleParameters:
    mass_kg: float = 798.0
    gravity_mps2: float = 9.81
    cg_height_m: float = 0.31
    wheelbase_m: float = 3.60
    front_track_m: float = 1.60
    rear_track_m: float = 1.58
    front_static_weight_fraction: float = 0.46
    front_roll_stiffness_fraction: float = 0.55
    aero_downforce_coeff_n_per_mps2: float = 1.85
    aero_front_fraction: float = 0.52
    wheel_radius_m: float = 0.33
    brake_bias_front: float = 0.58
    drive_bias_rear: float = 0.70
    min_wheel_load_n: float = 80.0
    min_speed_for_slip_mps: float = 2.0
    power_to_slip_gain: float = 0.28
    max_power_based_slip_ratio: float = 0.18
    slip_blend_speed_mps: float = 6.0


@dataclass(frozen=True)
class VehicleInputs:
    speed_mps: float
    ax_mps2: float = 0.0
    ay_mps2: float = 0.0
    steering_angle_rad: float = 0.0
    yaw_rate_radps: float = 0.0
    brake_power_w: float = 0.0
    drive_power_w: float = 0.0
    ambient_temp_k: float = celsius_to_kelvin(25.0)
    track_temp_k: float = celsius_to_kelvin(35.0)
    road_bulk_temp_k: float | None = None
    wind_mps: float = 0.0
    wind_yaw_rad: float = 0.0
    humidity_rel: float = 0.50
    solar_w_m2: float = 0.0
    road_moisture: float = 0.0
    rubbering_level: float = 0.0
    asphalt_roughness: float = 1.0
    asphalt_effusivity: float = 1.0
    brake_duct_cooling_factor_by_wheel: Mapping[WheelId, float] | None = None
    wheel_wake_factor_by_wheel: Mapping[WheelId, float] | None = None
    wheel_angular_speed_radps: Mapping[WheelId, float] | None = None
    camber_rad_by_wheel: Mapping[WheelId, float] | None = None
    toe_rad_by_wheel: Mapping[WheelId, float] | None = None


@dataclass(frozen=True)
class VehicleState:
    wheel_states: dict[WheelId, TireState]
    time_s: float = 0.0


@dataclass(frozen=True)
class VehicleDiagnostics:
    wheel_load_n: dict[WheelId, float]
    wheel_slip_ratio_cmd: dict[WheelId, float]
    wheel_slip_angle_cmd_rad: dict[WheelId, float]
    wheel_brake_power_w: dict[WheelId, float]
    wheel_drive_power_w: dict[WheelId, float]
    wheel_core_temp_c: dict[WheelId, float]
    wheel_surface_temp_c: dict[WheelId, float]
    wheel_pressure_bar_g: dict[WheelId, float]
    wheel_internal_htc_w_m2k: dict[WheelId, float]
    tire_diagnostics: dict[WheelId, TireDiagnostics]
    total_vertical_force_n: float
    load_conservation_error_pct: float
    front_axle_load_n: float
    rear_axle_load_n: float
    right_minus_left_load_n: float


class VehicleThermalSimulator:
    def __init__(
        self,
        parameters: VehicleParameters | None = None,
        tire_parameters_by_wheel: Mapping[WheelId, TireModelParameters] | None = None,
    ) -> None:
        self.parameters = parameters if parameters is not None else VehicleParameters()
        tire_params = tire_parameters_by_wheel if tire_parameters_by_wheel is not None else {}
        self._tire_parameters_by_wheel: dict[WheelId, TireModelParameters] = {
            wheel: tire_params.get(wheel, TireModelParameters()) for wheel in WHEEL_IDS
        }
        self._tire_simulators: dict[WheelId, TireThermalSimulator] = {
            wheel: TireThermalSimulator(self._tire_parameters_by_wheel[wheel]) for wheel in WHEEL_IDS
        }

    def initial_state(
        self,
        *,
        ambient_temp_k: float = celsius_to_kelvin(25.0),
        brake_temp_k: float | None = None,
        wear: float = 0.0,
        wear_by_wheel: Mapping[WheelId, float] | None = None,
    ) -> VehicleState:
        states: dict[WheelId, TireState] = {}
        for wheel in WHEEL_IDS:
            wheel_wear = wear if wear_by_wheel is None else wear_by_wheel.get(wheel, wear)
            states[wheel] = self._tire_simulators[wheel].initial_state(
                ambient_temp_k=ambient_temp_k,
                brake_temp_k=brake_temp_k,
                wear=wheel_wear,
            )
        return VehicleState(wheel_states=states, time_s=0.0)

    def step(self, state: VehicleState, inputs: VehicleInputs, dt_s: float) -> VehicleState:
        if dt_s <= 0.0:
            msg = f"dt_s must be positive, got {dt_s}"
            raise ValueError(msg)

        wheel_loads = self._wheel_vertical_loads(inputs)
        wheel_slip_ratio = self._wheel_slip_ratio_commands(inputs, wheel_loads)
        wheel_slip_angle = self._wheel_slip_angle_commands(inputs)
        wheel_brake_power = self._wheel_brake_power(inputs)
        wheel_speed = self._wheel_longitudinal_speeds(inputs)

        next_wheel_states: dict[WheelId, TireState] = {}
        for wheel in WHEEL_IDS:
            camber = 0.0 if inputs.camber_rad_by_wheel is None else inputs.camber_rad_by_wheel.get(wheel, 0.0)
            toe = 0.0 if inputs.toe_rad_by_wheel is None else inputs.toe_rad_by_wheel.get(wheel, 0.0)
            normal_load_n = max(wheel_loads[wheel], self.parameters.min_wheel_load_n)
            slip_ratio_cmd = wheel_slip_ratio[wheel]
            slip_angle_cmd = wheel_slip_angle[wheel]
            tire_inputs = TireInputs(
                speed_mps=abs(wheel_speed[wheel]),
                wheel_angular_speed_radps=self._wheel_angular_speed(
                    wheel=wheel,
                    wheel_speed_mps=wheel_speed[wheel],
                    slip_ratio_cmd=slip_ratio_cmd,
                    external_wheel_speed=inputs.wheel_angular_speed_radps,
                ),
                normal_load_n=normal_load_n,
                slip_ratio=slip_ratio_cmd,
                slip_angle_rad=slip_angle_cmd,
                slip_ratio_cmd=slip_ratio_cmd,
                slip_angle_cmd_rad=slip_angle_cmd,
                brake_power_w=wheel_brake_power[wheel],
                ambient_temp_k=inputs.ambient_temp_k,
                track_temp_k=inputs.track_temp_k,
                camber_rad=camber,
                toe_rad=toe,
                lateral_accel_mps2=inputs.ay_mps2,
                longitudinal_accel_mps2=inputs.ax_mps2,
                is_left_tire=wheel in ("FL", "RL"),
                is_front_tire=wheel in ("FL", "FR"),
            )
            next_wheel_states[wheel] = self._tire_simulators[wheel].step(
                state.wheel_states[wheel],
                tire_inputs,
                dt_s=dt_s,
            )

        return VehicleState(wheel_states=next_wheel_states, time_s=state.time_s + dt_s)

    def simulate(
        self,
        initial_state: VehicleState,
        inputs_stream: list[VehicleInputs],
        dt_s: float,
    ) -> list[VehicleState]:
        states = [initial_state]
        current = initial_state
        for inputs in inputs_stream:
            current = self.step(current, inputs, dt_s=dt_s)
            states.append(current)
        return states

    def diagnostics(self, state: VehicleState, inputs: VehicleInputs) -> VehicleDiagnostics:
        wheel_loads = self._wheel_vertical_loads(inputs)
        wheel_slip_ratio = self._wheel_slip_ratio_commands(inputs, wheel_loads)
        wheel_slip_angle = self._wheel_slip_angle_commands(inputs)
        wheel_brake_power = self._wheel_brake_power(inputs)
        wheel_drive_power = self._wheel_drive_power(inputs)
        wheel_speed = self._wheel_longitudinal_speeds(inputs)

        tire_diags: dict[WheelId, TireDiagnostics] = {}
        wheel_core_temp: dict[WheelId, float] = {}
        wheel_surface_temp: dict[WheelId, float] = {}
        wheel_pressure: dict[WheelId, float] = {}
        wheel_h_int: dict[WheelId, float] = {}
        for wheel in WHEEL_IDS:
            camber = 0.0 if inputs.camber_rad_by_wheel is None else inputs.camber_rad_by_wheel.get(wheel, 0.0)
            toe = 0.0 if inputs.toe_rad_by_wheel is None else inputs.toe_rad_by_wheel.get(wheel, 0.0)
            normal_load_n = max(wheel_loads[wheel], self.parameters.min_wheel_load_n)
            slip_ratio_cmd = wheel_slip_ratio[wheel]
            slip_angle_cmd = wheel_slip_angle[wheel]
            tire_inputs = TireInputs(
                speed_mps=abs(wheel_speed[wheel]),
                wheel_angular_speed_radps=self._wheel_angular_speed(
                    wheel=wheel,
                    wheel_speed_mps=wheel_speed[wheel],
                    slip_ratio_cmd=slip_ratio_cmd,
                    external_wheel_speed=inputs.wheel_angular_speed_radps,
                ),
                normal_load_n=normal_load_n,
                slip_ratio=slip_ratio_cmd,
                slip_angle_rad=slip_angle_cmd,
                slip_ratio_cmd=slip_ratio_cmd,
                slip_angle_cmd_rad=slip_angle_cmd,
                brake_power_w=wheel_brake_power[wheel],
                ambient_temp_k=inputs.ambient_temp_k,
                track_temp_k=inputs.track_temp_k,
                camber_rad=camber,
                toe_rad=toe,
                lateral_accel_mps2=inputs.ay_mps2,
                longitudinal_accel_mps2=inputs.ax_mps2,
                is_left_tire=wheel in ("FL", "RL"),
                is_front_tire=wheel in ("FL", "FR"),
            )
            diag = self._tire_simulators[wheel].diagnostics(state.wheel_states[wheel], tire_inputs)
            tire_diags[wheel] = diag
            wheel_core_temp[wheel] = state.wheel_states[wheel].core_temperature_c
            wheel_surface_temp[wheel] = state.wheel_states[wheel].t_surface_middle_k - 273.15
            wheel_pressure[wheel] = diag.dynamic_pressure_bar_gauge
            wheel_h_int[wheel] = diag.internal_htc_w_m2k

        total_vertical_force_n = self._total_vertical_force(inputs)
        load_sum = sum(wheel_loads.values())
        load_error_pct = abs(load_sum - total_vertical_force_n) / max(total_vertical_force_n, 1.0) * 100.0
        front_axle = wheel_loads["FL"] + wheel_loads["FR"]
        rear_axle = wheel_loads["RL"] + wheel_loads["RR"]
        right_minus_left = (wheel_loads["FR"] + wheel_loads["RR"]) - (wheel_loads["FL"] + wheel_loads["RL"])
        return VehicleDiagnostics(
            wheel_load_n=wheel_loads,
            wheel_slip_ratio_cmd=wheel_slip_ratio,
            wheel_slip_angle_cmd_rad=wheel_slip_angle,
            wheel_brake_power_w=wheel_brake_power,
            wheel_drive_power_w=wheel_drive_power,
            wheel_core_temp_c=wheel_core_temp,
            wheel_surface_temp_c=wheel_surface_temp,
            wheel_pressure_bar_g=wheel_pressure,
            wheel_internal_htc_w_m2k=wheel_h_int,
            tire_diagnostics=tire_diags,
            total_vertical_force_n=total_vertical_force_n,
            load_conservation_error_pct=load_error_pct,
            front_axle_load_n=front_axle,
            rear_axle_load_n=rear_axle,
            right_minus_left_load_n=right_minus_left,
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
        wheel_brake = self._wheel_brake_power(inputs)
        wheel_drive = self._wheel_drive_power(inputs)
        slip_commands: dict[WheelId, float] = {}
        for wheel in WHEEL_IDS:
            speed_abs = abs(wheel_speed_mps[wheel])
            power_based_force = (wheel_drive[wheel] - wheel_brake[wheel]) / max(speed_abs, p.min_speed_for_slip_mps)
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

    def _wheel_brake_power(self, inputs: VehicleInputs) -> dict[WheelId, float]:
        p = self.parameters
        brake_total = max(inputs.brake_power_w, 0.0)
        front_total = brake_total * params_clip(p.brake_bias_front, 0.0, 1.0)
        rear_total = brake_total - front_total
        return {
            "FL": 0.5 * front_total,
            "FR": 0.5 * front_total,
            "RL": 0.5 * rear_total,
            "RR": 0.5 * rear_total,
        }

    def _wheel_drive_power(self, inputs: VehicleInputs) -> dict[WheelId, float]:
        p = self.parameters
        drive_total = max(inputs.drive_power_w, 0.0)
        rear_total = drive_total * params_clip(p.drive_bias_rear, 0.0, 1.0)
        front_total = drive_total - rear_total
        return {
            "FL": 0.5 * front_total,
            "FR": 0.5 * front_total,
            "RL": 0.5 * rear_total,
            "RR": 0.5 * rear_total,
        }

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


def params_clip(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)
