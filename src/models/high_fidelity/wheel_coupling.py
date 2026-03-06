from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

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
    contact_patch_length_m: float
    contact_patch_width_m: float
    sliding_fraction: float
    effective_mu: float
    effective_contact_temperature_k: float = 0.0
    adhesion_power_w: float = 0.0
    sliding_power_w: float = 0.0
    contact_pressure_factor: float = 1.0
    zone_effective_mu: np.ndarray | None = None
    zone_friction_power_w: np.ndarray | None = None
    zone_sliding_fraction: np.ndarray | None = None


@dataclass(frozen=True)
class _PatchResponse:
    fx_n: float
    fy_n: float
    friction_power_w: float
    contact_patch_length_m: float
    contact_patch_width_m: float
    sliding_fraction: float
    effective_mu: float
    effective_contact_temperature_k: float
    adhesion_power_w: float
    sliding_power_w: float
    contact_pressure_factor: float
    zone_effective_mu: np.ndarray
    zone_friction_power_w: np.ndarray
    zone_sliding_fraction: np.ndarray


class WheelForceCouplingModel:
    """Closed-loop wheel force/slip coupling with a reduced pressure/shear patch model."""

    def __init__(self, parameters: HighFidelityTireModelParameters) -> None:
        self.parameters = parameters
        self._row_positions = np.linspace(-1.0, 1.0, self.parameters.contact_patch_rows, dtype=float)
        self._col_positions = np.linspace(-1.0, 1.0, self.parameters.contact_patch_cols, dtype=float)
        self._row_grid = self._row_positions[:, None]
        self._col_grid = self._col_positions[None, :]
        self._lead_trail_gain = 1.0 + self.parameters.trailing_edge_slip_gain * np.maximum(self._row_grid, 0.0)
        self._stick_gain = 1.0 - 0.30 * np.maximum(-self._row_grid, 0.0)
        self._local_contact_time_factor = 0.75 + 0.25 * (1.0 + self._row_grid)
        self._lateral_speed_shape = 0.85 + 0.20 * (1.0 - self._col_grid * self._col_grid)
        self._pressure_shape = self._build_pressure_shape()
        self._zone_interp = self._build_zone_interp()

    def solve(
        self,
        *,
        inputs: HighFidelityTireInputs,
        surface_temp_k: float,
        bulk_surface_temp_k: float | None = None,
        flash_surface_temp_k: float | None = None,
        zone_bulk_surface_temp_k: tuple[float, float, float] | None = None,
        zone_flash_surface_temp_k: tuple[float, float, float] | None = None,
        dynamic_pressure_pa: float | None = None,
        contact_patch_area_m2: float | None = None,
    ) -> WheelCouplingResult:
        params = self.parameters
        kappa = self._clamp(inputs.slip_ratio_cmd, -params.max_effective_slip_ratio, params.max_effective_slip_ratio)
        alpha = self._clamp(inputs.slip_angle_cmd_rad, -params.max_effective_slip_angle_rad, params.max_effective_slip_angle_rad)

        force_kwargs = {
            "normal_load_n": inputs.normal_load_n,
            "surface_temp_k": surface_temp_k,
            "bulk_surface_temp_k": bulk_surface_temp_k,
            "flash_surface_temp_k": flash_surface_temp_k,
            "zone_bulk_surface_temp_k": zone_bulk_surface_temp_k,
            "zone_flash_surface_temp_k": zone_flash_surface_temp_k,
            "speed_mps": inputs.speed_mps,
            "road_moisture": inputs.road_moisture,
            "rubbering_level": inputs.rubbering_level,
            "asphalt_roughness": inputs.asphalt_roughness,
            "dynamic_pressure_pa": dynamic_pressure_pa,
            "contact_patch_area_m2": contact_patch_area_m2,
        }

        target_torque_nm: float | None = None
        target_lateral_force_n: float | None = None

        best_result: WheelCouplingResult | None = None
        best_error = math.inf
        max_iterations = max(params.max_coupling_iterations, 1)
        previous_iterate: tuple[float, float, _PatchResponse] | None = None
        for iteration in range(1, max_iterations + 1):
            response = self._evaluate_patch_response(
                slip_ratio=kappa,
                slip_angle_rad=alpha,
                **force_kwargs,
            )
            if target_torque_nm is None:
                target_torque_nm = self._target_torque_nm(inputs=inputs, command_fx_n=response.fx_n)
            if target_lateral_force_n is None:
                target_lateral_force_n = response.fy_n if inputs.lateral_force_target_n is None else float(inputs.lateral_force_target_n)
            torque_residual_nm = target_torque_nm - response.fx_n * params.wheel_effective_radius_m
            lateral_force_residual_n = target_lateral_force_n - response.fy_n
            error_norm = self._normalized_error(
                torque_residual_nm=torque_residual_nm,
                lateral_force_residual_n=lateral_force_residual_n,
            )
            current = WheelCouplingResult(
                effective_slip_ratio=kappa,
                effective_slip_angle_rad=alpha,
                longitudinal_force_n=response.fx_n,
                lateral_force_n=response.fy_n,
                net_wheel_torque_nm=target_torque_nm,
                torque_residual_nm=torque_residual_nm,
                lateral_force_residual_n=lateral_force_residual_n,
                friction_power_w=response.friction_power_w,
                converged=(
                    abs(torque_residual_nm) <= params.coupling_torque_tolerance_nm
                    and abs(lateral_force_residual_n) <= params.coupling_force_tolerance_n
                ),
                iterations=iteration,
                contact_patch_length_m=response.contact_patch_length_m,
                contact_patch_width_m=response.contact_patch_width_m,
                sliding_fraction=response.sliding_fraction,
                effective_mu=response.effective_mu,
                effective_contact_temperature_k=response.effective_contact_temperature_k,
                adhesion_power_w=response.adhesion_power_w,
                sliding_power_w=response.sliding_power_w,
                contact_pressure_factor=response.contact_pressure_factor,
                zone_effective_mu=np.array(response.zone_effective_mu, dtype=float, copy=True),
                zone_friction_power_w=np.array(response.zone_friction_power_w, dtype=float, copy=True),
                zone_sliding_fraction=np.array(response.zone_sliding_fraction, dtype=float, copy=True),
            )
            if error_norm < best_error:
                best_error = error_norm
                best_result = current
            if current.converged:
                return current

            next_kappa = self._next_slip_ratio(
                current_slip_ratio=kappa,
                current_slip_angle_rad=alpha,
                torque_residual_nm=torque_residual_nm,
                force_kwargs=force_kwargs,
                current_response=response,
                previous_iterate=previous_iterate,
            )
            next_alpha = self._next_slip_angle(
                current_slip_ratio=kappa,
                current_slip_angle_rad=alpha,
                lateral_force_residual_n=lateral_force_residual_n,
                force_kwargs=force_kwargs,
                current_response=response,
                previous_iterate=previous_iterate,
            )
            previous_iterate = (kappa, alpha, response)
            kappa = next_kappa
            alpha = next_alpha

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

    def _evaluate_patch_response(
        self,
        *,
        slip_ratio: float,
        slip_angle_rad: float,
        normal_load_n: float,
        surface_temp_k: float,
        bulk_surface_temp_k: float | None,
        flash_surface_temp_k: float | None,
        zone_bulk_surface_temp_k: tuple[float, float, float] | None,
        zone_flash_surface_temp_k: tuple[float, float, float] | None,
        speed_mps: float,
        road_moisture: float,
        rubbering_level: float,
        asphalt_roughness: float,
        dynamic_pressure_pa: float | None,
        contact_patch_area_m2: float | None,
    ) -> _PatchResponse:
        params = self.parameters
        normal_load_n = max(normal_load_n, 0.0)
        if normal_load_n <= 0.0:
            zeros = np.zeros(3, dtype=float)
            return _PatchResponse(
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,
                zeros,
                zeros,
                zeros,
            )

        if not params.use_reduced_patch_mechanics:
            return self._evaluate_legacy_response(
                slip_ratio=slip_ratio,
                slip_angle_rad=slip_angle_rad,
                normal_load_n=normal_load_n,
                surface_temp_k=surface_temp_k,
                speed_mps=speed_mps,
                road_moisture=road_moisture,
                rubbering_level=rubbering_level,
                asphalt_roughness=asphalt_roughness,
            )

        patch_length_m, patch_width_m, area_m2 = self._contact_patch_geometry(
            normal_load_n=normal_load_n,
            dynamic_pressure_pa=dynamic_pressure_pa,
            contact_patch_area_m2=contact_patch_area_m2,
        )
        pressure_grid = self._pressure_distribution(normal_load_n=normal_load_n, area_m2=area_m2)
        bulk_profile_k, flash_profile_k = self._zone_temperature_profiles(
            surface_temp_k=surface_temp_k,
            bulk_surface_temp_k=bulk_surface_temp_k,
            flash_surface_temp_k=flash_surface_temp_k,
            zone_bulk_surface_temp_k=zone_bulk_surface_temp_k,
            zone_flash_surface_temp_k=zone_flash_surface_temp_k,
        )
        cell_area_m2 = area_m2 / max(params.contact_patch_rows * params.contact_patch_cols, 1)
        speed_abs = max(speed_mps, 0.0)
        tan_alpha = math.tan(slip_angle_rad)
        contact_time_s = patch_length_m / max(speed_abs, 3.0)
        local_vx = np.broadcast_to(speed_abs * slip_ratio * self._lead_trail_gain, pressure_grid.shape)
        local_vy = np.broadcast_to(speed_abs * tan_alpha * self._lateral_speed_shape, pressure_grid.shape)
        if params.local_contact.enabled:
            effective_temp_k = self._local_contact_temperature_grid_k(
                bulk_profile_k=bulk_profile_k,
                flash_profile_k=flash_profile_k,
            )
            mu_peak, mu_slide = self._local_friction_coefficients(
                normal_load_n=normal_load_n,
                contact_pressure_pa=pressure_grid,
                bulk_profile_k=bulk_profile_k,
                flash_profile_k=flash_profile_k,
                road_moisture=road_moisture,
                rubbering_level=rubbering_level,
                asphalt_roughness=asphalt_roughness,
            )
        else:
            mean_temp_k = self._mixed_contact_temperature_k(
                bulk_temp_k=float(np.mean(bulk_profile_k)),
                flash_temp_k=float(np.mean(flash_profile_k)),
            )
            effective_temp_k = np.full_like(pressure_grid, mean_temp_k, dtype=float)
            mu_peak = self._friction_coefficient_grid(
                normal_load_n=normal_load_n,
                surface_temp_k=effective_temp_k,
                road_moisture=road_moisture,
                rubbering_level=rubbering_level,
                asphalt_roughness=asphalt_roughness,
            )
            mu_slide = mu_peak * (1.0 - 0.6 * params.mu_sliding_drop_fraction)
        normal_cell_n = pressure_grid * cell_area_m2
        shear_limit_pa = mu_peak * pressure_grid
        mean_pressure_pa = normal_load_n / max(area_m2, 1e-9)
        contact_pressure_factor_grid = pressure_grid / max(mean_pressure_pa, 1e-9)
        local_contact_time_s = np.broadcast_to(contact_time_s * self._local_contact_time_factor, pressure_grid.shape)
        shear_cmd_x = params.shear_stiffness_longitudinal_pa_per_m * local_vx * local_contact_time_s * self._stick_gain
        shear_cmd_y = params.shear_stiffness_lateral_pa_per_m * local_vy * local_contact_time_s * self._stick_gain
        shear_cmd_mag = np.hypot(shear_cmd_x, shear_cmd_y)

        nonzero = shear_cmd_mag > 1e-12
        sliding_fraction = np.zeros_like(shear_cmd_mag)
        sliding_fraction[nonzero] = 1.0 - shear_limit_pa[nonzero] / np.maximum(shear_cmd_mag[nonzero], 1e-12)
        sliding_fraction = np.clip(sliding_fraction / max(params.partial_slip_relaxation, 1e-6), 0.0, 1.0)

        shear_act_x = np.zeros_like(shear_cmd_x)
        shear_act_y = np.zeros_like(shear_cmd_y)
        sticking = nonzero & (shear_cmd_mag <= shear_limit_pa)
        shear_act_x[sticking] = shear_cmd_x[sticking]
        shear_act_y[sticking] = shear_cmd_y[sticking]
        sliding = nonzero & ~sticking
        if np.any(sliding):
            sliding_drop = 1.0 - params.mu_sliding_drop_fraction * sliding_fraction[sliding]
            if params.local_contact.enabled:
                shear_cap = mu_slide[sliding] * pressure_grid[sliding] * np.maximum(sliding_drop, 0.45)
            else:
                shear_cap = shear_limit_pa[sliding] * np.maximum(sliding_drop, 0.5)
            direction_x = shear_cmd_x[sliding] / shear_cmd_mag[sliding]
            direction_y = shear_cmd_y[sliding] / shear_cmd_mag[sliding]
            shear_act_x[sliding] = shear_cap * direction_x
            shear_act_y[sliding] = shear_cap * direction_y

        work_partition = 0.25 + 0.75 * sliding_fraction
        cell_power_w = cell_area_m2 * ((np.abs(shear_act_x * local_vx) + np.abs(shear_act_y * local_vy)) * work_partition)
        fx_total = float(np.sum(shear_act_x) * cell_area_m2)
        fy_total = float(np.sum(shear_act_y) * cell_area_m2)
        friction_power_total = float(np.sum(cell_power_w))
        weighted_sliding = float(np.sum(sliding_fraction * normal_cell_n))
        weighted_mu = float(np.sum(mu_peak * normal_cell_n))
        weighted_temp = float(np.sum(effective_temp_k * normal_cell_n))
        weighted_pressure_factor = float(np.sum(contact_pressure_factor_grid * normal_cell_n))
        total_normal = float(np.sum(normal_cell_n))
        effective_mu = math.hypot(fx_total, fy_total) / max(normal_load_n, 1e-9)
        zone_normal = np.sum(normal_cell_n, axis=0)
        zone_mu = np.divide(
            np.sum(mu_peak * normal_cell_n, axis=0),
            np.maximum(zone_normal, 1e-9),
        )
        zone_friction_power = cell_area_m2 * np.sum(
            (np.abs(shear_act_x * local_vx) + np.abs(shear_act_y * local_vy)) * work_partition,
            axis=0,
        )
        zone_sliding = np.divide(
            np.sum(sliding_fraction * normal_cell_n, axis=0),
            np.maximum(zone_normal, 1e-9),
        )
        adhesion_power = float(np.sum(cell_power_w * (1.0 - sliding_fraction)))
        sliding_power = float(max(friction_power_total - adhesion_power, 0.0))
        return _PatchResponse(
            fx_n=float(fx_total),
            fy_n=float(fy_total),
            friction_power_w=float(max(friction_power_total, 0.0)),
            contact_patch_length_m=float(patch_length_m),
            contact_patch_width_m=float(patch_width_m),
            sliding_fraction=float(weighted_sliding / max(total_normal, 1e-9)),
            effective_mu=float(weighted_mu / max(total_normal, 1e-9) if total_normal > 0.0 else effective_mu),
            effective_contact_temperature_k=float(weighted_temp / max(total_normal, 1e-9)),
            adhesion_power_w=max(adhesion_power, 0.0),
            sliding_power_w=max(sliding_power, 0.0),
            contact_pressure_factor=float(weighted_pressure_factor / max(total_normal, 1e-9)),
            zone_effective_mu=np.array(zone_mu, dtype=float, copy=True),
            zone_friction_power_w=np.array(np.maximum(zone_friction_power, 0.0), dtype=float, copy=True),
            zone_sliding_fraction=np.array(np.clip(zone_sliding, 0.0, 1.0), dtype=float, copy=True),
        )

    def _evaluate_legacy_response(
        self,
        *,
        slip_ratio: float,
        slip_angle_rad: float,
        normal_load_n: float,
        surface_temp_k: float,
        speed_mps: float,
        road_moisture: float,
        rubbering_level: float,
        asphalt_roughness: float,
    ) -> _PatchResponse:
        params = self.parameters
        mu_peak = self._friction_coefficient(
            normal_load_n=normal_load_n,
            surface_temp_k=surface_temp_k,
            road_moisture=road_moisture,
            rubbering_level=rubbering_level,
            asphalt_roughness=asphalt_roughness,
        )
        tan_alpha = math.tan(slip_angle_rad)
        slip_mag = math.sqrt(
            (slip_ratio / max(params.force_slip_ratio_reference, 1e-6)) ** 2
            + (tan_alpha / max(math.tan(params.force_slip_angle_reference_rad), 1e-6)) ** 2
        )
        slip_utilization = 1.0 - math.exp(-params.force_combined_shape * abs(slip_mag))
        force_total_n = mu_peak * normal_load_n * slip_utilization
        weight_long = abs(slip_ratio) / max(
            abs(slip_ratio) + params.lateral_weight_gain * abs(tan_alpha) + 1e-12,
            1e-12,
        )
        weight_lat = 1.0 - weight_long
        fx_n = math.copysign(force_total_n * (weight_long ** (1.0 / max(params.longitudinal_force_shape, 1e-6))), slip_ratio)
        fy_n = math.copysign(force_total_n * (weight_lat ** (1.0 / max(params.lateral_force_shape, 1e-6))), slip_angle_rad)
        slip_speed_mps = math.sqrt(
            (max(speed_mps, 0.0) * abs(slip_ratio)) ** 2
            + (max(speed_mps, 0.0) * tan_alpha) ** 2
        )
        friction_power_w = abs(fx_n * max(speed_mps, 0.0) * slip_ratio) + abs(
            fy_n * max(speed_mps, 0.0) * tan_alpha
        )
        friction_power_w = max(friction_power_w, force_total_n * slip_speed_mps)
        zone_friction = np.full(3, friction_power_w / 3.0, dtype=float)
        zone_mu = np.full(3, mu_peak, dtype=float)
        zone_sliding = np.full(3, np.clip(slip_utilization, 0.0, 1.0), dtype=float)
        return _PatchResponse(
            fx_n=float(fx_n),
            fy_n=float(fy_n),
            friction_power_w=float(max(friction_power_w, 0.0)),
            contact_patch_length_m=0.0,
            contact_patch_width_m=0.0,
            sliding_fraction=float(np.mean(zone_sliding)),
            effective_mu=float(mu_peak),
            effective_contact_temperature_k=float(surface_temp_k),
            adhesion_power_w=float(0.35 * friction_power_w),
            sliding_power_w=float(0.65 * friction_power_w),
            contact_pressure_factor=1.0,
            zone_effective_mu=zone_mu,
            zone_friction_power_w=zone_friction,
            zone_sliding_fraction=zone_sliding,
        )

    def _contact_patch_geometry(
        self,
        *,
        normal_load_n: float,
        dynamic_pressure_pa: float | None,
        contact_patch_area_m2: float | None,
    ) -> tuple[float, float, float]:
        params = self.parameters
        pp = params.pressure_patch
        if contact_patch_area_m2 is not None and contact_patch_area_m2 > 1e-9:
            area_m2 = float(contact_patch_area_m2)
        else:
            gauge_pressure_pa = max((dynamic_pressure_pa or pp.reference_pressure_pa) - pp.atmospheric_pressure_pa, 20_000.0)
            support = pp.carcass_support_pressure_pa
            area_m2 = normal_load_n / max(gauge_pressure_pa + support, 1.0)
        area_m2 = self._clamp(area_m2, pp.min_contact_patch_area_m2, pp.max_contact_patch_area_m2)
        width_guess_m = self._clamp(
            params.tire_section_width_m * (0.78 + 0.12 * min(normal_load_n / max(params.reference_load_n, 1e-6), 1.5)),
            params.contact_patch_min_width_m,
            params.contact_patch_max_width_m,
        )
        length_guess_m = self._clamp(
            area_m2 / max(width_guess_m, 1e-6) * (0.90 + 0.10 * params.contact_patch_length_scale / 0.26),
            params.contact_patch_min_length_m,
            params.contact_patch_max_length_m,
        )
        width_m = self._clamp(area_m2 / max(length_guess_m, 1e-6), params.contact_patch_min_width_m, params.contact_patch_max_width_m)
        length_m = self._clamp(area_m2 / max(width_m, 1e-6), params.contact_patch_min_length_m, params.contact_patch_max_length_m)
        return length_m, width_m, length_m * width_m

    def _pressure_distribution(
        self,
        *,
        normal_load_n: float,
        area_m2: float,
    ) -> np.ndarray:
        mean_pressure_pa = normal_load_n / max(area_m2, 1e-9)
        return self._pressure_shape * mean_pressure_pa * self.parameters.contact_patch_rows * self.parameters.contact_patch_cols

    def _zone_temperature_profiles(
        self,
        *,
        surface_temp_k: float,
        bulk_surface_temp_k: float | None,
        flash_surface_temp_k: float | None,
        zone_bulk_surface_temp_k: tuple[float, float, float] | None,
        zone_flash_surface_temp_k: tuple[float, float, float] | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        bulk_zones = np.full(3, surface_temp_k if bulk_surface_temp_k is None else bulk_surface_temp_k, dtype=float)
        if zone_bulk_surface_temp_k is not None:
            bulk_zones = np.asarray(zone_bulk_surface_temp_k, dtype=float)
        flash_zones = np.full(3, surface_temp_k if flash_surface_temp_k is None else flash_surface_temp_k, dtype=float)
        if zone_flash_surface_temp_k is not None:
            flash_zones = np.asarray(zone_flash_surface_temp_k, dtype=float)
        return self._zone_interp @ bulk_zones, self._zone_interp @ flash_zones

    def _mixed_contact_temperature_k(self, *, bulk_temp_k: float, flash_temp_k: float) -> float:
        weight = self._clamp(self.parameters.flash_temperature_weight, 0.0, 1.0)
        return (1.0 - weight) * bulk_temp_k + weight * flash_temp_k

    def _mixed_contact_temperature_grid_k(
        self,
        *,
        bulk_profile_k: np.ndarray,
        flash_profile_k: np.ndarray,
    ) -> np.ndarray:
        weight = self._clamp(self.parameters.flash_temperature_weight, 0.0, 1.0)
        return (1.0 - weight) * bulk_profile_k[None, :] + weight * flash_profile_k[None, :]

    def _local_contact_temperature_grid_k(
        self,
        *,
        bulk_profile_k: np.ndarray,
        flash_profile_k: np.ndarray,
    ) -> np.ndarray:
        params = self.parameters.local_contact
        bulk_grid = bulk_profile_k[None, :]
        flash_grid = flash_profile_k[None, :]
        blend = self._clamp(params.adhesion_flash_weight, 0.0, 1.0)
        lead_trail = 0.65 + 0.35 * np.maximum(self._row_grid, 0.0)
        return (1.0 - blend * lead_trail) * bulk_grid + (blend * lead_trail) * flash_grid

    def _local_friction_coefficients(
        self,
        *,
        normal_load_n: float,
        contact_pressure_pa: np.ndarray,
        bulk_profile_k: np.ndarray,
        flash_profile_k: np.ndarray,
        road_moisture: float,
        rubbering_level: float,
        asphalt_roughness: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        params = self.parameters
        local = params.local_contact
        adhesion_temp_k = (1.0 - local.adhesion_flash_weight) * bulk_profile_k[None, :] + local.adhesion_flash_weight * flash_profile_k[None, :]
        sliding_temp_k = (1.0 - local.sliding_flash_weight) * bulk_profile_k[None, :] + local.sliding_flash_weight * flash_profile_k[None, :]
        mean_pressure_pa = max(float(np.mean(contact_pressure_pa)), 1e-9)
        pressure_factor = np.clip(contact_pressure_pa / mean_pressure_pa, 0.5, 1.8)
        base_adhesion = self._friction_coefficient_grid(
            normal_load_n=normal_load_n,
            surface_temp_k=adhesion_temp_k,
            road_moisture=road_moisture,
            rubbering_level=rubbering_level,
            asphalt_roughness=asphalt_roughness,
        )
        base_sliding = self._friction_coefficient_grid(
            normal_load_n=normal_load_n,
            surface_temp_k=sliding_temp_k,
            road_moisture=road_moisture,
            rubbering_level=rubbering_level,
            asphalt_roughness=asphalt_roughness,
        )
        pressure_gain = 1.0 + local.pressure_mu_sensitivity * (pressure_factor - 1.0)
        adhesion_peak_term = (adhesion_temp_k - local.adhesion_temperature_peak_k) / max(local.adhesion_temperature_width_k, 1e-6)
        adhesion_shape = local.adhesion_min_fraction + (1.0 - local.adhesion_min_fraction) * np.exp(-(adhesion_peak_term**2))
        sliding_peak_term = (sliding_temp_k - local.sliding_temperature_peak_k) / max(local.sliding_temperature_width_k, 1e-6)
        sliding_shape = local.sliding_min_fraction + (1.0 - local.sliding_min_fraction) * np.exp(-(sliding_peak_term**2))
        mu_adhesion = np.maximum(base_adhesion * pressure_gain * adhesion_shape, params.force_mu_peak * 0.08)
        mu_sliding = np.maximum(
            base_sliding * pressure_gain * sliding_shape * (1.0 - local.sliding_mu_drop_fraction),
            params.force_mu_peak * 0.06,
        )
        return mu_adhesion, mu_sliding

    def _next_slip_ratio(
        self,
        *,
        current_slip_ratio: float,
        current_slip_angle_rad: float,
        torque_residual_nm: float,
        force_kwargs: dict[str, float | tuple[float, float, float] | None],
        current_response: _PatchResponse,
        previous_iterate: tuple[float, float, _PatchResponse] | None,
    ) -> float:
        params = self.parameters
        derivative_nm_per_slip = self._torque_derivative(
            current_slip_ratio=current_slip_ratio,
            current_slip_angle_rad=current_slip_angle_rad,
            force_kwargs=force_kwargs,
            current_response=current_response,
            previous_iterate=previous_iterate,
        )
        if abs(derivative_nm_per_slip) < 1e-6:
            derivative_nm_per_slip = math.copysign(1e-6, torque_residual_nm if torque_residual_nm != 0.0 else 1.0)
        next_slip_ratio = current_slip_ratio + params.coupling_relaxation * torque_residual_nm / derivative_nm_per_slip
        return self._clamp(next_slip_ratio, -params.max_effective_slip_ratio, params.max_effective_slip_ratio)

    def _next_slip_angle(
        self,
        *,
        current_slip_ratio: float,
        current_slip_angle_rad: float,
        lateral_force_residual_n: float,
        force_kwargs: dict[str, float | tuple[float, float, float] | None],
        current_response: _PatchResponse,
        previous_iterate: tuple[float, float, _PatchResponse] | None,
    ) -> float:
        params = self.parameters
        derivative_n_per_rad = self._lateral_derivative(
            current_slip_ratio=current_slip_ratio,
            current_slip_angle_rad=current_slip_angle_rad,
            force_kwargs=force_kwargs,
            current_response=current_response,
            previous_iterate=previous_iterate,
        )
        if abs(derivative_n_per_rad) < 1e-6:
            derivative_n_per_rad = math.copysign(1e-6, lateral_force_residual_n if lateral_force_residual_n != 0.0 else 1.0)
        next_slip_angle_rad = current_slip_angle_rad + params.coupling_relaxation * lateral_force_residual_n / derivative_n_per_rad
        return self._clamp(next_slip_angle_rad, -params.max_effective_slip_angle_rad, params.max_effective_slip_angle_rad)

    def _torque_derivative(
        self,
        *,
        current_slip_ratio: float,
        current_slip_angle_rad: float,
        force_kwargs: dict[str, float | tuple[float, float, float] | None],
        current_response: _PatchResponse,
        previous_iterate: tuple[float, float, _PatchResponse] | None,
    ) -> float:
        params = self.parameters
        if previous_iterate is not None:
            prev_slip_ratio, prev_slip_angle, prev_response = previous_iterate
            if abs(current_slip_angle_rad - prev_slip_angle) <= params.coupling_angle_perturbation_rad:
                delta_slip = current_slip_ratio - prev_slip_ratio
                if abs(delta_slip) >= 0.25 * params.coupling_slip_perturbation:
                    return (current_response.fx_n - prev_response.fx_n) * params.wheel_effective_radius_m / delta_slip
        dkappa = params.coupling_slip_perturbation
        fx_plus = self._evaluate_patch_response(
            slip_ratio=current_slip_ratio + dkappa,
            slip_angle_rad=current_slip_angle_rad,
            **force_kwargs,
        ).fx_n
        return (fx_plus - current_response.fx_n) * params.wheel_effective_radius_m / max(dkappa, 1e-12)

    def _lateral_derivative(
        self,
        *,
        current_slip_ratio: float,
        current_slip_angle_rad: float,
        force_kwargs: dict[str, float | tuple[float, float, float] | None],
        current_response: _PatchResponse,
        previous_iterate: tuple[float, float, _PatchResponse] | None,
    ) -> float:
        params = self.parameters
        if previous_iterate is not None:
            prev_slip_ratio, prev_slip_angle, prev_response = previous_iterate
            if abs(current_slip_ratio - prev_slip_ratio) <= params.coupling_slip_perturbation:
                delta_angle = current_slip_angle_rad - prev_slip_angle
                if abs(delta_angle) >= 0.25 * params.coupling_angle_perturbation_rad:
                    return (current_response.fy_n - prev_response.fy_n) / delta_angle
        dalpha = params.coupling_angle_perturbation_rad
        fy_plus = self._evaluate_patch_response(
            slip_ratio=current_slip_ratio,
            slip_angle_rad=current_slip_angle_rad + dalpha,
            **force_kwargs,
        ).fy_n
        return (fy_plus - current_response.fy_n) / max(dalpha, 1e-12)

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
        road_moisture: float,
        rubbering_level: float,
        asphalt_roughness: float,
    ) -> float:
        params = self.parameters
        load_ratio = normal_load_n / max(params.reference_load_n, 1e-6)
        load_factor = 1.0 / (1.0 + params.force_mu_load_sensitivity * max(load_ratio - 1.0, -0.5))
        temp_term = (surface_temp_k - params.force_mu_temperature_peak_k) / max(params.force_mu_temperature_width_k, 1e-6)
        temp_factor = params.force_mu_min_fraction + (1.0 - params.force_mu_min_fraction) * math.exp(-(temp_term**2))
        road_factor = (
            (1.0 - 0.28 * self._clamp(road_moisture, 0.0, 1.0))
            * (1.0 + 0.05 * self._clamp(rubbering_level, 0.0, 1.0))
            * (1.0 + 0.08 * (asphalt_roughness - 1.0))
        )
        return max(params.force_mu_peak * load_factor * temp_factor * road_factor, params.force_mu_peak * 0.1)

    def _friction_coefficient_grid(
        self,
        *,
        normal_load_n: float,
        surface_temp_k: np.ndarray,
        road_moisture: float,
        rubbering_level: float,
        asphalt_roughness: float,
    ) -> np.ndarray:
        params = self.parameters
        load_ratio = normal_load_n / max(params.reference_load_n, 1e-6)
        load_factor = 1.0 / (1.0 + params.force_mu_load_sensitivity * max(load_ratio - 1.0, -0.5))
        temp_term = (surface_temp_k - params.force_mu_temperature_peak_k) / max(params.force_mu_temperature_width_k, 1e-6)
        temp_factor = params.force_mu_min_fraction + (1.0 - params.force_mu_min_fraction) * np.exp(-(temp_term**2))
        road_factor = (
            (1.0 - 0.28 * self._clamp(road_moisture, 0.0, 1.0))
            * (1.0 + 0.05 * self._clamp(rubbering_level, 0.0, 1.0))
            * (1.0 + 0.08 * (asphalt_roughness - 1.0))
        )
        return np.maximum(params.force_mu_peak * load_factor * temp_factor * road_factor, params.force_mu_peak * 0.1)

    def _build_pressure_shape(self) -> np.ndarray:
        params = self.parameters
        shape = np.zeros((params.contact_patch_rows, params.contact_patch_cols), dtype=float)
        for r_idx, x_val in enumerate(self._row_positions):
            long_term = max(1.0 - abs(x_val) ** (1.0 + params.pressure_shape_longitudinal), 0.05)
            for c_idx, y_val in enumerate(self._col_positions):
                lat_term = max(1.0 - abs(y_val) ** (1.0 + params.pressure_shape_lateral), 0.10)
                shape[r_idx, c_idx] = long_term * lat_term
        shape /= max(float(np.sum(shape)), 1e-12)
        return shape

    def _build_zone_interp(self) -> np.ndarray:
        interpolation = np.zeros((self.parameters.contact_patch_cols, 3), dtype=float)
        for idx, position in enumerate(self._col_positions):
            if position <= 0.0:
                weight = (position + 1.0) / max(1.0, 1e-12)
                interpolation[idx, 0] = 1.0 - weight
                interpolation[idx, 1] = weight
            else:
                weight = position / max(1.0, 1e-12)
                interpolation[idx, 1] = 1.0 - weight
                interpolation[idx, 2] = weight
        return interpolation

    def _clamp(self, value: float, lower: float, upper: float) -> float:
        return float(min(max(value, lower), upper))
