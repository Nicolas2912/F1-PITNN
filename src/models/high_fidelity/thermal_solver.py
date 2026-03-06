from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np

from .types import HighFidelityTireInputs, HighFidelityTireModelParameters


@dataclass(frozen=True)
class ThermalSolverStepResult:
    temperature_field_rtw_k: np.ndarray
    energy_residual_pct: float
    substeps: int
    max_cfl: float
    advection_time_s: float = 0.0
    diffusion_time_s: float = 0.0
    diffusion_iterations: int = 0

    @property
    def temperature_field_rt_k(self) -> np.ndarray:
        return np.mean(self.temperature_field_rtw_k, axis=2)


class ThermalFieldSolver2D:
    """
    Layered 3D thermal solver on a radial x circumferential x width grid.

    The historical class name is kept for compatibility with the rest of the HF module.
    """

    def __init__(self, parameters: HighFidelityTireModelParameters) -> None:
        self.parameters = parameters
        self._radial_edges_m = self._build_radial_edges_m()
        self._radial_centers_m = 0.5 * (self._radial_edges_m[:-1] + self._radial_edges_m[1:])
        self._dr_m = np.diff(self._radial_edges_m)
        self._theta_delta_rad = (2.0 * math.pi) / max(self.parameters.theta_cells, 1)
        self._width_edges = np.linspace(
            -0.5 * self.parameters.tire_section_width_m,
            0.5 * self.parameters.tire_section_width_m,
            self.parameters.width_zones + 1,
            dtype=float,
        )
        self._width_centers = 0.5 * (self._width_edges[:-1] + self._width_edges[1:])
        self._dw_m = np.diff(self._width_edges)
        self._width_indices = np.arange(self.parameters.width_zones, dtype=int)
        self._cell_volumes_m3 = self._build_cell_volumes_m3()
        self._surface_cell_areas_m2 = self._build_surface_cell_areas_m2()
        self._radial_coeff_minus, self._radial_coeff_plus = self._build_radial_diffusion_coefficients()
        self._theta_coeff = self._build_theta_diffusion_coefficients()
        self._width_coeff_minus, self._width_coeff_plus = self._build_width_diffusion_coefficients()
        self._scratch_shape = (
            self.parameters.radial_cells,
            self.parameters.theta_cells,
            self.parameters.width_zones,
        )
        self._scratch_radial_prev = np.zeros(self._scratch_shape, dtype=float)
        self._scratch_radial_next = np.zeros(self._scratch_shape, dtype=float)
        self._scratch_width_prev = np.zeros(self._scratch_shape, dtype=float)
        self._scratch_width_next = np.zeros(self._scratch_shape, dtype=float)
        self._scratch_theta_prev = np.zeros(self._scratch_shape, dtype=float)
        self._scratch_theta_next = np.zeros(self._scratch_shape, dtype=float)
        self._scratch_updated = np.zeros(self._scratch_shape, dtype=float)
        self._patch_theta_cells = max(1, int(round(self.parameters.theta_cells * self.parameters.source_patch_theta_fraction)))
        self._patch_radial_cells = max(1, int(round(self.parameters.radial_cells * self.parameters.source_patch_radial_fraction)))
        self._patch_radial_start = max(self.parameters.radial_cells - self._patch_radial_cells, 0)
        self._patch_radial_indices = np.arange(self._patch_radial_start, self.parameters.radial_cells, dtype=int)
        self._theta_offsets = np.arange(
            -(self._patch_theta_cells // 2),
            -(self._patch_theta_cells // 2) + self._patch_theta_cells,
            dtype=int,
        )

    @property
    def radial_centers_m(self) -> np.ndarray:
        return self._radial_centers_m.copy()

    @property
    def width_centers_m(self) -> np.ndarray:
        return self._width_centers.copy()

    @property
    def cell_volumes_m3(self) -> np.ndarray:
        return self._cell_volumes_m3.copy()

    @property
    def surface_cell_areas_m2(self) -> np.ndarray:
        return self._surface_cell_areas_m2.copy()

    def initial_temperature_field(self, ambient_temp_k: float) -> np.ndarray:
        return np.full(
            (self.parameters.radial_cells, self.parameters.theta_cells, self.parameters.width_zones),
            float(ambient_temp_k),
            dtype=float,
        )

    def source_field_w_per_m3(
        self,
        *,
        volumetric_source_w_per_m3: float,
        wheel_angular_speed_radps: float,
        time_s: float,
        zone_weights: np.ndarray | None = None,
        layer_source_weights: dict[str, float] | None = None,
    ) -> np.ndarray:
        params = self.parameters
        source = np.zeros(
            (params.radial_cells, params.theta_cells, params.width_zones),
            dtype=float,
        )

        if layer_source_weights:
            _, _, _, _, layer_index = self.layer_property_maps(wear=0.0)
            layer_code = {"tread": 0, "belt": 1, "carcass": 2, "sidewall": 2, "inner_liner": 3}
            total_weight = max(float(sum(max(value, 0.0) for value in layer_source_weights.values())), 1e-12)
            for layer_name, weight in layer_source_weights.items():
                if weight <= 0.0 or layer_name not in layer_code:
                    continue
                mask = layer_index == layer_code[layer_name]
                if not np.any(mask):
                    continue
                source[mask] += max(volumetric_source_w_per_m3, 0.0) * max(params.source_volumetric_fraction, 0.0) * (weight / total_weight)
        else:
            source.fill(max(volumetric_source_w_per_m3, 0.0) * max(params.source_volumetric_fraction, 0.0))

        source_remaining = max(volumetric_source_w_per_m3, 0.0) * max(1.0 - params.source_volumetric_fraction, 0.0)
        if source_remaining <= 0.0:
            return source

        radial_indices, theta_indices, width_indices = self.contact_patch_indices(
            wheel_angular_speed_radps=wheel_angular_speed_radps,
            time_s=time_s,
        )
        patch_index = np.ix_(radial_indices, theta_indices, width_indices)
        patch_cells = radial_indices.shape[0] * theta_indices.shape[0] * width_indices.shape[0]
        if zone_weights is None:
            zone_weights = np.full(params.width_zones, 1.0 / max(params.width_zones, 1), dtype=float)
        zone_weights = np.asarray(zone_weights, dtype=float)
        zone_weights = zone_weights / max(float(np.sum(zone_weights)), 1e-12)
        patch_extra_density = source_remaining * (params.radial_cells * params.theta_cells) / max(patch_cells, 1)
        source[patch_index] += patch_extra_density * zone_weights[None, None, :]
        return source

    def contact_patch_indices(
        self,
        *,
        wheel_angular_speed_radps: float,
        time_s: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        phase_theta = (wheel_angular_speed_radps * time_s) % (2.0 * math.pi)
        center_theta_idx = int(round(phase_theta / max(self._theta_delta_rad, 1e-9))) % self.parameters.theta_cells
        theta_indices = (center_theta_idx + self._theta_offsets) % self.parameters.theta_cells
        return self._patch_radial_indices, theta_indices, self._width_indices

    def patch_volume_m3(
        self,
        *,
        wheel_angular_speed_radps: float,
        time_s: float,
    ) -> float:
        radial_indices, theta_indices, width_indices = self.contact_patch_indices(
            wheel_angular_speed_radps=wheel_angular_speed_radps,
            time_s=time_s,
        )
        return float(np.sum(self._cell_volumes_m3[np.ix_(radial_indices, theta_indices, width_indices)]))

    def layer_property_maps(
        self,
        *,
        wear: float,
        grain_index_w: np.ndarray | None = None,
        blister_index_w: np.ndarray | None = None,
        age_index: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        params = self.parameters
        width_zones = params.width_zones
        grain_index = (
            np.zeros(width_zones, dtype=float)
            if grain_index_w is None
            else np.asarray(grain_index_w, dtype=float)
        )
        blister_index = (
            np.zeros(width_zones, dtype=float)
            if blister_index_w is None
            else np.asarray(blister_index_w, dtype=float)
        )

        radial_span = params.outer_radius_m - params.inner_radius_m
        tread_thickness_m = params.surface_state.tread_thickness_m(wear)
        stack = params.layer_stack
        thicknesses = np.array(
            [
                tread_thickness_m,
                stack.belt.thickness_m,
                stack.carcass.thickness_m,
                stack.inner_liner.thickness_m,
            ],
            dtype=float,
        )
        total_stack = max(float(np.sum(thicknesses)), 1e-9)
        thicknesses *= radial_span / total_stack
        cumulative_depth = np.cumsum(thicknesses)

        rho_cp = np.zeros_like(self._cell_volumes_m3)
        k_r = np.zeros_like(self._cell_volumes_m3)
        k_theta = np.zeros_like(self._cell_volumes_m3)
        k_w = np.zeros_like(self._cell_volumes_m3)
        layer_index = np.zeros_like(self._cell_volumes_m3, dtype=int)

        layer_params = (
            LayerEntry("tread", stack.tread),
            LayerEntry("belt", stack.belt),
            LayerEntry("carcass", stack.carcass),
            LayerEntry("inner_liner", stack.inner_liner),
        )
        for r_idx, radius_m in enumerate(self._radial_centers_m):
            depth_from_outer = params.outer_radius_m - radius_m
            if depth_from_outer <= cumulative_depth[0]:
                entry = layer_params[0]
                layer_code = 0
            elif depth_from_outer <= cumulative_depth[1]:
                entry = layer_params[1]
                layer_code = 1
            elif depth_from_outer <= cumulative_depth[2]:
                entry = layer_params[2]
                layer_code = 2
            else:
                entry = layer_params[3]
                layer_code = 3

            for width_idx in range(width_zones):
                grain_penalty = 1.0 - 0.08 * grain_index[width_idx] if entry.name == "tread" else 1.0
                blister_scale = params.surface_state.blister_conductivity_penalty if entry.name == "tread" else 0.10
                blister_penalty = 1.0 - blister_scale * blister_index[width_idx]
                age_gain = 1.0 + 0.06 * max(age_index, 0.0)
                rho_cp[r_idx, :, width_idx] = entry.material.volumetric_heat_capacity_j_per_m3k * age_gain
                k_r[r_idx, :, width_idx] = max(entry.material.k_r_w_per_mk * grain_penalty * blister_penalty, 1e-4)
                k_theta[r_idx, :, width_idx] = max(entry.material.k_theta_w_per_mk * grain_penalty * blister_penalty, 1e-4)
                k_w[r_idx, :, width_idx] = max(entry.material.k_w_w_per_mk * blister_penalty, 1e-4)
                layer_index[r_idx, :, width_idx] = layer_code

        return rho_cp, k_r, k_theta, k_w, layer_index

    def layer_mean_temperatures_k(
        self,
        *,
        temperature_field_rtw_k: np.ndarray,
        wear: float,
    ) -> dict[str, float]:
        _, _, _, _, layer_index = self.layer_property_maps(wear=wear)
        outputs: dict[str, float] = {}
        for layer_code, layer_name in enumerate(("tread", "belt", "carcass", "inner_liner")):
            mask = layer_index == layer_code
            outputs[layer_name] = float(np.mean(temperature_field_rtw_k[mask])) if np.any(mask) else float("nan")
        return outputs

    def step(
        self,
        *,
        temperature_field_rtw_k: np.ndarray | None = None,
        temperature_field_rt_k: np.ndarray | None = None,
        inputs: HighFidelityTireInputs,
        time_s: float,
        dt_s: float,
        volumetric_source_w_per_m3: float,
        extra_source_w_per_m3: np.ndarray | None = None,
        layer_source_weights: dict[str, float] | None = None,
        wear: float = 0.0,
        grain_index_w: np.ndarray | None = None,
        blister_index_w: np.ndarray | None = None,
        age_index: float = 0.0,
        zone_source_weights: np.ndarray | None = None,
    ) -> ThermalSolverStepResult:
        if dt_s <= 0.0:
            msg = f"dt_s must be positive, got {dt_s}"
            raise ValueError(msg)

        if temperature_field_rtw_k is None and temperature_field_rt_k is None:
            msg = "One of temperature_field_rtw_k or temperature_field_rt_k must be provided"
            raise ValueError(msg)
        base_field = temperature_field_rtw_k
        if base_field is None and temperature_field_rt_k is not None:
            rt = np.asarray(temperature_field_rt_k, dtype=float)
            if rt.ndim == 2:
                base_field = np.repeat(rt[:, :, None], self.parameters.width_zones, axis=2)
            else:
                base_field = rt
        field = np.array(base_field, dtype=float, copy=True)
        self._validate_field_shape(field)
        if extra_source_w_per_m3 is not None:
            self._validate_field_shape(extra_source_w_per_m3)

        rho_cp, k_r, k_theta, k_w, _ = self.layer_property_maps(
            wear=wear,
            grain_index_w=grain_index_w,
            blister_index_w=blister_index_w,
            age_index=age_index,
        )

        omega = inputs.wheel_angular_speed_radps
        max_cfl = abs(omega) * dt_s / max(self._theta_delta_rad, 1e-9)
        substeps = min(
            max(1, math.ceil(dt_s / max(self.parameters.internal_solver_dt_s, 1e-6))),
            max(self.parameters.max_solver_substeps, 1),
        )
        dt_sub = dt_s / substeps

        expected_energy_j = self._total_energy_j(field, rho_cp=rho_cp)
        advection_time_s = 0.0
        diffusion_time_s = 0.0
        diffusion_iterations = 0
        for sub_idx in range(substeps):
            t_sub = time_s + sub_idx * dt_sub
            source = self.source_field_w_per_m3(
                volumetric_source_w_per_m3=volumetric_source_w_per_m3,
                wheel_angular_speed_radps=omega,
                time_s=t_sub,
                zone_weights=zone_source_weights,
                layer_source_weights=layer_source_weights,
            )
            if extra_source_w_per_m3 is not None:
                source = source + extra_source_w_per_m3
            expected_energy_j += dt_sub * self._total_source_power_w(source)

            start = time.perf_counter() if self.parameters.enable_profiling else 0.0
            field = self._advect_theta_periodic(field, omega=omega, dt_s=dt_sub)
            if self.parameters.enable_profiling:
                advection_time_s += time.perf_counter() - start

            start = time.perf_counter() if self.parameters.enable_profiling else 0.0
            field, substep_iterations = self._diffuse_vectorized_implicit(
                field,
                source_w_per_m3=source,
                rho_cp=rho_cp,
                k_r=k_r,
                k_theta=k_theta,
                k_w=k_w,
                dt_s=dt_sub,
            )
            diffusion_iterations += substep_iterations
            if self.parameters.enable_profiling:
                diffusion_time_s += time.perf_counter() - start
            field = np.clip(field, self.parameters.minimum_temperature_k, self.parameters.maximum_temperature_k)

        actual_energy_j = self._total_energy_j(field, rho_cp=rho_cp)
        energy_residual_pct = abs(actual_energy_j - expected_energy_j) / max(abs(expected_energy_j), 1.0) * 100.0
        return ThermalSolverStepResult(
            temperature_field_rtw_k=field,
            energy_residual_pct=float(energy_residual_pct),
            substeps=substeps,
            max_cfl=float(max_cfl),
            advection_time_s=float(advection_time_s),
            diffusion_time_s=float(diffusion_time_s),
            diffusion_iterations=int(diffusion_iterations),
        )

    def _validate_field_shape(self, field: np.ndarray) -> None:
        expected = (
            self.parameters.radial_cells,
            self.parameters.theta_cells,
            self.parameters.width_zones,
        )
        if field.shape != expected:
            msg = f"Expected field shape {expected}, got {field.shape}"
            raise ValueError(msg)

    def _build_radial_edges_m(self) -> np.ndarray:
        params = self.parameters
        bias = max(params.radial_spacing_bias, 1e-6)
        x = np.linspace(0.0, 1.0, params.radial_cells + 1, dtype=float)
        stretched = 1.0 - np.power(1.0 - x, bias)
        return params.inner_radius_m + stretched * (params.outer_radius_m - params.inner_radius_m)

    def _build_cell_volumes_m3(self) -> np.ndarray:
        radial_cells = self.parameters.radial_cells
        theta_cells = self.parameters.theta_cells
        width_zones = self.parameters.width_zones
        volumes = np.zeros((radial_cells, theta_cells, width_zones), dtype=float)
        for i in range(radial_cells):
            arc_length_m = self._radial_centers_m[i] * self._theta_delta_rad
            for w in range(width_zones):
                volumes[i, :, w] = self._dr_m[i] * arc_length_m * self._dw_m[w]
        return volumes

    def _build_surface_cell_areas_m2(self) -> np.ndarray:
        theta_cells = self.parameters.theta_cells
        width_zones = self.parameters.width_zones
        areas = np.zeros((theta_cells, width_zones), dtype=float)
        surface_arc_length_m = self.parameters.outer_radius_m * self._theta_delta_rad
        for w in range(width_zones):
            areas[:, w] = surface_arc_length_m * self._dw_m[w]
        return areas

    def _build_radial_diffusion_coefficients(self) -> tuple[np.ndarray, np.ndarray]:
        radial_cells = self.parameters.radial_cells
        coeff_minus = np.zeros(radial_cells, dtype=float)
        coeff_plus = np.zeros(radial_cells, dtype=float)
        for i in range(radial_cells):
            if i == 0:
                dr_plus = self._radial_centers_m[i + 1] - self._radial_centers_m[i]
                coeff_plus[i] = 1.0 / max(dr_plus * dr_plus, 1e-12)
            elif i == radial_cells - 1:
                dr_minus = self._radial_centers_m[i] - self._radial_centers_m[i - 1]
                coeff_minus[i] = 1.0 / max(dr_minus * dr_minus, 1e-12)
            else:
                dr_minus = self._radial_centers_m[i] - self._radial_centers_m[i - 1]
                dr_plus = self._radial_centers_m[i + 1] - self._radial_centers_m[i]
                coeff_minus[i] = 2.0 / max(dr_minus * (dr_minus + dr_plus), 1e-12)
                coeff_plus[i] = 2.0 / max(dr_plus * (dr_minus + dr_plus), 1e-12)
        return coeff_minus, coeff_plus

    def _build_theta_diffusion_coefficients(self) -> np.ndarray:
        return 1.0 / np.maximum(self._radial_centers_m * self._theta_delta_rad, 1e-9) ** 2

    def _build_width_diffusion_coefficients(self) -> tuple[np.ndarray, np.ndarray]:
        width_zones = self.parameters.width_zones
        coeff_minus = np.zeros(width_zones, dtype=float)
        coeff_plus = np.zeros(width_zones, dtype=float)
        for i in range(width_zones):
            if i == 0:
                coeff_plus[i] = 1.0 / max(self._dw_m[i] * self._dw_m[i], 1e-12)
            elif i == width_zones - 1:
                coeff_minus[i] = 1.0 / max(self._dw_m[i - 1] * self._dw_m[i - 1], 1e-12)
            else:
                coeff_minus[i] = 2.0 / max(self._dw_m[i - 1] * (self._dw_m[i - 1] + self._dw_m[i]), 1e-12)
                coeff_plus[i] = 2.0 / max(self._dw_m[i] * (self._dw_m[i - 1] + self._dw_m[i]), 1e-12)
        return coeff_minus, coeff_plus

    def _advect_theta_periodic(self, field: np.ndarray, *, omega: float, dt_s: float) -> np.ndarray:
        if abs(omega) <= 1e-12:
            return field
        shift_cells = omega * dt_s / max(self._theta_delta_rad, 1e-12)
        if shift_cells >= 0.0:
            base_shift = int(math.floor(shift_cells))
            frac = shift_cells - base_shift
            primary = np.roll(field, base_shift, axis=1)
            secondary = np.roll(field, base_shift + 1, axis=1)
        else:
            source_shift = -shift_cells
            base_shift = int(math.floor(source_shift))
            frac = source_shift - base_shift
            primary = np.roll(field, -base_shift, axis=1)
            secondary = np.roll(field, -(base_shift + 1), axis=1)
        return (1.0 - frac) * primary + frac * secondary

    def _diffuse_vectorized_implicit(
        self,
        field: np.ndarray,
        *,
        source_w_per_m3: np.ndarray,
        rho_cp: np.ndarray,
        k_r: np.ndarray,
        k_theta: np.ndarray,
        k_w: np.ndarray,
        dt_s: float,
    ) -> tuple[np.ndarray, int]:
        rhs = field + dt_s * source_w_per_m3 / np.maximum(rho_cp, 1e-12)
        alpha_r = k_r / np.maximum(rho_cp, 1e-12)
        alpha_theta = k_theta / np.maximum(rho_cp, 1e-12)
        alpha_w = k_w / np.maximum(rho_cp, 1e-12)
        coeff_r_minus = dt_s * alpha_r * self._radial_coeff_minus[:, None, None]
        coeff_r_plus = dt_s * alpha_r * self._radial_coeff_plus[:, None, None]
        coeff_theta = dt_s * alpha_theta * self._theta_coeff[:, None, None]
        coeff_w_minus = dt_s * alpha_w * self._width_coeff_minus[None, None, :]
        coeff_w_plus = dt_s * alpha_w * self._width_coeff_plus[None, None, :]

        diagonal = 1.0 + coeff_r_minus + coeff_r_plus + 2.0 * coeff_theta + coeff_w_minus + coeff_w_plus
        estimate = rhs.copy()
        iterations = 0
        radial_prev = self._scratch_radial_prev
        radial_next = self._scratch_radial_next
        width_prev = self._scratch_width_prev
        width_next = self._scratch_width_next
        theta_prev = self._scratch_theta_prev
        theta_next = self._scratch_theta_next
        updated = self._scratch_updated
        for iterations in range(1, max(self.parameters.diffusion_max_iterations, 1) + 1):
            radial_prev.fill(0.0)
            radial_next.fill(0.0)
            radial_prev[1:, :, :] = estimate[:-1, :, :]
            radial_next[:-1, :, :] = estimate[1:, :, :]
            width_prev.fill(0.0)
            width_next.fill(0.0)
            width_prev[:, :, 1:] = estimate[:, :, :-1]
            width_next[:, :, :-1] = estimate[:, :, 1:]
            theta_prev[:, 0, :] = estimate[:, -1, :]
            theta_prev[:, 1:, :] = estimate[:, :-1, :]
            theta_next[:, -1, :] = estimate[:, 0, :]
            theta_next[:, :-1, :] = estimate[:, 1:, :]

            np.copyto(updated, rhs)
            updated += coeff_r_minus * radial_prev
            updated += coeff_r_plus * radial_next
            updated += coeff_theta * theta_prev
            updated += coeff_theta * theta_next
            updated += coeff_w_minus * width_prev
            updated += coeff_w_plus * width_next
            updated /= np.maximum(diagonal, 1e-12)
            max_delta = float(np.max(np.abs(updated - estimate)))
            estimate = updated.copy()
            if max_delta < self.parameters.diffusion_tolerance_k:
                break
        return estimate, iterations

    def _total_energy_j(self, field: np.ndarray, *, rho_cp: np.ndarray) -> float:
        return float(np.sum(field * rho_cp * self._cell_volumes_m3))

    def _total_source_power_w(self, source_w_per_m3: np.ndarray) -> float:
        return float(np.sum(source_w_per_m3 * self._cell_volumes_m3))


@dataclass(frozen=True)
class LayerEntry:
    name: str
    material: object
