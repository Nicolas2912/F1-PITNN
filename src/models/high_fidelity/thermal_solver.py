from __future__ import annotations

from dataclasses import dataclass
import math
import time

import numpy as np

from .native_diffusion import (
    native_diffusion_available,
    native_diffusion_enabled,
    run_native_build_source_and_diffuse_implicit,
    run_native_diffuse_vectorized_implicit,
)
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
        self._scratch_estimate = np.zeros(self._scratch_shape, dtype=float)
        self._scratch_updated = np.zeros(self._scratch_shape, dtype=float)
        self._scratch_source = np.zeros(self._scratch_shape, dtype=float)
        self._patch_theta_cells = max(1, int(round(self.parameters.theta_cells * self.parameters.source_patch_theta_fraction)))
        self._patch_radial_cells = max(1, int(round(self.parameters.radial_cells * self.parameters.source_patch_radial_fraction)))
        self._patch_radial_start = max(self.parameters.radial_cells - self._patch_radial_cells, 0)
        self._patch_radial_indices = np.arange(self._patch_radial_start, self.parameters.radial_cells, dtype=int)
        self._theta_offsets = np.arange(
            -(self._patch_theta_cells // 2),
            -(self._patch_theta_cells // 2) + self._patch_theta_cells,
            dtype=int,
        )
        self._width_positions = np.linspace(-1.0, 1.0, self.parameters.width_zones, dtype=float) if self.parameters.width_zones > 1 else np.zeros(1, dtype=float)
        self._source_layer_masks = self._build_source_layer_masks()

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
        source = self._scratch_source
        source.fill(0.0)

        if layer_source_weights:
            layer_code = {"tread": 0, "belt": 1, "carcass": 2, "sidewall": 2, "inner_liner": 3}
            total_weight = max(float(sum(max(value, 0.0) for value in layer_source_weights.values())), 1e-12)
            for layer_name, weight in layer_source_weights.items():
                if weight <= 0.0 or layer_name not in layer_code:
                    continue
                mask = self._source_layer_masks[layer_code[layer_name]]
                if mask is None:
                    continue
                source[mask] += max(volumetric_source_w_per_m3, 0.0) * max(params.source_volumetric_fraction, 0.0) * (weight / total_weight)
        else:
            source.fill(max(volumetric_source_w_per_m3, 0.0) * max(params.source_volumetric_fraction, 0.0))

        source_remaining = max(volumetric_source_w_per_m3, 0.0) * max(1.0 - params.source_volumetric_fraction, 0.0)
        if source_remaining <= 0.0:
            return np.array(source, copy=True)

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
        return np.array(source, copy=True)

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
        construction = params.construction

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
        depth_from_outer = params.outer_radius_m - self._radial_centers_m
        radial_layer_index = np.zeros(params.radial_cells, dtype=int)
        radial_layer_index[depth_from_outer > cumulative_depth[0]] = 1
        radial_layer_index[depth_from_outer > cumulative_depth[1]] = 2
        radial_layer_index[depth_from_outer > cumulative_depth[2]] = 3

        rho_cp = np.zeros_like(self._cell_volumes_m3)
        k_r = np.zeros_like(self._cell_volumes_m3)
        k_theta = np.zeros_like(self._cell_volumes_m3)
        k_w = np.zeros_like(self._cell_volumes_m3)

        layer_params = (
            LayerEntry("tread", stack.tread),
            LayerEntry("belt", stack.belt),
            LayerEntry("carcass", stack.carcass),
            LayerEntry("inner_liner", stack.inner_liner),
        )
        age_gain = 1.0 + 0.06 * max(age_index, 0.0)
        tread_grain_penalty = 1.0 - 0.08 * grain_index
        tread_blister_penalty = 1.0 - params.surface_state.blister_conductivity_penalty * blister_index
        other_blister_penalty = 1.0 - 0.10 * blister_index
        for layer_code, entry in enumerate(layer_params):
            radial_mask = radial_layer_index == layer_code
            if not np.any(radial_mask):
                continue
            grain_penalty = tread_grain_penalty if entry.name == "tread" else np.ones(width_zones, dtype=float)
            blister_penalty = tread_blister_penalty if entry.name == "tread" else other_blister_penalty
            k_r_scale, k_theta_scale, k_w_scale = self._construction_conductivity_scale_array(
                material=entry.material,
                wear=wear,
                temperature_k=params.construction.temp_reference_k,
                construction_enabled=construction.enabled,
            )
            rho_cp[radial_mask, :, :] = entry.material.volumetric_heat_capacity_j_per_m3k * age_gain
            k_r[radial_mask, :, :] = np.maximum(
                entry.material.k_r_w_per_mk * grain_penalty * blister_penalty * k_r_scale,
                1e-4,
            )[None, None, :]
            k_theta[radial_mask, :, :] = np.maximum(
                entry.material.k_theta_w_per_mk * grain_penalty * blister_penalty * k_theta_scale,
                1e-4,
            )[None, None, :]
            k_w[radial_mask, :, :] = np.maximum(
                entry.material.k_w_w_per_mk * blister_penalty * k_w_scale,
                1e-4,
            )[None, None, :]

        layer_index = np.broadcast_to(radial_layer_index[:, None, None], self._cell_volumes_m3.shape).copy()
        return rho_cp, k_r, k_theta, k_w, layer_index

    def layer_conductivity_scale_summary(
        self,
        *,
        wear: float,
    ) -> dict[str, float]:
        outputs: dict[str, float] = {}
        for layer_name, material in (
            ("tread", self.parameters.layer_stack.tread),
            ("belt", self.parameters.layer_stack.belt),
            ("carcass", self.parameters.layer_stack.carcass),
            ("inner_liner", self.parameters.layer_stack.inner_liner),
        ):
            scales = self._construction_conductivity_scale_array(
                material=material,
                wear=wear,
                temperature_k=self.parameters.construction.temp_reference_k,
                construction_enabled=self.parameters.construction.enabled,
            )
            outputs[layer_name] = float(np.mean(np.stack(scales, axis=0)))
        return outputs

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

        rho_cp, k_r, k_theta, k_w, layer_index = self.layer_property_maps(
            wear=wear,
            grain_index_w=grain_index_w,
            blister_index_w=blister_index_w,
            age_index=age_index,
        )

        omega = inputs.wheel_angular_speed_radps
        use_native_diffusion = native_diffusion_enabled() and native_diffusion_available()
        max_cfl = abs(omega) * dt_s / max(self._theta_delta_rad, 1e-9)
        substeps = min(
            max(1, math.ceil(dt_s / max(self.parameters.internal_solver_dt_s, 1e-6))),
            max(self.parameters.max_solver_substeps, 1),
        )
        dt_sub = dt_s / substeps
        layer_source_weights_array = self._layer_source_weights_array(layer_source_weights)
        diffusion_max_iterations = int(self.parameters.diffusion_max_iterations)
        diffusion_tolerance_k = float(self.parameters.diffusion_tolerance_k)
        source_volumetric_fraction = float(self.parameters.source_volumetric_fraction)
        theta_delta_rad = float(self._theta_delta_rad)
        minimum_temperature_k = self.parameters.minimum_temperature_k
        maximum_temperature_k = self.parameters.maximum_temperature_k

        expected_energy_j = self._total_energy_j(field, rho_cp=rho_cp)
        advection_time_s = 0.0
        diffusion_time_s = 0.0
        diffusion_iterations = 0
        for sub_idx in range(substeps):
            t_sub = time_s + sub_idx * dt_sub
            start = time.perf_counter() if self.parameters.enable_profiling else 0.0
            field = self._advect_theta_periodic(field, omega=omega, dt_s=dt_sub)
            if self.parameters.enable_profiling:
                advection_time_s += time.perf_counter() - start

            if use_native_diffusion:
                start = time.perf_counter() if self.parameters.enable_profiling else 0.0
                field, substep_iterations, source = run_native_build_source_and_diffuse_implicit(
                    field=field,
                    extra_source_w_per_m3=extra_source_w_per_m3,
                    rho_cp=rho_cp,
                    k_r=k_r,
                    k_theta=k_theta,
                    k_w=k_w,
                    dt_s=dt_sub,
                    radial_coeff_minus=self._radial_coeff_minus,
                    radial_coeff_plus=self._radial_coeff_plus,
                    theta_coeff=self._theta_coeff,
                    width_coeff_minus=self._width_coeff_minus,
                    width_coeff_plus=self._width_coeff_plus,
                    diffusion_max_iterations=diffusion_max_iterations,
                    diffusion_tolerance_k=diffusion_tolerance_k,
                    source_volumetric_fraction=source_volumetric_fraction,
                    volumetric_source_w_per_m3=float(volumetric_source_w_per_m3),
                    wheel_angular_speed_radps=float(omega),
                    time_s=float(t_sub),
                    theta_delta_rad=theta_delta_rad,
                    patch_radial_indices=self._patch_radial_indices,
                    theta_offsets=self._theta_offsets,
                    width_indices=self._width_indices,
                    layer_index=layer_index,
                    zone_weights=zone_source_weights,
                    layer_source_weights=layer_source_weights_array,
                )
            else:
                source = self.source_field_w_per_m3(
                    volumetric_source_w_per_m3=volumetric_source_w_per_m3,
                    wheel_angular_speed_radps=omega,
                    time_s=t_sub,
                    zone_weights=zone_source_weights,
                    layer_source_weights=layer_source_weights,
                )
                if extra_source_w_per_m3 is not None:
                    source = source + extra_source_w_per_m3

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
            expected_energy_j += dt_sub * self._total_source_power_w(source)
            if self.parameters.enable_profiling:
                diffusion_time_s += time.perf_counter() - start
            np.clip(field, minimum_temperature_k, maximum_temperature_k, out=field)

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

    def _build_source_layer_masks(self) -> tuple[np.ndarray | None, ...]:
        _, _, _, _, layer_index = self.layer_property_maps(wear=0.0)
        return tuple(
            mask if np.any(mask) else None
            for mask in (layer_index == layer_code for layer_code in range(4))
        )

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
        if native_diffusion_enabled() and native_diffusion_available():
            return run_native_diffuse_vectorized_implicit(
                field=field,
                source_w_per_m3=source_w_per_m3,
                rho_cp=rho_cp,
                k_r=k_r,
                k_theta=k_theta,
                k_w=k_w,
                dt_s=dt_s,
                radial_coeff_minus=self._radial_coeff_minus,
                radial_coeff_plus=self._radial_coeff_plus,
                theta_coeff=self._theta_coeff,
                width_coeff_minus=self._width_coeff_minus,
                width_coeff_plus=self._width_coeff_plus,
                diffusion_max_iterations=int(self.parameters.diffusion_max_iterations),
                diffusion_tolerance_k=float(self.parameters.diffusion_tolerance_k),
            )
        return self._diffuse_vectorized_implicit_python(
            field,
            source_w_per_m3=source_w_per_m3,
            rho_cp=rho_cp,
            k_r=k_r,
            k_theta=k_theta,
            k_w=k_w,
            dt_s=dt_s,
        )

    def _diffuse_vectorized_implicit_python(
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
        estimate = self._scratch_estimate
        np.copyto(estimate, rhs)
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
            estimate, updated = updated, estimate
            if max_delta < self.parameters.diffusion_tolerance_k:
                break
        return np.array(estimate, copy=True), iterations

    def _total_energy_j(self, field: np.ndarray, *, rho_cp: np.ndarray) -> float:
        return float(np.sum(field * rho_cp * self._cell_volumes_m3))

    def _total_source_power_w(self, source_w_per_m3: np.ndarray) -> float:
        return float(np.sum(source_w_per_m3 * self._cell_volumes_m3))

    def _layer_source_weights_array(
        self,
        layer_source_weights: dict[str, float] | None,
    ) -> np.ndarray | None:
        if not layer_source_weights:
            return None
        return np.asarray(
            [
                max(float(layer_source_weights.get("tread", 0.0)), 0.0),
                max(float(layer_source_weights.get("belt", 0.0)), 0.0),
                max(float(layer_source_weights.get("carcass", 0.0)), 0.0),
                max(float(layer_source_weights.get("sidewall", 0.0)), 0.0),
                max(float(layer_source_weights.get("inner_liner", 0.0)), 0.0),
            ],
            dtype=float,
        )

    def _construction_conductivity_scale(
        self,
        *,
        material: object,
        width_idx: int,
        width_zones: int,
        wear: float,
        temperature_k: float,
        construction_enabled: bool,
    ) -> tuple[float, float, float]:
        if not construction_enabled:
            return 1.0, 1.0, 1.0
        position = self._width_position(width_idx=width_idx, width_zones=width_zones)
        shoulder_weight = min(max(abs(position), 0.0), 1.0)
        center_weight = max(1.0 - abs(position), 0.0)
        bead_weight = max(1.0 - min(abs(position) / max(self.parameters.construction.bead_width_fraction, 1e-6), 1.0), 0.0)
        total_weight = max(shoulder_weight + center_weight + bead_weight, 1e-9)
        shoulder_weight /= total_weight
        center_weight /= total_weight
        bead_weight /= total_weight
        width_bias = (
            shoulder_weight * material.shoulder_conductivity_bias
            + center_weight * material.center_conductivity_bias
            + bead_weight * material.bead_conductivity_bias
        )
        temp_gain = 1.0 + material.temp_conductivity_sensitivity_per_k * (
            temperature_k - self.parameters.construction.temp_reference_k
        )
        wear_gain = 1.0 - material.wear_conductivity_sensitivity * np.clip(wear, 0.0, 1.0)
        reinforcement_delta = max(material.reinforcement_density_factor - 1.0, 0.0)
        angle_rad = math.radians(material.cord_angle_deg)
        radial_orient = 1.0 + 0.12 * reinforcement_delta * abs(math.cos(angle_rad))
        theta_orient = 1.0 + 0.18 * reinforcement_delta * abs(math.sin(angle_rad))
        width_orient = 1.0 + 0.08 * reinforcement_delta * abs(math.sin(2.0 * angle_rad))
        common = max(width_bias * temp_gain * wear_gain, 0.25)
        return (
            max(common * radial_orient, 0.25),
            max(common * theta_orient, 0.25),
            max(common * width_orient, 0.25),
        )

    def _construction_conductivity_scale_array(
        self,
        *,
        material: object,
        wear: float,
        temperature_k: float,
        construction_enabled: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if not construction_enabled:
            ones = np.ones(self.parameters.width_zones, dtype=float)
            return ones, ones, ones
        position = np.abs(self._width_positions)
        shoulder_weight = np.clip(position, 0.0, 1.0)
        center_weight = np.maximum(1.0 - position, 0.0)
        bead_limit = max(self.parameters.construction.bead_width_fraction, 1e-6)
        bead_weight = np.maximum(1.0 - np.minimum(position / bead_limit, 1.0), 0.0)
        total_weight = np.maximum(shoulder_weight + center_weight + bead_weight, 1e-9)
        shoulder_weight = shoulder_weight / total_weight
        center_weight = center_weight / total_weight
        bead_weight = bead_weight / total_weight
        width_bias = (
            shoulder_weight * material.shoulder_conductivity_bias
            + center_weight * material.center_conductivity_bias
            + bead_weight * material.bead_conductivity_bias
        )
        temp_gain = 1.0 + material.temp_conductivity_sensitivity_per_k * (
            temperature_k - self.parameters.construction.temp_reference_k
        )
        wear_gain = 1.0 - material.wear_conductivity_sensitivity * np.clip(wear, 0.0, 1.0)
        reinforcement_delta = max(material.reinforcement_density_factor - 1.0, 0.0)
        angle_rad = math.radians(material.cord_angle_deg)
        radial_orient = 1.0 + 0.12 * reinforcement_delta * abs(math.cos(angle_rad))
        theta_orient = 1.0 + 0.18 * reinforcement_delta * abs(math.sin(angle_rad))
        width_orient = 1.0 + 0.08 * reinforcement_delta * abs(math.sin(2.0 * angle_rad))
        common = np.maximum(width_bias * temp_gain * wear_gain, 0.25)
        return (
            np.maximum(common * radial_orient, 0.25),
            np.maximum(common * theta_orient, 0.25),
            np.maximum(common * width_orient, 0.25),
        )

    def _width_position(self, *, width_idx: int, width_zones: int) -> float:
        if width_zones <= 1:
            return 0.0
        return -1.0 + 2.0 * width_idx / max(width_zones - 1, 1)


@dataclass(frozen=True)
class LayerEntry:
    name: str
    material: object
