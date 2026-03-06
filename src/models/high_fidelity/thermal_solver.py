from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .types import HighFidelityTireInputs, HighFidelityTireModelParameters


@dataclass(frozen=True)
class ThermalSolverStepResult:
    temperature_field_rt_k: np.ndarray
    energy_residual_pct: float
    substeps: int
    max_cfl: float


class ThermalFieldSolver2D:
    """
    2D (radial x circumferential) thermal solver.

    Operator split per substep:
    1) Explicit upwind advection in theta.
    2) Implicit backward-Euler diffusion (iterative Gauss-Seidel solve).
    """

    def __init__(self, parameters: HighFidelityTireModelParameters) -> None:
        self.parameters = parameters
        self._radial_edges_m = self._build_radial_edges_m()
        self._radial_centers_m = 0.5 * (self._radial_edges_m[:-1] + self._radial_edges_m[1:])
        self._dr_m = np.diff(self._radial_edges_m)
        self._theta_delta_rad = (2.0 * math.pi) / max(self.parameters.theta_cells, 1)
        self._cell_volumes_m3 = self._build_cell_volumes_m3()
        self._radial_coeff_minus, self._radial_coeff_plus = self._build_radial_diffusion_coefficients()
        self._theta_coeff = self._build_theta_diffusion_coefficients()

    @property
    def radial_centers_m(self) -> np.ndarray:
        return self._radial_centers_m.copy()

    @property
    def cell_volumes_m3(self) -> np.ndarray:
        return self._cell_volumes_m3.copy()

    def initial_temperature_field(self, ambient_temp_k: float) -> np.ndarray:
        return np.full(
            (self.parameters.radial_cells, self.parameters.theta_cells),
            float(ambient_temp_k),
            dtype=float,
        )

    def source_field_w_per_m3(
        self,
        *,
        volumetric_source_w_per_m3: float,
        wheel_angular_speed_radps: float,
        time_s: float,
    ) -> np.ndarray:
        params = self.parameters
        radial_cells = params.radial_cells
        theta_cells = params.theta_cells
        source = np.full(
            (radial_cells, theta_cells),
            max(volumetric_source_w_per_m3, 0.0) * max(params.source_volumetric_fraction, 0.0),
            dtype=float,
        )

        source_remaining = max(volumetric_source_w_per_m3, 0.0) * max(
            1.0 - params.source_volumetric_fraction,
            0.0,
        )
        if source_remaining <= 0.0:
            return source

        radial_indices, theta_indices = self.contact_patch_indices(
            wheel_angular_speed_radps=wheel_angular_speed_radps,
            time_s=time_s,
        )
        patch_cells = radial_indices.shape[0] * theta_indices.shape[0]
        patch_extra_density = source_remaining * (radial_cells * theta_cells) / max(patch_cells, 1)

        source[np.ix_(radial_indices, theta_indices)] += patch_extra_density
        return source

    def contact_patch_indices(
        self,
        *,
        wheel_angular_speed_radps: float,
        time_s: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        params = self.parameters
        radial_cells = params.radial_cells
        theta_cells = params.theta_cells

        patch_theta_cells = max(1, int(round(theta_cells * params.source_patch_theta_fraction)))
        patch_radial_cells = max(1, int(round(radial_cells * params.source_patch_radial_fraction)))
        patch_radial_start = max(radial_cells - patch_radial_cells, 0)
        radial_indices = np.arange(patch_radial_start, radial_cells, dtype=int)

        phase_theta = (wheel_angular_speed_radps * time_s) % (2.0 * math.pi)
        center_theta_idx = int(round(phase_theta / max(self._theta_delta_rad, 1e-9))) % theta_cells
        theta_offsets = np.arange(
            -(patch_theta_cells // 2),
            -(patch_theta_cells // 2) + patch_theta_cells,
            dtype=int,
        )
        theta_indices = (center_theta_idx + theta_offsets) % theta_cells
        return radial_indices, theta_indices

    def patch_volume_m3(
        self,
        *,
        wheel_angular_speed_radps: float,
        time_s: float,
    ) -> float:
        radial_indices, theta_indices = self.contact_patch_indices(
            wheel_angular_speed_radps=wheel_angular_speed_radps,
            time_s=time_s,
        )
        return float(np.sum(self._cell_volumes_m3[np.ix_(radial_indices, theta_indices)]))

    def step(
        self,
        *,
        temperature_field_rt_k: np.ndarray,
        inputs: HighFidelityTireInputs,
        time_s: float,
        dt_s: float,
        volumetric_source_w_per_m3: float,
        extra_source_w_per_m3: np.ndarray | None = None,
    ) -> ThermalSolverStepResult:
        if dt_s <= 0.0:
            msg = f"dt_s must be positive, got {dt_s}"
            raise ValueError(msg)

        field = np.array(temperature_field_rt_k, dtype=float, copy=True)
        self._validate_field_shape(field)
        if extra_source_w_per_m3 is not None:
            self._validate_field_shape(extra_source_w_per_m3)
        omega = inputs.wheel_angular_speed_radps
        max_cfl = abs(omega) * dt_s / max(self._theta_delta_rad, 1e-9)

        substeps_dt = max(1, math.ceil(dt_s / max(self.parameters.internal_solver_dt_s, 1e-6)))
        substeps_cfl = max(
            1,
            math.ceil(max_cfl / max(self.parameters.advection_cfl_limit, 1e-6)),
        )
        substeps = min(
            max(substeps_dt, substeps_cfl),
            max(self.parameters.max_solver_substeps, 1),
        )
        dt_sub = dt_s / substeps

        expected_energy_j = self._total_energy_j(field)
        for sub_idx in range(substeps):
            t_sub = time_s + sub_idx * dt_sub
            source = self.source_field_w_per_m3(
                volumetric_source_w_per_m3=volumetric_source_w_per_m3,
                wheel_angular_speed_radps=omega,
                time_s=t_sub,
            )
            if extra_source_w_per_m3 is not None:
                source = source + extra_source_w_per_m3
            expected_energy_j += dt_sub * self._total_source_power_w(source)

            field = self._advect_theta_upwind(field, omega=omega, dt_s=dt_sub)
            field = self._diffuse_implicit(field, source_w_per_m3=source, dt_s=dt_sub)
            field = np.clip(
                field,
                self.parameters.minimum_temperature_k,
                self.parameters.maximum_temperature_k,
            )

        actual_energy_j = self._total_energy_j(field)
        energy_residual_pct = (
            abs(actual_energy_j - expected_energy_j) / max(abs(expected_energy_j), 1.0) * 100.0
        )
        return ThermalSolverStepResult(
            temperature_field_rt_k=field,
            energy_residual_pct=float(energy_residual_pct),
            substeps=substeps,
            max_cfl=float(max_cfl / max(substeps, 1)),
        )

    def _validate_field_shape(self, field: np.ndarray) -> None:
        expected = (self.parameters.radial_cells, self.parameters.theta_cells)
        if field.shape != expected:
            msg = f"Expected field shape {expected}, got {field.shape}"
            raise ValueError(msg)

    def _build_radial_edges_m(self) -> np.ndarray:
        params = self.parameters
        bias = max(params.radial_spacing_bias, 1e-6)
        x = np.linspace(0.0, 1.0, params.radial_cells + 1, dtype=float)
        # Bias > 1 clusters cells near the outer radius (tread side).
        stretched = 1.0 - np.power(1.0 - x, bias)
        return params.inner_radius_m + stretched * (params.outer_radius_m - params.inner_radius_m)

    def _build_cell_volumes_m3(self) -> np.ndarray:
        width_m = max(self.parameters.tire_section_width_m, 1e-3)
        radial_cells = self.parameters.radial_cells
        theta_cells = self.parameters.theta_cells
        volumes = np.zeros((radial_cells, theta_cells), dtype=float)
        for i in range(radial_cells):
            arc_length_m = self._radial_centers_m[i] * self._theta_delta_rad
            volumes[i, :] = self._dr_m[i] * arc_length_m * width_m
        return volumes

    def _build_radial_diffusion_coefficients(self) -> tuple[np.ndarray, np.ndarray]:
        radial_cells = self.parameters.radial_cells
        coeff_minus = np.zeros(radial_cells, dtype=float)
        coeff_plus = np.zeros(radial_cells, dtype=float)

        for i in range(radial_cells):
            if i == 0:
                dr_plus = self._radial_centers_m[i + 1] - self._radial_centers_m[i]
                coeff_minus[i] = 0.0
                coeff_plus[i] = 1.0 / max(dr_plus * dr_plus, 1e-12)
            elif i == radial_cells - 1:
                dr_minus = self._radial_centers_m[i] - self._radial_centers_m[i - 1]
                coeff_minus[i] = 1.0 / max(dr_minus * dr_minus, 1e-12)
                coeff_plus[i] = 0.0
            else:
                dr_minus = self._radial_centers_m[i] - self._radial_centers_m[i - 1]
                dr_plus = self._radial_centers_m[i + 1] - self._radial_centers_m[i]
                coeff_minus[i] = 2.0 / max(dr_minus * (dr_minus + dr_plus), 1e-12)
                coeff_plus[i] = 2.0 / max(dr_plus * (dr_minus + dr_plus), 1e-12)
        return coeff_minus, coeff_plus

    def _build_theta_diffusion_coefficients(self) -> np.ndarray:
        return 1.0 / np.maximum(self._radial_centers_m * self._theta_delta_rad, 1e-9) ** 2

    def _advect_theta_upwind(self, field: np.ndarray, *, omega: float, dt_s: float) -> np.ndarray:
        if abs(omega) <= 1e-12:
            return field
        cfl = abs(omega) * dt_s / max(self._theta_delta_rad, 1e-12)
        cfl = min(cfl, self.parameters.advection_cfl_limit)
        if omega >= 0.0:
            return field - cfl * (field - np.roll(field, 1, axis=1))
        return field - cfl * (np.roll(field, -1, axis=1) - field)

    def _diffuse_implicit(
        self,
        field: np.ndarray,
        *,
        source_w_per_m3: np.ndarray,
        dt_s: float,
    ) -> np.ndarray:
        params = self.parameters
        alpha = max(params.thermal_diffusivity_m2_per_s, 0.0)
        rho_cp = max(params.volumetric_heat_capacity_j_per_m3k, 1e-12)
        rhs = field + dt_s * source_w_per_m3 / rho_cp
        estimate = rhs.copy()

        for _ in range(max(params.diffusion_max_iterations, 1)):
            max_delta = 0.0
            for i in range(params.radial_cells):
                coeff_minus = self._radial_coeff_minus[i]
                coeff_plus = self._radial_coeff_plus[i]
                coeff_theta = self._theta_coeff[i]
                diagonal = 1.0 + dt_s * alpha * (coeff_minus + coeff_plus + 2.0 * coeff_theta)

                i_minus = max(i - 1, 0)
                i_plus = min(i + 1, params.radial_cells - 1)
                for j in range(params.theta_cells):
                    j_minus = (j - 1) % params.theta_cells
                    j_plus = (j + 1) % params.theta_cells
                    neighbor_sum = (
                        coeff_minus * estimate[i_minus, j]
                        + coeff_plus * estimate[i_plus, j]
                        + coeff_theta * estimate[i, j_minus]
                        + coeff_theta * estimate[i, j_plus]
                    )
                    updated = (rhs[i, j] + dt_s * alpha * neighbor_sum) / max(diagonal, 1e-12)
                    delta = abs(updated - estimate[i, j])
                    if delta > max_delta:
                        max_delta = delta
                    estimate[i, j] = updated
            if max_delta < params.diffusion_tolerance_k:
                break
        return estimate

    def _total_energy_j(self, field: np.ndarray) -> float:
        rho_cp = max(self.parameters.volumetric_heat_capacity_j_per_m3k, 1e-12)
        return float(np.sum(field * self._cell_volumes_m3) * rho_cp)

    def _total_source_power_w(self, source_w_per_m3: np.ndarray) -> float:
        return float(np.sum(source_w_per_m3 * self._cell_volumes_m3))
