from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any, Callable, Protocol, TypeVar, cast

import numpy as np

from .types import HighFidelityTireModelParameters


class DataclassInstance(Protocol):
    __dataclass_fields__: dict[str, Any]


DataclassT = TypeVar("DataclassT", bound=DataclassInstance)


@dataclass(frozen=True)
class ParameterPrior:
    name: str
    lower: float
    upper: float
    distribution: str = "uniform"


@dataclass(frozen=True)
class SobolSensitivityIndex:
    name: str
    first_order: float
    total_order: float


@dataclass(frozen=True)
class QuantileEnvelope:
    q05: np.ndarray
    q50: np.ndarray
    q95: np.ndarray


@dataclass(frozen=True)
class LHSResult:
    priors: tuple[ParameterPrior, ...]
    unit_samples: np.ndarray
    parameter_samples: dict[str, np.ndarray]
    outputs: np.ndarray
    envelope: QuantileEnvelope


@dataclass(frozen=True)
class SobolResult:
    priors: tuple[ParameterPrior, ...]
    indices: tuple[SobolSensitivityIndex, ...]
    variance: float


class HighFidelityUQ:
    """Seeded LHS and Sobol-style UQ helpers for the high-fidelity stack."""

    def random_unit_samples(
        self,
        *,
        priors: list[ParameterPrior] | tuple[ParameterPrior, ...],
        sample_count: int,
        seed: int,
    ) -> np.ndarray:
        if sample_count <= 0:
            msg = f"sample_count must be positive, got {sample_count}"
            raise ValueError(msg)
        if len(priors) == 0:
            msg = "priors must not be empty"
            raise ValueError(msg)
        rng = np.random.default_rng(seed)
        return rng.random((sample_count, len(priors)), dtype=float)

    def latin_hypercube(
        self,
        *,
        priors: list[ParameterPrior] | tuple[ParameterPrior, ...],
        sample_count: int = 400,
        seed: int = 12345,
    ) -> np.ndarray:
        if sample_count <= 0:
            msg = f"sample_count must be positive, got {sample_count}"
            raise ValueError(msg)
        if len(priors) == 0:
            msg = "priors must not be empty"
            raise ValueError(msg)

        rng = np.random.default_rng(seed)
        dim = len(priors)
        base = np.zeros((sample_count, dim), dtype=float)
        for col in range(dim):
            perm = rng.permutation(sample_count)
            jitter = rng.random(sample_count)
            base[:, col] = (perm + jitter) / sample_count
        return base

    def map_priors(
        self,
        *,
        unit_samples: np.ndarray,
        priors: list[ParameterPrior] | tuple[ParameterPrior, ...],
    ) -> dict[str, np.ndarray]:
        mapped: dict[str, np.ndarray] = {}
        for col, prior in enumerate(priors):
            mapped[prior.name] = self._transform_unit_samples(unit_samples[:, col], prior)
        return mapped

    def _map_prior_matrix(
        self,
        *,
        unit_samples: np.ndarray,
        priors: list[ParameterPrior] | tuple[ParameterPrior, ...],
    ) -> np.ndarray:
        return np.column_stack(
            [self._transform_unit_samples(unit_samples[:, col], prior) for col, prior in enumerate(priors)]
        )

    def _evaluate_scalar_model(
        self,
        *,
        model_fn: Callable[[dict[str, float]], float],
        prior_names: tuple[str, ...],
        sample_matrix: np.ndarray,
    ) -> np.ndarray:
        return np.fromiter(
            (
                model_fn({name: float(value) for name, value in zip(prior_names, row, strict=True)})
                for row in sample_matrix
            ),
            dtype=float,
            count=sample_matrix.shape[0],
        )

    def lhs_screen(
        self,
        *,
        priors: list[ParameterPrior] | tuple[ParameterPrior, ...],
        model_fn: Callable[[dict[str, float]], np.ndarray],
        sample_count: int = 400,
        seed: int = 12345,
    ) -> LHSResult:
        unit_samples = self.latin_hypercube(
            priors=priors,
            sample_count=sample_count,
            seed=seed,
        )
        mapped = self.map_priors(unit_samples=unit_samples, priors=priors)
        outputs = np.stack(
            [
                np.asarray(
                    model_fn({prior.name: float(mapped[prior.name][idx]) for prior in priors}),
                    dtype=float,
                )
                for idx in range(sample_count)
            ],
            axis=0,
        )
        return LHSResult(
            priors=tuple(priors),
            unit_samples=unit_samples,
            parameter_samples=mapped,
            outputs=outputs,
            envelope=self.quantile_envelope(outputs),
        )

    def sobol_indices(
        self,
        *,
        priors: list[ParameterPrior] | tuple[ParameterPrior, ...],
        model_fn: Callable[[dict[str, float]], float],
        sample_count: int = 2048,
        seed: int = 12345,
    ) -> SobolResult:
        if sample_count <= 1:
            msg = f"sample_count must be greater than 1, got {sample_count}"
            raise ValueError(msg)

        prior_names = tuple(prior.name for prior in priors)
        unit_a = self.random_unit_samples(priors=priors, sample_count=sample_count, seed=seed)
        unit_b = self.random_unit_samples(priors=priors, sample_count=sample_count, seed=seed + 1)
        matrix_a = self._map_prior_matrix(unit_samples=unit_a, priors=priors)
        matrix_b = self._map_prior_matrix(unit_samples=unit_b, priors=priors)

        y_a = self._evaluate_scalar_model(
            model_fn=model_fn,
            prior_names=prior_names,
            sample_matrix=matrix_a,
        )
        y_b = self._evaluate_scalar_model(
            model_fn=model_fn,
            prior_names=prior_names,
            sample_matrix=matrix_b,
        )
        variance = float(np.var(np.concatenate((y_a, y_b)), ddof=1))
        if variance <= 1e-12:
            zero = tuple(
                SobolSensitivityIndex(name=prior.name, first_order=0.0, total_order=0.0)
                for prior in priors
            )
            return SobolResult(priors=tuple(priors), indices=zero, variance=0.0)

        indices: list[SobolSensitivityIndex] = []
        for dim, prior in enumerate(priors):
            matrix_ab = matrix_a.copy()
            matrix_ab[:, dim] = matrix_b[:, dim]
            y_ab = self._evaluate_scalar_model(
                model_fn=model_fn,
                prior_names=prior_names,
                sample_matrix=matrix_ab,
            )

            first_order = float(1.0 - np.mean((y_b - y_ab) ** 2) / (2.0 * variance))
            total_order = float(np.mean((y_a - y_ab) ** 2) / (2.0 * variance))
            indices.append(
                SobolSensitivityIndex(
                    name=prior.name,
                    first_order=first_order,
                    total_order=total_order,
                )
            )

        indices.sort(key=lambda item: item.total_order, reverse=True)
        return SobolResult(priors=tuple(priors), indices=tuple(indices), variance=variance)

    def quantile_envelope(self, outputs: np.ndarray) -> QuantileEnvelope:
        outputs = np.asarray(outputs, dtype=float)
        return QuantileEnvelope(
            q05=np.quantile(outputs, 0.05, axis=0),
            q50=np.quantile(outputs, 0.50, axis=0),
            q95=np.quantile(outputs, 0.95, axis=0),
        )

    def apply_sample(
        self,
        *,
        base: Any,
        sample: dict[str, float],
    ) -> Any:
        updated = base
        for path, value in sample.items():
            updated = self._replace_dataclass_path(updated, path.split("."), value)
        return updated

    def default_tire_priors(
        self,
        *,
        parameters: HighFidelityTireModelParameters | None = None,
    ) -> tuple[ParameterPrior, ...]:
        params = parameters if parameters is not None else HighFidelityTireModelParameters()
        boundary = params.boundary
        pressure_patch = params.pressure_patch
        surface_state = params.surface_state
        core_sensor = params.core_sensor
        return (
            ParameterPrior(
                name="layer_stack.tread.k_r_w_per_mk",
                lower=0.70 * params.layer_stack.tread.k_r_w_per_mk,
                upper=1.30 * params.layer_stack.tread.k_r_w_per_mk,
            ),
            ParameterPrior(
                name="layer_stack.belt.k_theta_w_per_mk",
                lower=0.70 * params.layer_stack.belt.k_theta_w_per_mk,
                upper=1.30 * params.layer_stack.belt.k_theta_w_per_mk,
            ),
            ParameterPrior(
                name="layer_stack.carcass.volumetric_heat_capacity_j_per_m3k",
                lower=0.80 * params.layer_stack.carcass.volumetric_heat_capacity_j_per_m3k,
                upper=1.20 * params.layer_stack.carcass.volumetric_heat_capacity_j_per_m3k,
            ),
            ParameterPrior(
                name="strain_amplitude_reference",
                lower=0.75 * params.strain_amplitude_reference,
                upper=1.25 * params.strain_amplitude_reference,
            ),
            ParameterPrior(
                name="force_mu_peak",
                lower=0.80 * params.force_mu_peak,
                upper=1.20 * params.force_mu_peak,
            ),
            ParameterPrior(
                name="boundary.eta_tire",
                lower=max(0.40, boundary.eta_tire - 0.20),
                upper=min(0.95, boundary.eta_tire + 0.15),
            ),
            ParameterPrior(
                name="boundary.h_cp_w_per_m2k",
                lower=0.55 * boundary.h_cp_w_per_m2k,
                upper=1.65 * boundary.h_cp_w_per_m2k,
            ),
            ParameterPrior(
                name="boundary.h_c_bead_w_per_m2k",
                lower=0.50 * boundary.h_c_bead_w_per_m2k,
                upper=1.60 * boundary.h_c_bead_w_per_m2k,
            ),
            ParameterPrior(
                name="boundary.road_thermal_conductivity_w_per_mk",
                lower=0.70 * boundary.road_thermal_conductivity_w_per_mk,
                upper=1.35 * boundary.road_thermal_conductivity_w_per_mk,
            ),
            ParameterPrior(
                name="pressure_patch.base_volume_m3",
                lower=0.90 * pressure_patch.base_volume_m3,
                upper=1.08 * pressure_patch.base_volume_m3,
            ),
            ParameterPrior(
                name="pressure_patch.carcass_support_pressure_pa",
                lower=0.75 * pressure_patch.carcass_support_pressure_pa,
                upper=1.25 * pressure_patch.carcass_support_pressure_pa,
            ),
            ParameterPrior(
                name="pressure_patch.effective_radius_pressure_gain_m_per_bar",
                lower=1.25 * pressure_patch.effective_radius_pressure_gain_m_per_bar,
                upper=0.75 * pressure_patch.effective_radius_pressure_gain_m_per_bar,
            ),
            ParameterPrior(
                name="core_sensor.probe_depth_fraction_from_outer",
                lower=max(0.08, core_sensor.probe_depth_fraction_from_outer - 0.08),
                upper=min(0.60, core_sensor.probe_depth_fraction_from_outer + 0.12),
            ),
            ParameterPrior(
                name="core_sensor.response_time_s",
                lower=0.60 * core_sensor.response_time_s,
                upper=1.80 * core_sensor.response_time_s,
            ),
            ParameterPrior(
                name="surface_state.graining_gain",
                lower=0.70 * surface_state.graining_gain,
                upper=1.50 * surface_state.graining_gain,
            ),
            ParameterPrior(
                name="surface_state.blister_gain",
                lower=0.70 * surface_state.blister_gain,
                upper=1.50 * surface_state.blister_gain,
            ),
        )

    def _transform_unit_samples(self, unit_values: np.ndarray, prior: ParameterPrior) -> np.ndarray:
        if prior.upper < prior.lower:
            msg = f"Prior upper bound must be >= lower bound for {prior.name}"
            raise ValueError(msg)
        if prior.distribution == "uniform":
            return prior.lower + unit_values * (prior.upper - prior.lower)
        if prior.distribution == "loguniform":
            if prior.lower <= 0.0 or prior.upper <= 0.0:
                msg = f"loguniform prior {prior.name} requires positive bounds"
                raise ValueError(msg)
            log_lower = np.log(prior.lower)
            log_upper = np.log(prior.upper)
            return np.exp(log_lower + unit_values * (log_upper - log_lower))
        msg = f"Unsupported prior distribution: {prior.distribution}"
        raise ValueError(msg)

    def _replace_dataclass_path(
        self,
        obj: DataclassT,
        path_parts: list[str],
        value: float,
    ) -> DataclassT:
        if not is_dataclass(obj):
            msg = f"Expected dataclass while applying {'.'.join(path_parts)}"
            raise TypeError(msg)
        dataclass_obj: Any = obj
        field_name = path_parts[0]
        dataclass_field_names = {field.name for field in fields(dataclass_obj)}
        if field_name not in dataclass_field_names:
            msg = f"Unknown dataclass field {field_name} on {type(obj).__name__}"
            raise ValueError(msg)

        if len(path_parts) == 1:
            return cast(DataclassT, replace(dataclass_obj, **{field_name: value}))

        nested = getattr(obj, field_name)
        return cast(
            DataclassT,
            replace(
                dataclass_obj,
                **{field_name: self._replace_dataclass_path(cast(DataclassT, nested), path_parts[1:], value)},
            ),
        )
