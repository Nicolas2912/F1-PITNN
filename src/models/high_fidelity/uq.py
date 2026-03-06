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

        unit_a = self.random_unit_samples(priors=priors, sample_count=sample_count, seed=seed)
        unit_b = self.random_unit_samples(priors=priors, sample_count=sample_count, seed=seed + 1)
        mapped_a = self.map_priors(unit_samples=unit_a, priors=priors)
        mapped_b = self.map_priors(unit_samples=unit_b, priors=priors)

        y_a = np.array(
            [model_fn({prior.name: float(mapped_a[prior.name][idx]) for prior in priors}) for idx in range(sample_count)],
            dtype=float,
        )
        y_b = np.array(
            [model_fn({prior.name: float(mapped_b[prior.name][idx]) for prior in priors}) for idx in range(sample_count)],
            dtype=float,
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
            unit_ab = unit_a.copy()
            unit_ab[:, dim] = unit_b[:, dim]
            mapped_ab = self.map_priors(unit_samples=unit_ab, priors=priors)
            y_ab = np.array(
                [
                    model_fn({sample_prior.name: float(mapped_ab[sample_prior.name][idx]) for sample_prior in priors})
                    for idx in range(sample_count)
                ],
                dtype=float,
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
        return (
            ParameterPrior(
                name="thermal_diffusivity_m2_per_s",
                lower=0.75 * params.thermal_diffusivity_m2_per_s,
                upper=1.35 * params.thermal_diffusivity_m2_per_s,
            ),
            ParameterPrior(
                name="volumetric_heat_capacity_j_per_m3k",
                lower=0.85 * params.volumetric_heat_capacity_j_per_m3k,
                upper=1.20 * params.volumetric_heat_capacity_j_per_m3k,
            ),
            ParameterPrior(
                name="strain_amplitude_reference",
                lower=0.70 * params.strain_amplitude_reference,
                upper=1.30 * params.strain_amplitude_reference,
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
