from __future__ import annotations

import os
import numpy as np

try:
    from ._hf_diffusion_native import diffuse_vectorized_implicit as _native_diffuse_vectorized_implicit
except ImportError:
    _native_diffuse_vectorized_implicit = None


def native_diffusion_available() -> bool:
    return _native_diffuse_vectorized_implicit is not None


def native_diffusion_enabled() -> bool:
    value = os.getenv("PITNN_USE_NATIVE_DIFFUSION", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def run_native_diffuse_vectorized_implicit(
    *,
    field: np.ndarray,
    source_w_per_m3: np.ndarray,
    rho_cp: np.ndarray,
    k_r: np.ndarray,
    k_theta: np.ndarray,
    k_w: np.ndarray,
    dt_s: float,
    radial_coeff_minus: np.ndarray,
    radial_coeff_plus: np.ndarray,
    theta_coeff: np.ndarray,
    width_coeff_minus: np.ndarray,
    width_coeff_plus: np.ndarray,
    diffusion_max_iterations: int,
    diffusion_tolerance_k: float,
) -> tuple[np.ndarray, int]:
    if _native_diffuse_vectorized_implicit is None:
        msg = "Native diffusion extension is not available"
        raise RuntimeError(msg)
    result_field, iterations = _native_diffuse_vectorized_implicit(
        field,
        source_w_per_m3,
        rho_cp,
        k_r,
        k_theta,
        k_w,
        dt_s,
        radial_coeff_minus,
        radial_coeff_plus,
        theta_coeff,
        width_coeff_minus,
        width_coeff_plus,
        diffusion_max_iterations,
        diffusion_tolerance_k,
    )
    return np.asarray(result_field, dtype=float), int(iterations)
