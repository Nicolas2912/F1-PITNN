from __future__ import annotations

import math

import numpy as np

from models.high_fidelity import HighFidelityTireModelParameters, HighFidelityUQ, ParameterPrior


def test_p7_lhs_sampling_is_seeded_and_reproducible() -> None:
    uq = HighFidelityUQ()
    priors = (
        ParameterPrior(name="a", lower=0.0, upper=1.0),
        ParameterPrior(name="b", lower=5.0, upper=7.0),
    )

    first = uq.latin_hypercube(priors=priors, sample_count=32, seed=1234)
    second = uq.latin_hypercube(priors=priors, sample_count=32, seed=1234)
    third = uq.latin_hypercube(priors=priors, sample_count=32, seed=5678)

    assert np.array_equal(first, second)
    assert not np.array_equal(first, third)


def test_p7_quantile_envelope_and_nested_parameter_application_are_stable() -> None:
    uq = HighFidelityUQ()
    base = HighFidelityTireModelParameters()
    sample = {
        "thermal_diffusivity_m2_per_s": 2.0e-7,
        "boundary.eta_tire": 0.61,
        "boundary.h_cp_w_per_m2k": 1900.0,
    }

    updated = uq.apply_sample(base=base, sample=sample)
    assert math.isclose(updated.thermal_diffusivity_m2_per_s, 2.0e-7, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(updated.boundary.eta_tire, 0.61, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(updated.boundary.h_cp_w_per_m2k, 1900.0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(base.boundary.eta_tire, HighFidelityTireModelParameters().boundary.eta_tire)

    outputs = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
        ],
        dtype=float,
    )
    envelope = uq.quantile_envelope(outputs)
    assert np.allclose(envelope.q50, np.array([2.5, 3.5, 4.5]))
    assert np.all(envelope.q05 <= envelope.q50)
    assert np.all(envelope.q50 <= envelope.q95)


def test_p7_sobol_ranking_identifies_dominant_parameter() -> None:
    uq = HighFidelityUQ()
    priors = (
        ParameterPrior(name="x1", lower=0.0, upper=1.0),
        ParameterPrior(name="x2", lower=0.0, upper=1.0),
        ParameterPrior(name="x3", lower=0.0, upper=1.0),
    )

    result = uq.sobol_indices(
        priors=priors,
        sample_count=512,
        seed=2026,
        model_fn=lambda sample: 4.0 * sample["x1"] + 0.5 * sample["x2"] + 0.1 * sample["x3"],
    )

    assert result.variance > 0.0
    assert result.indices[0].name == "x1"
    assert result.indices[0].total_order > result.indices[1].total_order
    assert result.indices[1].total_order > result.indices[2].total_order
    for index in result.indices:
        assert 0.0 <= index.first_order <= 1.0
        assert 0.0 <= index.total_order <= 1.0


def test_p7_lhs_screen_returns_deterministic_envelopes() -> None:
    uq = HighFidelityUQ()
    priors = uq.default_tire_priors()

    def model_fn(sample: dict[str, float]) -> np.ndarray:
        x = sample["layer_stack.tread.k_r_w_per_mk"] * 10.0
        y = sample["boundary.eta_tire"]
        z = sample["force_mu_peak"]
        w = sample["core_sensor.probe_depth_fraction_from_outer"]
        return np.array([x + y, z - y, x * 0.1 + z + w], dtype=float)

    first = uq.lhs_screen(priors=priors, model_fn=model_fn, sample_count=64, seed=77)
    second = uq.lhs_screen(priors=priors, model_fn=model_fn, sample_count=64, seed=77)

    assert np.array_equal(first.unit_samples, second.unit_samples)
    assert np.allclose(first.outputs, second.outputs)
    assert np.allclose(first.envelope.q05, second.envelope.q05)
    assert np.allclose(first.envelope.q50, second.envelope.q50)
    assert np.allclose(first.envelope.q95, second.envelope.q95)
