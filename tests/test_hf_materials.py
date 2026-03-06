from __future__ import annotations

import math

import pytest

from models.high_fidelity import (
    HighFidelityTireInputs,
    HighFidelityTireModelParameters,
    ViscoelasticMaterialModel,
)
from models.physics import celsius_to_kelvin


def _inputs(**overrides: float) -> HighFidelityTireInputs:
    payload: dict[str, float] = {
        "speed_mps": 62.0,
        "wheel_angular_speed_radps": 195.0,
        "normal_load_n": 3_800.0,
        "slip_ratio_cmd": 0.08,
        "slip_angle_cmd_rad": 0.05,
        "ambient_temp_k": celsius_to_kelvin(30.0),
        "track_temp_k": celsius_to_kelvin(44.0),
    }
    payload.update(overrides)
    return HighFidelityTireInputs(**payload)


def test_p2_wlf_shift_factor_is_monotonic_around_reference_temperature() -> None:
    params = HighFidelityTireModelParameters(
        wlf_reference_temp_k=celsius_to_kelvin(80.0),
        wlf_c1=8.86,
        wlf_c2_k=101.6,
    )
    model = ViscoelasticMaterialModel(params)

    t_low = celsius_to_kelvin(60.0)
    t_ref = celsius_to_kelvin(80.0)
    t_high = celsius_to_kelvin(100.0)

    a_low = model.wlf_shift_factor(temperature_k=t_low)
    a_ref = model.wlf_shift_factor(temperature_k=t_ref)
    a_high = model.wlf_shift_factor(temperature_k=t_high)

    assert a_low > 1.0
    assert math.isclose(a_ref, 1.0, rel_tol=1e-12, abs_tol=1e-12)
    assert a_high < 1.0
    assert a_low > a_ref > a_high


def test_p2_loss_modulus_and_hysteresis_power_density_are_non_negative() -> None:
    model = ViscoelasticMaterialModel(HighFidelityTireModelParameters())
    control = _inputs()

    temperatures_k = (
        celsius_to_kelvin(40.0),
        celsius_to_kelvin(80.0),
        celsius_to_kelvin(110.0),
    )
    frequencies_hz = (0.3, 2.0, 8.0, 20.0)

    for temperature_k in temperatures_k:
        for frequency_hz in frequencies_hz:
            e_double_prime = model.loss_modulus_pa(
                temperature_k=temperature_k,
                frequency_hz=frequency_hz,
            )
            assert e_double_prime >= 0.0

        q_hyst = model.hysteresis_power_density_w_per_m3(
            temperature_k=temperature_k,
            inputs=control,
        )
        assert q_hyst >= 0.0


def test_p2_deformation_derived_strain_grows_with_load_and_slip() -> None:
    model = ViscoelasticMaterialModel(HighFidelityTireModelParameters(use_structural_hysteresis_model=True))
    mild = model.effective_strain_amplitude(inputs=_inputs(normal_load_n=3200.0, slip_ratio_cmd=0.02, slip_angle_cmd_rad=0.01))
    aggressive = model.effective_strain_amplitude(inputs=_inputs(normal_load_n=4200.0, slip_ratio_cmd=0.12, slip_angle_cmd_rad=0.08))

    assert aggressive.contact_patch_length_m >= mild.contact_patch_length_m
    assert aggressive.tread_strain > mild.tread_strain
    assert aggressive.belt_strain > mild.belt_strain
    assert aggressive.carcass_strain > mild.carcass_strain
    assert aggressive.sidewall_strain > mild.sidewall_strain
    assert aggressive.equivalent_strain_amplitude > mild.equivalent_strain_amplitude


def test_p2_layer_hysteresis_summary_conserves_positive_layer_contributions() -> None:
    model = ViscoelasticMaterialModel(HighFidelityTireModelParameters(use_structural_hysteresis_model=True))
    summary = model.layer_hysteresis_summary(
        temperature_k=celsius_to_kelvin(92.0),
        inputs=_inputs(normal_load_n=4000.0, slip_ratio_cmd=0.09, slip_angle_cmd_rad=0.06),
    )

    assert summary.total_power_density_w_per_m3 > 0.0
    assert set(summary.power_density_by_layer_w_per_m3) == {"tread", "belt", "carcass", "sidewall"}
    assert sum(summary.power_density_by_layer_w_per_m3.values()) == pytest.approx(summary.total_power_density_w_per_m3)
    assert summary.power_density_by_layer_w_per_m3["sidewall"] > 0.0


def test_p2_excitation_frequency_proxy_increases_with_slip_energy() -> None:
    model = ViscoelasticMaterialModel(HighFidelityTireModelParameters())
    mild = _inputs(slip_ratio_cmd=0.01, slip_angle_cmd_rad=0.01)
    aggressive = _inputs(slip_ratio_cmd=0.14, slip_angle_cmd_rad=0.09)

    f_mild = model.excitation_frequency_hz(inputs=mild)
    f_aggressive = model.excitation_frequency_hz(inputs=aggressive)

    assert f_mild > 0.0
    assert f_aggressive > f_mild
