import numpy as np
from pytest import approx

from nmrsim.math import (add_peaks, get_intensity, lorentz, reduce_peaks,
                         _normalize, normalize_peaklist, ppm_to_hz_from_nuclei_info)
from tests.dnmr_standards import TWOSPIN_SLOW


def test_add_peaks():
    peaklist = [(100, 1.1), (110, 1.2), (150, 1.6)]
    expected_result = (120.0, 3.9)
    result = add_peaks(peaklist)
    assert result == expected_result


def test_reduce_peaks():
    refspec = [(293.0, 0.75), (300.0, 1.5), (307.0, 0.75),
               (432.5, 0.0625), (439.5, 0.3125), (446.5, 0.625),
               (453.5, 0.625), (460.5, 0.3125), (467.5, 0.0625),
               (1193.0, 0.5), (1200.0, 1.0), (1207.0, 0.5)]
    tobereduced = [
        (1193.0, 0.5), (1200.0, 0.5), (1200.0, 0.5), (1207.0, 0.5),
        (432.5, 0.0625), (439.5, 0.0625), (439.5, 0.0625),
        (446.5, 0.0625), (439.5, 0.0625), (446.5, 0.0625),
        (446.5, 0.0625), (453.5, 0.0625), (439.5, 0.0625),
        (446.5, 0.0625), (446.5, 0.0625), (453.5, 0.0625),
        (446.5, 0.0625), (453.5, 0.0625), (453.5, 0.0625),
        (460.5, 0.0625), (439.5, 0.0625), (446.5, 0.0625),
        (446.5, 0.0625), (453.5, 0.0625), (446.5, 0.0625),
        (453.5, 0.0625), (453.5, 0.0625), (460.5, 0.0625),
        (446.5, 0.0625), (453.5, 0.0625), (453.5, 0.0625),
        (460.5, 0.0625), (453.5, 0.0625), (460.5, 0.0625),
        (460.5, 0.0625), (467.5, 0.0625),
        (293.0, 0.75), (300.0, 0.75), (300.0, 0.75), (307.0, 0.75)
    ]
    testspec = reduce_peaks(tobereduced)
    np.testing.assert_array_almost_equal(testspec, refspec, decimal=2)


def test_normalize():
    intensities = [1, 3, 4]
    _normalize(intensities)
    assert intensities == [0.125, 0.375, 0.5]
    double_intensities = [1, 3, 3, 1]
    _normalize(double_intensities, 2)
    assert double_intensities == [0.25, 0.75, 0.75, 0.25]


def test_normalize_spectrum():
    unnormalized = [(1200.0, 1.0), (500.0, 2.0)]
    expected = [(1200.0, 2.0), (500.0, 4.0)]
    result = normalize_peaklist(unnormalized, n=6)
    assert np.allclose(result, expected)


def test_lorentz_width():
    """Tests that w corresponds to width at half height"""
    v0 = 100
    I = 1
    w = 2
    max_height = lorentz(v0, v0, I, w)
    low_width_height = lorentz(v0 - w/2, v0, I, w)  # noqa: E226
    high_width_height = lorentz(v0 + w/2, v0, I, w)  # noqa: E226
    assert low_width_height / max_height == approx(0.5)
    assert high_width_height / max_height == approx(0.5)


def test_get_intensity():
    # 199.70588235,  199.86858573
    # 2.46553790e-05, 2.44294680e-05,
    assert get_intensity(TWOSPIN_SLOW, 199.75) == 2.46553790e-05
    assert get_intensity(TWOSPIN_SLOW, 199.80) == 2.44294680e-05


def test_ppm_to_hz_from_nuclei_info_basic():
    """
    Tests the ppm_to_hz_from_nuclei_info function with a 1H and 2H (D) example.
    """
    spectrometer_1H_MHz = 600
    ppm_positions = np.array([1.0, 3.0]) # 1H at 1 ppm, D at 3 ppm
    nuclei_types = ['1H', '2H'] # Using '2H' for Deuterium consistently
    # These are the exact gamma values that produced your expected frequencies
    gyromagnetic_ratios_MHz_per_T = [42.577478461, 6.53569888]

    # Expected frequencies (your 'v' output)
    # 1H: 1.0 ppm * (42.577478461 MHz/T * (600 MHz / 42.577478461 MHz/T)) / 1e6 = 1.0 * 600 = 600 Hz
    # 2H: 3.0 ppm * (6.53569888 MHz/T * (600 MHz / 42.577478461 MHz/T)) / 1e6 = 3.0 * (6.53569888 / 42.577478461) * 600
    #     = 3.0 * 0.153499446 * 600 = 276.30236475 Hz
    EXPECTED_FREQUENCIES_HZ = np.array([600.0, 276.30236475])

    actual_frequencies_hz = ppm_to_hz_from_nuclei_info(
        ppm_positions, nuclei_types, gyromagnetic_ratios_MHz_per_T, spectrometer_1H_MHz
    )

    np.testing.assert_allclose(actual_frequencies_hz, EXPECTED_FREQUENCIES_HZ, atol=1e-8)

def test_ppm_to_hz_from_nuclei_info_only_13C():
    """
    Tests ppm_to_hz_from_nuclei_info with only 13C nuclei to ensure B0 calculation is robust.
    """
    spectrometer_1H_MHz = 600
    ppm_positions = np.array([50.0, 55.0])
    nuclei_types = ['13C', '13C']
    gyromagnetic_ratios_MHz_per_T = [10.7083984, 10.7083984]  # Gamma for 13C

    # Calculate expected frequencies:
    # 13C Larmor Freq at 600MHz (1H) machine = gamma_13C * (600 / gamma_1H) * 1e6
    # = 10.7083984 * (600 / 42.57747892) * 1e6 Hz
    # = 10.7083984 * 14.0924976 Hz/MHz = 150.920950 MHz -> 150920950 Hz
    # Freq for 1 ppm (13C) = 150920950 Hz / 1e6 = 150.920950 Hz/ppm
    # Expected for 50 ppm: 50.0 * 150.920950 = 7546.0475 Hz
    # Expected for 55 ppm: 55.0 * 150.920950 = 8300.65225 Hz
    EXPECTED_FREQUENCIES_HZ_C = np.array([
        50.0 * (10.7083984 * (600 / 42.57747892)),
        55.0 * (10.7083984 * (600 / 42.57747892))
    ]) * (1_000_000 / 1_000_000)  # Keep in Hz

    actual_frequencies_hz_C = ppm_to_hz_from_nuclei_info(
        ppm_positions, nuclei_types, gyromagnetic_ratios_MHz_per_T, spectrometer_1H_MHz
    )

    np.testing.assert_allclose(actual_frequencies_hz_C, EXPECTED_FREQUENCIES_HZ_C, atol=1e-7)  # Adjust atol


def test_ppm_to_hz_from_nuclei_info_only_deuterium():
    """
    Tests ppm_to_hz_from_nuclei_info with only Deuterium (2H) nuclei to ensure
    B0 calculation and frequency conversion are robust.
    """
    spectrometer_1H_MHz = 600

    ppm_positions = np.array([0.5, 1.5]) # Example ppm positions for two Deuterium nuclei
    nuclei_types = ['2H', '2H']  # Both are Deuterium
    gamma_2H_MHz_per_T = 6.53569888  # Gyromagnetic ratio for Deuterium (2H)
    gyromagnetic_ratios_MHz_per_T = [gamma_2H_MHz_per_T, gamma_2H_MHz_per_T]

    # --- Calculate expected frequencies ---
    # The hardcoded 1H gamma used internally by ppm_to_hz_from_nuclei_info
    gamma_1H_MHz_per_T_internal = 42.57747892

    # 1. Calculate B0 magnetic field strength in Tesla
    B0_Tesla = spectrometer_1H_MHz / gamma_1H_MHz_per_T_internal

    # 2. Calculate Deuterium's Larmor frequency at this B0, for 1 ppm (Hz/ppm factor)
    #    This is gamma_2H * B0 (which would be in MHz if gamma is in MHz/T and B0 in T)
    #    Then convert to Hz/ppm by dividing by 1e6 (since ppm is parts per million)
    larmor_2H_hz_per_ppm_factor = gamma_2H_MHz_per_T * B0_Tesla

    # 3. Calculate expected frequencies for the given ppm positions
    #    Freq (Hz) = ppm_value * (Larmor_Freq_at_B0 (Hz) / 1e6)
    #    Or, more simply: ppm_value * (Hz/ppm factor for that nucleus)
    EXPECTED_FREQUENCIES_HZ_D = np.array([
        ppm_positions[0] * larmor_2H_hz_per_ppm_factor,
        ppm_positions[1] * larmor_2H_hz_per_ppm_factor
    ])

    actual_frequencies_hz_D = ppm_to_hz_from_nuclei_info(
        ppm_positions, nuclei_types, gyromagnetic_ratios_MHz_per_T, spectrometer_1H_MHz
    )

    np.testing.assert_allclose(actual_frequencies_hz_D, EXPECTED_FREQUENCIES_HZ_D, atol=1e-8)