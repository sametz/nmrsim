import numpy as np
from nmrtools.math import (add_peaks, reduce_peaks,
                           _normalize, normalize_spectrum)


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
    result = normalize_spectrum(unnormalized, n=6)
    assert np.allclose(result, expected)
