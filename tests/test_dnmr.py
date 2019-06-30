import numpy as np

from nmrtools.dnmr import d2s_func, dnmr_AB, DnmrTwoSinglets, DnmrAB
from tests.plottools import popplot
from tests.testdata import AB_WINDNMR, TWOSPIN_COALESCE, TWOSPIN_SLOW


def get_intensity(spectrum, x):
    """
    A quick and dirty method to get intensity of data point closest to
    frequency x. Better: interpolate between two data points if match isn't
    exact (TODO?)
    :param spectrum: tuple of (x, y) arrays for frequency, intensity data
    :param x: frequency lookup
    :return: the intensity at that frequency
    """
    nearest_x_index = np.abs(spectrum[0] - x).argmin()
    return spectrum[1][nearest_x_index]


def get_maxima(spectrum):
    """
    Crude function that returns maxima in the spectrum.
    :param spectrum: tuple of frequency, intensity arrays
    :return: a list of (frequency, intensity) tuples for individual maxima.
    """
    res = []
    for n, val in enumerate(spectrum[1][1:-2]):
        index = n+1  # start at spectrum[1][1]
        lastvalue = spectrum[1][index-1]
        nextvalue = spectrum[1][index+1]

        if lastvalue < val and nextvalue < val:
            print('MAXIMUM FOUND AT: ')
            print((spectrum[0][index], val))
            res.append((spectrum[0][index], val))
    return res


def test_d2s_func_slow_exchange():
    spectrum = TWOSPIN_SLOW
    peaks = get_maxima(spectrum)
    print("Maxima: ", peaks)

    intensity_calculator = d2s_func(165, 135, 1.5, 0.5, 0.5, 0.5)

    x = np.linspace(85, 215, 800)
    y = intensity_calculator(x)
    # popplot(x, y)  # replace with non-PyQtGraph popup graph if desired

    print('Testing intensity calculator on 135: ', intensity_calculator(135))
    print('Testing intensity calculator on 165: ', intensity_calculator(165))

    for peak in peaks:
        print('Testing vs. accepted peak at: ', peak)
        calculated_intensity = intensity_calculator(peak[0])

        print('i.e. input of frequency ', peak[0], ' should give output of '
              'intensity ', peak[1])
        print('Calculated intensity is actually: ', calculated_intensity)

        np.testing.assert_almost_equal(calculated_intensity,
                                       peak[1])


def test_DnmrTwoSinglets_instantiates():
    args = (165, 135, 1.5, 0.5, 0.5, 0.5, (215, 85))
    sim = DnmrTwoSinglets(*args)
    expected_args = (165, 135, 1.5, 0.5, 0.5, (85, 215))
    actual_args = (sim.va, sim.vb, sim.k, sim.wa, sim.wb, sim.limits)
    assert expected_args == actual_args


def test_DnmrTwoSinglets_properties():
    sim = DnmrTwoSinglets(1, 2, 3, 4, 5, 6, (7, 8))
    sim.va = 165.0
    sim.vb = 135.0
    sim.k = 1.5
    sim.wa = 0.5
    sim.wb = 0.5
    sim.pa = 0.5
    sim.limits = (85.0, 215.0)
    sim_args = (sim.va, sim.vb, sim.k, sim.wa, sim.wb, sim.pa, sim.limits)
    print('result: ', sim_args)
    assert sim_args == (165.0, 135.0, 1.5, 0.5, 0.5, 0.5, (85.0, 215.0))


def test_DnmrTwoSinglets_slow_exchange():
    sim = DnmrTwoSinglets(165, 135, 1.5, 0.5, 0.5, 0.5)
    assert np.allclose(sim.spectrum(), TWOSPIN_SLOW)
    popplot(*sim.spectrum())


def test_DnmrTwoSinglets_coalesce():
    sim = DnmrTwoSinglets(165, 135, 65.9, 0.5, 0.5, 0.5)
    assert np.allclose(sim.spectrum(), TWOSPIN_COALESCE)
    popplot(*sim.spectrum())


def test_Dnmr_TwoSinglets_limits():
    sim = DnmrTwoSinglets(165, 135, 1.5, 0.5, 0.5, 0.5)
    sim.limits = (500, 0)
    assert sim.limits == (0, 500)


def test_ab_WINDNMR_defaults():
    spectrum = AB_WINDNMR
    peaks = get_maxima(spectrum)
    print("Maxima: ", peaks)

    ab_args = (165, 135, 12, 12, 0.5)

    for peak in peaks:
        print('Testing vs. accepted peak at: ', peak)
        calculated_intensity = dnmr_AB(peak[0], *ab_args)

        print('i.e. input of frequency ', peak[0], ' should give output of '
                                                   'intensity ', peak[1])
        print('Calculated intensity is actually: ', calculated_intensity)

        assert np.allclose(calculated_intensity, peak[1])


def test_DnmrAB_instantiates():
    args = (165, 135, 12, 12, 0.5, (215, 85))
    sim = DnmrAB(*args)
    expected_args = (165, 135, 12, 12, 0.5, (85, 215))
    actual_args = (sim.v1, sim.v2, sim.J, sim.k, sim.W, sim.limits)
    assert expected_args == actual_args


def test_DnmrAB_properties():
    sim = DnmrAB(1, 2, 3, 4, 5, (6, 7))
    sim.v1 = 165.0
    sim.v2 = 135.0
    sim.J = 12.0
    sim.k = 12.0
    sim.W = 0.5
    sim.limits = (85.0, 215.0)
    sim_args = (sim.v1, sim.v2, sim.J, sim.k, sim.W, sim.limits)
    print('result: ', sim_args)
    assert sim_args == (165.0, 135.0, 12.0, 12.0, 0.5, (85.0, 215.0))
    assert np.allclose(sim.spectrum(), AB_WINDNMR)


def test_DnmrAB_WINDNMR_defaults():
    expected = np.array(AB_WINDNMR)
    sim = DnmrAB(165, 135, 12, 12, 0.5)
    result = np.array(sim.spectrum())
    print(expected.shape, result.shape)
    print(expected[:, -10:])
    print(result[:, -10:])
    assert np.allclose(result, expected)