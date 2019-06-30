import numpy as np

from nmrtools.dnmr import DnmrTwoSinglets
from tests.plottools import popplot
from tests.testdata import TWOSPIN_COALESCE, TWOSPIN_SLOW


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


def test_DnmrTwoSinglets_instantiation():
    args = (165, 135, 1.5, 0.5, 0.5, 0.5, (200, 0))
    sim = DnmrTwoSinglets(*args)
    expected_args = (165, 135, 1.5, 0.5, 0.5, (0, 200))
    actual_args = (sim.va, sim.vb, sim.k, sim.wa, sim.wb, sim.limits)
    assert expected_args == actual_args


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