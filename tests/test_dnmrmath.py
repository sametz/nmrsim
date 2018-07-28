# TODO: Write more tests!
import numpy as np

from tests.plottools import popplot
from nmrtools.nmrplot import dnmrplot_2spin
from tests.testdata import TWOSPIN_SLOW, AB_WINDNMR  # , TWOSPIN_COALESCE, TWOSPIN_FAST
from nmrtools.nmrmath import two_spin, d2s_func, TwoSinglets, dnmr_AB

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


def test_two_spin_slow_exchange():
    spectrum = TWOSPIN_SLOW
    peaks = get_maxima(spectrum)
    print("Maxima: ", peaks)

    args = (165, 135, 1.5, 0.5, 0.5, 0.5)
    x = np.linspace(85, 215, 800)
    y = two_spin(x, *args)
    popplot(x, y)

    for peak in peaks:
        print('Testing vs. accepted peak at: ', peak)
        calculated_intensity = two_spin(peak[0], *args)

        print('i.e. input of frequency ', peak[0], ' should give output of '
              'intensity ', peak[1])
        print('Calculated intensity is actually: ', calculated_intensity)

        np.testing.assert_almost_equal(calculated_intensity,
                                       peak[1])


def test_d2s_func_slow_exchange():
    spectrum = TWOSPIN_SLOW
    peaks = get_maxima(spectrum)
    print("Maxima: ", peaks)

    intensity_calculator = d2s_func(165, 135, 1.5, 0.5, 0.5, 0.5)

    x = np.linspace(85, 215, 800)
    y = intensity_calculator(x)
    popplot(x, y)

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


def test_TwoSinglets_slow_exchange():
    spectrum = TWOSPIN_SLOW
    peaks = get_maxima(spectrum)
    print("Maxima: ", peaks)

    Simulation = TwoSinglets(165, 135, 1.5, 0.5, 0.5, 50)
    popplot(*Simulation.spectrum())

    print('Testing intensity calculator on 135: ', Simulation.intensity(135))
    print('Testing intensity calculator on 165: ', Simulation.intensity(165))

    for peak in peaks:
        print('Testing vs. accepted peak at: ', peak)
        calculated_intensity = Simulation.intensity(peak[0])

        print('i.e. input of frequency ', peak[0], ' should give output of '
              'intensity ', peak[1])
        print('Calculated intensity is actually: ', calculated_intensity)

        np.testing.assert_almost_equal(calculated_intensity,
                                       peak[1])


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

        np.testing.assert_almost_equal(calculated_intensity,
                                       peak[1])
