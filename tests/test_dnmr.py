import numpy as np
import pytest

from nmrtools.dnmr import (dnmr_two_singlets, dnmr_AB_func, dnmr_AB,
                           DnmrTwoSinglets, DnmrAB)
from nmrtools.math import get_maxima
from tests.plottools import popplot
from tests.testdata import (AB_WINDNMR, TWOSPIN_COALESCE, TWOSPIN_FAST,
                            TWOSPIN_SLOW)


class TestDnmrTwoSinglets:
    def test_dnmr_two_singlets_commute(self):
        freqorder_ab = dnmr_two_singlets(165.00, 135.00, 1.50, 2.50, 0.50, 0.75)
        freqorder_ba = dnmr_two_singlets(135.00, 165.00, 1.50, 0.50, 2.50, 0.25)
        popplot(*freqorder_ab)
        popplot(*freqorder_ba)
        np.testing.assert_array_almost_equal(freqorder_ab, freqorder_ba)

    def test_dnmr_two_singlets_limits(self):
        WINDNMR_DEFAULT = (165.00, 135.00, 1.50, 0.50, 0.50, 0.50)
        limits = (215, 85)  # also test limits, with inverted (hi, lo) order
        x, y = dnmr_two_singlets(*WINDNMR_DEFAULT, limits=limits)
        accepted_x, accepted_y = TWOSPIN_SLOW
        np.testing.assert_array_almost_equal(x, accepted_x)
        np.testing.assert_array_almost_equal(y, accepted_y)

    def test_dnmr_two_singlets_slow_exchange(self):
        WINDNMR_DEFAULT = (165.00, 135.00, 1.50, 0.50, 0.50, 0.50)
        x, y = dnmr_two_singlets(*WINDNMR_DEFAULT)
        accepted_x, accepted_y = TWOSPIN_SLOW
        np.testing.assert_array_almost_equal(x, accepted_x)
        np.testing.assert_array_almost_equal(y, accepted_y)

    def test_dnmr_two_singlets_coalesce(self):

        WINDNMR_DEFAULT = (165.00, 135.00, 65.9, 0.50, 0.50, 0.50)
        x, y = dnmr_two_singlets(*WINDNMR_DEFAULT)
        accepted_x, accepted_y = TWOSPIN_COALESCE
        np.testing.assert_array_almost_equal(x, accepted_x)
        np.testing.assert_array_almost_equal(y, accepted_y)

    def test_dnmr_two_singlets_fastexchange(self):

        WINDNMR_DEFAULT = (165.00, 135.00, 1000.00, 0.50, 0.50, 0.50)
        x, y = dnmr_two_singlets(*WINDNMR_DEFAULT)
        accepted_x, accepted_y = TWOSPIN_FAST
        np.testing.assert_array_almost_equal(x, accepted_x)
        np.testing.assert_array_almost_equal(y, accepted_y)


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


@pytest.mark.parametrize('limits', ['foo', (1,), (1, 'foo'), (1, 2, 3)])
def test_DnmrTwoSinglets_limit_error(limits):
    with pytest.raises((AttributeError, TypeError, ValueError)):
        sim = DnmrTwoSinglets(limits=limits)
        assert sim


def test_DnmrTwoSinglets_slow_exchange():
    sim = DnmrTwoSinglets(165, 135, 1.5, 0.5, 0.5, 0.5)
    assert np.allclose(sim.spectrum(), TWOSPIN_SLOW)
    popplot(*sim.spectrum())


def test_DnmrTwoSinglets_coalesce():
    sim = DnmrTwoSinglets(165, 135, 65.9, 0.5, 0.5, 0.5)
    assert np.allclose(sim.spectrum(), TWOSPIN_COALESCE)
    popplot(*sim.spectrum())


def test_DnmrTwoSinglets_fastexchange():
    sim = DnmrTwoSinglets(165, 135, 1000, 0.5, 0.5, 0.5)
    assert np.allclose(sim.spectrum(), TWOSPIN_FAST)
    popplot(*sim.spectrum())


def test_DnmrTwoSinglets_limits():
    sim = DnmrTwoSinglets(165, 135, 1.5, 0.5, 0.5, 0.5)
    sim.limits = (500, 0)
    assert sim.limits == (0, 500)


def test_DnmrTwoSinglets_frequencies_commute():
    ab = DnmrTwoSinglets(165.00, 135.00, 1.50, 2.50, 0.50, 0.75)
    ba = DnmrTwoSinglets(135.00, 165.00, 1.50, 0.50, 2.50, 0.25)
    ab_spec = ab.spectrum()  # doing this saves a function call (faster test)
    ba_spec = ba.spectrum()
    popplot(*ab_spec)
    popplot(*ba_spec)
    np.testing.assert_array_almost_equal(ab_spec, ba_spec)


def test_dnmr_ab_func_WINDNMR_defaults():
    spectrum = AB_WINDNMR
    peaks = get_maxima(spectrum)
    print("Maxima: ", peaks)

    ab_args = (165, 135, 12, 12, 0.5)

    for peak in peaks:
        print('Testing vs. accepted peak at: ', peak)
        calculated_intensity = dnmr_AB_func(peak[0], *ab_args)

        print('i.e. input of frequency ', peak[0], ' should give output of '
                                                   'intensity ', peak[1])
        print('Calculated intensity is actually: ', calculated_intensity)

        assert np.allclose(calculated_intensity, peak[1])


def test_dnmr_AB_frequencies_commute():
    freqorder_ab = dnmr_AB(165.00, 135.00, 12.00, 12.00, 0.50)
    freqorder_ba = dnmr_AB(135.00, 165.00, 12.00, 12.00, 0.50)
    popplot(*freqorder_ab)
    popplot(*freqorder_ba)
    np.testing.assert_array_almost_equal(freqorder_ab, freqorder_ba)


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


@pytest.mark.parametrize('limits', ['foo', (1,), (1, 'foo'), (1, 2, 3)])
def test_DnmrAB_limit_error(limits):
    with pytest.raises((AttributeError, TypeError, ValueError)):
        sim = DnmrAB(limits=limits)
        assert sim


def test_DnmrAB_WINDNMR_defaults():
    expected = np.array(AB_WINDNMR)
    sim = DnmrAB(165, 135, 12, 12, 0.5)
    result = np.array(sim.spectrum())
    print(expected.shape, result.shape)
    print(expected[:, -10:])
    print(result[:, -10:])
    assert np.allclose(result, expected)
