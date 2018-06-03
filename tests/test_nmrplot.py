import numpy as np
from pytest import approx
from uw_dnmr.model.nmrplot import (lorentz, add_signals,
                                     dnmrplot_2spin, dnmrplot_AB)
from . import testdata
from .accepted_data import ADD_SIGNALS_DATASET
from .plottools import popplot


def test_lorentz_width():
    """Tests that w corresponds to width at half height"""
    v0 = 100
    I = 1
    w = 2

    assert lorentz(v0 - w/2, v0, I, w) == approx(0.5)
    assert lorentz(v0 + w/2, v0, I, w) == approx(0.5)


def test_add_signals():
    """
    Tests that current nmrplot.add_signals output agrees with an accepted
    dataset.
    """
    x = np.linspace(390, 410, 200)
    doublet = [(399, 1), (401, 1)]
    y = add_signals(x, doublet, 1)
    X = np.array([x for x, _ in ADD_SIGNALS_DATASET])
    Y = np.array([y for _, y in ADD_SIGNALS_DATASET])

    assert np.array_equal(x, X)
    assert np.array_equal(y, Y)


def test_dnmrplot_2spin_slowexchange():

    WINDNMR_DEFAULT = (165.00, 135.00, 1.50, 0.50, 0.50, 0.50)
    x, y = dnmrplot_2spin(*WINDNMR_DEFAULT)
    accepted_x, accepted_y = testdata.TWOSPIN_SLOW
    np.testing.assert_array_almost_equal(x, accepted_x)
    np.testing.assert_array_almost_equal(y, accepted_y)


def test_dnmrplot_2spin_coalesce():

    WINDNMR_DEFAULT = (165.00, 135.00, 65.9, 0.50, 0.50, 0.50)
    x, y = dnmrplot_2spin(*WINDNMR_DEFAULT)
    accepted_x, accepted_y = testdata.TWOSPIN_COALESCE
    np.testing.assert_array_almost_equal(x, accepted_x)
    np.testing.assert_array_almost_equal(y, accepted_y)


def test_dnmrplot_2spin_fastexchange():

    WINDNMR_DEFAULT = (165.00, 135.00, 1000.00, 0.50, 0.50, 0.50)
    x, y = dnmrplot_2spin(*WINDNMR_DEFAULT)
    accepted_x, accepted_y = testdata.TWOSPIN_FAST
    np.testing.assert_array_almost_equal(x, accepted_x)
    np.testing.assert_array_almost_equal(y, accepted_y)


def test_dnmrplot_2spin_frequencies_commute():

    freqorder_ab = dnmrplot_2spin(165.00, 135.00, 1.50, 2.50, 0.50, 0.75)
    freqorder_ba = dnmrplot_2spin(135.00, 165.00, 1.50, 0.50, 2.50, 0.25)
    popplot(*freqorder_ab)
    popplot(*freqorder_ba)
    np.testing.assert_array_almost_equal(freqorder_ab, freqorder_ba)


def test_dnmrplot_AB():

    WINDNMR_DEFAULT = (165.00, 135.00, 12.00, 12.00, 0.50)
    x, y = dnmrplot_AB(*WINDNMR_DEFAULT)
    accepted_x, accepted_y = testdata.AB_WINDNMR
    np.testing.assert_array_almost_equal(x, accepted_x)
    np.testing.assert_array_almost_equal(y, accepted_y)


def test_dnmrplot_AB_frequencies_commute():
    freqorder_ab = dnmrplot_AB(165.00, 135.00, 12.00, 12.00, 0.50)
    freqorder_ba = dnmrplot_AB(135.00, 165.00, 12.00, 12.00, 0.50)
    popplot(*freqorder_ab)
    popplot(*freqorder_ba)
    np.testing.assert_array_almost_equal(freqorder_ab, freqorder_ba)
