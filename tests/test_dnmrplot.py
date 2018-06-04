# TODO: Write more tests!

import numpy as np

from tests import testdata
from tests.plottools import popplot
from pydnmr.dnmrplot import dnmrplot_2spin, dnmrplot_AB


def test_dnmrplot_2spin_slowexchange():

    WINDNMR_DEFAULT = (165.00, 135.00, 1.50, 0.50, 0.50, 50.00)
    x, y = dnmrplot_2spin(*WINDNMR_DEFAULT)
    accepted_x, accepted_y = testdata.TWOSPIN_SLOW
    np.testing.assert_array_almost_equal(x, accepted_x)
    np.testing.assert_array_almost_equal(y, accepted_y)


def test_dnmrplot_2spin_coalesce():

    WINDNMR_DEFAULT = (165.00, 135.00, 65.9, 0.50, 0.50, 50.00)
    x, y = dnmrplot_2spin(*WINDNMR_DEFAULT)
    accepted_x, accepted_y = testdata.TWOSPIN_COALESCE
    np.testing.assert_array_almost_equal(x, accepted_x)
    np.testing.assert_array_almost_equal(y, accepted_y)


def test_dnmrplot_2spin_fastexchange():

    WINDNMR_DEFAULT = (165.00, 135.00, 1000.00, 0.50, 0.50, 50.00)
    x, y = dnmrplot_2spin(*WINDNMR_DEFAULT)
    accepted_x, accepted_y = testdata.TWOSPIN_FAST
    np.testing.assert_array_almost_equal(x, accepted_x)
    np.testing.assert_array_almost_equal(y, accepted_y)

def test_dnmrplot_2spin_frequencies_commute():

    freqorder_ab = dnmrplot_2spin(165.00, 135.00, 1.50, 2.50, 0.50, 75.00)
    freqorder_ba = dnmrplot_2spin(135.00, 165.00, 1.50, 0.50, 2.50, 25.00)
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


# if __name__ == "__main__":
#     import plottools as pt
#
#     WINDNMR_DEFAULTS = (165.00, 135.00, 1.50, 0.50, 0.50, 50.00)
#     spectrum = dnmrplot_2spin(*WINDNMR_DEFAULTS)
#     pt.popplot(*spectrum)
