import numpy as np
import pytest

from nmrtools.firstorder import *
from nmrtools.plt import mplplot_stick
from tests.simulation_data import rioux


def test_multiplet_allows_singlet():
    refspec = [(1200.0, 2.0)]
    # GIVEN a signal
    # WHEN multiplet is called with an empty J list
    testspec = multiplet((1200.0, 2.0), [])
    # THEN the returned peaklist only contains the original signal
    assert np.allclose(refspec, testspec)


def test_multiplet():
    refspec = [(293.0, 0.75), (300.0, 1.5), (307.0, 0.75),
               (432.5, 0.0625), (439.5, 0.3125), (446.5, 0.625),
               (453.5, 0.625), (460.5, 0.3125), (467.5, 0.0625),
               (1193.0, 0.5), (1200.0, 1.0), (1207.0, 0.5)]
    v1 = (1200, 2)
    v2 = (450, 2)
    v3 = (300, 3)
    J12 = 7
    J23 = 7
    m1 = multiplet(v1, [(J12, 2)])
    m2 = multiplet(v2, [(J12, 2), (J23, 3)])
    m3 = multiplet(v3, [(J23, 2)])
    testspec = reduce_peaks(sorted(m1 + m2 + m3))
    np.testing.assert_array_almost_equal(testspec, refspec, decimal=2)


def test_first_order_spin_system():
    v, J = rioux()
    spectrum = first_order_spin_system(v, J)
    x = np.array([i[0] for i in spectrum])
    y = np.array([i[1] for i in spectrum])
    mplplot_stick(spectrum)
    assert 1 == 1
