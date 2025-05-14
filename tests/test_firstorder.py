import numpy as np

from nmrsim.firstorder import multiplet, first_order_spin_system
from nmrsim.math import reduce_peaks
from tests.qm_arguments import rioux


def test_multiplet_allows_singlet():
    refspec = [(1200.0, 2.0, 0.5)]
    # GIVEN a signal
    # WHEN multiplet is called with an empty J list
    testspec = multiplet((1200.0, 2.0, 0.5), [])
    # THEN the returned peaklist only contains the original signal
    assert np.allclose(refspec, testspec)


def test_multiplet():
    refspec = [(293.0, 0.75, 1.0),
               (300.0, 1.5, 1.0),
               (307.0, 0.75, 1.0),
               (432.5, 0.0625, 0.5),
               (439.5, 0.3125, 0.5),
               (446.5, 0.625, 0.5),
               (453.5, 0.625, 0.5),
               (460.5, 0.3125, 0.5),
               (467.5, 0.0625, 0.5),
               (1193.0, 0.5, 0.5),
               (1200.0, 1.0, 0.5),
               (1207.0, 0.5, 0.5)]
    v1 = (1200, 2, 0.5)
    v2 = (450, 2, 0.5)
    v3 = (300, 3, 1.0)
    J12 = 7
    J23 = 7
    m1 = multiplet(v1, [(J12, 2)])
    m2 = multiplet(v2, [(J12, 2), (J23, 3)])
    m3 = multiplet(v3, [(J23, 2)])
    testspec = reduce_peaks(sorted(m1 + m2 + m3))
    np.testing.assert_array_almost_equal(testspec, refspec, decimal=2)


def test_first_order_spin_system():
    v, J = rioux()
    spectrum = first_order_spin_system(v, J, 0.5)
    m1 = multiplet((430, 1, 0.5), [(7, 1), (15, 1)])
    m2 = multiplet((265, 1, 0.5), [(7, 1), (1.5, 1)])
    m3 = multiplet((300, 1, 0.5), [(15, 1), (1.5, 1)])
    m = reduce_peaks(sorted(m1 + m2 + m3))
    # x = np.array([i[0] for i in spectrum])
    # y = np.array([i[1] for i in spectrum])
    # mplplot_stick(spectrum)
    assert np.allclose(spectrum, m)
