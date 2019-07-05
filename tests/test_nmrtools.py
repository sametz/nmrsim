import numpy as np

from nmrtools import SpinSystem, Spectrum
from nmrtools.firstorder import first_order_spin_system, Multiplet
from tests.accepted_data import SPECTRUM_RIOUX
from tests.simulation_data import rioux


def test_SpinSystem_ABX():
    v, J = rioux()
    ss = SpinSystem(v, J)
    assert np.allclose(ss.peaklist(), SPECTRUM_RIOUX)


def test_SpinSystem_firstorder():
    v, J = rioux()
    expected_result = first_order_spin_system(v, J)
    ss = SpinSystem(v, J, second_order=False)
    assert np.allclose(expected_result, ss.peaklist())


def test_Spectrum_instantiates_with_multiplet():
    m1 = Multiplet(100, 1, [(10, 2)])
    m2 = Multiplet(80, 1, [(10, 2)])
    s = Spectrum([m1, m2])
    expected_peaklist = sorted([(110, 0.25), (100, 0.5), (90, 0.5), (80, 0.5),
                                (70, 0.25)])
    result = s.peaklist()
    print('result: ', result)
    assert np.array_equal(expected_peaklist, result)
