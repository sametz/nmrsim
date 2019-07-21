import numpy as np
import pytest

from nmrtools import Multiplet, SpinSystem, Spectrum
from nmrtools.firstorder import first_order_spin_system
from tests.accepted_data import SPECTRUM_RIOUX
from tests.simulation_data import rioux


def test_AutoStorage_get_without_instance():
    # See L. Ramalho, "Fluent Python", p. 637. Testing instance=None
    assert Multiplet.v is not None


@pytest.fixture()
def td():
    """Return multiplet data for a 1200 Hz, 2H
    triplet of doublets, J=7.1, 1.1 Hz.
    """
    v = 1200.0
    I = 2
    J = [(7.1, 2), (1.1, 1)]
    return v, I, J


@pytest.fixture()
def dummy_multiplet(td):
    return Multiplet(*td)


@pytest.fixture()
def abx():
    return rioux()


@pytest.fixture()
def spinsystem(abx):
    return SpinSystem(*abx)


class TestMultiplet:
    def test_instantiates(self, td):
        v, I, J = td
        td_multiplet = Multiplet(v, I, J)
        assert td_multiplet.v == 1200.0
        assert td_multiplet.I == 2
        assert td_multiplet.J == [(7.1, 2), (1.1, 1)]
        expected_peaklist = [
            (1192.35, 0.25), (1193.45, 0.25),
            (1199.45, 0.5), (1200.55, 0.5),
            (1206.55, 0.25), (1207.65, 0.25)]
        assert np.allclose(td_multiplet.peaklist(), expected_peaklist)

    def test_dummy_multiplet(self, dummy_multiplet):
        assert dummy_multiplet.v == 1200.0
        assert dummy_multiplet.I == 2
        assert dummy_multiplet.J == [(7.1, 2), (1.1, 1)]
        expected_peaklist = [
            (1192.35, 0.25), (1193.45, 0.25),
            (1199.45, 0.5), (1200.55, 0.5),
            (1206.55, 0.25), (1207.65, 0.25)]
        assert np.allclose(dummy_multiplet.peaklist(), expected_peaklist)

    def test_v_setter(self, dummy_multiplet):
        dummy_multiplet.v = 200.0
        assert dummy_multiplet.v == 200.0  # tests Autostorage instance=None
        expected_peaklist = [
            (192.35, 0.25), (193.45, 0.25),
            (199.45, 0.5), (200.55, 0.5),
            (206.55, 0.25), (207.65, 0.25)]
        assert np.allclose(dummy_multiplet.peaklist(), expected_peaklist)

        with pytest.raises(TypeError):
            dummy_multiplet.v = 'foo'
            assert dummy_multiplet.v == 'foo'

    def test_I_setter(self, dummy_multiplet):
        dummy_multiplet.I = 4
        expected_peaklist = [
            (1192.35, 0.5), (1193.45, 0.5),
            (1199.45, 1.0), (1200.55, 1.0),
            (1206.55, 0.5), (1207.65, 0.5)]
        assert np.allclose(dummy_multiplet.peaklist(), expected_peaklist)

        with pytest.raises(TypeError):
            dummy_multiplet.I = 'foo'

    def test_J_setter(self, dummy_multiplet):
        dummy_multiplet.J = [(10.0, 1)]
        expected_peaklist = [(1195.0, 1), (1205.0, 1)]
        assert np.allclose(dummy_multiplet.peaklist(), expected_peaklist)
        dummy_multiplet.J = []
        expected_singlet = [(1200, 2)]
        assert np.allclose(dummy_multiplet.peaklist(), expected_singlet)

        with pytest.raises(TypeError):
            dummy_multiplet.J = [((1, 2), (3, 4)), ((5, 6), (7, 8))]

        with pytest.raises(ValueError):
            dummy_multiplet.J = [(1, 2, 3)]

    def test_eq(self, dummy_multiplet):
        m1 = Multiplet(100, 1, [(10, 2)])
        m2 = m1 * 2
        m3 = m2 / 2
        assert m1 is not m3
        assert m1 == m3

    def test_add(self, dummy_multiplet):
        m1 = Multiplet(100, 1, [(10, 2)])
        m2 = Multiplet(80, 1, [(10, 2)])
        result = m1 + m2
        # assert 1 == 1
        expected_peaklist = sorted([(110, 0.25), (100, 0.5), (90, 0.5), (80, 0.5),
                                    (70, 0.25)])
        assert np.array_equal(result.peaklist(), expected_peaklist)
        with pytest.raises(TypeError):
            bad_result = m1 + 3

    def test_mul(self, dummy_multiplet):
        doubled = dummy_multiplet * 2.0
        assert doubled is not dummy_multiplet  # should not modify original
        expected_peaklist = [
            (1192.35, 0.5), (1193.45, 0.5),
            (1199.45, 1.0), (1200.55, 1.0),
            (1206.55, 0.5), (1207.65, 0.5)]
        assert np.allclose(doubled.peaklist(), expected_peaklist)
        with pytest.raises(TypeError):
            assert dummy_multiplet * 'foo' == NotImplemented

    def test_imul(self, dummy_multiplet):
        original_m = dummy_multiplet
        dummy_multiplet *= 2
        assert dummy_multiplet is original_m
        expected_peaklist = [
            (1192.35, 0.5), (1193.45, 0.5),
            (1199.45, 1.0), (1200.55, 1.0),
            (1206.55, 0.5), (1207.65, 0.5)]
        assert np.allclose(dummy_multiplet.peaklist(), expected_peaklist)
        with pytest.raises(TypeError):
            dummy_multiplet *= 'foo'

    def test_truediv(self, dummy_multiplet):
        doubled = dummy_multiplet * 2.0
        divtest = dummy_multiplet / 0.5
        assert doubled is not divtest
        assert doubled == divtest
        with pytest.raises(ZeroDivisionError):
            dummy_multiplet / 0

    def test_itruediv(self, dummy_multiplet):
        original = dummy_multiplet
        doubled = dummy_multiplet * 2.0
        dummy_multiplet /= 0.5
        assert original is dummy_multiplet
        assert doubled is not original
        assert doubled == original
        with pytest.raises(ZeroDivisionError):
            dummy_multiplet /= 0

# @pytest.mark.usefixtures('abx')
class TestSpinSystem:
    def test_instantiates(self, abx):
        v, J = abx
        ss = SpinSystem(v, J, second_order=False)
        assert np.allclose(ss.v, v)
        assert np.allclose(ss.J, J)
        assert ss.second_order is False

    def test_v_validation(self, spinsystem):
        new_v = [1, 2, 3]
        spinsystem.v = new_v
        assert np.array_equal(spinsystem.v, new_v)
        with pytest.raises(ValueError):
            spinsystem.v = [1, 2]
        with pytest.raises(TypeError):
            spinsystem.v = ['foo', 'bar', 'baz']

    def test_J_validation(self, spinsystem):
        new_J = [[0, 1, 2],
                 [1, 0, 3],
                 [2, 3, 0]]
        spinsystem.J = new_J
        assert np.array_equal(spinsystem.J, new_J)

        wrongsize_J = [[0, 1], [1, 0]]
        with pytest.raises(TypeError):
            spinsystem.J = wrongsize_J

        nozeros_J = [[0, 1, 2],
                     [1, 0, 3],
                     [2, 3, 1]]
        with pytest.raises(ValueError):
            spinsystem.J = nozeros_J

        notdiagonal_J = new_J = [[0, 1, 2],
                                 [1, 0, 3],
                                 [2, 4, 0]]
        with pytest.raises(ValueError):
            spinsystem.J = notdiagonal_J

    def test_second_order_validation(self, spinsystem):
        spinsystem.second_order = False
        assert spinsystem.second_order is False
        with pytest.raises(TypeError):
            spinsystem.second_order = "second order"

    def test_SpinSystem_ABX(self, abx):
        v, J = abx
        ss = SpinSystem(v, J)
        assert np.allclose(ss.peaklist(), SPECTRUM_RIOUX)

    def test_SpinSystem_firstorder(self, abx):
        v, J = abx
        expected_result = first_order_spin_system(v, J)
        ss = SpinSystem(v, J, second_order=False)
        assert np.allclose(expected_result, ss.peaklist())

    def test_eq(self, abx):
        ss1 = SpinSystem(*abx)
        ss2 = SpinSystem(*abx)
        assert ss1 is not ss2
        assert ss1 == ss2

    def test_add(self, abx, dummy_multiplet):
        ss = SpinSystem(*abx)
        spectrum_1 = dummy_multiplet + ss
        spectrum_2 = ss + dummy_multiplet
        assert spectrum_1 == spectrum_2
        with pytest.raises(TypeError):
            bad_ss = ss + 1


class TestSpectrum:
    def test_Spectrum_instantiates_with_multiplet(self):
        m1 = Multiplet(100, 1, [(10, 2)])
        m2 = Multiplet(80, 1, [(10, 2)])
        s = Spectrum([m1, m2])
        expected_peaklist = sorted([(110, 0.25), (100, 0.5), (90, 0.5), (80, 0.5),
                                    (70, 0.25)])
        result = s.peaklist()
        assert np.array_equal(expected_peaklist, result)

    def test_add_and_eq(self):
        m1 = Multiplet(100, 1, [(10, 2)])
        m2 = Multiplet(80, 1, [(10, 2)])
        s = Spectrum([m1, m2])
        s2 = m1 + m2
        assert s == s2
        s3 = m1 + m1 + m2 + m2  # test for more than two objects being added
        expected_peaklist = sorted([(110, 0.5), (100, 1), (90, 1), (80, 1),
                                    (70, 0.5)])
        assert np.allclose(s3.peaklist(), expected_peaklist)
        with pytest.raises(TypeError):
            s4 = s + 1

    def test_add_appends_to_components(self):
        m1 = Multiplet(100, 1, [(10, 2)])
        m2 = Multiplet(80, 1, [(10, 2)])
        s = Spectrum([m1])
        s2 = s + m2
        assert s2._components == [m1, m2]
