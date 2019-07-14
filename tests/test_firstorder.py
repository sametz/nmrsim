import numpy as np
import pytest

from nmrtools.firstorder import *
from nmrtools.plt import mplplot_stick
from tests.simulation_data import rioux


def test_doublet():
    refspec = [(95, 0.5), (105, 0.5)]
    testspec = sorted(doublet([(100, 1)], 10))
    np.testing.assert_array_almost_equal(testspec, refspec, decimal=2)


def test_multiplet():
    # Inputs: simulate n-propanol
    refspec = sorted([
        (1193.0, 0.5), (1200.0, 0.5), (1200.0, 0.5), (1207.0, 0.5),
        (432.5, 0.0625), (439.5, 0.0625), (439.5, 0.0625),
        (446.5, 0.0625), (439.5, 0.0625), (446.5, 0.0625),
        (446.5, 0.0625), (453.5, 0.0625), (439.5, 0.0625),
        (446.5, 0.0625), (446.5, 0.0625), (453.5, 0.0625),
        (446.5, 0.0625), (453.5, 0.0625), (453.5, 0.0625),
        (460.5, 0.0625), (439.5, 0.0625), (446.5, 0.0625),
        (446.5, 0.0625), (453.5, 0.0625), (446.5, 0.0625),
        (453.5, 0.0625), (453.5, 0.0625), (460.5, 0.0625),
        (446.5, 0.0625), (453.5, 0.0625), (453.5, 0.0625),
        (460.5, 0.0625), (453.5, 0.0625), (460.5, 0.0625),
        (460.5, 0.0625), (467.5, 0.0625),
        (293.0, 0.75), (300.0, 0.75), (300.0, 0.75), (307.0, 0.75)])
    v1 = (1200, 2)
    v2 = (450, 2)
    v3 = (300, 3)
    J12 = 7
    J23 = 7
    m1 = multiplet(v1, [(J12, 2)])
    m2 = multiplet(v2, [(J12, 2), (J23, 3)])
    m3 = multiplet(v3, [(J23, 2)])

    testspec = sorted(m1 + m2 + m3)
    np.testing.assert_array_almost_equal(testspec, refspec, decimal=2)


def test_first_order_allows_singlet():
    refspec = [(1200.0, 2.0)]
    testspec = first_order((1200.0, 2.0), [])
    assert np.allclose(refspec, testspec)


def test_first_order():
    refspec = [(293.0, 0.75), (300.0, 1.5), (307.0, 0.75),
               (432.5, 0.0625), (439.5, 0.3125), (446.5, 0.625),
               (453.5, 0.625), (460.5, 0.3125), (467.5, 0.0625),
               (1193.0, 0.5), (1200.0, 1.0), (1207.0, 0.5)]
    v1 = (1200, 2)
    v2 = (450, 2)
    v3 = (300, 3)
    J12 = 7
    J23 = 7
    m1 = first_order(v1, [(J12, 2)])
    m2 = first_order(v2, [(J12, 2), (J23, 3)])
    m3 = first_order(v3, [(J23, 2)])
    testspec = reduce_peaks(sorted(m1 + m2 + m3))
    np.testing.assert_array_almost_equal(testspec, refspec, decimal=2)


def test_first_order_spin_system():
    v, J = rioux()
    spectrum = first_order_spin_system(v, J)
    x = np.array([i[0] for i in spectrum])
    y = np.array([i[1] for i in spectrum])
    mplplot_stick(spectrum)
    assert 1 == 1


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


class TestMultiplet:
    def test_instantiates(self, td):
        v, I, J = td
        td_multiplet = Multiplet(v, I, J)
        assert td_multiplet.v ==1200.0
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
