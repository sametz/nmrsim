import copy
import pathlib
import numpy as np
import pytest

from nmrsim.qm import (_tm_cache, hamiltonian_dense, hamiltonian_sparse,  # noqa
                       secondorder_dense, secondorder_sparse, _so_sparse,  # noqa
                       qm_spinsystem)
from tests.accepted_data import HAMILTONIAN_RIOUX, SPECTRUM_RIOUX
from tests.qm_arguments import rioux

@pytest.mark.xfail(reason="sparse array boolean conversion issue")
def test_so_sparse_creates_files(fs):
    test_bin = (pathlib.Path(__file__)
                .resolve()
                .parent.parent
                .joinpath('src', 'nmrsim', 'bin'))
    fs.add_real_directory(test_bin, read_only=False)
    expected_Lz = test_bin.joinpath('Lz3.npz')
    expected_Lproduct = test_bin.joinpath('Lproduct3.npz')
    assert expected_Lz.exists()
    assert expected_Lproduct.exists()
    fs.remove_object(str(expected_Lz))
    fs.remove_object(str(expected_Lproduct))
    assert not expected_Lz.exists()
    assert not expected_Lproduct.exists()
    Lz, Lproduct = _so_sparse(3)  # noqa
    assert Lz, Lproduct
    assert expected_Lz.exists()
    assert expected_Lproduct.exists()

@pytest.mark.xfail(reason="sparse array boolean conversion issue")
def test_tm_cache_creates_file(fs):
    test_bin = (pathlib.Path(__file__)
                .resolve()
                .parent.parent
                .joinpath('src', 'nmrsim', 'bin'))
    fs.add_real_directory(test_bin, read_only=False)
    expected_T = test_bin.joinpath('T3.npz')
    assert expected_T.exists()
    fs.remove_object(str(expected_T))
    assert not expected_T.exists()
    T = _tm_cache(3)
    assert T
    assert expected_T.exists()


def test_hamiltonian_dense():
    # GIVEN v and J inputs for the Rioux 3-spin system
    v, J = rioux()
    # Define the spins array for a 3-spin-1/2 system (Rioux's default)
    spins = np.array([0.5, 0.5, 0.5]) 
    # WHEN hamiltonian_dense is used to calculate the Hamiltonian
    H_dense = hamiltonian_dense(v, J, spins=spins)
    # THEN it matches the Hamiltonian result using the old accepted algorithm
    assert np.array_equal(H_dense, HAMILTONIAN_RIOUX)


def test_hamiltonian_sparse():
    # GIVEN v and J inputs for the Rioux 3-spin system
    v, J = rioux()
    spins = np.array([0.5, 0.5, 0.5])
    # WHEN hamiltonian_sparse is used to calculate the Hamiltonian
    H_sparse = hamiltonian_sparse(v, J, spins=spins)
    # THEN it matches the Hamiltonian result using the old accepted algorithm
    assert np.array_equal(H_sparse.todense(), HAMILTONIAN_RIOUX)  # noqa


def test_secondorder_dense():
    # GIVEN v and J inputs for the Rioux 3-spin system
    v, J = rioux()
    # WHEN secondorder_dense is called with those inputs
    result = secondorder_dense(v, J)
    # THEN the resulting spectrum matches that using the old algorithm
    assert np.allclose(result, SPECTRUM_RIOUX)


def test_secondorder_sparse():
    # GIVEN v and J inputs for the Rioux 3-spin system
    v, J = rioux()
    # WHEN secondorder_sparse is called with those inputs
    result = secondorder_sparse(v, J)
    # THEN the resulting spectrum matches that using the old algorithm
    assert np.allclose(result, SPECTRUM_RIOUX)


def test_qm_spinsystem_peaklist():
    # GIVEN v and J inputs
    v, J = rioux()
    # WHEN qm_spinsystem is called with v and J and all possible cache/sparse
    spectrum_TT = qm_spinsystem(v, J, cache=True, sparse=True, normalize=True)
    spectrum_FT = qm_spinsystem(v, J, cache=False, sparse=True, normalize=True)
    spectrum_TF = qm_spinsystem(v, J, cache=True, sparse=False, normalize=True)
    spectrum_FF = qm_spinsystem(v, J, cache=False, sparse=False, normalize=True)
    # THEN they all match the expected result
    for s in [spectrum_TT, spectrum_FT, spectrum_TF, spectrum_FF]:
        assert np.allclose(s, SPECTRUM_RIOUX)


def test_vj_can_be_lists():
    # Allows user to use Python lists as inputs for QM calculations
    # GIVEN v and J arguments as lists
    v = [110.5, 125.5, 200]
    J = [[0, 12, 2],
         [12, 0, 8],
         [2, 8, 0]]
    v_id = id(v)
    j_id = id(J)
    v_original = copy.deepcopy(v)
    j_original = copy.deepcopy(J)
    # WHEN a peaklist calculation is run with v and J as arguments
    peaklist_dense = secondorder_dense(v, J)
    peaklist_sparse = secondorder_sparse(v, J)
    # dense and sparse Hamiltonians apply separate conversions, so check
    # both are consistent:
    assert np.allclose(peaklist_dense, peaklist_sparse)
    # THEN there are no errors
    # AND the original v/J objects have not mutated
    assert v_id == id(v)
    assert j_id == id(J)
    assert v == v_original
    assert J == j_original
