import copy
import pathlib
import numpy as np
import pytest
import sparse
from sparse import COO

from sparse._utils import assert_eq
import scipy

from nmrsim.qm import (_tm_cache, hamiltonian_dense, hamiltonian_sparse,
                       secondorder_dense, secondorder_sparse, _so_sparse,
                       qm_spinsystem)
from tests.accepted_data import HAMILTONIAN_RIOUX, SPECTRUM_RIOUX
from tests.qm_arguments import rioux


def test_so_sparse_creates_files(fs):
    test_bin = (pathlib.Path(__file__)
                .resolve()
                .parent.parent
                .joinpath('nmrsim', 'bin'))
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


def test_tm_cache_creates_file(fs):
    test_bin = (pathlib.Path(__file__)
                .resolve()
                .parent.parent
                .joinpath('nmrsim', 'bin'))
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
    # WHEN hamiltonian_dense is used to calculate the Hamiltonian
    H_dense = hamiltonian_dense(v, J)
    # THEN it matches the Hamiltonian result using the old accepted algorithm
    assert np.array_equal(H_dense, HAMILTONIAN_RIOUX)


def test_hamiltonian_sparse():
    # GIVEN v and J inputs for the Rioux 3-spin system
    v, J = rioux()
    for arg in [v, J]:
        print(arg)
        print(type(arg))
        assert isinstance(arg, (sparse.COO, np.ndarray, scipy.sparse.spmatrix))
    # WHEN hamiltonian_sparse is used to calculate the Hamiltonian
    H_sparse = hamiltonian_sparse(v, J)
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


# tests below are copied from the Sparse project
@pytest.mark.skip()
@pytest.mark.parametrize(
    "a_shape,b_shape,axes",
    [
        [(3, 4), (4, 3), (1, 0)],
        [(3, 4), (4, 3), (0, 1)],
        [(3, 4, 5), (4, 3), (1, 0)],
        [(3, 4), (5, 4, 3), (1, 1)],
        [(3, 4), (5, 4, 3), ((0, 1), (2, 1))],
        [(3, 4), (5, 4, 3), ((1, 0), (1, 2))],
        [(3, 4, 5), (4,), (1, 0)],
        [(4,), (3, 4, 5), (0, 1)],
        [(4,), (4,), (0, 0)],
        [(4,), (4,), 0],
    ],
)
@pytest.mark.parametrize(
    "a_format, b_format",
    [("coo", "coo"), ("coo", "gcxs"), ("gcxs", "coo"), ("gcxs", "gcxs")],
)
def test_tensordot(a_shape, b_shape, axes, a_format, b_format):
    if sparse.__version__ > "0.10.0":
        from sparse._compressed import GCXS
    else:
        assert 1 == 2
    sa = sparse.random(a_shape, density=0.5, format=a_format)
    sb = sparse.random(b_shape, density=0.5, format=b_format)

    a = sa.todense()
    b = sb.todense()

    a_b = np.tensordot(a, b, axes)

    # tests for return_type=None
    sa_sb = sparse.tensordot(sa, sb, axes)
    sa_b = sparse.tensordot(sa, b, axes)
    a_sb = sparse.tensordot(a, sb, axes)

    assert_eq(a_b, sa_sb)
    assert_eq(a_b, sa_b)
    assert_eq(a_b, a_sb)
    if all(isinstance(arr, COO) for arr in [sa, sb]):
        assert isinstance(sa_sb, COO)
    else:
        assert isinstance(sa_sb, GCXS)
    assert isinstance(sa_b, np.ndarray)
    assert isinstance(a_sb, np.ndarray)

    # tests for return_type=COO
    sa_b = sparse.tensordot(sa, b, axes, return_type=COO)
    a_sb = sparse.tensordot(a, sb, axes, return_type=COO)

    assert_eq(a_b, sa_b)
    assert_eq(a_b, a_sb)
    assert isinstance(sa_b, COO)
    assert isinstance(a_sb, COO)

    # tests form return_type=GCXS
    sa_b = sparse.tensordot(sa, b, axes, return_type=GCXS)
    a_sb = sparse.tensordot(a, sb, axes, return_type=GCXS)

    assert_eq(a_b, sa_b)
    assert_eq(a_b, a_sb)
    assert isinstance(sa_b, GCXS)
    assert isinstance(a_sb, GCXS)

    # tests for return_type=np.ndarray
    sa_sb = sparse.tensordot(sa, sb, axes, return_type=np.ndarray)

    assert_eq(a_b, sa_sb)
    assert isinstance(sa_sb, np.ndarray)


def test_tensordot_empty():
    x1 = np.empty((0, 0, 0))
    x2 = np.empty((0, 0, 0))
    s1 = sparse.COO.from_numpy(x1)
    s2 = sparse.COO.from_numpy(x2)

    assert_eq(np.tensordot(x1, x2), sparse.tensordot(s1, s2))


def test_tensordot_valueerror():
    x1 = sparse.COO(np.array(1))
    x2 = sparse.COO(np.array(1))

    with pytest.raises(ValueError):
        x1 @ x2