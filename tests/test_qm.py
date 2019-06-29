import pathlib
import numpy as np

from nmrtools.qm import (cache_tm, hamiltonian_dense, hamiltonian_sparse,
                         nspinspec_dense, nspinspec_sparse, so_sparse,
                         spectrum)
from tests.accepted_data import HAMILTONIAN_RIOUX, SPECTRUM_RIOUX
from tests.simulation_data import rioux


def test_so_sparse_creates_files(fs):
    test_bin = (pathlib.Path(__file__)
                .resolve()
                .parent.parent
                .joinpath('nmrtools', 'bin'))
    fs.create_dir(test_bin)
    expected_Lz = test_bin.joinpath('Lz3.npz')
    expected_Lproduct = test_bin.joinpath('Lproduct3.npz')
    assert not expected_Lz.exists()
    assert not expected_Lproduct.exists()
    Lz, Lproduct = so_sparse(3)  # noqa
    assert expected_Lz.exists()
    assert expected_Lproduct.exists()


def test_cache_tm_creates_file(fs):
    test_bin = (pathlib.Path(__file__)
                .resolve()
                .parent.parent
                .joinpath('nmrtools', 'bin'))
    fs.create_dir(test_bin)
    expected_T = test_bin.joinpath('T3.npz')
    assert not expected_T.exists()
    T = cache_tm(3)
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
    # WHEN hamiltonian_dense is used to calculate the Hamiltonian
    H_sparse = hamiltonian_sparse(v, J)
    # THEN it matches the Hamiltonian result using the old accepted algorithm
    assert np.array_equal(H_sparse.todense(), HAMILTONIAN_RIOUX)  # noqa


def test_nspinspec_dense():
    # GIVEN v and J inputs for the Rioux 3-spin system
    v, J = rioux()
    # WHEN nspinspec_dense is called with those inputs
    result = nspinspec_dense(v, J)
    # THEN the resulting spectrum matches that using the old algorithm
    assert np.allclose(result, SPECTRUM_RIOUX)


def test_nspinspec_sparse():
    # GIVEN v and J inputs for the Rioux 3-spin system
    v, J = rioux()
    # WHEN nspinspec_sparse is called with those inputs
    result = nspinspec_sparse(v, J)
    # THEN the resulting spectrum matches that using the old algorithm
    assert np.allclose(result, SPECTRUM_RIOUX)


def test_nspinspec_spectrum():
    # GIVEN v and J inputs
    v, J = rioux()
    # WHEN spectrum is called with v and J and all possible cache/sparse
    spectrum_TT = spectrum(v, J, cache=True, sparse=True, normalize=True)
    spectrum_FT = spectrum(v, J, cache=False, sparse=True, normalize=True)
    spectrum_TF = spectrum(v, J, cache=True, sparse=False, normalize=True)
    spectrum_FF = spectrum(v, J, cache=False, sparse=False, normalize=True)
    # THEN they all match the expected result
    for s in [spectrum_TT, spectrum_FT, spectrum_TF, spectrum_FF]:
        assert np.allclose(s, SPECTRUM_RIOUX)
