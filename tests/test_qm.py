import numpy as np

from nmrtools.nmrmath import hamiltonian, nspinspec
from nmrtools.qm import (hamiltonian_dense, hamiltonian_sparse,
                         nspinspec_dense, nspinspec_sparse, spectrum)
from tests.simulation_data import rioux

H_RIOUX = hamiltonian(*rioux())
SPECTRUM_RIOUX = nspinspec(*rioux())

def test_hamiltonian_dense():
    # GIVEN v and J inputs for the Rioux 3-spin system
    v, J = rioux()
    # WHEN hamiltonian_dense is used to calculate the Hamiltonian
    H_dense = hamiltonian_dense(v, J)
    # THEN it matches the Hamiltonian result using the old accepted algorithm
    assert np.array_equal(H_dense, H_RIOUX)


def test_hamiltonian_sparse():
    # GIVEN v and J inputs for the Rioux 3-spin system
    v, J = rioux()
    # WHEN hamiltonian_dense is used to calculate the Hamiltonian
    H_sparse = hamiltonian_sparse(v, J)
    # THEN it matches the Hamiltonian result using the old accepted algorithm
    assert np.array_equal(H_sparse.todense(), H_RIOUX)


def test_nspinspec_dense():
    # GIVEN v and J inputs for the Rioux 3-spin system
    v, J = rioux()
    # WHEN nspinspec_dense is called with those inputs
    spectrum = nspinspec_dense(v, J)
    # THEN the resulting spectrum matches that using the old algorithm
    assert np.allclose(spectrum, SPECTRUM_RIOUX)


def test_nspinspec_sparse():
    # GIVEN v and J inputs for the Rioux 3-spin system
    v, J = rioux()
    # WHEN nspinspec_sparse is called with those inputs
    spectrum = nspinspec_sparse(v, J)
    # THEN the resulting spectrum matches that using the old algorithm
    assert np.allclose(spectrum, SPECTRUM_RIOUX)


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
