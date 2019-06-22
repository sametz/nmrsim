import numpy as np

from nmrtools.nmrmath import hamiltonian, nspinspec
from nmrtools.qm import (hamiltonian_dense, hamiltonian_sparse,
                         nspinspec_dense, nspinspec_sparse)
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
