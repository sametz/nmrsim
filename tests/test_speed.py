import time

import numpy as np
import pytest

from nmrtools.nmrmath import *


def spin8():
    v = np.array([85, 120, 160, 185, 205, 215, 235, 260])
    # Note: previous version used a scipy lil_matrix for J, but hamiltonian
    # gave a dimension mismatch. Changed to a np matrix and worked.
    J = np.zeros((8, 8))
    J[0, 1] = -12
    J[0, 2] = 6
    J[0, 3] = 2
    # J[0, 4] = 0
    # J[0, 5] = 0
    # J[0, 6] = 0
    # J[0, 7] = 0
    # J[1, 2] = 0
    # J[1, 3] = 0
    J[1, 4] = 14
    # J[1, 5] = 0
    # J[1, 6] = 0
    J[1, 7] = 3
    # J[2, 3] = 0
    # J[2, 4] = 0
    J[2, 5] = 3
    # J[2, 6] = 0
    # J[2, 7] = 0
    # J[3, 4] = 0
    J[3, 5] = 5
    # J[3, 6] = 0
    # J[3, 7] = 0
    J[4, 5] = 2
    # J[4, 6] = 0
    # J[4, 7] = 0
    # J[5, 6] = 0
    # J[5, 7] = 0
    J[6, 7] = 12
    J = J + J.T
    return v, J


def test_8spin():
    v, J = spin8()
    start1 = time.time()
    H1 = new_hamiltonian(v, J)
    end1 = time.time()

    start2 = time.time()
    H2 = new_hamiltonian(v, J)
    end2 = time.time()

    print('first run t: ', end1 - start1)
    print('second run t: ', end2 - start2)
    print((end1 - start1) / (end2 - start2), ' speedup')
    np.testing.assert_array_almost_equal(H1, H2, decimal=3)


def test_new_hamiltonian_speed():
    v, J = spin8()
    start1 = time.time()
    H = new_hamiltonian(v, J)
    end1 = time.time()
    print('time: ', end1 - start1)
    assert H is not None


def test_hamiltonian_speed():
    v, J = spin8()
    start1 = time.time()
    H = hamiltonian(v, J)
    end1 = time.time()
    print('time: ', end1 - start1)
    assert H is not None
