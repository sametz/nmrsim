import builtins
import io
import os
import time

from nmrtools.nmrmath import *
import numpy as np
import pytest


# monkeypatching: https://stackoverflow.com/questions/51737334
def patch_open(open_func, files):
    def open_patched(path, mode='r', buffering=-1, encoding=None, errors=None,
                     newline=None, closefd=True, opener=None):
        if 'w' in mode and not os.path.isfile(path):
            files.append(path)
        return open_func(path, mode=mode, buffering=buffering,
                         encoding=encoding, errors=errors, newline=newline,
                         closefd=closefd, opener=opener)
    return open_patched


@pytest.fixture(autouse=True)
def cleanup_files(monkeypatch):
    files = []
    monkeypatch.setattr(builtins, 'open', patch_open(builtins.open, files))
    monkeypatch.setattr(io, 'open', patch_open(io.open, files))
    yield
    for file in files:
        os.remove(file)


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


def test_h_save_and_load():
    """Check that a Hamiltonian matrix can be pickled and unpickled."""
    # GIVEN a numpy matrix of integers
    test_h = np.random.rand(8, 8)

    # WHEN the matrix is pickled and reloaded
    h_save(3, test_h)
    result_h = h_load(3)

    # THEN the reloaded matrix matches the original matrix
    np.testing.assert_array_equal(test_h, result_h)


def test_new_h():
    freqlist = [430, 265, 300]
    freqarray = np.array(freqlist)
    J = np.zeros((3, 3))
    J[0, 1] = 7
    J[0, 2] = 15
    J[1, 2] = 1.5
    J = J + J.T
    # print(freqlist)
    # print(J.todense())
    v = [-491.625, -230.963, -200.306, -72.106, 61.883, 195.524, 234.217,
         503.375]
    H = new_h(freqarray, J)
    # print(H).real
    eigvals = np.linalg.eigvals(H)
    eigvals.sort()
    np.testing.assert_array_equal(eigvals, sorted(eigvals))
    np.testing.assert_array_almost_equal(eigvals, v, decimal=3)


def test_8spin():
    v, J = spin8()
    start1 = time.time()
    H1 = new_h(v, J)
    end1 = time.time()

    start2 = time.time()
    H2 = new_h(v, J)
    end2 = time.time()

    print('first run t: ', end1 - start1)
    print('second run t: ', end2 - start2)
    print((end1 - start1) / (end2 - start2), ' speedup')
    np.testing.assert_array_almost_equal(H1, H2, decimal=3)
