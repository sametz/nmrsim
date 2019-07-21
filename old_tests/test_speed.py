import time

from old_modules.nmrmath import *


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


def spin10():
    # Based on cyclohept-2-enone. See Taber's "Organic Spectroscopic
    # Structure Determination, pp. 117-118.
    d = np.array([1.7, 1.7, 1.9, 1.9, 2.4, 2.4, 2.6, 2.6, 6.0, 6.6])
    v = d * 300  # MHz
    J = np.zeros((10, 10))
    J[0, 1] = -12
    J[0, 2] = 7
    J[0, 3] = 7
    J[0, 4] = 7
    J[0, 5] = 7
    # J[0, 6] = 0
    # J[0, 7] = 0
    J[1, 2] = 7
    J[1, 3] = 7
    J[1, 4] = 7
    J[1, 5] = 7
    # J[1, 6] = 0
    # J[1, 7] = 3
    J[2, 3] = -12
    # J[2, 4] = 0
    # J[2, 5] = 3
    J[2, 6] = 6.4
    J[2, 7] = 6.4
    # J[2, 8] = 7.5
    # J[3, 4] = 0
    # J[3, 5] = 5
    J[3, 6] = 6.4
    J[3, 7] = 6.4
    # J[3, 8] = 7.5
    J[4, 5] = -12
    # J[4, 6] = 0
    # J[4, 7] = 0
    J[4, 9] = 5.8
    # J[4, 10] = 7.5
    # J[5, 6] = 0
    # J[5, 7] = 0
    J[5, 9] = 5.8
    # J[5, 10] = 7.5
    # J[6, 7] = 1
    # J[6, 8] = 15.8
    # J[7, 8] = 10.5
    J[8, 9] = 11.5
    # J[9, 10] = 11.5
    J = J + J.T
    return v, J


def spin11():
    # Based on 2Z-octa-2, 7-dienoate. See Taber's "Organic Spectroscopic
    # Structure Determination, pp. 119-120.
    d = np.array([1.7, 1.7, 2.2, 2.2, 2.5, 2.5, 4.89, 5.01, 5.77, 5.68, 6.14])
    v = d * 300  # MHz
    J = np.zeros((11, 11))
    J[0, 1] = -12
    J[0, 2] = 7
    J[0, 3] = 7
    J[0, 4] = 7
    J[0, 5] = 7
    # J[0, 6] = 0
    # J[0, 7] = 0
    J[1, 2] = 7
    J[1, 3] = 7
    J[1, 4] = 7
    J[1, 5] = 7
    # J[1, 6] = 0
    # J[1, 7] = 3
    J[2, 3] = -12
    # J[2, 4] = 0
    J[2, 5] = 3
    # J[2, 6] = 0
    # J[2, 7] = 0
    J[2, 8] = 7.5
    # J[3, 4] = 0
    # J[3, 5] = 5
    # J[3, 6] = 0
    # J[3, 7] = 0
    J[3, 8] = 7.5
    J[4, 5] = -12
    # J[4, 6] = 0
    # J[4, 7] = 0
    J[4, 10] = 7.5
    # J[5, 6] = 0
    # J[5, 7] = 0
    J[5, 10] = 7.5
    J[6, 7] = 1
    J[6, 8] = 15.8
    J[7, 8] = 10.5
    J[9, 10] = 11.5
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


def test_sparse_hamiltonian_speed():
    v, J = spin8()
    start1 = time.time()
    H = sparse_hamiltonian(v, J)
    end1 = time.time()
    print('time: ', end1 - start1)
    assert H is not None