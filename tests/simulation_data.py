import numpy as np


def spin3():
    v = np.array([115, 140, 190])
    J = np.zeros((3, 3))
    J[0, 1] = 6
    J[0, 2] = 12
    J[1, 2] = 3
    J = J + J.T
    return v, J


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


def spin11():
    """Fox() pared down by 1 nuclei, for testing spin-11 system."""

    # memory error crash on 12 nuclei (on my home PC). Hashing out nuclei 10/11 data to reduce it to a
    # 10- or 11-nuclei test case works.
    v = np.array([1.63, 1.63, 2.2, 2.2, 2.5, 2.5, 2.5, 2.5, 2.5,
                  2.5,
                  5.71 #,
                  # 5.77
                  ]) * 400
    J = np.zeros((len(v), len(v)))
    J[0, 1] = -12
    J[0, 2] = 1
    J[0, 3] = 10
    J[0, 8] = 1
    J[0, 9] = 8.5
    J[1, 2] = 10
    J[1, 3] = 1
    J[1, 8] = 8.5
    J[1, 9] = 1
    J[2, 3] = -12
    J[2, 10] = 9
    J[3, 10] = 8.5
    J[4, 5] = -12
    J[4, 6] = 3.5
    J[4, 7] = 7
    # J[4, 11] = 8.5
    J[5, 6] = 7
    J[5, 7] = 3.5
    # J[5, 11] = 7.5
    J[6, 7] = -12
    J = J + J.T
    return v, J


def fox():
    """Joe Fox had an interesting spectrum for cyclooct-4-enone. Here are estimated parameters for this simulation.

    A stress test for a 12-nuclei simulation.
    """

    # memory error crash on 12 nuclei (on my home PC). Hashing out nuclei 10/11 data to reduce it to a
    # 10- or 11-nuclei test case works.
    v = np.array([1.63, 1.63, 2.2, 2.2, 2.5, 2.5, 2.5, 2.5, 2.5,
                  2.5,
                  5.71 #,
                  # 5.77
                  ]) * 400
    J = np.zeros((len(v), len(v)))
    J[0, 1] = -12
    J[0, 2] = 1
    J[0, 3] = 10
    J[0, 8] = 1
    J[0, 9] = 8.5
    J[1, 2] = 10
    J[1, 3] = 1
    J[1, 8] = 8.5
    J[1, 9] = 1
    J[2, 3] = -12
    J[2, 10] = 9
    J[3, 10] = 8.5
    J[4, 5] = -12
    J[4, 6] = 3.5
    J[4, 7] = 7
    # J[4, 11] = 8.5
    J[5, 6] = 7
    J[5, 7] = 3.5
    # J[5, 11] = 7.5
    J[6, 7] = -12
    J = J + J.T
    return v, J


def rioux():
    """http://www.users.csbsju.edu/~frioux/nmr/ABC-NMR-Tensor.pdf

    Returns
    -------

    """
    v = np.array([430.0, 265.0, 300.0])
    J = np.zeros((3, 3))
    J[0, 1] = 7.0
    J[0, 2] = 15.0
    J[1, 2] = 1.50
    J = J + J.T
    return v, J
