from simsignal_optimize import *
import numpy as np
from scipy.sparse import lil_matrix
from scipy.linalg import eigh


def test_nlist():
    assert nlist(3) == [[], [], []]
    assert nlist(1) == [[]]
    assert nlist(0) == []


def test_popcount():
    assert popcount(0) == 0
    for n in range(1, 10):
        assert popcount(2 ** n) == 1


def test_is_allowed():
    assert is_allowed(0, 0) == False
    assert is_allowed(0, 3) == False
    assert is_allowed(0, 1) == True
    assert is_allowed(255, 254) == True


def test_transition_matrix():
    T = np.array([
        [0, 1, 1, 0, 1, 0, 0, 0],
        [1, 0, 0, 1, 0, 1, 0, 0],
        [1, 0, 0, 1, 0, 0, 1, 0],
        [0, 1, 1, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1, 1, 0],
        [0, 1, 0, 0, 1, 0, 0, 1],
        [0, 0, 1, 0, 1, 0, 0, 1],
        [0, 0, 0, 1, 0, 1, 1, 0]
    ])
    np.testing.assert_array_equal(T,  transition_matrix(8).toarray())


def test_hamiltonian():
    freqlist = [430, 265, 300]
    J = lil_matrix((3, 3))
    J[0, 1] = 7
    J[0, 2] = 15
    J[1, 2] = 1.5
    J = J + J.T
    # print(freqlist)
    # print(J.todense())
    v = [-491.625, -230.963, -200.306, -72.106, 61.883, 195.524, 234.217,
         503.375]
    H = hamiltonian(freqlist, J)
    # print(H).real
    eigvals = eigh(H.todense(), eigvals_only=True)
    np.testing.assert_array_equal(eigvals, sorted(eigvals))
    np.testing.assert_array_almost_equal(eigvals, v, decimal=3)


def test_simsignals():
    refspec = [(260.66152857482973, 0.92044386594717353),
               (262.18930344673686, 0.99503587544800565),
               (267.62991550888137, 0.99421657034922251),
               (269.15769038078849, 1.0902494458059944),
               (291.31911366903159, 0.91527406734942929),
               (292.84688854093866, 0.85524357901564929),
               (306.32295186307283, 1.1700594579603014),
               (307.85072673497996, 1.0594650613364776),
               (419.51935775613867, 1.1642820667033968),
               (426.48774469019031, 1.0651876493998582),
               (434.52319595017991, 0.92017735031402692),
               (441.49158288423155, 0.85028549285752886)]
    freqlist = [430, 265, 300]
    J = lil_matrix((3, 3))
    J[0, 1] = 7
    J[0, 2] = 15
    J[1, 2] = 1.5
    J = J + J.T
    H = hamiltonian(freqlist, J)
    testspec = sorted(simsignals(H, 3))
    np.testing.assert_array_almost_equal(testspec, refspec, decimal=2)
    testss2 = sorted(ss2(H, 3))
    np.testing.assert_array_almost_equal(testss2, refspec, decimal=2)
    testss3 = sorted(ss3(H, 3))
    np.testing.assert_array_almost_equal(testss3, refspec, decimal=2)

# def test_derp():
#     assert 1 + 1 == 3


