"""Sparse 0.11.0 broke nmrsim. This test suite contains smoke tests that
third-party modules (sparse/numba/numpy) are still behaving as intended.

When this test suite was created, the scipy version of tensordot was failing,
even for sparse 0.10.0.

Because sparse is not Version 1 yet, tests for its functions are a good idea.
"""
import numpy as np
import sparse


def test_numpy_tensordot():
    """A stepping stone to finding the problem with sparse 0.11.0's tensordot"""
    # GIVEN accepted numpy doc example for tensordot
    a = np.arange(60.).reshape(3, 4, 5)
    b = np.arange(24.).reshape(4, 3, 2)
    # WHEN the following tensordot is performed
    c = np.tensordot(a, b, axes=([1, 0], [0, 1]))
    # THEN the expected results are seen
    assert c.shape == (5, 2)
    assert np.array_equal(c,
                          np.array(
                              [[4400., 4730.],
                               [4532., 4874.],
                               [4664., 5018.],
                               [4796., 5162.],
                               [4928., 5306.]]
                          ))


def test_sparse_tensordot():
    # GIVEN accepted numpy doc example for tensordot
    a = np.arange(60.).reshape(3, 4, 5)
    b = np.arange(24.).reshape(4, 3, 2)
    # WHEN the sparse version of tensordot is performed
    c_sparse = sparse.tensordot(a, b, axes=([1, 0], [0, 1]))
    # THEN the expected results are seen
    c = c_sparse.todense()
    assert c.shape == (5, 2)
    assert np.array_equal(c,
                          np.array(
                              [[4400., 4730.],
                               [4532., 4874.],
                               [4664., 5018.],
                               [4796., 5162.],
                               [4928., 5306.]]
                          ))
