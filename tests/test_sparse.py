"""Sparse 0.11.0 broke nmrsim. This test suite contains smoke tests that
third-party modules (sparse/numba/numpy) are still behaving as intended.

When this test suite was created, the scipy version of tensordot was failing,
even for sparse 0.10.0.

Because sparse is not Version 1 yet, tests for its functions are a good idea.
"""
import numpy as np
import pytest
import sparse
from sparse import COO
from sparse._utils import assert_eq  # noqa

from .qm_arguments import spin2

pytestmark = pytest.mark.skip(
    "tests of broken sparse 0.11/0.12 are being retained in case there's a regression in the future with sparse>0.13.0"
)


# The following three tests are copied over from the sparse project.
# They pass in sparse >=0.11.0
@pytest.mark.skipif(sparse.__version__ < "0.11.0",
                    reason="This test from the sparse library requires v0.11 or greater")
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
    from sparse._compressed import GCXS
    # else:
    #     assert 1 == 2
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


def test_numpy_tensordot():
    """A stepping stone to finding the problem with sparse 0.11.0's tensordot.

    This is taken from the Numpy tensordot documentation ("traditional" example):
    https://numpy.org/doc/stable/reference/generated/numpy.tensordot.html
    """
    # GIVEN accepted numpy doc example for tensordot
    a = np.arange(60.).reshape(3, 4, 5)  # noqa
    b = np.arange(24.).reshape(4, 3, 2)  # noqa
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


@pytest.mark.xfail(reason="sparse bug")
def test_sparse_tensordot():
    """Tests to see if sparse_tensordot can pass the numpy tensordot test above.
    Fails for all sparse versions. See:
    https://github.com/pydata/sparse/issues/493

    nmrsim luckily always worked prior to sparse v0.11 because
    np arrays were never used as inputs for sparse.tensordot,
    so this bug with sparse was never encountered.
    """
    # GIVEN accepted numpy doc example for tensordot
    a = np.arange(60.).reshape(3, 4, 5)  # noqa
    b = np.arange(24.).reshape(4, 3, 2)  # noqa
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


@pytest.mark.xfail(reason="sparse bug will cause second parametrized test to fail")
@pytest.mark.parametrize("tensordot", [np.tensordot, sparse.tensordot])
def test_twospin_v(tensordot):
    """A test using one of the sub-calculations for the Hamiltonian
    to see if sparse.tensordot can do the same calculation as np.tensordot.

    Redundant with the above two tests.
    """
    v, J = spin2()
    Lz = np.array(
        [[[0.5 + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
          [0. + 0.j, 0.5 + 0.j, 0. + 0.j, 0. + 0.j],
          [0. + 0.j, 0. + 0.j, -0.5 + 0.j, -0. + 0.j],
          [0. + 0.j, 0. + 0.j, -0. + 0.j, -0.5 + 0.j]],

         [[0.5 + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
          [0. + 0.j, -0.5 + 0.j, 0. + 0.j, -0. + 0.j],
          [0. + 0.j, 0. + 0.j, 0.5 + 0.j, 0. + 0.j],
          [0. + 0.j, -0. + 0.j, 0. + 0.j, -0.5 + 0.j]]]
    )

    H = tensordot(v, Lz, axes=1)
    assert np.allclose(H,
                       np.array(
                           [[15. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                            [0. + 0.j, -5. + 0.j, 0. + 0.j, 0. + 0.j],
                            [0. + 0.j, 0. + 0.j, 5. + 0.j, 0. + 0.j],
                            [0. + 0.j, 0. + 0.j, 0. + 0.j, -15. + 0.j]]
                       ))


def test_twospin_v_coo():
    """Tests to see if sparse.tensordot works with COO arrays instead.

    This test passes for sparse <=0.10.0, but fails for >=0.11.0,
    and generates the same nmrsim error that was observed when sparse was upgraded.
    """
    v, J = spin2()
    Lz = np.array(
        [[[0.5 + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
          [0. + 0.j, 0.5 + 0.j, 0. + 0.j, 0. + 0.j],
          [0. + 0.j, 0. + 0.j, -0.5 + 0.j, -0. + 0.j],
          [0. + 0.j, 0. + 0.j, -0. + 0.j, -0.5 + 0.j]],

         [[0.5 + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
          [0. + 0.j, -0.5 + 0.j, 0. + 0.j, -0. + 0.j],
          [0. + 0.j, 0. + 0.j, 0.5 + 0.j, 0. + 0.j],
          [0. + 0.j, -0. + 0.j, 0. + 0.j, -0.5 + 0.j]]]
    )
    v_coo = sparse.COO(v)
    Lz_coo = sparse.COO(Lz)
    H = sparse.tensordot(v_coo, Lz_coo, axes=1)
    assert np.allclose(
        H.todense(),
        np.array(
            [[15. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
             [0. + 0.j, -5. + 0.j, 0. + 0.j, 0. + 0.j],
             [0. + 0.j, 0. + 0.j, 5. + 0.j, 0. + 0.j],
             [0. + 0.j, 0. + 0.j, 0. + 0.j, -15. + 0.j]]
        ))


# v2, J2 = spin2()

v2 = np.array([10.0, 20.0])
J2 = None

Lz2 = np.array(
    [[[0.5 + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
      [0. + 0.j, 0.5 + 0.j, 0. + 0.j, 0. + 0.j],
      [0. + 0.j, 0. + 0.j, -0.5 + 0.j, -0. + 0.j],
      [0. + 0.j, 0. + 0.j, -0. + 0.j, -0.5 + 0.j]],

     [[0.5 + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
      [0. + 0.j, -0.5 + 0.j, 0. + 0.j, -0. + 0.j],
      [0. + 0.j, 0. + 0.j, 0.5 + 0.j, 0. + 0.j],
      [0. + 0.j, -0. + 0.j, 0. + 0.j, -0.5 + 0.j]]]
)
v2_coo = sparse.COO(v2)
Lz2_coo = sparse.COO(Lz2)


@pytest.mark.xfail(reason="sparse bug will cause double-ndarray case to fail")
@pytest.mark.parametrize("v, Lz", [
    (v2, Lz2),
    (v2_coo, Lz2),
    (v2, Lz2_coo),
    (v2_coo, Lz2_coo)
])
def test_twospin_v_Lz(v, Lz):
    """Tests to see what combinations of argument types cause which errors.

    Different errors for: both ndarray; one ndarray and one COO; both COO.
    See: https://github.com/pydata/sparse/issues/499
    """
    H = sparse.tensordot(v, Lz, axes=1)
    if not isinstance(H, np.ndarray):
        H = H.todense()
    assert np.allclose(H,
                       np.array(
                           [[15. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                            [0. + 0.j, -5. + 0.j, 0. + 0.j, 0. + 0.j],
                            [0. + 0.j, 0. + 0.j, 5. + 0.j, 0. + 0.j],
                            [0. + 0.j, 0. + 0.j, 0. + 0.j, -15. + 0.j]]
                       ))


a = np.arange(60.).reshape(3, 4, 5)  # noqa
b = np.arange(24.).reshape(4, 3, 2)  # noqa
a_coo = sparse.COO(a)
b_coo = sparse.COO(b)


@pytest.mark.xfail(reason="sparse bug will cause double ndarray case to fail")
@pytest.mark.parametrize("a, b", [
    (a, b),
    (a_coo, b),
    (a, b_coo),
    (a_coo, b_coo)
])
def test_sparse_tensordot_inputs(a, b):
    """Tests the numpy "typical" example versus different argument type
    combinations.

    As in other tests, double-ndarray gives an error across all versions of
    sparse; one ndarray and one COO gives one numba error; and both COO gives
    another numba error. See: https://github.com/pydata/sparse/issues/499
    """
    # GIVEN accepted numpy doc example for tensordot

    # WHEN the sparse version of tensordot is performed
    c = sparse.tensordot(a, b, axes=([1, 0], [0, 1]))

    # THEN the expected results are seen
    if not isinstance(c, np.ndarray):
        c = c.todense()
    assert c.shape == (5, 2)
    assert np.array_equal(c,
                          np.array(
                              [[4400., 4730.],
                               [4532., 4874.],
                               [4664., 5018.],
                               [4796., 5162.],
                               [4928., 5306.]]
                          ))
