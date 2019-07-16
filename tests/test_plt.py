import numpy as np
import pytest

from nmrtools.plt import add_signals, mplplot, mplplot_stick, mplplot_lineshape
from tests.accepted_data import ADD_SIGNALS_DATASET
from tests.testdata import TWOSPIN_SLOW

# TODO: currently plot routines are returning x, y data. Change to returning
# the plt object, and introspect it for this x, y data if needed for tests.


def test_add_signals():
    """
    Tests that current nmrplot.add_signals output agrees with an accepted
    dataset.
    """
    # test was written before normalization of height vs width was built into
    #  lorentz(). Fudge-factor added to scale old accepted data.
    x = np.linspace(390, 410, 200)
    doublet = [(399, 1), (401, 1)]
    y = add_signals(x, doublet, 1)
    X = np.array([x for x, _ in ADD_SIGNALS_DATASET])
    Y = np.array([y * 0.5 for _, y in ADD_SIGNALS_DATASET])

    assert np.array_equal(x, X)
    assert np.array_equal(y, Y)


def test_mplplot_defaults():
    doublet = [(399, 1), (401, 1)]
    x, y = mplplot(doublet)
    assert len(x) == 800
    assert x[0] == 399 - 50
    assert x[-1] == 401 + 50


@pytest.mark.parametrize('limits', ['foo', (1,), (1, 'foo'), (1, 2, 3)])
def test_mplplot_limit_error(limits):
    doublet = [(399, 1), (401, 1)]
    with pytest.raises((AttributeError, TypeError, ValueError)):
        mplplot(doublet, limits=limits)  # noqa


def test_mplplot():
    doublet = [(399, 1), (401, 1)]
    limits = (410, 390)  # deliberately opposite order
    x, y = mplplot(doublet, points=200, limits=limits)
    # test was written before normalization of height vs width was built into
    #  lorentz(). Fudge-factor added to scale old accepted data.
    y = y * 2
    print(np.array([x, y]).T)
    print(np.array(ADD_SIGNALS_DATASET))
    assert np.allclose(np.array([x, y]).T, np.array(ADD_SIGNALS_DATASET))


def test_mplplot_stick_defaults():
    doublet = [(399, 1), (401, 1)]
    x, y = mplplot_stick(doublet)
    assert len(x) == 4
    assert x[-2] == 399 - 50
    assert x[-1] == 401 + 50


@pytest.mark.parametrize('limits', ['foo', (1,), (1, 'foo'), (1, 2, 3)])
def test_mplplot_stick_limit_error(limits):
    doublet = [(399, 1), (401, 1)]
    with pytest.raises((AttributeError, TypeError, ValueError)):
        mplplot_stick(doublet, limits=limits)


def test_mplplot_stick():
    doublet = [(399, 1), (401, 1)]
    limits = (410, 390)  # deliberately opposite order
    x, y = mplplot_stick(doublet, limits=limits)
    result_xy = list(zip(x, y))
    expected_xy = [(399, 1), (401, 1), (390, 0.001), (410, 0.001)]
    assert np.allclose(np.array(result_xy), np.array(expected_xy))


def test_mpl_lineshape_defaults():
    # Currently a human-checked test.
    # TODO: have plt object returned from mplplot_lineshape and introspect
    x, y = TWOSPIN_SLOW
    mplplot_lineshape(x, y, limits=(100, 200))
    # Check that x goes from 200 to 100 (i.e. is also reversed)
    assert 1 == 1


def test_mpl_lineshape_swaps_limits():
    # Currently a human-checked test
    x, y = TWOSPIN_SLOW
    mplplot_lineshape(x, y, limits=(200, 100))
    # Check that x goes from 200 to 100 (i.e. is also reversed)
    assert 1 == 1


@pytest.mark.parametrize('limits', ['foo', (1, ), (1, 'foo'), (1, 2, 3)])
def test_mpl_lineshape_limit_error(limits):
    x, y = TWOSPIN_SLOW
    with pytest.raises((AttributeError, TypeError, ValueError)):
        mplplot_lineshape(x, y, limits=limits)


def test_mpl_lineshape():
    # Currently a human-checked test.
    # TODO: have plt object returned from mplplot_lineshape and introspect
    x, y = TWOSPIN_SLOW
    y_max = max(y) * 1.1
    mplplot_lineshape(x, y, y_min=0, y_max=y_max)
    # check that y goes from 0 to max+10%
    assert 1 == 1
