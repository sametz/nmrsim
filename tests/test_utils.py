import pytest

from nmrtools._utils import is_integer, is_decimal_fraction, is_tuple_of_two_numbers, is_positive


def test_is_integer():
    assert 2 == is_integer(2)
    with pytest.raises(TypeError):
        assert 2.0 == is_integer(2.0)


@pytest.mark.parametrize('n', [0, 0.5, 1])
def test_is_decimal_fraction(n):
    assert n is is_decimal_fraction(n)


@pytest.mark.parametrize('n', [-0.1, 1.1, '0.5', 'foo'])
def test_is_decimal_fraction_raises(n):
    with pytest.raises((ValueError, TypeError)):
        is_decimal_fraction(n)


def test_is_tuple_of_two_numbers():
    assert (1, 2.0) == is_tuple_of_two_numbers((1, 2.0))


@pytest.mark.parametrize('t', [2, (2,), ('foo', 'bar')])
def test_is_tuple_of_two_numbers_raises(t):
    with pytest.raises((TypeError, ValueError)):
        assert t == is_tuple_of_two_numbers(t)


@pytest.mark.parametrize('good, bad', [(1, -1), (0.1, -0.1), (0.000001, 0)])
def test_is_positive(good, bad):
    assert good == is_positive(good)
    with pytest.raises(ValueError):
        assert bad == is_positive(bad)