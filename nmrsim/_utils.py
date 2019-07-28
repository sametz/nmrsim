"""Utilities used elsewhere in the nmrsim package."""

import numbers


def is_number(n):
    if isinstance(n, numbers.Real):
        return n
    else:
        raise TypeError('Must be a real number.')


def is_integer(n):
    if isinstance(n, numbers.Integral):
        return n
    else:
        raise TypeError('Must be an integer.')


def is_decimal_fraction(n):
    if 0 <= is_number(n) <= 1:
        return n
    else:
        raise ValueError('Number must be >=0 and <=1')


def is_tuple_of_two_numbers(t):
    m, n = t
    return is_number(m), is_number(n)


def is_positive(n):
    if n > 0:
        return n
    raise ValueError('Number must be positive.')


def low_high(t):
    two_numbers = is_tuple_of_two_numbers(t)
    return min(two_numbers), max(two_numbers)
