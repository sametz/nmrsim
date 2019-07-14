""""Functions for calculating first-order spectra will appear here."""
import abc
import numbers

import numpy as np

from nmrtools.math import reduce_peaks


def doublet(plist, J):
    """
    Applies a *J* coupling to each signal in a list of (frequency, intensity)
    signals, creating two half-intensity signals at +/- *J*/2.

    Parameters
    ---------
    plist : [(float, float)...]
        a list of (frequency{Hz}, intensity) tuples.
    J : float
        The coupling constant in Hz.

    Returns
    -------
    [(float, float)...]
        a list of (frequency, intensity) tuples.
    """
    res = []
    for v, i in plist:
        res.append((v - J / 2, i / 2))
        res.append((v + J / 2, i / 2))
    return res


def multiplet(signal, couplings):
    """
    Splits a set of signals into first-order multiplets.

    Parameters
    ---------
    signal : (float, float)
        a (frequency{Hz}, intensity) tuple;
    couplings : [(float, int)...]
        A list of (*J*, # of nuclei) tuples. The order of the tuples in
        couplings does not matter.
        e.g. to split a signal into a *dt, J* = 8, 5 Hz, use:
        ``couplings = [(8, 2), (5, 3)]``

    Returns
    -------
    [(float, float)...]
        a plist of the multiplet that results from splitting the plist
        signal(s) by each J.

    """
    res = [signal]
    for coupling in couplings:
        for i in range(coupling[1]):
            res = doublet(res, coupling[0])
    return res


def first_order(signal, couplings):  # Wa, RightHz, WdthHz not implemented yet
    """
    Splits a signal into a first-order multiplet.

    Parameters
    ---------
    signal : (float, float)
        a (frequency, intensity) tuple.
    couplings : [(float, int)...]
        a list of (J, # of nuclei) tuples.

    Returns
    -------
    [(float, float)...}
        a plist-style spectrum (list of (frequency, intensity) tuples)
    """
    return reduce_peaks(sorted(multiplet(signal, couplings)))


def first_order_spin_system(v, J):
    """
    Can create a first-order peaklist from the same v/J arguments used for
    qm calculations.

    Parameters
    ----------
    v : array-like [float...]
        an array of frequencies
    J : 2D array-like (square)
        a matrix of J coupling constants

    Returns
    -------

    """
    result = []
    for i, v_ in enumerate(v):
        couplings = ((j, 1) for j in J[i] if j != 0)
        signal = multiplet((v_, 1), couplings)
        result += signal
    return reduce_peaks(sorted(result))


# work in progress
# def spectrum_from_signals(signals):
#     spectrum = []
#     for signal in signals:
#         spectrum += multiplet(signals)
#     return reduce_peaks(spectrum)


# https://realpython.com/python-type-checking/
# https://blog.florimond.dev/reconciling-dataclasses-and-properties-in-python


class AutoStorage:
    """A descriptor class for managing getters and setters.

    See L. Ramalho, "Fluent Python", Ch. 20.
    """
    """Downside of this method is non-informative docstrings for attributes.
    Consider another implementation.
    """
    __counter = 0

    def __init__(self):
        cls = self.__class__
        prefix = cls.__name__
        index = cls.__counter
        self.storage_name = '_{}#{}'.format(prefix, index)
        cls.__counter += 1

    def __get__(self, instance, owner):
        if instance is None:
            return self
        else:
            return getattr(instance, self.storage_name)

    def __set__(self, instance, value):
        setattr(instance, self.storage_name, value)


class Validated(abc.ABC, AutoStorage):
    """An abstract subclass of AutoStorage that provides validation for
    setters.

    See L. Ramalho, "Fluent Python", Ch. 20.
    """
    def __set__(self, instance, value):
        value = self.validate(instance, value)
        super().__set__(instance, value)

    @abc.abstractmethod
    def validate(self, instance, value):
        """return validated value or raise ValueError"""


class Number(Validated):
    """test that a value is a real number"""

    def validate(self, instance, value):
        if not isinstance(value, numbers.Real):
            raise TypeError('value must be a real number')
        return value


class Couplings(Validated):
    """test that J resembles an array of number pairs (for each J/# of nuclei
    entry.
    """
    def validate(self, instance, value):
        testarray = np.array(value)
        if testarray.shape == (0,):  # empty list
            return value
        if len(testarray.shape) != 2:
            print('first entry in array is: ', value[0])
            raise TypeError('J should be 2D array-like')

        _, n = testarray.shape
        if n != 2:
            raise ValueError('J should have a second dimension of 2 '
                             'for J value, # of nuclei.')
        return value


class Multiplet:
    """A representation of a first-order multiplet.

    """
    v = Number()
    I = Number()
    J = Couplings()

    def __init__(self, v, I, J):
        self.v = v
        self.I = I
        self.J = J
        self._peaklist = first_order((v, I), J)

    def __mul__(self, scalar):
        if isinstance(scalar, numbers.Real):
            return Multiplet(self.v, self.I * scalar, self.J)
        else:
            return NotImplemented

    def __imul__(self, scalar):
        if isinstance(scalar, numbers.Real):
            self.I = self.I * scalar
            return self
        else:
            return NotImplemented

    def _refresh(self):
        self._peaklist = first_order((self.v, self.I), self.J)

    def peaklist(self):
        """Return a list of (frequency, intensity) signals."""
        self._refresh()
        return self._peaklist


""" API ideas:
    firstorder.multiplet for one signal
    firstorder.spectrum for multiple signals
    firstorder.spinsystem takes qm-style v and J arguments, and returns a
        firstorder.spectrum?
    thinking bigger picture: an nmrtools.spectrum class that holds all data
    needed for a complete spectrum, including spectrometer frequency and
    whether to calculate it as first-order or second-order.

    Convenience functions for how users will supply arguments? In particular,
    J couplings.
    J = [(7.1, 3), (1.0, 2)]
    J = {'J12': 7.1,
         'J13': 7.1,
         'J14': 1.0} and parse into a matrix?

    standardize nomenclature: signals vs peaks vs spectrum vs...
"""
