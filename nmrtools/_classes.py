"""This module provides classes to abstract NMR concepts such as spin systems
and spectra.
"""
import itertools
import numbers

import numpy as np

from nmrtools.firstorder import first_order, first_order_spin_system
from nmrtools.math import reduce_peaks
from nmrtools.qm import spectrum

from ._descriptors import Number, Couplings


# work in progress
# def spectrum_from_signals(signals):
#     spectrum = []
#     for signal in signals:
#         spectrum += multiplet(signals)
#     return reduce_peaks(spectrum)


# https://realpython.com/python-type-checking/
# https://blog.florimond.dev/reconciling-dataclasses-and-properties-in-python



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

    def __add__(self, other):
        if hasattr(other, 'peaklist') and callable(other.peaklist):
            return Spectrum([self, other])
        else:
            return NotImplemented

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



class SpinSystem:
    """Stub implementation of SpinSystem.
    Flesh out API (e.g. getters/setters; dunder methods) later."""
    def __init__(self, v, J, second_order=True):
        self._nuclei_number = len(v)
        self.v = v
        self.J = J
        self._second_order = second_order
        self._peaklist = self.peaklist()
        print(self._v, self._J, self._second_order, self.peaklist)

    @property
    def v(self):
        return self._v

    @v.setter
    def v(self, vlist):
        if len(vlist) != self._nuclei_number:
            raise ValueError('v length must match J shape.')
        if not isinstance(vlist[0], numbers.Real):
            raise TypeError('v must be an array of numbers.')
        self._v = vlist

    @property
    def J(self):
        return self._J

    @J.setter
    def J(self, J_array):
        J = np.array(J_array)
        m, n = J.shape
        if (m != n) or (m != self._nuclei_number):
            raise TypeError("J dimensions don't match v length.")
        if not np.allclose(J, J.T):
            raise ValueError('J must be diagonal-symmetric.')
        for i in range(m):
            if J[i, i] != 0:
                raise ValueError('Diagonal elements of J must be 0.')
        self._J = J_array

    @property
    def second_order(self):
        return self._second_order

    @second_order.setter
    def second_order(self, boolean):
        if isinstance(boolean, bool):
            self._second_order = boolean
        else:
            raise TypeError('second_order must be a boolean')

    def peaklist(self):
        if self._second_order:
            return spectrum(self._v, self._J)
        else:
            return first_order_spin_system(self._v, self._J)


class Spectrum:
    """Stub implementation of Spectrum.
    Flesh out API (e.g. getter/setters; dunder methods) later."""
    def __init__(self, components, mhz=300, limits=None):
        self.components = components[:]
        self.mhz = mhz
        peaklists = [c.peaklist() for c in self.components]
        peaklists_merged = itertools.chain.from_iterable(peaklists)
        self._peaklist = reduce_peaks(peaklists_merged)

    def peaklist(self):
        return self._peaklist
    # def __init__(self):
    #     pass
