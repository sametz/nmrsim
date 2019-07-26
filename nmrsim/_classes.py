"""This module provides high-level API classes for abstract NMR concepts such
as spin systems and spectra.
"""
import itertools
import numbers

import numpy as np

from nmrsim.firstorder import first_order_spin_system, multiplet
from nmrsim.math import reduce_peaks
from nmrsim.qm import qm_spinsystem

from ._descriptors import Number, Couplings

# TODO: Multiplet "belongs" in the nmrsim.firstorder namespace, but has a
# cyclic dependency with Spectrum and multiplet. Until a solution found, have
# to define both in the same module.


class Multiplet:
    """A representation of a first-order multiplet.

    Attributes
    ----------
    v : float or int
        The frequency of the center of the multiplet.
    I : float or int
        The total intensity ("integration") of the multiplet.
    J : 2D array-like, e.g. [(int or float, int)...] for [(J, # of nuclei)...].
        For example, a td, *J* = 7.0, 2.5 Hz would have:
        J = [(7.0, 2), (2.5, 1)].
    """
    v = Number()
    I = Number()
    J = Couplings()

    def __init__(self, v, I, J):
        self.v = v
        self.I = I
        self.J = J
        self._peaklist = multiplet((v, I), J)

    def __eq__(self, other):
        if hasattr(other, 'peaklist') and callable(other.peaklist):
            return np.allclose(self.peaklist(), other.peaklist())

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

    def __truediv__(self, scalar):
        return self.__mul__(1 / scalar)

    def __itruediv__(self, scalar):
        return self.__imul__(1 / scalar)

    def _refresh(self):
        self._peaklist = multiplet((self.v, self.I), self.J)

    def peaklist(self):
        """Return a list of (frequency, intensity) signals.

        Returns
        -------
        [(float, float)...]
            Array of (frequency, intensity) signals.
        """
        self._refresh()
        return self._peaklist


class SpinSystem:
    """Stub implementation of SpinSystem.
    Flesh out API (e.g. getters/setters; dunder methods) later."""
    def __init__(self, v, J, second_order=True):
        self._nuclei_number = len(v)
        self.v = v
        self.J = J
        self._second_order = second_order
        self._peaklist = self.peaklist()

    @property
    def v(self):
        """An array of the frequency of each nucleus (in the absence of
        coupling).

        Returns
        -------
        float or int
        """
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
        """A 2D array of coupling constants.

        J[m][n] corresponds to the coupling between the nuclei of frequencies
        v[m] and v[n].

        Returns
        -------
        np.array
            The array of coupling constants.
        """
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
        """Whether the spin system should use second-order simulation (instead
        of a first-order simulation). If False, will perform a first-order
        calculation instead.

        Returns
        -------
        bool
            Whether the spin system should use a second-order simulation
            (instead of a first-order simulation).
        """
        return self._second_order

    @second_order.setter
    def second_order(self, boolean):
        if isinstance(boolean, bool):
            self._second_order = boolean
        else:
            raise TypeError('second_order must be a boolean')

    def peaklist(self):
        """Return a list of (frequency, intensity) signals.

        Returns
        -------
        [(float, float)...]
            Array of (frequency, intensity) signals.
        """
        if self._second_order:
            return qm_spinsystem(self._v, self._J)
        else:
            return first_order_spin_system(self._v, self._J)

    def __eq__(self, other):
        if hasattr(other, 'peaklist') and callable(other.peaklist):
            return np.allclose(self.peaklist(), other.peaklist())

    def __add__(self, other):
        if hasattr(other, 'peaklist') and callable(other.peaklist):
            return Spectrum([self, other])
        else:
            return NotImplemented


class Spectrum:
    """A collection of spectral features (SpinSystem; Multiplet).

    Flesh out API (e.g. getter/setters; dunder methods) later."""
    def __init__(self, components):
        self._components = components[:]
        peaklists = [c.peaklist() for c in self._components]
        peaklists_merged = itertools.chain.from_iterable(peaklists)
        self._peaklist = reduce_peaks(peaklists_merged)

    def _add_peaklist(self, other):
        self._peaklist = reduce_peaks(
            itertools.chain(self._peaklist, other.peaklist()))

    def __eq__(self, other):
        if hasattr(other, 'peaklist') and callable(other.peaklist):
            return np.allclose(self.peaklist(), other.peaklist())

    def __add__(self, other):
        if hasattr(other, 'peaklist') and callable(other.peaklist):
            self._add_peaklist(other)
            self._components.append(other)
            return self
        else:
            return NotImplemented

    def peaklist(self):
        """Return the peaklist for the spectrum.

        Returns
        -------
        [(float, float)...]
            Array of (frequency, intensity) signals.
        """
        return self._peaklist
