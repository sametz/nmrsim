"""
This module provides high-level API classes for abstract NMR concepts such as spin systems and spectra.
"""
import itertools
import numbers

import numpy as np

from nmrsim.firstorder import first_order_spin_system, multiplet
from nmrsim.math import reduce_peaks, add_lorentzians
from nmrsim.qm import qm_spinsystem
from nmrsim._utils import low_high

from ._descriptors import Number, Couplings

# TODO: Multiplet "belongs" in the nmrsim.firstorder namespace, but has a
# cyclic dependency with Spectrum and multiplet. Until a solution found, have
# to define both in the same module.
# TODO: consider a base class that has common features such as .peaklist()


class Multiplet:
    """
    A representation of a first-order multiplet.

    Parameters
    ----------
    v : float or int
        The frequency of the center of the multiplet.
    I : float or int
        The total intensity ("integration") of the multiplet.
    J : 2D array-like, e.g. [(int or float, int)...] for [(J, # of nuclei)...].
        For example, a td, *J* = 7.0, 2.5 Hz would have:
        J = [(7.0, 2), (2.5, 1)].
    w : float or int (optional)
        the peak width at half-height. Currently only used when Multiplet is a
        component of a nmrsim.Spectrum object.

    Attributes
    ----------
    v : float or int
        The frequency of the center of the multiplet.
    I : float or int
        The total intensity ("integration") of the multiplet.
    J : 2D array-like, e.g. [(int or float, int)...] for [(J, # of nuclei)...].
        For example, a td, *J* = 7.0, 2.5 Hz would have:
        J = [(7.0, 2), (2.5, 1)].
    w : float or int (optional)
        the peak width at half-height. Currently only used when Multiplet is a
        component of a nmrsim.Spectrum object.

    Notes
    -----
    Multiplet equates equal to other objects if their peaklists are identical.

    Multiplets can be added to other Multiplet/SpinSystem/Spectrum objects.
    Addition returns a Spectrum object with the Multiplet as a component.

    Multiplets can be multiplied by a scalar to scale their intensities and
    return a new Multiplet object, or be multiplied in-place to modify and
    return `self`.

    Similarly, multiplets can be divided or divided in-place by a scalar.

    """
    v = Number()
    I = Number()
    J = Couplings()

    def __init__(self, v, I, J, w=0.5):
        self.v = v
        self.I = I
        self.J = J
        self.w = w
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
        """
        Return a peaklist for the multiplet.

        Returns
        -------
        [(float, float)...]
            List of (frequency, intensity) peaks.
        """
        self._refresh()
        return self._peaklist


class SpinSystem:
    """
    A representation of an NMR spectrum composed of one or more
    Multiplet/SpinSystem components.

    Parameters
    ----------
    v : [float or int...]
        a list of *n* nuclei frequencies in Hz
    J : 2D array-like
        An *n, n* array of couplings in Hz. The order of nuclei in the list
        corresponds to the column and row order in the matrix, e.g.
        couplings[0][1] and [1]0] are the J coupling between the nuclei of
        freqs[0] and freqs[1].
    w : float or int (optional, default = 0.5)
        the peak width (in Hz) at half-height.
        Currently only used when SpinSystem is a component of a nmrsim.Spectrum
        object.
    second_order : bool (optional, default = True)
        Whether the SpinSystem should be simulated as second-order. If false,
        a first-order simulation will be used.

    Attributes
    ----------
    v
    J
    w : float or int (optional, default = 0.5)
        the peak width (in Hz) at half-height.
        Currently only used when SpinSystem is a component of a nmrsim.Spectrum
        object.
    second_order

    Notes
    -----
    SpinSystem equates equal to other objects if their peaklists are identical.

    SpinSystem objects can be added to Multiplet/SpinSystem/Spectrum objects.
    Addition returns a Spectrum object with the SpinSystem as a component.

    """
    def __init__(self, v, J, w=0.5, second_order=True):
        self._nuclei_number = len(v)
        self.v = v
        self.J = J
        self.w = w
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
    """
    A collection of spectral features (SpinSystem; Multiplet).

    Parameters
    ----------
    components : list
        A list of Multiplet and/or SpinSystem objects that the Spectrum is
        composed from.
    vmin, vmax : float or int (optional)
        The minimum and maximum frequencies for the Spectrum window.
        The defaults give a 50 Hz margin from the lowest- and highest-
        frequency peaks in the spectrum, respectively.

    Attributes
    ----------
    vmin, vmax : float or int (optional)
        The minimum and maximum frequencies for the Spectrum window.

    Notes
    -----
    Spectrum objects can be added to other Multiplet/SpinSystem/Spectrum
    objects, to return a new Spectrum object. In-place addition is also
    recognized, modifying the Spectrum object in-place and returning self.
    """
    def __init__(self, components, vmin=None, vmax=None):
        combo = [extract_components(c) for c in components]
        result = list(itertools.chain.from_iterable(combo))
        self._components = result
        peaklists = [c.peaklist() for c in self._components]
        peaklists_merged = itertools.chain.from_iterable(peaklists)
        self._peaklist = sorted(reduce_peaks(peaklists_merged))
        if vmin is None:  # Must compare to None so 0 is a valid input
            self._reset_vmin()
        else:
            self.vmin = vmin
        if vmax is None:
            self._reset_vmax()
        else:
            self.vmax = vmax

    def _add_peaklist(self, other):
        self._peaklist = sorted(reduce_peaks(
            itertools.chain(self._peaklist, other.peaklist())))
        self._reset_minmax()

    def _reset_minmax(self):
        self._reset_vmin()
        self._reset_vmax()

    def _reset_vmin(self):
        self.vmin = min(self._peaklist)[0] - 50

    def _reset_vmax(self):
        self.vmax = max(self._peaklist)[0] + 50

    def default_limits(self):
        """Reset vmin and vmax to defaults.

        Returns
        -------
        float or int, float or int
            Spectrum.vmin, Spectrum.vmax
        """
        self._reset_minmax()
        return self.vmin, self.vmax

    def __eq__(self, other):
        if hasattr(other, 'peaklist') and callable(other.peaklist):
            return np.allclose(self.peaklist(), other.peaklist())

    def __add__(self, other):
        new_spectrum = Spectrum(self._components[:], vmin=self.vmin, vmax=self.vmax)
        new_spectrum += other
        return new_spectrum

    def __iadd__(self, other):
        if hasattr(other, 'peaklist') and callable(other.peaklist):
            if isinstance(other, Spectrum):
                for component in other._components:
                    self.__iadd__(component)
            else:
                self._add_peaklist(other)
                self._components.append(other)
            return self
        else:
            raise TypeError('Item being added to Spectrum object not compatible')

    def peaklist(self):
        """Return the peaklist for the spectrum.

        Returns
        -------
        [(float, float)...]
            Array of (frequency, intensity) signals.
        """
        return self._peaklist

    def lineshape(self, points=800):
        """Return the x and y arrays for the spectrum's lineshape.

        Returns
        -------
        [float...], [float...]
            a tuple of x array, y array.
        """
        vmin, vmax = low_high((self.vmin, self.vmax))
        x = np.linspace(vmin, vmax, points)
        y = [add_lorentzians(x, c.peaklist(), c.w)
             for c in self._components]
        y_sum = np.sum(y, 0)
        return x, y_sum


def extract_components(nmr_object, clist=None):
    """
    Flatten the list of components comprising an nmrsim object.

    An nmrsim.Spectrum can be composed from "atomic" objects such as nmrsim.Multiplet or nmrsim.SpinSystem, or from
    other Spectrum objects. This function recursively un-nests any Spectrum sub-components to return a list of
    atomic objects.

    Parameters
    ----------
    nmr_object : nmrsim class
        the nmrsim subclass to be parsed
    clist : [obj...]
        the list of atomic objects being compiled

    Returns
    -------
    [obj...]

    """
    if clist is None:
        clist = []
    if isinstance(nmr_object, Spectrum):
        for c in nmr_object._components:
            extract_components(c, clist)
    else:
        clist.append(nmr_object)
    return clist
