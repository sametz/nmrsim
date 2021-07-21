"""The `dnmr` module provides functions for calculating DNMR line shapes, and
classes to describe DNMR systems.

The dnmr module provides the following classes:

* `DnmrTwoSinglets`: a sumulation of the lineshape for two uncoupled nuclei
  undergoing exchange.
* `DnmrAB`: a simulation of the lineshape for two coupled nuclei undergoing
  exchange (i.e. an AB (or AX) pattern at the slow exchange limit).

The `dnmr` module provides the following functions:

* `dnmr_two_singlets`: for simulating the lineshape for two uncoupled nuclei
  undergoing exchange [3]_.
* `dnmr_AB` : for simulating the lineshape for two coupled nuclei undergoing
  exchange (i.e. an AB (or AX) pattern at the slow exchange limit) [4]_.

References
----------
.. [3] Sandström, J. Dynamic NMR Spectroscopy; Academic Press: New York, 1982.
.. [4] a) Brown, K.C.; Tyson, R.L.; Weil, J.A. J. Chem. Educ. 1998, 75, 1632.
   b) an important math correction to the previous reference:

    TODO: add reference to correction

"""

import numpy as np

from nmrsim._utils import is_number, is_decimal_fraction, is_tuple_of_two_numbers, is_positive, is_integer


def _dnmr_two_singlets_func(va, vb, ka, wa, wb, pa):
    """
    Create a function that requires only frequency as an argurment, for
    calculating the lineshape of a DNMR spectrum for two uncoupled spin-half
    nuclei.

    This allows the expressions that are independent of frequency to be
    calculated only once, outside the returned function. The returned function
    can then be applied to a list of frequency (x) coordinates (e.g. a numpy
    linspace) to provide a list of the corresponding intensity (y) coordinates.

    Parameters
    ----------
    va : int or float
        The frequency (Hz) of nucleus 'a' at the slow exchange limit. va > vb
    vb : int or float
        The frequency (Hz) of nucleus 'b' at the slow exchange limit. vb < va
    ka : int or float
        The rate constant (Hz) for state a--> state b
    wa : int or float
        The width at half height of the signal for nucleus a (at the slow
        exchange limit).
    wb : int or float
        The width at half height of the signal for nucleus b (at the slow
        exchange limit).
    pa : float (0 <= pa <= 1)
        The fraction of the population in state a.

    Returns
    -------
    _maker: function

     Notes
    -----
    The nmrsim.dnmr module gives a reference for the algorithm used here.

    """
    pi = np.pi
    pi_squared = pi ** 2
    T2a = 1 / (pi * wa)
    T2b = 1 / (pi * wb)
    pb = 1 - pa
    tau = pb / ka
    dv = va - vb
    Dv = (va + vb) / 2
    P = tau * (1 / (T2a * T2b) + pi_squared * (dv ** 2)) + (pa / T2a + pb / T2b)
    p = 1 + tau * ((pb / T2a) + (pa / T2b))
    Q = tau * (- pi * dv * (pa - pb))
    R = pi * dv * tau * ((1 / T2b) - (1 / T2a)) + pi * dv * (pa - pb)
    r = 2 * pi * (1 + tau * ((1 / T2a) + (1 / T2b)))

    def _maker(v):
        """Calculate the intensity (y coordinate) at a given frequency v(x coordinate)."""
        # TODO: fix docstring, explain _P _Q etc correlate to P, Q etc in lit.
        # FIXED: previous version of this function used
        # nonlocal Dv, P, Q, R
        # but apparently when function called repeatedly these values would
        # become corrupted (turning into arrays?!)
        # Solution: add underscores to create copies of any variables in
        # outer scope whose values are changed in the inner scope.

        _Dv = Dv - v
        _P = P - tau * 4 * pi_squared * (_Dv ** 2)
        _Q = Q + tau * 2 * pi * _Dv
        _R = R + _Dv * r
        return(_P * p + _Q * _R) / (_P ** 2 + _R ** 2)
    return _maker


def dnmr_two_singlets(va, vb, ka, wa, wb, pa, limits=None, points=800):
    """
    Create a the lineshape for a DNMR spectrum of two uncoupled spin-half nuclei.

    Parameters
    ----------
    va, vb : int or float
        The frequencies (Hz) of nuclei 'a' and 'b' at the slow exchange limit.
    ka : int or float
        The rate constant (Hz) for state a--> state b
    wa, wb : int or float
        The peak widths at half height for the 'a' and 'b' singlets at the
        slow-exchange limit.
    pa : float (0 <= pa <= 1)
        The fraction of the population in state a
    limits : (int or float, int or float), optional
        The minimum and maximum frequencies (in any order) for the simulation.
    points : int
        The length of the returned arrays (i.e. the number of points plotted).

    Returns
    -------
    x, y : numpy.array, numpy.array
        Arrays for the x (frequency) and y (intensity) lineshape data points.

    See Also
    --------
    DnmrTwoSinglets : A class representation for this simulation.

    References
    ----------
    See the documentation for the nmrsim.dnmr module.

    """
    if vb > va:
        va, vb = vb, va
        wa, wb = wb, wa
        pa = 1 - pa
    if limits:
        l_limit = min(limits)
        r_limit = max(limits)
    else:
        l_limit = vb - 50
        r_limit = va + 50
    x = np.linspace(l_limit, r_limit, points)
    func = _dnmr_two_singlets_func(va, vb, ka, wa, wb, pa)
    y = func(x)
    return x, y


class DnmrTwoSinglets:
    """
    A DNMR simulation for two uncoupled nuclei undergoing exchange.

    Parameters
    ----------
    va, vb : int or float
        The frequencies (Hz) of nuclei 'a' and 'b' at the slow exchange limit.
    k : int or float
        The rate constant (Hz) for state a--> state b
    wa, wb : int or float
        The peak widths at half height for the 'a' and 'b' singlets at the
        slow-exchange limit.
    pa : float (0 <= pa <= 1)
        The fraction of the population in state a
    limits : (int or float, int or float), optional
        The minimum and maximum frequencies (in any order) for the simulation.
    points : int
        The length of the returned arrays (i.e. the number of points plotted).

    See Also
    --------
    DnmrTwoSinglets : A class representation for this simulation

    """

    def __init__(self, va=1, vb=0, k=0.01, wa=0.5, wb=0.5, pa=0.5,
                 limits=None, points=800):
        # rethink default kwargs for v/k/w

        self.va = va
        self.vb = vb
        self.k = k
        self.wa = wa
        self.wb = wb
        self.pa = pa
        if limits:
            self.limits = limits
        else:
            self._vmin = min([va, vb]) - 50
            self._vmax = max([va, vb]) + 50
        self.points = points

    @property
    def va(self):
        """
        The frequency of nucleus "a" (Hz) at the slow-exchange limit.

        Returns
        -------
        int or float

        """
        return self._va

    @va.setter
    def va(self, value):
        self._va = is_number(value)

    @property
    def vb(self):
        """
        The frequency of nucleus "b" (Hz) at the slow-exchange limit.

        Returns
        -------
        int or float

        """
        return self._vb

    @vb.setter
    def vb(self, value):
        self._vb = is_number(value)

    @property
    def k(self):
        """
        The rate constant (Hz) for state A--> state B (must be >0).

        Returns
        -------
        int or float

        """
        return self._k

    @k.setter
    def k(self, value):
        self._k = is_positive(value)

    @property
    def wa(self):
        """
        The peak width at half height (Hz) for the 'a' singlet at the
        slow-exchange limit.

        Returns
        -------
        int or float

        """
        return self._wa

    @wa.setter
    def wa(self, value):
        self._wa = is_number(value)

    @property
    def wb(self):
        """
        The peak width at half height (Hz) for the 'b' singlet at the
        slow-exchange limit.

        Returns
        -------
        int or float

        """
        return self._wb

    @wb.setter
    def wb(self, value):
        self._wb = is_number(value)

    @property
    def pa(self):
        """
        The fraction of the population in state a. Must be >=0 and <=1.

        Returns
        -------
        float

        """
        return self._pa

    @pa.setter
    def pa(self, value):
        self._pa = is_decimal_fraction(value)

    @property
    def limits(self):
        """
        The minimum and maximum frequencies for the simulated lineshape.

        Returns
        -------
        (int or float, int or float)

        """
        return self._vmin, self._vmax

    @limits.setter
    def limits(self, limits):
        limits = is_tuple_of_two_numbers(limits)
        self._vmin = min(limits)
        self._vmax = max(limits)

    @property
    def points(self):
        """
        The length of the returned arrays (i.e. the number of points plotted).

        Returns
        -------
        int

        """
        return self._points

    @points.setter
    def points(self, value):
        self._points = is_integer(value)

    def lineshape(self):
        """
        Calculate and return the lineshape for the DNMR spectrum.

        Returns
        -------
        x, y : numpy.array, numpy.array
            Arrays for the x (frequency) and y (intensity) lineshape data
            points.

        """
        x = np.linspace(self._vmin, self._vmax, self.points)
        x, y = dnmr_two_singlets(self.va, self.vb, self.k, self.wa, self.wb, self.pa,
                                 limits=self.limits, points=self.points)
        return x, y


def _dnmr_AB_func(v, v1, v2, J, k, w):
    """
    Implement the equation from Weil et al for simulation of the DNMR lineshape
    for two coupled nuclei undergoing exchange (AB or AX pattern at the
    slow-exchange limit).

    Parameters
    ----------
    v : float or array-like
        a frequency (x coordinate) or array of frequencies at which an
        amplitude (y coordinate) is to be calculated.
    v1, v2 : float
        frequencies of a and b nuclei (at the slow exchange limit,
        in the absence of coupling)
    J : float
        the coupling constant between the two nuclei.
    k : float
        rate constant for state A--> state B
    w : float
        peak widths at half height (slow exchange limit).

    Returns
    -------
    float
        amplitude at frequency `v`.

    See Also
    --------
    DnmrAB : A class representation for this simulation.

    References
    ----------
    See the documentation for the nmrsim.dnmr module.

    """
    pi = np.pi
    vo = (v1 + v2) / 2
    tau = 1 / k
    tau2 = 1 / (pi * w)
    a1_plus = 4 * pi ** 2 * (vo - v + J / 2) ** 2
    a1_minus = 4 * pi ** 2 * (vo - v - J / 2) ** 2
    a2 = - ((1 / tau) + (1 / tau2)) ** 2
    a3 = - pi ** 2 * (v1 - v2) ** 2
    a4 = - pi ** 2 * J ** 2 + (1 / tau ** 2)
    a_plus = a1_plus + a2 + a3 + a4
    a_minus = a1_minus + a2 + a3 + a4

    b_plus = 4 * pi * (vo - v + J / 2) * (
        (1 / tau) + (1 / tau2)) - 2 * pi * J / tau
    b_minus = 4 * pi * (vo - v - J / 2) * (
        (1 / tau) + (1 / tau2)) + 2 * pi * J / tau

    r_plus = 2 * pi * (vo - v + J)
    r_minus = 2 * pi * (vo - v - J)

    s = (2 / tau) + (1 / tau2)

    n1 = r_plus * b_plus - s * a_plus
    d1 = a_plus ** 2 + b_plus ** 2
    n2 = r_minus * b_minus - s * a_minus
    d2 = a_minus ** 2 + b_minus ** 2

    I = (n1 / d1) + (n2 / d2)
    return I


def dnmr_AB(va, vb, J, k, w, limits=None, points=800):
    """
    Simulate the DNMR lineshape for two coupled nuclei undergoing exchange
    (AB or AX pattern at the slow-exchange limit).

    Parameters
    ---------
    va, vb : float
        frequencies of a and b nuclei (at the slow exchange limit,
        in the absence of coupling)
    J : float
        the coupling constant between the two nuclei.
    k : float
        rate constant for state A--> state B
    w : float
        peak widths at half height (at the slow-exchange limit).
    limits : (int or float, int or float), optional
        The minimum and maximum frequencies (in any order) for the simulation.
    points : int
        The length of the returned arrays (i.e. the number of points plotted).

    Returns
    -------
    x, y : numpy.array, numpy.array
        Arrays for the x (frequency) and y (intensity) lineshape data points.

    See Also
    --------
    DnmrAB : A class representation for this simulation.

    References
    ----------
    See the documentation for the nmrsim.dnmr module.

    """
    if limits:
        l_limit = min(limits)
        r_limit = max(limits)
    else:
        l_limit = min(va, vb) - 50
        r_limit = max(va, vb) + 50
    x = np.linspace(l_limit, r_limit, points)
    y = _dnmr_AB_func(x, va, vb, J, k, w)
    return x, y


class DnmrAB:
    """
    Simulate the DNMR lineshape for two coupled nuclei undergoing exchange
    (AB or AX pattern at the slow-exchange limit).

    Parameters
    ----------
    va, vb : int or float
        frequencies of a and b nuclei (at the slow exchange limit,
        in the absence of coupling)
    J : int or float
        the coupling constant between the two nuclei.
    k : int or float
        rate constant for state A--> state B
    w : int or float
        peak widths at half height (at the slow-exchange limit).
    limits : (int or float, int or float), optional
        The minimum and maximum frequencies (in any order) for the simulation.
    points : int
        The length of the returned arrays (i.e. the number of points plotted).

    See Also
    --------
    DnmrAB : A class representation for this simulation.

    References
    ----------
    See the documentation for the nmrsim.dnmr module.

    """

    def __init__(self, va=165.0, vb=135.0, J=12.0, k=12.0, w=0.5,
                 limits=None, points=800):
        self.va = va
        self.vb = vb
        self.J = J
        self.k = k
        self.w = w
        if limits:
            self.limits = limits
        else:
            self._vmin = min([va, vb]) - 50
            self._vmax = max([va, vb]) + 50
        self.points = points

    @property
    def va(self):
        """
        The frequency of nucleus "a" (Hz) at the slow-exchange limit, in the absence of coupling.

        Returns
        -------
        int or float

        """
        return self._va

    @va.setter
    def va(self, value):
        self._va = is_number(value)

    @property
    def vb(self):
        """
        The frequency of nucleus "b" (Hz) at the slow-exchange limit, in the absence of coupling.

        Returns
        -------
        int or float

        """
        return self._vb

    @vb.setter
    def vb(self, value):
        self._vb = is_number(value)

    @property
    def J(self):
        """
        The coupling constant (Hz) between the two nuclei.

        Returns
        -------
        int or float

        """
        return self._J

    @J.setter
    def J(self, value):
        self._J = is_number(value)

    @property
    def k(self):
        """
        The rate constant (Hz) for state A--> state B (must be >0).

        Returns
        -------
        int or float

        """
        return self._k

    @k.setter
    def k(self, value):
        self._k = is_positive(value)

    @property
    def w(self):
        """
        The peak width (Hz) at half height (at the slow-exchange limit).

        Returns
        -------
        int or float

        """
        return self._w

    @w.setter
    def w(self, value):
        self._w = is_number(value)

    @property
    def limits(self):
        """
        Give minimum and maximum frequencies for the simulated lineshape.

        Returns
        -------
        (int or float, int or float)

        """
        return self._vmin, self._vmax

    @limits.setter
    def limits(self, limits):
        limits = is_tuple_of_two_numbers(limits)
        self._vmin = min(limits)
        self._vmax = max(limits)

    @property
    def points(self):
        """
        Give the length of the returned arrays (i.e. the number of points plotted).

        Returns
        -------
        int

        """
        return self._points

    @points.setter
    def points(self, value):
        self._points = is_integer(value)

    def lineshape(self):
        """Return the x, y lineshape data for the simulation.

        Returns
        -------
        x, y : numpy.array, numpy.array
            Arrays for the x (frequency) and y (intensity) lineshape data
            points.
        """
        x = np.linspace(self._vmin, self._vmax, self.points)
        y = _dnmr_AB_func(x, self.va, self.vb, self.J, self.k, self.w)
        return x, y
