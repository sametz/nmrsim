"""The dnmr module provides functions for calculating DNMR line shapes, and
classes to describe DNMR systems.

The dnmr module provides the following classes:

* DnmrTwoSinglets: for simulating the lineshape for two uncoupled nuclei
undergoing exchange.

* DnmrAB: for simulating the lineshape for two coupled nuclei undergoing
exchange (i.e. an AB (or AX) pattern at the slow exchange limit).

The dnmr module provides the following functions:


The function for computing the DNMR lineshape for two uncoupled nuclei is
taken from:
    Sandström, J. Dynamic NMR Spectroscopy; Academic Press: New York, 1982

and similar (non-Pythonic) variable names are used here.

The function for computing the DNMR lineshape for two coupled nuclei is
taken from:
    Brown, K.C.; Tyson, R.L.; Weil, J.A. J. Chem. Educ. 1998, 75, 1632
and the related, important correction:
    TODO: add reference to correction

TODO: complete documentation.
"""
import numpy as np
# TODO: think about naming the intensity-returning functions better. Private?


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
        """
        Calculate the intensity (y coordinate)
        Scheduled for refactoring.
        :param v: frequency
        :return: function that calculates the intensity at v
        """
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
    Create a the lineshape for a DNMR spectrum of two uncoupled spin-half
    nuclei.

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

    Returns
    -------
    x, y : numpy.array, numpy.array
        Arrays for the x (frequency) and y (intensity) lineshape data points.
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
    """ A DNMR simulation for two uncoupled nuclei undergoing exchange.

    TODO: finish the docs.

    """

    _pi = np.pi
    _pi_squared = _pi ** 2

    def __init__(self, va=1, vb=0, k=0.01, wa=0.5, wb=0.5, pa=0.5,
                 limits=None):
        """
        Initialize the system with the required parameters:
        :param va: Frequency of nucleus a
        :param vb: Frequency of nucleus b (must be < va)
        :param k: Rate of nuclear exchange
        :param wa: With at half height for va signal at the slow exchange limit
        :param wb: With at half height for vb signal at the slow exchange limit
        :param pa: Fractional population of state 'a'
        """
        # Idea is to complete the frequency-independent calculations when the
        #  class is instantiated, and thus calculations may be faster.
        if vb > va:
            va, vb = vb, va
            wa, wb = wb, wa
            pa = 1 - pa
        self._va = va
        self._vb = vb
        self._k = k
        self._wa = wa
        self._wb = wb
        self._pa = pa
        if limits:
            self.limits = limits
        else:
            self._vmin = min([va, vb]) - 50
            self._vmax = max([va, vb]) + 50

        self._set_T2a()
        self._set_T2b()
        self._set_pb()
        self._set_tau()
        self._set_dv()
        self._set_Dv()
        self._set_P()
        self._set_p()
        self._set_Q()
        self._set_R()
        self._set_r()

    def _set_T2a(self):
        self._T2a = 1 / (self._pi * self._wa)

    def _set_T2b(self):
        self._T2b = 1 / (self._pi * self._wb)

    def _set_pb(self):
        self._pb = 1 - self._pa

    def _set_tau(self):
        self._tau = self._pb / self._k

    def _set_dv(self):
        self._dv = self._va - self._vb

    def _set_Dv(self):
        self._Dv = (self._va + self._vb) / 2

    def _set_P(self):
        self._P = self._tau * (1 / (self._T2a * self._T2b) + self._pi_squared * (self._dv ** 2)) \
                  + (self._pa / self._T2a + self._pb / self._T2b)

    def _set_p(self):
        self._p = 1 + self._tau * ((self._pb / self._T2a) + (self._pa / self._T2b))

    def _set_Q(self):
        self._Q = self._tau * (- self._pi * self._dv * (self._pa - self._pb))

    def _set_R(self):
        self._R = self._pi * self._dv * self._tau * ((1 / self._T2b) - (1 / self._T2a)) \
                  + self._pi * self._dv * (self._pa - self._pb)

    def _set_r(self):
        self._r = 2 * self._pi * (1 + self._tau * ((1 / self._T2a) + (1 / self._T2b)))

    @property
    def va(self):
        return self._va

    @va.setter
    def va(self, value):
        self._va = value
        self._set_vab_dependencies()

    def _set_vab_dependencies(self):
        self._set_dv()
        self._set_Dv()
        self._set_P()
        self._set_Q()
        self._set_R()

    @property
    def vb(self):
        return self._vb

    @vb.setter
    def vb(self, value):
        self._vb = value
        self._set_vab_dependencies()

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, value):
        self._k = value
        self._set_tau()
        self._set_p()
        self._set_P()
        self._set_Q()
        self._set_R()
        self._set_r()

    @property
    def wa(self):
        return self._wa

    @wa.setter
    def wa(self, value):
        self._wa = value
        self._set_T2a()
        self._set_wab_dependencies()

    def _set_wab_dependencies(self):
        self._set_p()
        self._set_P()
        self._set_R()
        self._set_r()

    @property
    def wb(self):
        return self._wb

    @wb.setter
    def wb(self, value):
        self._wb = value
        self._set_T2b()
        self._set_wab_dependencies()

    @property
    def pa(self):
        return self._pa

    @pa.setter
    def pa(self, value):
        self._pa = value
        self._set_pb()
        self._set_tau()
        self._set_p()
        self._set_P()
        self._set_Q()
        self._set_R()
        self._set_r()

    @property
    def limits(self):
        return self._vmin, self._vmax

    @limits.setter
    def limits(self, limits):
        try:
            vmin, vmax = limits
            vmin = float(vmin)
            vmax = float(vmax)
        except Exception as e:
            print(e)
            print('limits must be a tuple of two numbers')
            raise

        if vmax < vmin:
            vmin, vmax = vmax, vmin
        self._vmin = vmin
        self._vmax = vmax

    def intensity(self, v):
        """
        Yield a function for the lineshape for TwoSinglets
        :param v: frequency
        :return: a frequency-dependent function that returns the intensity of
        the spectrum at frequency v
        """
        # TODO: add to docstring
        p = self._p
        Dv = self._Dv
        P = self._P
        Q = self._Q
        R = self._R
        r = self._r
        tau = self._tau
        Dv -= v
        P -= tau * 4 * self._pi_squared * (Dv ** 2)
        Q += tau * 2 * self._pi * Dv
        R += Dv * r
        return (P * p + Q * R) / (P ** 2 + R ** 2)

    def spectrum(self):
        """
        Calculate a DNMR spectrum, using the parameters TwoSinglets was
        instantiated with.
        :return: a tuple of numpy arrays (x = numpy linspace representing
        frequencies, y = numpy array of intensities along those frequencies)pwd
        """
        x = np.linspace(self._vmin, self._vmax, 800)
        y = self.intensity(x)

        return x, y


def dnmr_AB_func(v, v1, v2, J, k, w):
    """
    A translation of the equation from Weil's JCE paper (NOTE: Reich pointed
    out that it has a sign typo!).
    p. 14, for the uncoupled 2-site exchange simulation.

    Parameters
    ---------

    v : float or array-like
        a frequency (x coordinate) or array of frequencies at which an
        amplitude (y coordinate) is to be calculated.
    v1, v2 : float
        frequencies of a and b nuclei (at the slow exchange limit,
        in the absence of coupling; `va` > `vb`)
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

    Notes
    -----

    References
    ----------
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


def dnmr_AB(v1, v2, J, k, w):
    """
    TODO: finish the docs.
    Parameters
    ----------
    v1
    v2
    J
    k
    w

    Returns
    -------

    """
    if v2 > v1:
        v1, v2 = v2, v1
    l_limit = v2 - 50
    r_limit = v1 + 50
    x = np.linspace(l_limit, r_limit, 800)
    y = dnmr_AB_func(x, v1, v2, J, k, w)
    return x, y


class DnmrAB:
    """
    TODO: finish the docs.
    """

    _pi = np.pi
    _pi_squared = _pi ** 2
    
    def __init__(self, v1=165.0, v2=135.0, J=12.0, k=12.0, W=0.5,
                 limits=None):
        self._v1 = v1
        self._v2 = v2
        self._J = J
        self._k = k
        self._W = W
        if limits:
            self.limits = limits
        else:
            self._vmin = min([v1, v2]) - 50
            self._vmax = max([v1, v2]) + 50

        self._set_vo()
        self._set_tau()
        self._set_tau2()
        self._set_a2()
        self._set_a3()
        self._set_a4()
        self._set_s()

    def _set_vo(self):
        self._vo = (self._v1 + self._v2) / 2

    def _set_tau(self):
        self._tau = 1 / self._k

    def _set_tau2(self):
        self._tau2 = 1 / (self._pi * self._W)

    def _set_a2(self):
        self._a2 = - ((1 / self._tau) + (1 / self._tau2)) ** 2

    def _set_a3(self):
        self._a3 = - self._pi_squared * (self._v1 - self._v2) ** 2

    def _set_a4(self):
        self._a4 = - self._pi_squared * self._J ** 2 + (1 / self._tau ** 2)

    def _set_s(self):
        self._s = (2 / self._tau) + (1 / self._tau2)

    @property
    def v1(self):
        return self._v1

    @v1.setter
    def v1(self, value):
        self._v1 = value
        self._set_vo()
        self._set_a3()

    @property
    def v2(self):
        return self._v2

    @v2.setter
    def v2(self, value):
        self._v2 = value
        self._set_vo()
        self._set_a3()

    @property
    def J(self):
        return self._J

    @J.setter
    def J(self, value):
        self._J = value
        self._set_a4()

    @property
    def k(self):
        return self._k

    @k.setter
    def k(self, value):
        self._k = value
        self._set_tau()
        self._set_a2()
        self._set_a4()
        self._set_s()

    @property
    def W(self):
        return self._W

    @W.setter
    def W(self, value):
        self._W = value
        self._set_tau2()
        self._set_a2()
        self._set_s()

    @property
    def limits(self):
        return self._vmin, self._vmax

    @limits.setter
    def limits(self, limits):
        try:
            vmin, vmax = limits
            vmin = float(vmin)
            vmax = float(vmax)
        except Exception as e:
            print(e)
            print('limits must be a tuple of two numbers')
            raise

        if vmax < vmin:
            vmin, vmax = vmax, vmin
        self._vmin = vmin
        self._vmax = vmax

    def intensity(self, v):
        a1_plus = 4 * self._pi_squared * (self._vo - v + self._J / 2) ** 2
        a1_minus = 4 * self._pi_squared * (self._vo - v - self._J / 2) ** 2
        a_plus = a1_plus + self._a2 + self._a3 + self._a4
        a_minus = a1_minus + self._a2 + self._a3 + self._a4

        b_plus = 4 * self._pi * (self._vo - v + self._J / 2) * (
                (1 / self._tau) + (1 / self._tau2)) - 2 * self._pi * self._J / self._tau
        b_minus = 4 * self._pi * (self._vo - v - self._J / 2) * (
                (1 / self._tau) + (1 / self._tau2)) + 2 * self._pi * self._J / self._tau

        r_plus = 2 * self._pi * (self._vo - v + self._J)
        r_minus = 2 * self._pi * (self._vo - v - self._J)
        n1 = r_plus * b_plus - self._s * a_plus
        d1 = a_plus ** 2 + b_plus ** 2
        n2 = r_minus * b_minus - self._s * a_minus
        d2 = a_minus ** 2 + b_minus ** 2
        I = (n1 / d1) + (n2 / d2)
        return I

    def spectrum(self):
        x = np.linspace(self._vmin, self._vmax, 800)
        y = self.intensity(x)
        return x, y
